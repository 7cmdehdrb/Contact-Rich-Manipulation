"""Simple runtime, flattened from the effective v0_2_2_4 lifecycle."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class _RecentEpisodeOutcomes:
    """Bounded episode ring; columns are success and timeout, including early failures."""

    def __init__(self, capacity, device):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('timeout_metric_window must be a positive integer')
        self.capacity = capacity
        self.buffer = torch.zeros((capacity, 2), dtype=torch.long, device=device)
        self.totals = torch.zeros(2, dtype=torch.long, device=device)
        self.cursor = 0
        self.filled = 0

    def append(self, outcomes):
        count = len(outcomes)
        if count == 0:
            return
        if count >= self.capacity:
            self.buffer[:] = outcomes[-self.capacity:]
            self.totals[:] = self.buffer.sum(0)
            self.cursor = 0
            self.filled = self.capacity
            return
        slots = (torch.arange(count, device=self.buffer.device) + self.cursor) % self.capacity
        self.totals += outcomes.sum(0) - self.buffer[slots].sum(0)
        self.buffer[slots] = outcomes
        self.cursor = (self.cursor + count) % self.capacity
        self.filled = min(self.capacity, self.filled + count)


class SweepSimpleEnv(ManagerBasedRLEnv):
    """Independent Fast runtime with bounded timeout metrics."""

    _mean_metrics = ('target_goal_distance_xy_m', 'eef_linear_speed_m_s',
                     'target_linear_speed_m_s', 'target_within_goal_radius_fraction')

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._timeout_outcomes = _RecentEpisodeOutcomes(cfg.timeout_metric_window, cfg.sim.device)
        # Preserve IK runtime configuration; Holding rewards have no EEF height gate.
        cfg.contact_height_limit = cfg.goal_reward_height_limit
        count, device = cfg.scene.num_envs, cfg.sim.device
        fast = getattr(cfg, 'reset_ik_backend', None) == 'warp'
        if fast and (cfg.max_obstacles != 6 or cfg.obstacle_observation_slots != 16):
            raise ValueError('Fast reset requires six obstacles and sixteen observation slots')
        if not fast and (cfg.max_gray_obstacles, cfg.max_white_obstacles, cfg.obstacle_observation_slots) != (3, 3, 16):
            raise ValueError('Wrench Refactor requires Gray3/White3 capacity and 16 observation slots')
        self.learning_iteration = 0
        self.target_id = torch.zeros((count, 1), device=device, dtype=torch.long)
        self.target_width = torch.full((count, 1), .04, device=device)
        self.sweep_dir = torch.zeros((count, 3), device=device)
        self.sensor_data_fresh = torch.zeros(count, device=device, dtype=torch.bool)
        self._active_object_mask = torch.zeros((count, 17), device=device, dtype=torch.bool)
        self._active_object_mask[:, 0] = True
        self._initial_obstacle_observation = torch.zeros((count, 64), device=device)
        self._initial_object_positions_w = torch.zeros((count, 17, 3), device=device)
        self._revision_spawn_positions = torch.zeros_like(self._initial_object_positions_w)
        self._final_actual_population = torch.zeros((count,) if fast else (count, 2), device=device, dtype=torch.long)
        self._revision_layout_complete = torch.zeros(count, device=device, dtype=torch.bool)
        self._revision_layout_rng = torch.Generator(device=device).manual_seed(int(cfg.seed or 0))
        self._metric_sums = torch.zeros((count, 4), device=device)
        self._metric_counts = torch.zeros_like(self._metric_sums)
        self._metric_peaks = torch.zeros((count, 2), device=device)
        self._metric_final_distance = torch.zeros(count, device=device)
        self._metric_min_distance = torch.full((count,), float('inf'), device=device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def load_managers(self):
        shelf = self.scene['shelf'].data
        self._revision_initial_shelf_pos = shelf.root_pos_w.clone()
        self._revision_initial_shelf_quat = shelf.root_quat_w.clone()
        self._metric_eef_body = self.scene['robot'].body_names.index('axia80_link')
        super().load_managers()
        limits = getattr(self.cfg, 'wrist_wrench_observation_limits', None)
        if limits is not None:
            bounds = torch.as_tensor(limits, dtype=torch.float32, device=self.device)
            if bounds.shape != (6,) or not bool((torch.isfinite(bounds) & (bounds > 0)).all()):
                raise ValueError('Wrist wrench observation limits must be six positive finite values')
            # ObservationManager owns a deep copy of the serialized config.
            # Bind device tensors only on its live terms, leaving PureWrenchReader
            # and direct reward/telemetry reads untouched. Built-in clip runs
            # after noise and also runs when observation corruption is disabled.
            for group, terms in self.observation_manager.active_terms.items():
                if 'wrist_wrench' in terms:
                    term = getattr(getattr(self.observation_manager.cfg, group), 'wrist_wrench')
                    term.clip = (-bounds, bounds)

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            self._goal_reset_seed = seed + 1707
            if hasattr(self, '_goal_reset_rng'):
                self._goal_reset_rng.manual_seed(self._goal_reset_seed)
        if seed is not None:
            self._revision_layout_rng.manual_seed(seed)
            if getattr(self.cfg.events, 'robot_initial_pose', None) is not None:
                self._goal_reset_seed = seed + 1707
                if hasattr(self, '_goal_reset_rng'):
                    self._goal_reset_rng.manual_seed(self._goal_reset_seed)
        result = super().reset(seed=seed, env_ids=env_ids, options=options)
        return result

    def step(self, action):
        self.sensor_data_fresh[:] = True
        result = super().step(action)
        return result


    def _update_initial_ik_bounds(self, ids, target_local, default_local, shelf_pos, shelf_quat,
                                  front_clearance, lateral_offset, height_jitter):
        """Draw both reset sampling boxes as non-physical 1 mm wire edges."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.utils.math import quat_apply

        if not hasattr(self, '_initial_ik_bounds_marker'):
            self._initial_ik_bounds_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path='/Visuals/InspireWrenchRefactor/initial_ik_bounds',
                markers={name: sim_utils.CuboidCfg(
                    size=(1., 1., 1.), rigid_props=None, collision_props=None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ) for name, color in [('right', (0.1, 0.8, 1.0)), ('left', (1.0, 0.5, 0.1))]},
            ))
            self._ik_bounds_positions = torch.zeros((self.num_envs, 24, 3), device=self.device)
            self._ik_bounds_rotations = torch.zeros((self.num_envs, 24, 4), device=self.device)
            self._ik_bounds_scales = torch.zeros((self.num_envs, 24, 3), device=self.device)
            self._ik_bounds_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        size = target_local.new_tensor((front_clearance[1] - front_clearance[0],
                                       lateral_offset[1] - lateral_offset[0], 2 * height_jitter))
        positions, scales = [], []
        for side in (1., -1.):
            center = default_local.clone()
            center[:, 0] = .20 + sum(front_clearance) / 2
            center[:, 1] = target_local[:, 1] + side * sum(lateral_offset) / 2
            for axis in range(3):
                other = [i for i in range(3) if i != axis]
                for a, b in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                    edge = center.clone()
                    edge[:, other[0]] += a * size[other[0]] / 2
                    edge[:, other[1]] += b * size[other[1]] / 2
                    positions.append(shelf_pos + quat_apply(shelf_quat, edge))
                    scale = size.new_full((3,), .001)
                    scale[axis] = size[axis].clamp_min(.001)
                    scales.append(scale)
        self._ik_bounds_positions[ids] = torch.stack(positions, dim=1)
        self._ik_bounds_rotations[ids] = shelf_quat[:, None, :]
        self._ik_bounds_scales[ids] = torch.stack(scales)
        self._ik_bounds_active[ids] = True
        active = self._ik_bounds_active
        indices = torch.arange(24, device=self.device).div(12, rounding_mode='floor')
        self._initial_ik_bounds_marker.visualize(
            translations=self._ik_bounds_positions[active].reshape(-1, 3),
            orientations=self._ik_bounds_rotations[active].reshape(-1, 4),
            scales=self._ik_bounds_scales[active].reshape(-1, 3),
            marker_indices=indices.repeat(int(active.sum())),
        )

    def set_learning_iteration(self, iteration):
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError('Learning iteration must be a nonnegative integer')
        self.learning_iteration = iteration
        self.curriculum_manager.compute(env_ids=None)

    def record_goal_accuracy_metrics(self):
        """One device-only sample per policy step, including the terminal state.

        Read existing pose/velocity buffers; no acceleration, sensor readback,
        CPU transfer, or GPU-to-host scalar synchronization in the step path.
        """
        data = self.scene['object_collection'].data
        # This environment always uses slot zero as its target.
        robot = self.scene['robot']
        body = self._metric_eef_body
        offset = self.scene['ee_frame'].data.target_pos_w[:, 0] - robot.data.body_link_pos_w[:, body]
        velocity = robot.data.body_link_lin_vel_w[:, body] + torch.cross(
            robot.data.body_link_ang_vel_w[:, body], offset, dim=-1)
        goal = self.command_manager.get_command('target_goal_pos')
        distance = (data.object_pos_w[:, 0, :2] - goal[:, :2]).norm(dim=-1)
        values = torch.stack((distance, velocity.norm(dim=-1),
                              data.object_lin_vel_w[:, 0].norm(dim=-1),
                              (distance <= self.cfg.goal_success_radius).float()), dim=-1)
        valid = torch.isfinite(values)
        valid[:, 3] = valid[:, 0]
        clean = torch.where(valid, values, 0.)
        self._metric_sums += clean
        self._metric_counts += valid
        self._metric_peaks = torch.maximum(self._metric_peaks, clean[:, 1:3])
        self._metric_final_distance = clean[:, 0]
        self._metric_min_distance = torch.minimum(
            self._metric_min_distance, torch.where(valid[:, 0], distance, float('inf')))

    def _reset_idx(self, env_ids):
        # Read the terminal outcome before parent reset overwrites the goal/poses.
        completed = ((self.episode_length_buf[env_ids] > 0)
                     & self.termination_manager.dones[env_ids])
        # Same-step ties use ascending env ID, including if a batch exceeds N.
        ids = env_ids[completed].sort().values
        due = self.episode_length_buf[ids] >= self.max_episode_length
        success = due & self.termination_manager.get_term('success')[ids]
        self._timeout_outcomes.append(torch.stack((success, due), dim=-1).long())
        # Log per-episode temporal means and peaks, averaged over reset episodes.
        counts = self._metric_counts[env_ids]
        means = self._metric_sums[env_ids] / counts.clamp_min(1)
        valid = counts > 0
        batch_means = (means * valid).sum(0) / valid.sum(0).clamp_min(1)
        metrics = {f'Metrics/{key}_mean': batch_means[i] for i, key in enumerate(self._mean_metrics)}
        speed_valid = valid[:, 1:3]
        peaks = (self._metric_peaks[env_ids] * speed_valid).sum(0) / speed_valid.sum(0).clamp_min(1)
        metrics['Metrics/eef_linear_speed_m_s_episode_max'] = peaks[0]
        metrics['Metrics/target_linear_speed_m_s_episode_max'] = peaks[1]
        distance_valid = valid[:, 0]
        denom = distance_valid.sum().clamp_min(1)
        metrics['Metrics/target_goal_distance_xy_m_final'] = (
            self._metric_final_distance[env_ids] * distance_valid).sum() / denom
        metrics['Metrics/target_goal_distance_xy_m_min'] = torch.where(
            distance_valid, self._metric_min_distance[env_ids], 0.).sum() / denom
        self._metric_sums[env_ids] = 0.
        self._metric_counts[env_ids] = 0.
        self._metric_peaks[env_ids] = 0.
        self._metric_final_distance[env_ids] = 0.
        self._metric_min_distance[env_ids] = float('inf')
        self.sensor_data_fresh[env_ids] = False
        super()._reset_idx(env_ids)
        self.action_manager._action[env_ids, -2:] = 1.
        self.action_manager._prev_action[env_ids, -2:] = 1.
        self._initial_object_positions_w[env_ids] = self.scene['object_collection'].data.object_pos_w[env_ids]
        self.extras['log'].update(metrics)
        totals = self._timeout_outcomes.totals
        self.extras['log'].update({
            'Metrics/timeout_success_rate': totals[0].float() / totals[1].clamp_min(1),
            'Metrics/timeout_success_count': totals[0].clone(),
            'Metrics/timeout_episode_count': totals[1].clone(),
        })



class SweepSimpleFingerEnv(SweepSimpleEnv):
    """Keep the parent lifecycle, with zero-radian hand reset commands."""

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.action_manager._action[env_ids, 6:] = 0.
        self.action_manager._prev_action[env_ids, 6:] = 0.


class SweepSimpleFingerV1Env(SweepSimpleFingerEnv):
    def load_managers(self):
        from ..assets.hand_kinematics import finger_action_bounds
        self._finger_action_lower, self._finger_action_upper = finger_action_bounds(self.device)
        super().load_managers()

    def step(self, action):
        from ..assets.hand_kinematics import clip_finger_actions
        # Out-of-place: PPO retains its original Gaussian sample and log-prob.
        # ActionManager stores the execution command for Last Action and both
        # current/previous action-rate penalties, before applying OSC scales.
        if self.cfg.clip_raw_actions:
            action = clip_finger_actions(action, self._finger_action_lower, self._finger_action_upper)
        return super().step(action)


class SweepSimpleFingerV2Env(SweepSimpleFingerV1Env):
    def load_managers(self):
        from ..mdp.common.sweep_simple import FingerRetreatState
        self._finger_retreat = FingerRetreatState(self)
        super().load_managers()

    def _reset_idx(self, env_ids):
        # Parent records terminal success before clearing the episode state.
        super()._reset_idx(env_ids)
        self._finger_retreat.reset(self, env_ids)


class SweepSimpleSemiProfEnv(SweepSimpleFingerV1Env):
    """Sample target contact after each scene update without copying the RL loop."""

    def load_managers(self):
        super().load_managers()
        from ..mdp.common.sweep_simple_force import SemiProfForceSamples
        self._semi_prof_forces = SemiProfForceSamples(self)
        self._semi_collecting = False

    def step(self, action):
        self._semi_prof_forces.begin(self)
        original_update = self.scene.update

        def update_and_sample(*args, **kwargs):
            original_update(*args, **kwargs)
            if self._semi_collecting:
                self._semi_prof_forces.capture(self)

        # Instance-local hook, scoped to this call; all parent reset/recorder/
        # reward/observation ordering stays intact, including exceptions.
        self.scene.update = update_and_sample
        self._semi_collecting = True
        try:
            return super().step(action)
        finally:
            self._semi_collecting = False
            self.scene.update = original_update

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._semi_prof_forces.reset(env_ids)


class SweepSimpleFingerForceDirectionEnv(SweepSimpleFingerEnv):
    """Test lifecycle plus force sampling; never enter Finger-v1 raw clipping."""

    def load_managers(self):
        super().load_managers()
        from ..mdp.common.sweep_simple_force import SemiProfForceSamples
        self._semi_prof_forces = SemiProfForceSamples(self)
        self._semi_collecting = False

    def step(self, action):
        self._semi_prof_forces.begin(self)
        original_update = self.scene.update

        def update_and_sample(*args, **kwargs):
            original_update(*args, **kwargs)
            if self._semi_collecting:
                self._semi_prof_forces.capture(self)

        # Instance-local hook, scoped to this call; all parent reset/recorder/
        # reward/observation ordering stays intact, including exceptions.
        self.scene.update = update_and_sample
        self._semi_collecting = True
        try:
            return super().step(action)
        finally:
            self._semi_collecting = False
            self.scene.update = original_update

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._semi_prof_forces.reset(env_ids)
