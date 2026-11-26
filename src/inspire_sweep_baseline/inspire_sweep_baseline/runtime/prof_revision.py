import torch
from isaaclab.envs import ManagerBasedRLEnv
from ..mdp.common.stabilize_sweep_contact import ContactSamples
from ..mdp.common.prof_revision_state import ProfRevisionState, ProfRevisionPrecisionState

from .stabilize_sweep import _RecentEpisodeOutcomes
from ..mdp.common.prof_revision_geometry import initialize_fingers, update_geometry


class ProfRevisionEnv(ManagerBasedRLEnv):
    """Independent observed-OBB lifecycle; only physics helpers are shared."""

    _state_type = ProfRevisionState
    _mean_metrics = ('target_goal_distance_xy_m', 'eef_linear_speed_m_s',
                     'target_linear_speed_m_s', 'target_within_goal_radius_fraction')

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._timeout_outcomes = _RecentEpisodeOutcomes(cfg.timeout_metric_window, cfg.sim.device)
        count, device = cfg.scene.num_envs, cfg.sim.device
        fast = getattr(cfg, 'reset_ik_backend', None) == 'warp'
        if fast and (cfg.max_obstacles != cfg._layout_capacity() or cfg.obstacle_observation_slots != 16):
            raise ValueError('Fast reset requires the configured layout capacity and sixteen observation slots')
        self._obb_target_pos_w = torch.zeros((count, 3), device=device)
        self._obb_target_quat_w = torch.zeros((count, 4), device=device)
        self._obb_target_quat_w[:, 0] = 1.
        self._obb_target_size = torch.zeros((count, 3), device=device)
        self._obb_target_local_pos = torch.zeros((count, 3), device=device)
        self._obb_target_local_quat = self._obb_target_quat_w.clone()
        self._obb_live_pos_w = torch.zeros((count, 3), device=device)
        self._obb_live_quat_w = self._obb_target_quat_w.clone()
        self._obb_obstacle_pos_w = torch.zeros((count, 16, 3), device=device)
        self._obb_obstacle_quat_w = torch.zeros((count, 16, 4), device=device)
        self._obb_obstacle_quat_w[..., 0] = 1.
        self._obb_obstacle_size = torch.zeros((count, 16, 3), device=device)
        self._obb_valid = torch.zeros((count, 16), device=device, dtype=torch.bool)
        self._obb_sources = torch.full((count, 16), -1, device=device, dtype=torch.long)
        self._finger_pos_w = torch.zeros((count, 5, 3), device=device)
        self._finger_pose_e = torch.zeros((count, 5, 7), device=device)
        self._finger_pose_e[..., 6] = 1.
        self._finger_gaps = torch.zeros((count, 5), device=device)
        self._formation = torch.zeros(count, device=device)
        self._stabilize_state = self._create_state(count, device, cfg)
        self._sweep_phase=self._stabilize_state.achieved
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
        self._stabilize_contacts=ContactSamples(self)

    def _create_state(self, count, device, cfg):
        return self._state_type(count, device)

    def _additional_step_metrics(self):
        return {}

    def load_managers(self):
        shelf = self.scene['shelf'].data
        self._revision_initial_shelf_pos = shelf.root_pos_w.clone()
        self._revision_initial_shelf_quat = shelf.root_quat_w.clone()
        self._metric_eef_body = self.scene['robot'].body_names.index('axia80_link')
        initialize_fingers(self)
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
            self._revision_layout_rng.manual_seed(seed)
        result = super().reset(seed=seed, env_ids=env_ids, options=options)
        return result

    def step(self, action):
        self.sensor_data_fresh[:] = True
        self._stabilize_contacts.begin()
        update=self.scene.update
        def capture(dt):
            update(dt)
            contacts=self._stabilize_contacts
            contacts.capture(self)
            if contacts.samples == self.cfg.decimation:
                update_geometry(self)
                self._stabilize_state.finish(
                    self._obb_live_pos_w, self.command_manager.get_command('target_goal_pos'),
                    contacts.force_sum/contacts.samples, contacts.contact, contacts.all_valid,
                    self.scene['ee_frame'].data.target_pos_w[:, 0], self.cfg.reach_near_radius)
                state = self._stabilize_state
                values = torch.stack((state.reach_distance,
                    state.reach_excess, state.reach_delta, self._finger_gaps.sum(-1),
                    self._formation, state.distance, state.positive, state.retreat,
                    state.contact_recognized.float(), state.alignment,
                    *self._finger_gaps.unbind(-1)), -1).mean(0)
                names = ('reach_distance_m', 'reach_excess_m', 'reach_delta_m',
                    'finger_gap_sum_m', 'formation_raw', 'goal_distance_xy_m',
                    'sweep_progress_m', 'sweep_retreat_m', 'contact_recognized', 'force_alignment',
                    'thumb_gap_m', 'index_gap_m', 'middle_gap_m', 'ring_gap_m', 'little_gap_m')
                self._prof_step_metrics = {f'ProfRevision/{name}': values[i] for i, name in enumerate(names)}
                self._prof_step_metrics.update(self._additional_step_metrics())
        self.scene.update=capture
        try:
            result = super().step(action)
            # Captured before automatic reset, including the terminal transition.
            self.extras['log'].update(self._prof_step_metrics)
            for name in ('reaching', 'contact_formation', 'sweep_progress', 'sweep_retreat'):
                index = self.reward_manager.active_terms.index(name)
                self.extras['log'][f'ProfRevision/{name}_step_reward'] = (
                    self.reward_manager._step_reward[:, index].mean()*self.step_dt)
            return result
        finally:
            self.scene.update=update


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
        distance = (self._obb_live_pos_w[:, :2] - goal[:, :2]).norm(dim=-1)
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
        if hasattr(self,"_stabilize_contacts"):self._stabilize_contacts.reset(env_ids)
        update_geometry(self, env_ids)
        self._stabilize_state.reset(env_ids, self._obb_live_pos_w[env_ids],
            self.command_manager.get_command('target_goal_pos')[env_ids],
            self.scene['ee_frame'].data.target_pos_w[env_ids, 0], self.cfg.reach_near_radius)
        self.action_manager._action[env_ids, 6:] = 0.
        self.action_manager._prev_action[env_ids, 6:] = 0.
        self._initial_object_positions_w[env_ids] = self.scene['object_collection'].data.object_pos_w[env_ids]
        self.extras['log'].update(metrics)
        self.extras['log'].update(self._reset_diagnostics)
        totals = self._timeout_outcomes.totals
        self.extras['log'].update({
            'Metrics/timeout_success_rate': totals[0].float() / totals[1].clamp_min(1),
            'Metrics/timeout_success_count': totals[0].clone(),
            'Metrics/timeout_episode_count': totals[1].clone(),
        })


class ProfRevisionV0_3_2Env(ProfRevisionEnv):
    """Precision state and telemetry only; lifecycle remains the parent's."""

    def _create_state(self, count, device, cfg):
        return ProfRevisionPrecisionState(
            count, device, cfg.sweep_force_fade_inner_m, cfg.sweep_force_fade_outer_m,
            cfg.sweep_precision_radius_m, cfg.sweep_precision_max_multiplier)

    def _additional_step_metrics(self):
        state = self._stabilize_state
        values = torch.stack((state.force_influence, state.distance_sensitivity,
                              state.effective_alignment, state.weighted_delta,
                              state.precision_positive, state.precision_retreat,
                              (state.distance <= self.cfg.sweep_force_fade_inner_m).float(),
                              ((state.distance > self.cfg.sweep_force_fade_inner_m)
                               & (state.distance < self.cfg.sweep_force_fade_outer_m)).float(),
                              (state.distance >= self.cfg.sweep_force_fade_outer_m).float()), -1).mean(0)
        names = ('force_influence', 'distance_sensitivity', 'effective_alignment',
                 'weighted_distance_delta_m', 'precision_progress_m', 'precision_retreat_m',
                 'near_fraction', 'transition_fraction', 'far_fraction')
        metrics = {f'ProfRevisionPrecision/{name}': values[i] for i, name in enumerate(names)}
        for name in ('sweep_force_fade_inner_m', 'sweep_force_fade_outer_m',
                     'sweep_precision_radius_m', 'sweep_precision_max_multiplier'):
            metrics[f'ProfRevisionPrecision/{name}'] = getattr(self.cfg, name)
        return metrics
