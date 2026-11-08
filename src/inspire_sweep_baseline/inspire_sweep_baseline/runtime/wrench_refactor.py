"""Independent Wrench runtime, with seventeen fixed bodies and cached spawn heights."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class WrenchRefactorEnv(ManagerBasedRLEnv):
    """Own reset snapshots, iteration and telemetry without previous task inheritance."""

    _mean_metrics = ('target_goal_distance_xy', 'eef_linear_speed',
                     'target_linear_speed', 'target_linear_acceleration')

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Hydra overrides run after Cfg.__post_init__; keep all three height gates aligned.
        cfg.contact_height_limit = cfg.goal_reward_height_limit
        count, device = cfg.scene.num_envs, cfg.sim.device
        if (cfg.max_gray_obstacles, cfg.max_white_obstacles, cfg.obstacle_observation_slots) != (3, 3, 16):
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
        self._final_actual_population = torch.zeros((count, 2), device=device, dtype=torch.long)
        self._revision_layout_complete = torch.zeros(count, device=device, dtype=torch.bool)
        self._revision_layout_rng = torch.Generator(device=device).manual_seed(int(cfg.seed or 0))
        self._metric_sums = {key: torch.zeros(count, device=device) for key in self._mean_metrics}
        self._metric_counts = {key: torch.zeros(count, device=device) for key in self._mean_metrics}
        self._max_force = torch.zeros(count, device=device)
        self._max_moment = torch.zeros(count, device=device)
        self._force_samples = torch.zeros(count, device=device, dtype=torch.bool)
        self._moment_samples = torch.zeros(count, device=device, dtype=torch.bool)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self._update_reaching_point_marker()

    def load_managers(self):
        shelf = self.scene['shelf'].data
        self._revision_initial_shelf_pos = shelf.root_pos_w.clone()
        self._revision_initial_shelf_quat = shelf.root_quat_w.clone()
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
            self._revision_layout_rng.manual_seed(seed)
            if getattr(self.cfg.events, 'robot_initial_pose', None) is not None:
                self._goal_reset_seed = seed + 1707
                if hasattr(self, '_goal_reset_rng'):
                    self._goal_reset_rng.manual_seed(self._goal_reset_seed)
        result = super().reset(seed=seed, env_ids=env_ids, options=options)
        self._update_reaching_point_marker()
        return result

    def step(self, action):
        self.sensor_data_fresh[:] = True
        result = super().step(action)
        self._update_reaching_point_marker()
        return result

    def _update_reaching_point_marker(self):
        """Render the reward's live point without adding any physical body."""
        if not getattr(self.cfg, "debug_reaching_point", False):
            if hasattr(self, "_reaching_point_marker"):
                self._reaching_point_marker.set_visibility(False)
            return
        if not hasattr(self, "_reaching_point_marker"):
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            self._reaching_point_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/InspireWrenchRefactor/reaching_point",
                markers={"point": sim_utils.SphereCfg(
                    radius=0.008,
                    rigid_props=None,
                    collision_props=None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.8)),
                )},
            ))
        from ..mdp.reward_functions.wrench_refactor import opposite_reaching_point

        params = self.reward_manager.get_term_cfg("target_reaching").params
        point = opposite_reaching_point(
            self, params.get("offset", 0.05), params.get("command_name", "target_goal_pos")
        )
        self._reaching_point_marker.set_visibility(True)
        self._reaching_point_marker.visualize(translations=point)

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
        data = self.scene['object_collection'].data
        rows = torch.arange(self.num_envs, device=self.device)
        target = self.target_id[:, 0]
        robot = self.scene['robot']
        body = robot.body_names.index('axia80_link')
        offset = self.scene['ee_frame'].data.target_pos_w[:, 0] - robot.data.body_link_pos_w[:, body]
        eef_velocity = robot.data.body_link_lin_vel_w[:, body] + torch.cross(
            robot.data.body_link_ang_vel_w[:, body], offset, dim=-1)
        goal = self.command_manager.get_command('target_goal_pos')
        values = {
            'target_goal_distance_xy': (data.object_pos_w[rows, target, :2] - goal[:, :2]).norm(dim=-1),
            'eef_linear_speed': eef_velocity.norm(dim=-1),
            'target_linear_speed': data.object_lin_vel_w[rows, target].norm(dim=-1),
            'target_linear_acceleration': data.object_com_lin_acc_w[rows, target].norm(dim=-1),
        }
        for key, value in values.items():
            valid = torch.isfinite(value)
            self._metric_sums[key] += torch.where(valid, value, 0.)
            self._metric_counts[key] += valid.float()
        wrench = self.observation_manager.cfg.policy.wrist_wrench.func(self)
        force, moment = wrench[:, :3].norm(dim=-1), wrench[:, 3:].norm(dim=-1)
        force_valid, moment_valid = torch.isfinite(force), torch.isfinite(moment)
        self._max_force = torch.maximum(self._max_force, torch.where(force_valid, force, 0.))
        self._max_moment = torch.maximum(self._max_moment, torch.where(moment_valid, moment, 0.))
        self._force_samples |= force_valid
        self._moment_samples |= moment_valid

    def _reset_idx(self, env_ids):
        # TEMP v3-Test diagnostics: read evaluated terms before reset clears them.
        if type(self.cfg).__name__ == 'WrenchRefactorV3TestEnvCfg':
            for env_id in env_ids.tolist():
                reasons = [
                    name for name in self.termination_manager.active_terms
                    if bool(self.termination_manager.get_term(name)[env_id])
                ]
                if reasons:
                    print(
                        f'[Wrench-Refactor-v3-Test termination] env={env_id} '
                        f'step={int(self.episode_length_buf[env_id])} '
                        f'reasons={", ".join(reasons)}',
                        flush=True,
                    )
        # END TEMP v3-Test diagnostics.
        metrics = {}
        for key in self._mean_metrics:
            sampled = self._metric_counts[key][env_ids] > 0
            if bool(sampled.any()):
                metrics[f'Metrics/{key}_mean'] = (
                    self._metric_sums[key][env_ids][sampled] /
                    self._metric_counts[key][env_ids][sampled]).mean()
            self._metric_sums[key][env_ids] = 0.
            self._metric_counts[key][env_ids] = 0.
        for key, values, valid in (('max_force', self._max_force, self._force_samples),
                                   ('max_moment', self._max_moment, self._moment_samples)):
            if bool(valid[env_ids].any()):
                metrics[f'Metrics/{key}'] = values[env_ids][valid[env_ids]].mean()
            values[env_ids] = 0.
            valid[env_ids] = False
        self.sensor_data_fresh[env_ids] = False
        super()._reset_idx(env_ids)
        self.action_manager._action[env_ids, -2:] = 1.
        self.action_manager._prev_action[env_ids, -2:] = 1.
        self._initial_object_positions_w[env_ids] = self.scene['object_collection'].data.object_pos_w[env_ids]
        for index, color in enumerate(('gray', 'white')):
            metrics[f'Metrics/obstacle_actual_{color}'] = self._final_actual_population[env_ids, index].float().mean()
        metrics['Metrics/layout_complete'] = self._revision_layout_complete[env_ids].float().mean()
        self.extras['log'].update(metrics)
