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

    def load_managers(self):
        shelf = self.scene['shelf'].data
        self._revision_initial_shelf_pos = shelf.root_pos_w.clone()
        self._revision_initial_shelf_quat = shelf.root_quat_w.clone()
        super().load_managers()

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            self._revision_layout_rng.manual_seed(seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    def step(self, action):
        self.sensor_data_fresh[:] = True
        return super().step(action)

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
