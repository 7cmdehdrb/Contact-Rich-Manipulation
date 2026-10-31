"""Final v0 lifecycle directly on ManagerBasedRLEnv, matching ObjectRandom v3."""

from ..assets.scenes.contact_sweep_clutter_layout import ClutterLayoutGenerator
from isaaclab.envs import ManagerBasedRLEnv
import torch as torch

class FinalEnv(ManagerBasedRLEnv):
    """Own reset snapshots and rollout iteration without task inheritance."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.scene.configure_population_capacity(
            cfg.max_gray_obstacles,
            cfg.max_white_obstacles,
            cfg.obstacle_observation_slots,
        )
        self._active_object_mask = torch.zeros(
            (cfg.scene.num_envs, len(cfg.scene.object_collection.rigid_objects)),
            dtype=torch.bool,
            device=cfg.sim.device,
        )
        self._active_object_mask[:, 0] = True
        self._population_layout_seed = cfg.seed
        self._population_layout_generators = {}
        count, device = (cfg.scene.num_envs, cfg.sim.device)
        self.learning_iteration = 0
        self.target_id = torch.zeros((count, 1), device=device, dtype=torch.long)
        self.target_width = torch.full((count, 1), 0.04, device=device)
        self.sweep_dir = torch.zeros((count, 3), device=device)
        self.sensor_data_fresh = torch.zeros(count, device=device, dtype=torch.bool)
        self._initial_obstacle_observation = torch.zeros((count, 4 * cfg.obstacle_observation_slots), device=device)
        self._initial_object_positions_w = torch.zeros((count, len(cfg.scene.object_collection.rigid_objects), 3), device=device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            self._population_layout_seed = seed
            self._population_layout_generators.clear()
            if hasattr(self, '_gray_padding_layout_generators'):
                self._gray_padding_layout_generators.clear()
        if seed is not None:
            seed = self.seed(seed)
            self._clutter_layout_generator = ClutterLayoutGenerator(seed=seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    def _reset_idx(self, env_ids):
        self.sensor_data_fresh[env_ids] = False
        super()._reset_idx(env_ids)
        self.action_manager._action[env_ids, -2:] = 1.0
        self.action_manager._prev_action[env_ids, -2:] = 1.0
        self._initial_object_positions_w[env_ids] = self.scene['object_collection'].data.object_pos_w[env_ids]

    def step(self, action):
        self.sensor_data_fresh[:] = True
        return super().step(action)

    def set_learning_iteration(self, iteration: int):
        """Use RSL-RL's zero-based iteration; apply schedules before rollout."""
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError('Learning iteration must be a nonnegative integer')
        self.learning_iteration = iteration
        self.curriculum_manager.compute(env_ids=None)


class FinalV1Env(FinalEnv):
    """Final v1/v2 extend lifecycle only for deterministic layout reseeding."""

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            import numpy as np
            self._final_layout_rng = np.random.default_rng(seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, '_final_actual_population'):
            counts = self._final_actual_population[env_ids].float()
            self.extras['log']['Curriculum/obstacle_actual_gray'] = counts[:, 0].mean()
            self.extras['log']['Curriculum/obstacle_actual_white'] = counts[:, 1].mean()


class FinalGoalAccuracyEnv(FinalV1Env):
    """Collect episode metrics at each termination check, before automatic reset."""

    _mean_metrics = ('target_goal_distance_xy', 'eef_linear_speed',
                     'target_linear_speed', 'target_linear_acceleration')

    def __init__(self, cfg, render_mode=None, **kwargs):
        count, device = cfg.scene.num_envs, cfg.sim.device
        self._metric_sums = {key: torch.zeros(count, device=device) for key in self._mean_metrics}
        self._metric_counts = {key: torch.zeros(count, device=device) for key in self._mean_metrics}
        self._max_force = torch.zeros(count, device=device)
        self._max_moment = torch.zeros(count, device=device)
        self._force_samples = torch.zeros(count, device=device, dtype=torch.bool)
        self._moment_samples = torch.zeros(count, device=device, dtype=torch.bool)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def record_goal_accuracy_metrics(self):
        data = self.scene['object_collection'].data
        rows = torch.arange(self.num_envs, device=self.device)
        target_ids = self.target_id[:, 0].long()
        target_pos = data.object_pos_w[rows, target_ids]
        goal = self.command_manager.get_command('target_goal_pos')
        robot = self.scene['robot']
        body_index = robot.body_names.index('axia80_link')
        offset = self.scene['ee_frame'].data.target_pos_w[:, 0] - robot.data.body_link_pos_w[:, body_index]
        eef_velocity = robot.data.body_link_lin_vel_w[:, body_index] + torch.cross(
            robot.data.body_link_ang_vel_w[:, body_index], offset, dim=-1)
        values = {
            'target_goal_distance_xy': torch.linalg.vector_norm(target_pos[:, :2] - goal[:, :2], dim=-1),
            'eef_linear_speed': torch.linalg.vector_norm(eef_velocity, dim=-1),
            'target_linear_speed': torch.linalg.vector_norm(data.object_lin_vel_w[rows, target_ids], dim=-1),
            'target_linear_acceleration': torch.linalg.vector_norm(
                data.object_com_lin_acc_w[rows, target_ids], dim=-1),
        }
        for key, value in values.items():
            finite = torch.isfinite(value)
            self._metric_sums[key] += torch.where(finite, value, 0.0)
            self._metric_counts[key] += finite.float()

        wrench = self.observation_manager.cfg.policy.wrist_wrench.func(self)
        force = torch.linalg.vector_norm(wrench[:, :3], dim=-1)
        moment = torch.linalg.vector_norm(wrench[:, 3:], dim=-1)
        force_finite = torch.isfinite(force)
        moment_finite = torch.isfinite(moment)
        self._max_force = torch.maximum(self._max_force, torch.where(force_finite, force, 0.0))
        self._max_moment = torch.maximum(self._max_moment, torch.where(moment_finite, moment, 0.0))
        self._force_samples |= force_finite
        self._moment_samples |= moment_finite

    def _reset_idx(self, env_ids):
        metrics = {}
        for key in self._mean_metrics:
            sampled = self._metric_counts[key][env_ids] > 0
            if bool(sampled.any()):
                metrics[f'Metrics/{key}_mean'] = (
                    self._metric_sums[key][env_ids][sampled]
                    / self._metric_counts[key][env_ids][sampled]).mean()
            self._metric_sums[key][env_ids] = 0.0
            self._metric_counts[key][env_ids] = 0.0
        for name, maxima, sampled in (
            ('max_force', self._max_force, self._force_samples),
            ('max_moment', self._max_moment, self._moment_samples),
        ):
            valid = sampled[env_ids]
            if bool(valid.any()):
                metrics[f'Metrics/{name}'] = maxima[env_ids][valid].mean()
        self._max_force[env_ids] = 0.0
        self._max_moment[env_ids] = 0.0
        self._force_samples[env_ids] = False
        self._moment_samples[env_ids] = False
        super()._reset_idx(env_ids)
        for color in ('gray', 'white'):
            old = f'Curriculum/obstacle_actual_{color}'
            if old in self.extras['log']:
                self.extras['log'][f'Metrics/obstacle_actual_{color}'] = self.extras['log'].pop(old)
        self.extras['log'].update(metrics)


class FinalGoalAccuracyV0_2_1Env(FinalGoalAccuracyEnv):
    """Expose the selected physical candidate as the parent's logical target 0."""

    def load_managers(self):
        from ..mdp.common.final_target_pool import (
            LogicalTargetCollection, SelectedTargetSensor, TargetChoice,
        )
        from ..mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name

        physical = self.scene['object_collection']
        count = 1 + self.cfg.max_gray_obstacles + self.cfg.max_white_obstacles
        logical = LogicalTargetCollection(physical, count)
        self._target_choice = TargetChoice(self.cfg.seed, len(logical.pool_ids))
        self._target_pool_bounds = self._final_object_bounds.copy()
        self._target_pool_scales = self._final_object_scales.copy()
        self._final_object_bounds = self._target_pool_bounds[:, :count].copy()
        self._final_object_scales = self._target_pool_scales[:, :count].copy()
        self._active_object_mask = self._active_object_mask[:, :count].clone()
        self._initial_object_positions_w = self._initial_object_positions_w[:, :count].clone()
        sensors = [self.scene[object_hand_sensor_name(physical.object_names[i])]
                   for i in logical.pool_ids.cpu().tolist()]
        self.scene.rigid_object_collections['object_collection'] = logical
        self.scene.sensors[object_hand_sensor_name('mass_cylinder')] = SelectedTargetSensor(logical, sensors)
        super().load_managers()

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            self._target_choice.reseed(seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    @property
    def target_kind_id(self):
        """Diagnostic only; never appended to actor or critic observations."""
        return self.scene['object_collection'].selected
