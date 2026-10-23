"""Independent Manager-based lifecycle for the fused clutter environment.

Fusion lifecycle with fixed capacity and episode-local obstacle populations."""

from ..assets.scenes.contact_sweep_clutter_layout import ClutterLayoutGenerator
from isaaclab.envs import ManagerBasedRLEnv
import torch as torch

class ContactSweepClutterFusionEnv(ManagerBasedRLEnv):
    """Own reset snapshots and rollout iteration without task inheritance."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        count, device = cfg.scene.num_envs, cfg.sim.device
        self.learning_iteration = 0
        self.target_id = torch.zeros((count, 1), device=device, dtype=torch.long)
        self.target_width = torch.full((count, 1), 0.04, device=device)
        self.sweep_dir = torch.zeros((count, 3), device=device)
        self.sensor_data_fresh = torch.zeros(count, device=device, dtype=torch.bool)
        self._initial_obstacle_observation = torch.zeros((count, 4 * cfg.obstacle_observation_slots), device=device)
        self._initial_object_positions_w = torch.zeros(
            (count, len(cfg.scene.object_collection.rigid_objects), 3), device=device,
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            seed = self.seed(seed)
            self._clutter_layout_generator = ClutterLayoutGenerator(seed=seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    def _reset_idx(self, env_ids):
        self.sensor_data_fresh[env_ids] = False
        super()._reset_idx(env_ids)
        self.action_manager._action[env_ids, -2:] = 1.0
        self.action_manager._prev_action[env_ids, -2:] = 1.0
        self._initial_object_positions_w[env_ids] = (
            self.scene["object_collection"].data.object_pos_w[env_ids]
        )

    def step(self, action):
        self.sensor_data_fresh[:] = True
        return super().step(action)

    def set_learning_iteration(self, iteration: int):
        """Use RSL-RL's zero-based iteration; apply schedules before rollout."""
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError("Learning iteration must be a nonnegative integer")
        self.learning_iteration = iteration
        self.curriculum_manager.compute(env_ids=None)


class ContactSweepClutterFusionV1_0Env(ContactSweepClutterFusionEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # Hydra applies overrides after Cfg construction, so rebuild capacity
        # here before any assets, sensors or reward histories are instantiated.
        cfg.scene.configure_population_capacity(
            cfg.max_gray_obstacles, cfg.max_white_obstacles, cfg.obstacle_observation_slots,
        )
        self._active_object_mask = torch.zeros(
            (cfg.scene.num_envs, len(cfg.scene.object_collection.rigid_objects)),
            dtype=torch.bool, device=cfg.sim.device,
        )
        self._active_object_mask[:, 0] = True
        self._population_layout_seed = cfg.seed
        self._population_layout_generators = {}
        super().__init__(cfg, render_mode=render_mode, **kwargs)

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            self._population_layout_seed = seed
            self._population_layout_generators.clear()
        return super().reset(seed=seed, env_ids=env_ids, options=options)
