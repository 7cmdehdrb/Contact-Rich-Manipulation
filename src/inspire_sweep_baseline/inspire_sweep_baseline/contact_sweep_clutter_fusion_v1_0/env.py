"""Fusion lifecycle with fixed capacity and episode-local obstacle populations."""

import torch

from ..contact_sweep_clutter_fusion.env import ContactSweepClutterFusionEnv


class ContactSweepClutterFusionV1_0Env(ContactSweepClutterFusionEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # Hydra applies overrides after Cfg construction, so rebuild capacity
        # here before any assets, sensors or reward histories are instantiated.
        cfg.scene.configure_population_capacity(cfg.max_gray_obstacles, cfg.max_white_obstacles)
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

    # def step(self, action):
    #     result = super().step(action)
    #     # TerminationManager keeps this step's term masks through auto-reset.
    #     # Read those results rather than evaluating conditions on reset states.
    #     done_env_ids = (result[2] | result[3]).nonzero(as_tuple=True)[0]
    #     if done_env_ids.numel() > 0:
    #         for name in self.termination_manager.active_terms:
    #             env_ids = done_env_ids[self.termination_manager.get_term(name)[done_env_ids]]
    #             if env_ids.numel() > 0:
    #                 detail = " (arm joint velocity)" if name == "hand_velocity" else ""
    #                 print(
    #                     f"[Termination][{type(self).__name__}] step={self.common_step_counter} "
    #                     f"term={name}{detail} env_ids={env_ids.tolist()}",
    #                     flush=True,
    #                 )
    #     return result
