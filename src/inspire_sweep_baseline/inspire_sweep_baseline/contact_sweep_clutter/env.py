"""Contact Sweep with episode-fixed initial obstacle observations."""

import torch

from ..contact_sweep.env import ContactSweepEnv
from .layout import ClutterLayoutGenerator


class ContactSweepClutterEnv(ContactSweepEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # ObservationManager probes shapes before the first episode reset.
        self._initial_obstacle_observation = torch.zeros(
            (cfg.scene.num_envs, 40), device=cfg.sim.device,
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def reset(self, seed=None, env_ids=None, options=None):
        if seed is not None:
            # The inherited seeder covers Torch/global NumPy, while layout
            # placement deliberately owns a separate persistent Generator.
            seed = self.seed(seed)
            self._clutter_layout_generator = ClutterLayoutGenerator(seed=seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)
