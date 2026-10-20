"""Keep exact reset geometry separate from noisy policy observations."""

import torch

from ..contact_sweep_clutter.env import ContactSweepClutterEnv


class ContactSweepClutterV1Env(ContactSweepClutterEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self._initial_object_positions_w = torch.zeros(
            (cfg.scene.num_envs, len(cfg.scene.object_collection.rigid_objects), 3),
            device=cfg.sim.device,
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # All placement events have finished. Capture physical link origins,
        # without the initial-observation noise, before any new physics step.
        self._initial_object_positions_w[env_ids] = (
            self.scene["object_collection"].data.object_pos_w[env_ids]
        )
