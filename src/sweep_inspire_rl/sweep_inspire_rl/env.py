"""Manager-based sweep with an explicit mixed arm/hand action space."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from sweeping_policy.env import ShelfSweepEnv


class InspireShelfSweepEnv(ShelfSweepEnv):
    """Keep Sweep's vectorized target state and expose normalized hand inputs."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Joint teleports update FK without a new contact/wrench sample. Keep
        # each reset row masked until its first actual physics step.
        self.sensor_data_fresh = torch.zeros(
            cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if self.action_manager.total_action_dim != 8:
            raise RuntimeError("Expected six OSC and two Inspire synergy action values")
        if self.scene["object_collection"].num_objects != 1:
            raise RuntimeError("The Inspire sweep must instantiate exactly one object per environment")
        self.single_action_space = gym.spaces.Box(
            low=np.array([-np.inf] * 6 + [0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf] * 6 + [1.0, 1.0], dtype=np.float32),
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.sensor_data_fresh[env_ids] = False

    def step(self, action):
        self.sensor_data_fresh[:] = True
        return super().step(action)
