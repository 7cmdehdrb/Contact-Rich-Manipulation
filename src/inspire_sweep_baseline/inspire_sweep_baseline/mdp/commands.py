"""Episode-fixed object goal command, independent of other project packages."""

from __future__ import annotations

import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class SweepGoalCommand(CommandTerm):
    """Snapshot the reset object pose and translate its goal by the sweep vector."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._objects = env.scene[cfg.asset_name]
        target_ids = env.target_id[:, 0].long()
        rows = torch.arange(env.num_envs, device=env.device)
        self.target_init_state_w = self._objects.data.object_link_state_w[rows, target_ids].clone()
        self.pos_command_w = self.target_init_state_w[:, :3] + env.sweep_dir

    @property
    def command(self):
        return torch.cat((self.pos_command_w, self.target_init_state_w[:, 3:7]), dim=-1)

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        target_ids = self._env.target_id[env_ids, 0].long()
        self.target_init_state_w[env_ids] = self._objects.data.object_link_state_w[env_ids, target_ids]
        self.pos_command_w[env_ids] = (
            self.target_init_state_w[env_ids, :3] + self._env.sweep_dir[env_ids]
        )

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass


@configclass
class SweepGoalCommandCfg(CommandTermCfg):
    class_type: type[CommandTerm] = SweepGoalCommand
    asset_name: str = "object_collection"
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    debug_vis: bool = False
