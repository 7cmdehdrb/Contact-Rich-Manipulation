"""Local manager-based environment with reset-safe Hand and sensor state."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class InspireSweepEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        # Managers evaluate commands/observations during base construction.
        count, device = cfg.scene.num_envs, cfg.sim.device
        self.target_id = torch.zeros((count, 1), device=device, dtype=torch.long)
        self.target_width = torch.full((count, 1), 0.04, device=device)
        self.sweep_dir = torch.zeros((count, 3), device=device)
        self.sensor_data_fresh = torch.zeros(count, device=device, dtype=torch.bool)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _reset_idx(self, env_ids):
        # manager = getattr(self, "termination_manager", None)
        # if manager is not None:
        #     for name in manager.active_terms:
        #         fired_ids = env_ids[manager.get_term(name)[env_ids]]
        #         if fired_ids.numel():
        #             cfg = manager.get_term_cfg(name)
        #             function = getattr(cfg.func, "__name__", type(cfg.func).__name__)
        #             print(
        #                 f"[TERMINATION] term={name} function={function} timeout={cfg.time_out} "
        #                 f"env_ids={fired_ids.tolist()} "
        #                 f"episode_steps={self.episode_length_buf[fired_ids].tolist()}",
        #                 flush=True,
        #             )
        self.sensor_data_fresh[env_ids] = False
        super()._reset_idx(env_ids)
        # Arm history remains zero; the last two absolute commands mean OPEN.
        self.action_manager._action[env_ids, -2:] = 1.0
        self.action_manager._prev_action[env_ids, -2:] = 1.0

    def step(self, action):
        # Parent auto-reset invalidates only reset rows before observations.
        self.sensor_data_fresh[:] = True
        return super().step(action)
