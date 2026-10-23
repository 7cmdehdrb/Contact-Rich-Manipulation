"""Episode-fixed object goal command, independent of other project packages."""

from __future__ import annotations

from isaaclab.managers import CommandTerm
import torch as torch

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

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis and not hasattr(self, "_goal_marker"):
            # Lazy imports keep non-visual command use independent of markers.
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers import VisualizationMarkersCfg

            self._goal_marker = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/InspireSweepBaseline/target_goal",
                markers={"goal": sim_utils.SphereCfg(
                    radius=0.025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.2)),
                )},
            ))
        if hasattr(self, "_goal_marker"):
            self._goal_marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        if hasattr(self, "pos_command_w"):
            self._goal_marker.visualize(translations=self.pos_command_w)
