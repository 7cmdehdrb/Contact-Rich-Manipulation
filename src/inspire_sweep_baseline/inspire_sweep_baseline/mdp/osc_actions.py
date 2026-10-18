"""Relative OSC commands with separate physical translation/rotation bounds."""

from __future__ import annotations

import math

import torch
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass


class BoundedOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """Bound the pose delta after scaling, retaining raw policy actions.

    The reference pose is the current EEF at each policy step. These bounds
    constrain one command, rather than the arm's total reachable workspace.
    They are not joint-velocity guarantees near singularities or contacts.
    """

    def __init__(self, cfg, env):
        if cfg.controller_cfg.target_types != ["pose_rel"] or cfg.controller_cfg.impedance_mode != "fixed":
            raise ValueError("Bounded OSC requires fixed-impedance pose_rel commands")
        for limit in (cfg.position_command_limit, cfg.orientation_command_limit):
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("OSC command limits must be positive and finite")
        super().__init__(cfg, env)

    def _preprocess_actions(self, actions):
        super()._preprocess_actions(actions)
        # The upstream OSC action does not apply ActionTermCfg.clip. Limit the
        # scaled translation and axis-angle vectors before set_command runs.
        for component, limit in (
            (slice(0, 3), self.cfg.position_command_limit),
            (slice(3, 6), self.cfg.orientation_command_limit),
        ):
            delta = self._processed_actions[:, component]
            magnitude = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            delta *= limit / magnitude.clamp_min(limit)


@configclass
class BoundedOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = BoundedOperationalSpaceControllerAction
    position_command_limit: float = 0.005
    """Maximum translation delta norm per policy command, in meters."""
    orientation_command_limit: float = 0.025
    """Maximum rotation-vector norm per policy command, in radians."""
