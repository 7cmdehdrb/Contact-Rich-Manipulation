"""Relative OSC commands expressed in the EEF axes sampled at each policy step."""

from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass


class EEFRelativeOSCAction(OperationalSpaceControllerAction):
    """Use the full offset EEF pose as a frozen task frame for each command."""

    def __init__(self, cfg, env):
        if cfg.task_frame_rel_path is not None or cfg.controller_cfg.target_types != ["pose_rel"]:
            raise ValueError("EEF-relative OSC requires pose_rel and no external task frame")
        super().__init__(cfg, env)

    def _compute_task_frame_pose(self):
        # process_actions updates _ee_pose_b first, including body_offset.
        # Clone: physics substeps update _ee_pose_b, but this command's frame
        # must remain at its policy-step pose until the next action arrives.
        self._task_frame_pose_b = self._ee_pose_b.clone()


@configclass
class EEFRelativeOSCActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = EEFRelativeOSCAction
