"""Finger-v1 defensive OSC input bounds before the inherited scales."""
from .wrench_refactor import WrenchRefactorOSCAction
from ...assets.hand_kinematics import finger_action_bounds, clip_finger_actions


class BoundedFingerOSCAction(WrenchRefactorOSCAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        lower, upper = finger_action_bounds(self.device)
        self._finger_lower, self._finger_upper = lower[:6], upper[:6]

    def process_actions(self, actions):
        super().process_actions(clip_finger_actions(actions, self._finger_lower, self._finger_upper))
