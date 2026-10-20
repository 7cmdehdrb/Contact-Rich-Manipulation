"""Apply manager curricula before a rollout, including episodes in progress."""

from ..contact_sweep_clutter_v1.env import ContactSweepClutterV1Env


class ContactSweepClutterV1CurriculumEnv(ContactSweepClutterV1Env):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.learning_iteration = 0
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def set_learning_iteration(self, iteration: int):
        """Set the same zero-based index printed by RSL-RL's learning loop."""
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Learning iteration must be a nonnegative integer")
        self.learning_iteration = iteration
        # RewardManager owns the live term configs. Do not wait for an episode
        # reset: all environments must collect a rollout under the same values.
        self.curriculum_manager.compute(env_ids=None)
