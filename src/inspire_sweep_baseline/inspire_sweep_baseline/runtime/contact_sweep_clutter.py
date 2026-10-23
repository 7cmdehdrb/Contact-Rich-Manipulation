"""Contact Sweep with episode-fixed initial obstacle observations.

Keep exact reset geometry separate from noisy policy observations.

Apply manager curricula before a rollout, including episodes in progress.

Capture terminal transitions before reset without extra observation calls.

Keep the v1 lifecycle while replacing the wrist measurement pipeline."""

from ..assets.scenes.contact_sweep_clutter_layout import ClutterLayoutGenerator
from .contact_sweep import ContactSweepEnv
from .contact_sweep_clutter_telemetry import WrenchTelemetry
import torch as torch

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


class ContactSweepClutterFTTestEnv(ContactSweepClutterV1CurriculumEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.ft_telemetry = None
        self._ft_in_step = False
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.ft_telemetry = WrenchTelemetry(self)

    def _reset_idx(self, env_ids):
        telemetry = self.ft_telemetry
        if telemetry is not None and self._ft_in_step:
            # Reward/termination managers have finished; physical state and
            # flags still belong to the episode that is about to end.
            telemetry.capture(self, env_ids)
        super()._reset_idx(env_ids)
        if telemetry is not None:
            telemetry.reset(env_ids)

    def step(self, action):
        self.ft_telemetry.begin_step()
        self._ft_in_step = True
        try:
            result = super().step(action)
        finally:
            self._ft_in_step = False
        # Reset rows have already been captured above. Their fresh flag is
        # false now, so they cannot overwrite terminal samples with reset zeros.
        self.ft_telemetry.capture(self, self.sensor_data_fresh.nonzero().flatten())
        self.ft_telemetry.finish_step()
        return result


class ContactSweepClutterV2Env(ContactSweepClutterV1Env):
    """Sensor and reward managers own compensation and episode-local history."""
