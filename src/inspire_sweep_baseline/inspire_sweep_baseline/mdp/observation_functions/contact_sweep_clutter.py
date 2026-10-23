"""Clutter-specific observations, rewards, and failure conditions.

World-XY goal occupancy, shared tactile rewards, and bounded scene movement.

Share the same compensated physical sample between observations and reward."""

from ..common.compensated_wrench import PureWrenchReader
from .tactile_observations import ProjectedTactileContactBits
from isaaclab.managers import ManagerTermBase
import torch as torch

def initial_obstacle_positions(env):
    """Return the reset-time 10 x [present, x_root, y_root, z_root] snapshot.

    The scene reset owns this cache. It excludes the target and zero-pads absent
    obstacle slots. Reading observations never refreshes positions from physics.
    """
    return env._initial_obstacle_observation


class SharedTactileContactBits(ProjectedTactileContactBits):
    """Read current contacts once for both reward terms and the observation.

    Rewards run before observations. Never read the previous observation buffer.
    Automatic/partial reset invalidates this cache before reset observations.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._cached_step = None
        self._cached_bits = None

    def reset(self, env_ids=None):
        self._cached_step = None

    def __call__(self, env, sensor_name="hand_tactile", threshold_n=0.01):
        key = (env.common_step_counter, sensor_name, threshold_n)
        if self._cached_step != key:
            self._cached_bits = super().__call__(env, sensor_name, threshold_n)
            self._cached_step = key
        # Observation modifiers must not change the shared physical reading.
        return self._cached_bits.clone()


class PureWristWrench(ManagerTermBase):
    """Return external reaction at EEF, expressed in EEF axes, in N and Nm.

    No object states or contact identities enter the compensation. Only the
    downstream Hand model and its kinematics are used. Invalid reset readings
    are masked; they are never used as a tare that could erase real contact.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._reader = None
        self._sample = None
        self._cached_step = None

    def reset(self, env_ids=None):
        self._cached_step = None

    def _read(self, env):
        key = (env.common_step_counter, env._sim_step_counter)
        if key != self._cached_step:
            if self._reader is None:
                self._reader = PureWrenchReader(env.scene["robot"], gravity_w=env.cfg.sim.gravity)
            frame = env.scene["ee_frame"].data
            self._sample = self._reader.read(
                frame.target_pos_w[:, 0], frame.target_quat_w[:, 0],
                valid_env_mask=env.sensor_data_fresh,
            )
            self._cached_step = key
        return self._sample

    def __call__(self, env):
        if not bool(env.sensor_data_fresh.any()):
            return torch.zeros((env.num_envs, 6), device=env.device)
        return self._read(env).pure_eef.clone()

    def world_wrench(self, env):
        """Same EEF reference point, world axes: avoid frame-rotation deltas."""
        if not bool(env.sensor_data_fresh.any()):
            return torch.zeros((env.num_envs, 6), device=env.device)
        return self._read(env).pure_world.clone()
