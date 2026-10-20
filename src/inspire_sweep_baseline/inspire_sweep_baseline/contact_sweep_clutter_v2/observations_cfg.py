"""Share the same compensated physical sample between observations and reward."""

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

from ..contact_sweep_clutter_v1.observations_cfg import ContactSweepClutterV1ObservationsCfg
from .wrench import PureWrenchReader


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


@configclass
class ContactSweepClutterV2ObservationsCfg(ContactSweepClutterV1ObservationsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.wrist_wrench.func = PureWristWrench
        self.policy.wrist_wrench.params = {}
