"""Simple family: pinned effective v0_2_2_4 configuration, without historical Cfg inheritance."""
from ..termination_functions.sweep_simple import retreat_success_at_timeout
from ..termination_functions.wrench_refactor_relative_ik_test import timeout_with_metrics
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..termination_functions.final import shelf_termination
from ..termination_functions.revision import any_active_object_fall_or_tilt
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.wrench_refactor_relative_ik_test import success_at_timeout

@configclass
class SweepSimpleTerminationsCfg:
    object_fall_or_tilt = DoneTerm(func=any_active_object_fall_or_tilt, params={'bottom_clearance': 0.01, 'fail_angle': 0.9})
    shelf_failure = DoneTerm(func=shelf_termination)
    arm_joint_velocity = DoneTerm(func=arm_velocity, params={'asset_cfg': arm_entity(), 'threshold': 4.0})
    timeout = DoneTerm(func=timeout_with_metrics, time_out=True)
    success = DoneTerm(func=success_at_timeout, time_out=True)


@configclass
class SweepSimpleProfV2TerminationsCfg(SweepSimpleTerminationsCfg):
    def __post_init__(self):
        self.arm_joint_velocity.params['threshold'] = 1.0


@configclass
class SweepSimpleFingerV0TestTerminationsCfg(SweepSimpleTerminationsCfg):
    def __post_init__(self):
        self.arm_joint_velocity.params['threshold'] = 1.0


@configclass
class SweepSimpleFingerV1TerminationsCfg(SweepSimpleTerminationsCfg):
    def __post_init__(self):
        self.arm_joint_velocity.params['threshold'] = 1.


@configclass
class SweepSimpleFingerV2TerminationsCfg(SweepSimpleFingerV1TerminationsCfg):
    success = DoneTerm(func=retreat_success_at_timeout, time_out=True)
