from ..termination_functions.wrench_refactor_relative_ik_test import timeout_with_metrics
from isaaclab.managers import TerminationTermCfg as DoneTerm, SceneEntityCfg
from isaaclab.utils import configclass
from ..common.contact_sweep_clutter_fusion import arm_entity
from ..termination_functions.final import shelf_termination
from ..termination_functions.revision import any_active_object_fall_or_tilt
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.prof_revision import success_at_timeout, tcp_linear_velocity, success_at_timeout_tcp

@configclass
class ProfRevisionTerminationsCfg:
    object_fall_or_tilt = DoneTerm(func=any_active_object_fall_or_tilt, params={'bottom_clearance': 0.01, 'fail_angle': 0.9})
    shelf_failure = DoneTerm(func=shelf_termination)
    arm_joint_velocity = DoneTerm(func=arm_velocity, params={'asset_cfg': arm_entity(), 'threshold': 1.0})
    timeout = DoneTerm(func=timeout_with_metrics, time_out=True)
    success = DoneTerm(func=success_at_timeout, time_out=True)


@configclass
class ProfRevisionV0_3TerminationsCfg(ProfRevisionTerminationsCfg):
    arm_joint_velocity = None
    tcp_linear_velocity = DoneTerm(
        func=tcp_linear_velocity,
        params={'asset_cfg': SceneEntityCfg('robot', body_names=['wrist_3_link']), 'threshold': 4.0},
    )
    success = DoneTerm(func=success_at_timeout_tcp, time_out=True)
