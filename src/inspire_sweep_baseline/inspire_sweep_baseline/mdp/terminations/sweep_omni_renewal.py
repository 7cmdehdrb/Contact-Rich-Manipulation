"""Renewal composition with shared exponential excess-speed penalties.

Dex Tactile v1 with goal occupancy rewards continuing until timeout."""

from ..common.sweep import _arm
from ..termination_functions.goal_terminations import object_fall_or_tilt
from ..termination_functions.goal_terminations import shelf_displacement
from ..termination_functions.sweep import arm_velocity
from ..termination_functions.sweep import push_fast
from ..termination_functions.sweep import shelf_collision
from .sweep import TerminationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

    # time = None
    # success = None
    # failure = None


@configclass
class OmniRenewalTerminationsCfg(TerminationsCfg):
    shelf_collision = DoneTerm(
        func=shelf_collision, params={"threshold": 0.1, "check_fingers": False}
    )
    object_drop = DoneTerm(
        func=object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )

    # 실제로는 0.3 이상부터 Termination이 발생해야 하지만, 0.3은 페널티로서 부여하고, 학습 안정성을 위해 0.8로 대폭 상향
    push_fast = DoneTerm(func=push_fast, params={"speed_condition": 0.8})
    # 실제로는 1.0 이상부터 Termination이 발생해야 하지만, 1.0은 페널티로서 부여하고, 학습 안정성을 위해 5.0으로 대폭 상향
    hand_velocity = DoneTerm(
        func=arm_velocity, params={"asset_cfg": _arm(), "threshold": 5.0}
    )

    object_outside_shelf = None
    shelf_displacement = DoneTerm(
        func=shelf_displacement, params={"threshold": 0.05}
    )


@configclass
class DexTactileOmniRenewalV2TerminationsCfg(OmniRenewalTerminationsCfg):
    # Achievement is a reward event. Only inherited failures/timeout end runs.
    success = None
