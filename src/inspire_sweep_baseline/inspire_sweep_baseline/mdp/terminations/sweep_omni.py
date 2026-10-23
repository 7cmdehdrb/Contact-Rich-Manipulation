"""Baseline mechanics with realistic physics and goal-based omnidirectional rewards."""

from ..common.sweep_omni import FAILURE_TERMS
from ..common.sweep_omni import OBJECT_RADIUS
from ..common.sweep_omni import SHELF_HALF_EXTENTS
from ..common.sweep_omni import SHELF_SURFACE_HEIGHT
from ..termination_functions.goal_terminations import StableGoalSuccess
from ..termination_functions.goal_terminations import object_fall_or_tilt
from ..termination_functions.goal_terminations import object_outside_shelf
from ..termination_functions.goal_terminations import shelf_displacement
from .sweep import TerminationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
import math as math

@configclass
class OmniTerminationsCfg(TerminationsCfg):
    object_drop = DoneTerm(
        func=object_fall_or_tilt,
        params={"height_condition": 1.04, "fail_angle": 0.9},
    )
    object_outside_shelf = DoneTerm(
        func=object_outside_shelf,
        params={"half_extents": SHELF_HALF_EXTENTS},
    )
    shelf_displacement = DoneTerm(
        func=shelf_displacement, params={"threshold": 0.05}
    )
    # Keep this after all failure terms: the hold counter reads their current results.
    success = DoneTerm(
        func=StableGoalSuccess,
        params={
            "position_tolerance": 0.01,
            "speed_tolerance": 0.02,
            "upright_angle": math.radians(10.0),
            "hold_time": 0.3,
            "surface_height": SHELF_SURFACE_HEIGHT,
            "support_tolerance": 0.01,
            "half_extents": SHELF_HALF_EXTENTS,
            "object_radius": OBJECT_RADIUS,
            "failure_terms": FAILURE_TERMS,
        },
    )
