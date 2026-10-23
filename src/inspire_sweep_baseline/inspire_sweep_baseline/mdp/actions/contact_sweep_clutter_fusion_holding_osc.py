# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Relative OSC commands expressed in the EEF axes sampled at each policy step.

Local adaptation of the checked-in UR10 Reach OSC action.

Preserves RPY conversion, nonfinite-effort rejection and effort saturation.
Robot joints and the control point are supplied by the Holding configuration."""

from __future__ import annotations

from ..action_functions.contact_sweep_clutter_fusion_holding_osc import EEFRelativeOSCAction
from ..action_functions.contact_sweep_clutter_fusion_holding_osc import UR10OperationalSpaceControllerAction
from isaaclab.envs.mdp.actions import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

@configclass
class EEFRelativeOSCActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = EEFRelativeOSCAction


@configclass
class UR10OperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """Configuration for UR10 OSC reach actions."""

    class_type: type = UR10OperationalSpaceControllerAction

    effort_limit_scale: float = 1.0
    """Fraction of the articulation effort limits available to OSC commands."""

    minimum_effort_limit: float = 1.0e-6
    """Lower bound used when clamping effort limits."""

    saturation_tolerance: float = 1.0e-6
    """Tolerance used to detect effort saturation."""
