"""Target-only palm/dorsal contact preference; no noisy observation is consumed."""

import math

import torch
from isaaclab.utils.math import quat_apply_inverse

from ...assets.robot import HAND_CONTACT_BODY_NAMES, PALM_SENSOR_BODY_NAMES
from ..common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
from ..observation_functions.tactile_observations import unpack_contact_points
from ..observation_functions.wrench_refactor import RightHandTactile
from .stabilize_sweep import sweep_progress


def sweep_region(direction_shelf):
    """Robot faces shelf -X: 12=-X, 3=+Y, 6=+X, 9=-Y.

    Regions 1/3 span 60 degrees, including their +/-30-degree boundaries.
    Regions 2/4 fill the remaining right/left 120 degrees. No deadzone.
    """
    angle = torch.atan2(direction_shelf[..., 1], -direction_shelf[..., 0])
    absolute = angle.abs()
    return torch.where(
        absolute <= math.pi / 6, 1,
        torch.where(absolute >= 5 * math.pi / 6, 3, torch.where(angle > 0, 2, 4)),
    )


def contact_side_multiplier(regions, palm_load, dorsal_load, preferred_multiplier=1.5,
                            other_multiplier=0.7):
    """Normal-load-weighted mixture; unclassified contacts are neutral (1)."""
    if not (math.isfinite(preferred_multiplier) and math.isfinite(other_multiplier)
            and preferred_multiplier > 0 and other_multiplier > 0):
        raise ValueError('Contact-side multipliers must be positive and finite')
    total = palm_load + dorsal_load
    preferred_load = torch.where(regions <= 2, dorsal_load, palm_load)
    weighted = (preferred_multiplier * preferred_load
                + other_multiplier * (total - preferred_load)) / total.clamp_min(1e-12)
    return torch.where(total > 1e-12, weighted, 1.)


def target_contact_buffer(raw):
    """Parent views filter the fixed target first, then the sixteen obstacles."""
    scalars, points, normals, separation, counts, starts = raw
    return scalars, points, normals, separation, counts[:, :1], starts[:, :1]


class ContactSideSweepProgress(RightHandTactile):
    """Reuse real paired surface geometry, restricted to contacts with the target.

    Direct pad contacts contribute palm load; parent convex-hull contacts are
    localized to the same front/back surfaces used by the existing 34-channel
    tactile implementation. Neither bit thresholds nor observation noise enter
    the load mixture. Unmapped edges contribute no side preference.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        target_name = env.scene['object_collection'].object_names[0]
        self._target_sensor = env.scene[object_hand_sensor_name(target_name)]
        self._pad_filters = torch.tensor(
            [HAND_CONTACT_BODY_NAMES.index(name) for name in PALM_SENSOR_BODY_NAMES],
            device=env.device, dtype=torch.long,
        )

    def _surface_loads(self, env):
        magnitude = torch.zeros((env.num_envs, 34), device=env.device)
        fresh = env.sensor_data_fresh
        if not bool(fresh.any()):
            return magnitude[:, :17].sum(-1), magnitude[:, 17:].sum(-1)
        # This object-reporting sensor is filtered to hand bodies only. Avoid
        # the unfiltered pad net_forces, which also include shelf/obstacles.
        matrix = self._target_sensor.data.force_matrix_w
        if matrix.shape != (env.num_envs, 1, len(HAND_CONTACT_BODY_NAMES), 3):
            raise RuntimeError('Expected target-to-hand contact matrix in hand filter order')
        direct = matrix[:, 0].index_select(1, self._pad_filters).norm(dim=-1)
        magnitude[:, :17] = torch.where(fresh[:, None], direct, 0.)
        raw = [target_contact_buffer(env.scene[name].contact_physx_view.get_contact_data(dt=env.physics_dt))
               for name, _ in self._parents]
        totals = torch.stack([torch.where(fresh[:, None], item[4].long(), 0).sum()
                              for item in raw]).cpu().tolist()
        data = env.scene['robot'].data
        pos, rot = data.body_link_pos_w[:, self._robot_ids], data.body_link_quat_w[:, self._robot_ids]
        live_pos = torch.where(fresh[:, None, None], pos, 0.)
        live_rot = torch.where(fresh[:, None, None], rot, rot.new_tensor([1., 0., 0., 0.]))
        for (_, regions), contact, total in zip(self._parents, raw, totals):
            if total:
                points, forces, normals, ids = unpack_contact_points(contact, fresh, env.num_envs, total)
                regions.accumulate(magnitude, points, forces, normals, ids, live_pos, live_rot)
        return magnitude[:, :17].sum(-1), magnitude[:, 17:].sum(-1)

    def __call__(self, env, preferred_multiplier=1.5, other_multiplier=0.7):
        # Use the episode's intended sweep direction, not a near-goal vector
        # whose angle can flip after an overshoot. Rotation removes shelf yaw.
        direction = quat_apply_inverse(env.scene['shelf'].data.root_quat_w, env.sweep_dir)
        regions = sweep_region(direction)
        palm, dorsal = self._surface_loads(env)
        factor = contact_side_multiplier(regions, palm, dorsal, preferred_multiplier, other_multiplier)
        return sweep_progress(env) * factor


class SignedContactSideSweepProgress(ContactSideSweepProgress):
    """Keep surface evidence and positive preference; bad angles use unit cost."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._step = -1
        self._value = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._value[env_ids if env_ids is not None else slice(None)] = 0.

    def __call__(self, env, preferred_multiplier=1.5, other_multiplier=0.7):
        if self._step != env.common_step_counter:
            state = env._stabilize_state
            palm, dorsal = self._surface_loads(env)
            factor = contact_side_multiplier(state.reach_region, palm, dorsal,
                                             preferred_multiplier, other_multiplier)
            state.sweep_side_multiplier.copy_(factor)
            state.sweep_positive.copy_(state.signed_progress.clamp_min(0.) * factor)
            state.sweep_direction_negative.copy_(state.signed_progress.clamp_max(0.))
            self._value.copy_((state.sweep_positive + state.sweep_direction_negative) / env.step_dt)
            self._step = env.common_step_counter
        return self._value
