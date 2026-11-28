"""Device-only virtual fingertip frames and frozen/live OBB transforms."""
import torch
import json
from pathlib import Path
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

FINGER_BODIES = ('inspire_thumb_force_sensor_4', 'inspire_index_force_sensor_3',
                'inspire_middle_force_sensor_3', 'inspire_ring_force_sensor_3',
                'inspire_little_force_sensor_3')


def canonical_quaternion(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Scalar first; on an exact half turn use the first nonzero vector entry.
    sign = torch.ones_like(q[..., 0])
    pending = torch.ones_like(sign, dtype=torch.bool)
    for index in range(4):
        value = q[..., index]
        sign = torch.where(pending & (value < 0.), -1., sign)
        pending = pending & (value == 0.)
    return q * sign[..., None]


def initialize_fingers(env):
    robot = env.scene['robot']
    env._finger_body_ids = torch.tensor([robot.body_names.index(name) for name in FINGER_BODIES], device=env.device)
    path = Path(__file__).resolve().parents[2] / 'assets/data/wrench_right_hand/tactile_regions.json'
    regions = {r['name']: r for r in json.loads(path.read_text())['regions']}
    offsets = []
    for name in FINGER_BODIES:
        bounds = regions[name]['pad_bounds_m']
        lo, hi = bounds['min'], bounds['max']
        offsets.append(((lo[0]+hi[0])/2., (lo[1]+hi[1])/2., hi[2]))
    env._finger_offsets = torch.tensor(offsets, device=env.device)


def distance_angle_formation(gaps, normals_e, toward_center_e, distance_scale):
    """Per-pad distance times cos² alignment; both sides of a pad are valid.

    Return the components as well as their sum for inspectable diagnostics.
    Degenerate normals/directions have zero alignment, with safe denominators.
    """
    normal_norm = normals_e.norm(dim=-1)
    direction_norm = toward_center_e.norm(dim=-1)
    cosine = ((normals_e / normal_norm.clamp_min(1e-8)[..., None]) *
              (toward_center_e / direction_norm.clamp_min(1e-8)[..., None])).sum(-1)
    valid = (normal_norm > 1e-8) & (direction_norm > 1e-8)
    alignment = torch.where(valid, cosine.clamp(-1., 1.).square(), 0.)
    proximity = distance_scale / (distance_scale + gaps)
    contribution = proximity * alignment
    return contribution.sum(-1), proximity, alignment, contribution


def update_geometry(env, ids=None, target_index=0):
    # One common path for all fingers; reset only selected rows.
    rows = slice(None) if ids is None else ids
    data = env.scene['robot'].data
    pos = data.body_link_pos_w[rows][:, env._finger_body_ids]
    quat = data.body_link_quat_w[rows][:, env._finger_body_ids]
    finger = pos + quat_apply(quat, env._finger_offsets.expand(len(pos), -1, -1))
    frame = env.scene['ee_frame'].data
    epos, equat = frame.target_pos_w[rows, 0], frame.target_quat_w[rows, 0]
    inverse = quat_conjugate(equat)[:, None].expand(-1, 5, -1)
    relative_pos = quat_apply(inverse, finger-epos[:, None])
    relative_quat = canonical_quaternion(quat_mul(inverse, quat))
    env._finger_pos_w[rows] = finger
    env._finger_pose_e[rows] = torch.cat((relative_pos, relative_quat[..., 1:], relative_quat[..., :1]), -1)
    objects = env.scene['object_collection'].data
    oq = objects.object_link_quat_w[rows, target_index]
    center = objects.object_link_pos_w[rows, target_index]+quat_apply(oq, env._obb_target_local_pos[rows])
    rotation = quat_mul(oq, env._obb_target_local_quat[rows])
    env._obb_live_pos_w[rows] = center
    env._obb_live_quat_w[rows] = rotation
    local = quat_apply_inverse(rotation[:, None].expand(-1, 5, -1), finger-center[:, None])
    gaps = (local.abs()-env._obb_target_size[rows, None]*.5).clamp_min(0.).norm(dim=-1)
    env._finger_gaps[rows] = gaps
    if getattr(env.cfg, 'formation_mode', None) == 'distance_angle_sum':
        # relative_quat remains wxyz; the stored observation uses xyzw.
        pad_z = torch.zeros_like(relative_pos)
        pad_z[..., 2] = 1.
        normals_e = quat_apply(relative_quat, pad_z)
        toward_center_e = quat_apply(inverse, center[:, None]-finger)
        env._formation[rows] = distance_angle_formation(
            gaps, normals_e, toward_center_e, env.cfg.formation_distance_scale)[0]
    else:
        env._formation[rows] = env.cfg.formation_distance_scale / (env.cfg.formation_distance_scale+gaps.sum(-1))


def obb_in_eef(env, positions, rotations, sizes):
    frame = env.scene['ee_frame'].data
    inverse = quat_conjugate(frame.target_quat_w[:, 0])
    pos = frame.target_pos_w[:, 0]
    if positions.ndim == 3:
        inverse = inverse[:, None].expand(-1, positions.shape[1], -1)
        pos = pos[:, None]
    centers = quat_apply(inverse, positions-pos)
    rotation = canonical_quaternion(quat_mul(inverse, rotations))
    # OBB sizes stay in their own axes; retain full EEF-relative 3D rotation.
    return torch.cat((centers, sizes, rotation[..., 1:], rotation[..., :1]), -1)
