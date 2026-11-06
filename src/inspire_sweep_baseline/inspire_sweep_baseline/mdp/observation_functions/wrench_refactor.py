"""Right-hand paired front/back sensing from individual PhysX contact points."""
import json
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from ...assets.robot import PALM_SENSOR_BODY_NAMES
from ...assets.tactile_regions import parent_sensor_name
from ..common.sensors import contact_magnitudes_and_bits
from .tactile_observations import TactileContactBits, unpack_contact_points
from .asymmetric_observations import InitialObjectPositionInRoot
from .final import tactile_observation_noise


class PairedContactRegions:
    """Extrude each pad footprint to BOTH surfaces of its parent's convex hull.

    Surface depths are computed at each contact's XY, not from a global bounding
    box. Overlapping footprints assign a contact to the nearest pad. Magnitudes
    accumulate per point, so opposing forces and multiple contacts from the same
    body pair cannot cancel a channel or be collapsed into one centroid.
    """

    def __init__(self, regions, channels, device):
        self.channels = torch.tensor(channels, device=device, dtype=torch.long)
        count = len(regions)
        self.footprint = torch.zeros((count, max(len(r['footprint_halfspaces']) for r in regions), 3), device=device)
        self.footprint[..., 2] = -1
        self.hull = torch.zeros((count, max(len(r['parent_halfspaces']) for r in regions), 4), device=device)
        self.hull[..., 3] = -1
        for i, r in enumerate(regions):
            self.footprint[i, :len(r['footprint_halfspaces'])] = torch.tensor(r['footprint_halfspaces'], device=device)
            self.hull[i, :len(r['parent_halfspaces'])] = torch.tensor(r['parent_halfspaces'], device=device)
        lo = torch.tensor([r['pad_bounds_m']['min'][:2] for r in regions], device=device)
        hi = torch.tensor([r['pad_bounds_m']['max'][:2] for r in regions], device=device)
        self.center, self.radius = (lo + hi) / 2, ((hi - lo) / 2).clamp_min(1e-6)

    def accumulate(self, output, points, forces, normals, env_ids, positions, quaternions):
        # Vendor hulls contain thousands of facets. Bound temporary memory even
        # when many environments contact the same parent simultaneously.
        for start in range(0, len(points), 256):
            end = start + 256
            self._accumulate_chunk(output, points[start:end], forces[start:end], normals[start:end],
                                   env_ids[start:end], positions, quaternions)

    def _accumulate_chunk(self, output, points, forces, normals, env_ids, positions, quaternions):
        if not len(points):
            return
        channels = self.channels
        q = quaternions[env_ids][:, channels]
        local = quat_apply_inverse(q, points[:, None] - positions[env_ids][:, channels])
        normal = quat_apply_inverse(q, normals[:, None].expand_as(local))
        xy = local[..., :2]
        footprint = (torch.einsum('mpi,pei->mpe', xy, self.footprint[..., :2]) + self.footprint[..., 2] <= .0005).all(-1)
        c = torch.einsum('mpi,pei->mpe', xy, self.hull[..., :2]) + self.hull[..., 3]
        nz = self.hull[..., 2]
        ray = -c / torch.where(nz.abs() > 1e-8, nz, 1.)
        upper = torch.where(nz > 1e-8, ray, torch.inf).amin(-1)
        lower = torch.where(nz < -1e-8, ray, -torch.inf).amax(-1)
        vertical_valid = torch.where(nz.abs() <= 1e-8, c <= .0005, True).all(-1)
        valid = footprint & vertical_valid & (upper >= lower) & (normal[..., 2].abs() >= .2)
        front_distance = (local[..., 2] - upper).abs()
        back_distance = (local[..., 2] - lower).abs()
        # A point belongs to its nearer surface, never both on a thin hull.
        is_back = back_distance < front_distance
        valid &= torch.minimum(front_distance, back_distance) <= .0015
        score = ((xy - self.center) / self.radius).square().sum(-1)
        selected = torch.where(valid, score, torch.inf).argmin(-1)
        row = torch.arange(len(points), device=points.device)
        channel = channels[selected] + 17 * is_back[row, selected].long()
        magnitude = forces.norm(dim=-1)
        index = env_ids * 34 + channel
        output.flatten().index_add_(0, index, torch.where(valid.any(-1), magnitude, 0.))


class RightHandTactile(TactileContactBits):
    """17 front channels followed by their 17 matched back channels."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        path = Path(__file__).resolve().parents[2] / 'assets/data/wrench_right_hand/tactile_regions.json'
        regions = json.loads(path.read_text())['regions']
        if tuple(r['name'] for r in regions) != PALM_SENSOR_BODY_NAMES:
            raise RuntimeError('Right tactile metadata channel mismatch')
        names = tuple(env.scene['robot'].body_names)
        self._robot_ids = torch.tensor([names.index(n) for n in PALM_SENSOR_BODY_NAMES], device=env.device)
        self._parents = []
        for parent in dict.fromkeys(r['parent_name'] for r in regions):
            channels = [i for i, r in enumerate(regions) if r['parent_name'] == parent]
            self._parents.append((parent_sensor_name(parent), PairedContactRegions([regions[i] for i in channels], channels, env.device)))

    def reset(self, env_ids=None):
        self._env._refactor_tactile_sample = None

    def __call__(self, env, sensor_name='hand_tactile', threshold_n=.05):
        key = (env.common_step_counter, env._sim_step_counter, threshold_n)
        cached = getattr(env, '_refactor_tactile_sample', None)
        if cached is not None and cached[0] == key:
            return cached[1].clone()
        direct = self._read_forces(env, sensor_name)
        magnitude = torch.zeros((env.num_envs, 34), device=env.device)
        magnitude[:, :17], _ = contact_magnitudes_and_bits(direct, threshold_n)
        if bool(env.sensor_data_fresh.any()):
            raw = [env.scene[name].contact_physx_view.get_contact_data(dt=env.physics_dt) for name, _ in self._parents]
            totals = torch.stack([torch.where(env.sensor_data_fresh[:, None], r[4].long(), 0).sum() for r in raw]).cpu().tolist()
            data = env.scene['robot'].data
            pos, rot = data.body_link_pos_w[:, self._robot_ids], data.body_link_quat_w[:, self._robot_ids]
            live_pos = torch.where(env.sensor_data_fresh[:, None, None], pos, 0.)
            live_rot = torch.where(env.sensor_data_fresh[:, None, None], rot, rot.new_tensor([1.,0.,0.,0.]))
            torch._assert_async(torch.isfinite(live_pos).all() & torch.isfinite(live_rot).all()
                                & ((live_rot.square().sum(-1) - 1.).abs() < 1e-3).all(),
                                'Invalid right-hand tactile poses')
            for (_, regions), contact, total in zip(self._parents, raw, totals):
                if total:
                    points, forces, normals, ids = unpack_contact_points(contact, env.sensor_data_fresh, env.num_envs, total)
                    regions.accumulate(magnitude, points, forces, normals, ids, live_pos, live_rot)
        bits = ((magnitude >= threshold_n) & env.sensor_data_fresh[:, None]).float()
        env._refactor_tactile_sample = (key, bits)
        return bits.clone()


def root_positions_in_eef(env, positions):
    """Transform fixed-base positions into the current offset EEF frame."""
    robot, eef = env.scene['robot'].data, env.scene['ee_frame'].data
    root_p, root_q = robot.root_pos_w, robot.root_quat_w
    eef_p, eef_q = eef.target_pos_w[:, 0], eef.target_quat_w[:, 0]
    if positions.ndim == 3:
        root_p, root_q = root_p[:, None], root_q[:, None]
        eef_p, eef_q = eef_p[:, None], eef_q[:, None]
    world = root_p + quat_apply(root_q.expand(*positions.shape[:-1], 4), positions)
    return quat_apply_inverse(eef_q.expand(*positions.shape[:-1], 4), world - eef_p)


class InitialObjectPositionInEEF(InitialObjectPositionInRoot):
    """Re-express the Actor snapshot, including its fixed root-frame bias.

    Bias is sampled once per reset, before changing frames. Repeated EEF
    motion cannot expose live target state or resample the visual estimate.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._bias_b = torch.zeros_like(self._position_b)

    def __call__(self, env, command_name='target_goal_pos', coordinate_noise=.01):
        ids = self._pending.nonzero(as_tuple=False).flatten()
        if ids.numel():
            self._bias_b[ids] = (2 * torch.rand_like(self._bias_b[ids]) - 1) * coordinate_noise
        position = super().__call__(env, command_name)
        if env.cfg.observations.policy.enable_corruption:
            position = position + self._bias_b
        return root_positions_in_eef(env, position)


def goal_position_in_eef(env, command_name='target_goal_pos'):
    """Episode-fixed goal XYZ relative to the current EEF, without live target GT."""
    eef = env.scene['ee_frame'].data
    goal = env.command_manager.get_command(command_name)[:, :3]
    return quat_apply_inverse(eef.target_quat_w[:, 0], goal - eef.target_pos_w[:, 0])


def initial_obstacle_positions_in_eef(env):
    """Change only snapshot coordinates; preserve visibility, noise and row order."""
    rows = env._initial_obstacle_observation.unflatten(-1, (16, 4))
    positions = root_positions_in_eef(env, rows[..., 1:])
    relative = torch.cat((rows[..., :1], positions), dim=-1)
    return torch.where(rows[..., :1] > .5, relative, 0.).flatten(1)


def encode_paired_tactile(bits):
    palm, dorsal = bits[..., :17], bits[..., 17:]
    return palm - dorsal + 2 * palm * dorsal


class RelativeRightHandTactile(RightHandTactile):
    """Seventeen paired states: none=0, palm=1, dorsal=-1, both=2."""

    def __call__(self, env, sensor_name='hand_tactile', threshold_n=.05):
        return encode_paired_tactile(super().__call__(env, sensor_name, threshold_n))


def paired_tactile_observation_noise(data, cfg):
    """Preserve the parent's independent 1% bit flips before paired encoding."""
    palm = ((data == 1) | (data == 2)).to(data.dtype)
    dorsal = ((data == -1) | (data == 2)).to(data.dtype)
    return encode_paired_tactile(tactile_observation_noise(torch.cat((palm, dorsal), -1), cfg))
