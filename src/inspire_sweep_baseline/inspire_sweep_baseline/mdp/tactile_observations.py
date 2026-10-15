"""Independent contact bits in the existing Hand sensor-link order."""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse

from ..assets.robot import PALM_SENSOR_BODY_NAMES
from .sensors import contact_magnitudes_and_bits


class TactileContactBits(ManagerTermBase):
    """Read each pad separately and keep its policy index stable across USD order.

    One ContactSensor groups the existing rigid bodies; its force vectors are
    never summed across pads. Reset rows are masked before force validation.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._sensor_name = cfg.params.get("sensor_name", "hand_tactile")
        names = tuple(env.scene[self._sensor_name].body_names)
        expected = PALM_SENSOR_BODY_NAMES
        if len(names) != len(expected) or set(names) != set(expected):
            raise RuntimeError(
                f"Expected exactly the {len(expected)} tactile sensor bodies; "
                f"missing={sorted(set(expected) - set(names))}, "
                f"extra={sorted(set(names) - set(expected))}, actual={names}"
            )
        self._body_ids = torch.tensor(
            [names.index(name) for name in expected], device=env.device, dtype=torch.long,
        )

    def __call__(self, env, sensor_name="hand_tactile", threshold_n=0.05):
        _, bits = contact_magnitudes_and_bits(self._read_forces(env, sensor_name), threshold_n)
        return bits.to(torch.float32)

    def _read_forces(self, env, sensor_name):
        if sensor_name != self._sensor_name:
            raise ValueError("sensor_name must match the initialized tactile sensor")
        count = len(PALM_SENSOR_BODY_NAMES)
        fresh = env.sensor_data_fresh
        if not bool(fresh.any()):
            return torch.zeros((env.num_envs, count, 3), device=env.device)
        forces = env.scene[sensor_name].data.net_forces_w
        if forces.shape != (env.num_envs, count, 3):
            raise RuntimeError(
                f"Expected tactile force shape ({env.num_envs}, {count}, 3), got {tuple(forces.shape)}"
            )
        forces = torch.where(fresh[:, None, None], forces, torch.zeros_like(forces))
        return forces.index_select(1, self._body_ids)


class ParentContactRegions:
    """Map individual parent contacts onto immutable sensor-local footprints.

    Only points in the outward extrusion of an original pad polygon count.
    Overlapping footprints assign each point once, to the nearest normalized
    pad center. All kernels batch contacts/environments; no per-env CPU loop.
    """

    def __init__(self, regions, channels, device):
        self.channels = torch.tensor(channels, dtype=torch.long, device=device)
        count, edges = len(regions), max(len(region["footprint_halfspaces"]) for region in regions)
        self.planes = torch.zeros((count, edges, 3), device=device)
        # Padding planes always pass the <= tolerance test.
        self.planes[..., 2] = -1.0
        for index, region in enumerate(regions):
            plane = torch.tensor(region["footprint_halfspaces"], device=device)
            self.planes[index, :len(plane)] = plane
        lower = torch.tensor([region["pad_bounds_m"]["min"][:2] for region in regions], device=device)
        upper = torch.tensor([region["pad_bounds_m"]["max"][:2] for region in regions], device=device)
        self.center_xy = (lower + upper) / 2
        self.radius_xy = ((upper - lower) / 2).clamp_min(1.0e-6)
        self.z_min = torch.tensor([region["outward_z_min_m"] for region in regions], device=device)
        self.z_max = torch.tensor([region["outward_z_max_m"] for region in regions], device=device)

    def accumulate(self, output, points, forces, normals, env_ids, positions, quaternions):
        if not points.numel():
            return
        count = len(self.channels)
        quat = quaternions[env_ids][:, self.channels]
        delta = points[:, None, :] - positions[env_ids][:, self.channels]
        local = quat_apply_inverse(quat.reshape(-1, 4), delta.reshape(-1, 3)).reshape(-1, count, 3)
        normal = normals[:, None, :].expand(-1, count, -1)
        local_normal = quat_apply_inverse(quat.reshape(-1, 4), normal.reshape(-1, 3)).reshape(-1, count, 3)
        plane_distance = torch.einsum("mpd,ped->mpe", local[..., :2], self.planes[..., :2]) + self.planes[..., 2]
        inside = (
            (plane_distance <= 0.0005).all(dim=-1)
            & (local[..., 2] >= self.z_min)
            & (local[..., 2] <= self.z_max)
            # Normal orientation conventions differ; front/back is determined
            # by z bounds. Reject nearly tangential parent-side contacts.
            & (local_normal[..., 2].abs() >= 0.2)
        )
        score = ((local[..., :2] - self.center_xy) / self.radius_xy).square().sum(dim=-1)
        selected = torch.where(inside, score, torch.inf).argmin(dim=-1)
        valid = inside.any(dim=-1)
        flat_ids = env_ids * output.shape[1] + self.channels[selected]
        # Keep fixed-size GPU buffers: boolean indexing would synchronize CUDA
        # to allocate a dynamically sized contact list on every parent view.
        output.view(-1, 3).index_add_(0, flat_ids, torch.where(valid[:, None], forces, 0.0))


def unpack_contact_points(raw, fresh, num_envs, total_contacts=None):
    """Gather live per-pair slices with a known allocation size.

    The runtime batches all parent counts into one host readback, then supplies
    total_contacts to avoid a separate CUDA synchronization per parent view.
    Unused/reset slices are never gathered or classified.
    """
    scalars, points, normals, _, counts, starts = raw
    if counts.ndim != 2 or counts.shape[0] != num_envs or starts.shape != counts.shape:
        raise RuntimeError("Expected contact counts/starts shaped (num_envs, filters) for one parent body")
    if counts.shape[1] == 0 or points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or scalars.numel() != len(points):
        raise RuntimeError("Expected a filtered tactile contact buffer with scalar forces and 3D points/normals")
    # PhysX provides uint32 indices; torch indexing requires signed long.
    counts, starts = counts.to(torch.long), starts.to(torch.long)
    filter_count = counts.shape[1]
    counts = torch.where(fresh[:, None], counts, 0)
    live = counts > 0
    torch._assert_async(
        ~((counts < 0) | (live & ((starts < 0) | (starts + counts > len(points))))).any(),
        "Invalid or truncated tactile contact buffer",
    )
    total = int(counts.sum()) if total_contacts is None else total_contacts
    if total == 0:
        return points[:0], points[:0], normals[:0], torch.empty(0, dtype=torch.long, device=points.device)
    torch._assert_async(counts.sum() < len(points), "Tactile contact buffer full; increase max_contact_data_count_per_prim")
    counts = counts.flatten()
    rows = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts, output_size=total,
    )
    block_starts = counts.cumsum(0) - counts
    offsets = torch.arange(total, device=counts.device) - block_starts[rows]
    indices = starts.flatten()[rows] + offsets
    position, normal = points[indices], normals[indices]
    force = scalars[indices].reshape(-1, 1) * normal
    torch._assert_async(
        torch.isfinite(position).all() & torch.isfinite(normal).all() & torch.isfinite(force).all(),
        "Tactile contact point data contains non-finite values",
    )
    return position, force, normal, rows // filter_count


class ProjectedTactileContactBits(TactileContactBits):
    """Direct pad forces plus localized parent contacts, without physics edits.

    Parent contacts are filtered to the task's primitive cylinder. Imported
    shelf/static ground filters are unsupported on the GPU tensor backend.
    Points outside all pad footprints never activate a channel. Unfiltered
    direct pad contacts remain supported for every collision counterpart.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from ..assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, TACTILE_REGIONS, parent_sensor_name

        robot_names = tuple(env.scene["robot"].body_names)
        if any(robot_names.count(name) != 1 for name in PALM_SENSOR_BODY_NAMES):
            raise RuntimeError("Robot must contain each tactile sensor body exactly once")
        self._robot_ids = torch.tensor(
            [robot_names.index(name) for name in PALM_SENSOR_BODY_NAMES], device=env.device, dtype=torch.long,
        )
        self._parents = []
        for parent in TACTILE_PARENT_BODY_NAMES:
            sensor_name = parent_sensor_name(parent)
            if tuple(env.scene[sensor_name].body_names) != (parent,):
                raise RuntimeError(f"Expected one tactile parent body {parent}")
            channels = [index for index, region in enumerate(TACTILE_REGIONS) if region["parent_name"] == parent]
            regions = [TACTILE_REGIONS[index] for index in channels]
            self._parents.append((sensor_name, ParentContactRegions(regions, channels, env.device)))

    def __call__(self, env, sensor_name="hand_tactile", threshold_n=0.05):
        forces = self._read_forces(env, sensor_name)
        if bool(env.sensor_data_fresh.any()):
            # One count readback for all parents; no full-capacity polygon work
            # or per-parent synchronizations for scenes without contacts.
            raw_contacts = [
                env.scene[name].contact_physx_view.get_contact_data(dt=env.physics_dt)
                for name, _ in self._parents
            ]
            totals = torch.stack([
                torch.where(env.sensor_data_fresh[:, None], raw[4].to(torch.long), 0).sum()
                for raw in raw_contacts
            ]).cpu().tolist()
            if not any(totals):
                _, bits = contact_magnitudes_and_bits(forces, threshold_n)
                return bits.to(torch.float32)
            data = env.scene["robot"].data
            positions = data.body_link_pos_w.index_select(1, self._robot_ids)
            quaternions = data.body_link_quat_w.index_select(1, self._robot_ids)
            live_positions = torch.where(env.sensor_data_fresh[:, None, None], positions, 0.0)
            live_quaternions = torch.where(
                env.sensor_data_fresh[:, None, None], quaternions,
                quaternions.new_tensor([1.0, 0.0, 0.0, 0.0]),
            )
            torch._assert_async(
                torch.isfinite(live_positions).all() & torch.isfinite(live_quaternions).all()
                & ((live_quaternions.square().sum(dim=-1) - 1.0).abs() < 1.0e-3).all(),
                "Invalid tactile sensor link poses",
            )
            mapped = torch.zeros_like(forces)
            for (_, regions), raw, total in zip(self._parents, raw_contacts, totals):
                if total == 0:
                    continue
                points, point_forces, normals, env_ids = unpack_contact_points(
                    raw, env.sensor_data_fresh, env.num_envs, total_contacts=total,
                )
                regions.accumulate(mapped, points, point_forces, normals, env_ids, live_positions, live_quaternions)
            _, direct_bits = contact_magnitudes_and_bits(forces, threshold_n)
            _, projected_bits = contact_magnitudes_and_bits(mapped, threshold_n)
            return (direct_bits | projected_bits).to(torch.float32)
        _, bits = contact_magnitudes_and_bits(forces, threshold_n)
        return bits.to(torch.float32)
