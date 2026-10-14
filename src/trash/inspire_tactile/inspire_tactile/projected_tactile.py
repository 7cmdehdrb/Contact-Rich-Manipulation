"""Attribute actual parent-body contacts to virtual pad regions, without physics edits."""

import math
import numpy as np
import torch

from .regions import load_regions, observer_key
from .tactile import TactileReader, binary_contact


def assign_regions(points, positions, quaternions, equations, tolerance):
    """Return a local region index per world point, or -1 when outside all regions.

    Quaternion order is wxyz. Overlaps go to the smallest signed plane-distance
    score, with stable channel ordering as tie-breaker. No force duplication.
    """
    scores = []
    for pos, quat, planes in zip(positions, quaternions, equations):
        delta = points - pos
        qvec = -quat[1:]
        cross = 2 * np.cross(qvec, delta)
        local = delta + quat[0] * cross + np.cross(qvec, cross)
        scores.append((local @ planes[:, :3].T + planes[:, 3]).max(axis=-1))
    scores = np.stack(scores, axis=-1)
    selected = scores.argmin(axis=-1)
    return np.where(scores.min(axis=-1) <= tolerance, selected, -1)


class ProjectedTactileReader(TactileReader):
    force_source = "contact_region_projection"

    def __init__(self, scene, urdf, names, threshold=0.01, tolerance=0.002, dt=1 / 240):
        super().__init__(scene["tactile"], names, threshold)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("region tolerance must be finite and non-negative")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("physics dt must be finite and positive")
        self.tolerance, self.dt = tolerance, dt
        self.regions = []
        for parent, regions in load_regions(urdf, names).items():
            observer = scene[observer_key(parent)]
            if observer.body_names != [parent]:
                raise RuntimeError(f"Expected one parent body {parent}: {observer.body_names}")
            self.regions.append((observer, [r[0] for r in regions],
                                 [r[1].astype(np.float32) for r in regions]))
        self.markers = None
        if self.sensor.cfg.debug_vis:
            from isaaclab.markers import VisualizationMarkers
            self.sensor.set_debug_vis(False)
            self.markers = VisualizationMarkers(self.sensor.cfg.visualizer_cfg.replace(
                prim_path="/Visuals/ContactRegionProjection"))

    def read(self):
        raw = self.sensor.data.net_forces_w[:, self.indices, :]
        # A handful of contact points is cheaper to classify on CPU than to
        # launch/synchronize many tiny CUDA kernels for each pad/pair.
        mapped = np.zeros(tuple(raw.shape), dtype=np.float32)
        unmapped = np.zeros((raw.shape[0], 3), dtype=np.float32)
        unobserved = np.zeros_like(unmapped)
        unmapped_count = np.zeros(raw.shape[0], dtype=np.int64)
        poses = self.sensor.data
        positions = poses.pos_w[:, self.indices].cpu().numpy()
        quaternions = poses.quat_w[:, self.indices].cpu().numpy()
        if (not np.isfinite(positions).all() or not np.isfinite(quaternions).all()
                or not np.allclose(np.linalg.norm(quaternions, axis=-1), 1.0, atol=1e-3)):
            raise RuntimeError("Invalid sensor poses; cannot classify contact points")
        for observer, channels, planes in self.regions:
            data = observer.data
            parent_net = data.net_forces_w[:, 0].cpu().numpy()
            if not np.isfinite(parent_net).all():
                raise RuntimeError("Non-finite parent contact force")
            if data.force_matrix_w is None:
                # No cylinder in this scene: no supported per-point opponent.
                unobserved += parent_net
                unmapped += parent_net
                continue
            matrix = data.force_matrix_w[:, 0].cpu().numpy()
            # Forces outside the supported opponent filters must stay visible
            # in diagnostics, not disappear or be assigned to a guessed pad.
            residual = parent_net - matrix.sum(axis=1)
            unobserved += residual
            unmapped += residual
            view = observer.contact_physx_view
            scalars, points, normals, _, counts, starts = view.get_contact_data(dt=self.dt)
            counts_cpu, starts_cpu = counts.cpu().tolist(), starts.cpu().tolist()
            if sum(sum(row) for row in counts_cpu) >= len(scalars):
                raise RuntimeError("Contact point buffer full; increase max_contact_data_count_per_prim")
            if any(any(row) for row in counts_cpu):
                scalars = scalars.cpu().numpy()
                points = points.cpu().numpy()
                normals = normals.cpu().numpy()
            for env_id, row in enumerate(counts_cpu):
                for filter_id, count in enumerate(row):
                    start = starts_cpu[env_id][filter_id]
                    if count == 0:
                        if not np.allclose(matrix[env_id, filter_id], 0, atol=1e-5):
                            raise RuntimeError("Missing contact points for nonzero force")
                        continue
                    if start < 0 or start + count > len(scalars):
                        raise RuntimeError("Invalid/truncated contact point buffer")
                    force = scalars[start:start + count].reshape(-1, 1) * normals[start:start + count]
                    contact = points[start:start + count]
                    if not np.isfinite(contact).all() or not np.isfinite(force).all():
                        raise RuntimeError("Non-finite contact point data")
                    if not np.allclose(force.sum(0), matrix[env_id, filter_id], atol=1e-4, rtol=1e-3):
                        raise RuntimeError("Contact point forces do not reconstruct pair force; possible buffer overflow")
                    selected = assign_regions(contact, positions[env_id, channels],
                                              quaternions[env_id, channels], planes, self.tolerance)
                    for local_id, channel in enumerate(channels):
                        mapped[env_id, channel] += force[selected == local_id].sum(0)
                    unmapped[env_id] += force[selected < 0].sum(0)
                    unmapped_count[env_id] += (selected < 0).sum()
        self.raw_forces_w = raw
        self.mapped_forces_w = torch.as_tensor(mapped, device=raw.device)
        self.unmapped_forces_w = torch.as_tensor(unmapped, device=raw.device)
        self.unmapped_count = torch.as_tensor(unmapped_count, device=raw.device)
        self.unobserved_forces_w = torch.as_tensor(unobserved, device=raw.device)
        total = raw + self.mapped_forces_w
        norms, bits = binary_contact(total, self.threshold)
        if self.markers is not None:
            # Isaac Lab contact marker order: 0=contact, 1=no contact.
            self.markers.visualize(positions.reshape(-1, 3), marker_indices=(1 - bits).reshape(-1).long())
        return total, norms, bits

    def diagnostics(self):
        return {**super().diagnostics(), "region_tolerance_m": self.tolerance,
                "mapped_force_w_N": self.mapped_forces_w.cpu().tolist(),
                "unmapped_parent_force_w_N": self.unmapped_forces_w.cpu().tolist(),
                "unobserved_parent_force_w_N": self.unobserved_forces_w.cpu().tolist(),
                "unmapped_parent_contact_count": self.unmapped_count.cpu().tolist()}


def make_reader(scene, urdf, model, args, dt):
    if args.tactile_mode == "raw":
        # Match debug colors to the same threshold as the transmitted binary bits.
        scene["tactile"].cfg.force_threshold = args.threshold
        return TactileReader(scene["tactile"], model["sensor_names"], args.threshold)
    return ProjectedTactileReader(scene, urdf, model["sensor_names"], args.threshold, args.region_tolerance, dt)


def add_arguments(parser):
    parser.add_argument("--tactile_mode", choices=("projected", "raw"), default="projected",
                        help="Observation only; raw: sensor rigid bodies; projected: also map parent contact points")
    parser.add_argument("--region_tolerance", type=float, default=0.002,
                        help="Virtual pad halfspace tolerance [m], NOT a physics contact offset")
