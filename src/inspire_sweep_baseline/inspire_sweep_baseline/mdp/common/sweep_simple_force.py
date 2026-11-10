"""Target-side normal + friction samples for Semi-Prof's direction reward."""
import torch
import warp as wp

from .contact_sweep_clutter_fusion_contacts import object_hand_sensor_name


@wp.kernel
def _sum_friction_pairs(
    forces: wp.array(dtype=wp.vec3),
    counts: wp.array2d(dtype=wp.int64),
    starts: wp.array2d(dtype=wp.int64),
    result: wp.array2d(dtype=wp.vec3),
    valid: wp.array2d(dtype=wp.bool),
):
    row, pair = wp.tid()
    count = counts[row, pair]
    start = starts[row, pair]
    total = wp.vec3(0.)
    ok = count >= 0 and (count == 0 or (start >= 0 and start + count <= forces.shape[0]))
    if ok and count > 0:
        for index in range(int(start), int(start + count)):
            value = forces[index]
            ok = ok and wp.isfinite(value[0]) and wp.isfinite(value[1]) and wp.isfinite(value[2])
            total += value
    ok = ok and wp.isfinite(total[0]) and wp.isfinite(total[1]) and wp.isfinite(total[2])
    if not ok:
        total = wp.vec3(0.)
    result[row, pair] = total
    valid[row, pair] = ok


def aggregate_friction(forces, counts, starts):
    """Sum only occupied PhysX anchor ranges on the existing tensor device.

    One Warp thread per sensor/filter pair, no CPU count readback or padding
    scan. Unused buffer contents cannot contaminate other pairs' force sums.
    """
    if forces.dtype != torch.float32 or forces.ndim != 2 or forces.shape[-1] != 3:
        raise ValueError('Native friction forces must be float32 [capacity,3]')
    if counts.ndim != 2 or starts.shape != counts.shape:
        raise ValueError('Native friction counts/starts must be matching [sensor,filter] matrices')
    wp.init()
    counts, starts = counts.long().contiguous(), starts.long().contiguous()
    result = forces.new_empty((*counts.shape, 3))
    valid = torch.empty_like(counts, dtype=torch.bool)
    stream = wp.stream_from_torch(torch.cuda.current_stream(forces.device)) if forces.is_cuda else None
    wp.launch(_sum_friction_pairs, dim=tuple(counts.shape), inputs=[
        wp.from_torch(forces.contiguous(), dtype=wp.vec3), wp.from_torch(counts),
        wp.from_torch(starts), wp.from_torch(result, dtype=wp.vec3), wp.from_torch(valid)],
        device=str(forces.device), stream=stream)
    return result, valid


class SemiProfForceSamples:
    """Exactly one sample per physics step, cleared before every policy step."""

    def __init__(self, env):
        name = env.scene['object_collection'].object_names[0]
        self.view = env.scene[object_hand_sensor_name(name)].contact_physx_view
        if self.view.sensor_count != env.num_envs or self.view.filter_count < 1:
            raise RuntimeError('Semi-Prof requires one target reporting body per environment and hand filters')
        self.normal_sum = torch.zeros((env.num_envs, 3), device=env.device)
        self.friction_sum = torch.zeros_like(self.normal_sum)
        self.contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self.valid = torch.zeros_like(self.contact)
        self.samples = 0
        self.step = -1
        self.last_physics_step = -1

    def begin(self, env):
        self.normal_sum.zero_()
        self.friction_sum.zero_()
        self.contact.zero_()
        self.valid.fill_(True)
        self.samples = 0
        self.step = env.common_step_counter + 1
        self.last_physics_step = env._sim_step_counter

    def capture(self, env):
        if self.last_physics_step == env._sim_step_counter:
            return
        normal = self.view.get_contact_force_matrix(dt=env.physics_dt)
        forces, _, counts, starts = self.view.get_friction_data(dt=env.physics_dt)
        friction, valid = aggregate_friction(forces, counts, starts)
        finite = torch.isfinite(normal).all(dim=(-1, -2)) & valid.all(-1)
        self.valid &= finite
        self.contact |= finite & (normal.norm(dim=-1).sum(-1) >= .01)
        self.normal_sum += torch.where(finite[:, None], normal.sum(1), 0.)
        self.friction_sum += torch.where(finite[:, None], friction.sum(1), 0.)
        self.samples += 1
        self.last_physics_step = env._sim_step_counter

    def reset(self, env_ids):
        self.normal_sum[env_ids] = 0.
        self.friction_sum[env_ids] = 0.
        self.contact[env_ids] = False
        self.valid[env_ids] = False

    def mean(self, env):
        fresh = self.step == env.common_step_counter and self.samples == env.cfg.decimation
        force = (self.normal_sum + self.friction_sum) / max(self.samples, 1)
        return force, self.valid & self.contact & fresh
