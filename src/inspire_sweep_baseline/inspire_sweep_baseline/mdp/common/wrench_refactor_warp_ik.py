"""Fused reset IK using the original USD-chain quaternion/DLS equations.

Imported only by Fast resets. The CPU Warp device is an explicit test backend;
production follows the tensor device and never transfers a CUDA batch to CPU.
"""
import torch
import warp as wp

V6 = wp.types.vector(length=6, dtype=wp.float32)
M6 = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
P6 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)


@wp.func
def mul(a: wp.vec4, b: wp.vec4):
    av, bv = wp.vec3(a[1], a[2], a[3]), wp.vec3(b[1], b[2], b[3])
    v = a[0] * bv + b[0] * av + wp.cross(av, bv)
    return wp.vec4(a[0] * b[0] - wp.dot(av, bv), v[0], v[1], v[2])


@wp.func
def conjugate(q: wp.vec4):
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def rotate(q: wp.vec4, v: wp.vec3):
    xyz = wp.vec3(q[1], q[2], q[3])
    t = 2.0 * wp.cross(xyz, v)
    return v + q[0] * t + wp.cross(xyz, t)


@wp.func
def angular_error(q: wp.vec4):
    value = q
    if value[0] < 0.0:
        value = -value
    xyz = wp.vec3(value[1], value[2], value[3])
    half = wp.atan2(wp.length(xyz), value[0])
    angle = 2.0 * half
    scale = 0.5 - angle * angle / 48.0
    if wp.abs(angle) > 1.0e-6:
        scale = wp.sin(half) / angle
    return xyz / scale


@wp.func
def fk(q: V6, p0: wp.array(dtype=wp.vec3), r0: wp.array(dtype=wp.vec4),
       p1: wp.array(dtype=wp.vec3), r1: wp.array(dtype=wp.vec4),
       axes: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
       offset_p: wp.vec3, offset_q: wp.vec4):
    p, r = wp.vec3(0.0), wp.vec4(1.0, 0.0, 0.0, 0.0)
    origins, directions = P6(), P6()
    for j in range(columns.shape[0]):
        p = p + rotate(r, p0[j])
        r = mul(r, r0[j])
        col = columns[j]
        if col >= 0:
            axis = wp.vec3(0.0)
            axis[axes[j]] = 1.0
            direction = rotate(r, axis)
            for k in range(3):
                origins[col, k] = p[k]
                directions[col, k] = direction[k]
            motion = wp.vec4(wp.cos(q[col] * 0.5), 0.0, 0.0, 0.0)
            motion[axes[j] + 1] = wp.sin(q[col] * 0.5)
            r = mul(r, motion)
        r = mul(r, conjugate(r1[j]))
        p = p - rotate(r, p1[j])
    p = p + rotate(r, offset_p)
    r = mul(r, offset_q)
    jacobian = M6()
    for j in range(6):
        axis = wp.vec3(directions[j, 0], directions[j, 1], directions[j, 2])
        origin = wp.vec3(origins[j, 0], origins[j, 1], origins[j, 2])
        linear = wp.cross(axis, p - origin)
        for k in range(3):
            jacobian[k, j] = linear[k]
            jacobian[k + 3, j] = axis[k]
    return p, r, jacobian


@wp.kernel(enable_backward=False)
def solve_kernel(target: wp.array(dtype=wp.vec3), rotation: wp.array(dtype=wp.vec4),
                 seed: wp.array(dtype=V6), limits: wp.array2d(dtype=wp.vec2),
                 p0: wp.array(dtype=wp.vec3), r0: wp.array(dtype=wp.vec4),
                 p1: wp.array(dtype=wp.vec3), r1: wp.array(dtype=wp.vec4),
                 axes: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
                 offset_p: wp.vec3, offset_q: wp.vec4, iterations: int,
                 position_tolerance: float, rotation_tolerance: float,
                 result: wp.array(dtype=V6), valid: wp.array(dtype=wp.bool),
                 errors: wp.array(dtype=wp.vec2)):
    row = wp.tid()
    q = seed[row]
    for iteration in range(iterations):
        p, r, j = fk(q, p0, r0, p1, r1, axes, columns, offset_p, offset_q)
        ep = target[row] - p
        er = angular_error(mul(rotation[row], conjugate(r)))
        if wp.length(ep) < position_tolerance and wp.length(er) < rotation_tolerance:
            break
        error = V6(ep[0], ep[1], ep[2], er[0], er[1], er[2])
        system = j * wp.transpose(j)
        for k in range(6):
            system[k, k] += 0.0004
        lower = M6()
        for a in range(6):
            for b in range(a + 1):
                value = system[a, b]
                for k in range(b):
                    value -= lower[a, k] * lower[b, k]
                if a == b:
                    lower[a, b] = wp.sqrt(value)
                else:
                    lower[a, b] = value / lower[b, b]
        forward, correction = V6(), V6()
        for a in range(6):
            value = error[a]
            for b in range(a):
                value -= lower[a, b] * forward[b]
            forward[a] = value / lower[a, a]
        for reverse in range(6):
            a = 5 - reverse
            value = forward[a]
            for b in range(a + 1, 6):
                value -= lower[b, a] * correction[b]
            correction[a] = value / lower[a, a]
        delta = wp.transpose(j) * correction
        maximum = float(0.15)
        for k in range(6):
            maximum = wp.max(maximum, wp.abs(delta[k]))
        for k in range(6):
            q[k] = wp.clamp(q[k] + delta[k] * (0.15 / maximum), limits[row, k][0], limits[row, k][1])
    p, r, j = fk(q, p0, r0, p1, r1, axes, columns, offset_p, offset_q)
    pe = wp.length(target[row] - p)
    re = wp.length(angular_error(mul(rotation[row], conjugate(r))))
    finite = bool(True)
    for k in range(6):
        finite = finite and wp.isfinite(q[k])
    result[row] = q
    valid[row] = finite and pe < position_tolerance and re < rotation_tolerance
    errors[row] = wp.vec2(pe, re)


class WarpResetArmIK:
    def __init__(self, chain):
        wp.init()
        self.device = chain.offset_pos.device
        # Owned tensors keep Warp views alive for the entire solver lifetime.
        self.frames = [torch.stack([j[k] for j in chain.joints]).contiguous() for k in range(4)]
        self.axes = torch.tensor([j[4] if j[4] is not None else 0 for j in chain.joints],
                                 device=self.device, dtype=torch.int32)
        self.columns = torch.tensor([j[5] if j[5] is not None else -1 for j in chain.joints],
                                    device=self.device, dtype=torch.int32)
        self.inputs = [wp.from_torch(t, dtype=d) for t, d in zip(self.frames, (wp.vec3, wp.vec4, wp.vec3, wp.vec4))]
        self.inputs += [wp.from_torch(self.axes), wp.from_torch(self.columns),
                        wp.vec3(*chain.offset_pos.cpu().tolist()), wp.vec4(*chain.offset_quat.cpu().tolist())]

    def solve(self, desired_pos, desired_quat, initial, limits, iterations=80,
              position_tolerance=0.0005, rotation_tolerance=0.003):
        tensors = (desired_pos, desired_quat, initial, limits)
        n = len(initial)
        if any(t.device != self.device or t.dtype != torch.float32 for t in tensors):
            raise ValueError('Warp IK requires float32 inputs on the chain device')
        if tuple(t.shape for t in tensors) != ((n, 3), (n, 4), (n, 6), (n, 6, 2)):
            raise ValueError('Invalid Warp IK tensor shapes')
        result, valid = torch.empty_like(initial), torch.empty(n, device=self.device, dtype=torch.bool)
        errors = torch.empty((n, 2), device=self.device)
        if n:
            inputs = [wp.from_torch(t.contiguous(), dtype=d) for t, d in zip(tensors, (wp.vec3, wp.vec4, V6, wp.vec2))]
            stream = wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.device.type == 'cuda' else None
            wp.launch(solve_kernel, dim=n, inputs=[*inputs, *self.inputs, iterations,
                      position_tolerance, rotation_tolerance, wp.from_torch(result, dtype=V6),
                      wp.from_torch(valid), wp.from_torch(errors, dtype=wp.vec2)],
                      device=str(self.device), stream=stream)
        return result, valid, errors[:, 0], errors[:, 1]
