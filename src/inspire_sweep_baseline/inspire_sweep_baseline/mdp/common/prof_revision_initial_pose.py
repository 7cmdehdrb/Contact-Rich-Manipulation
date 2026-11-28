"""Bounded GPU reset candidates and conservative collision checks; no physics steps."""
import math
import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate, quat_from_matrix, matrix_from_quat


def separated_obbs(a, qa, ha, b, qb, hb, margin=.002):
    """15-axis OBB SAT; conservative positive gap margin, scalar-first quats."""
    ra, rb = matrix_from_quat(qa), matrix_from_quat(qb)
    delta = b-a
    separated = torch.zeros(len(a), dtype=torch.bool, device=a.device)
    for i in range(15):
        if i < 3:
            axis = ra[:, :, i]
        elif i < 6:
            axis = rb[:, :, i-3]
        else:
            j, k = divmod(i-6, 3)
            axis = torch.cross(ra[:, :, j], rb[:, :, k], dim=-1)
        norm = axis.norm(dim=-1)
        pa = (torch.bmm(axis[:, None], ra).squeeze(1).abs()*ha).sum(-1)
        pb = (torch.bmm(axis[:, None], rb).squeeze(1).abs()*hb).sum(-1)
        separated |= (norm > 1e-7) & ((delta*axis).sum(-1).abs() > pa+pb+margin*norm)
    return separated


class InitialHandRandomizer:
    """Cached hand axes, mimic mapping and reset-only nonadjacent OBB pairs."""

    def __init__(self, env, model):
        from ...assets.robot import HAND_JOINT_NAMES
        from ...assets.hand_kinematics import HAND_DRIVE_UPPER_LIMITS, HAND_MIMIC_DRIVE_INDICES, HAND_MIMIC_MULTIPLIERS
        self.env, self.model = env, model
        robot = env.scene['robot']
        self.hand_ids = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)[0]
        self.upper = torch.tensor(HAND_DRIVE_UPPER_LIMITS, device=env.device)
        self.mimic = torch.tensor(HAND_MIMIC_DRIVE_INDICES, device=env.device)
        self.ratios = torch.tensor(HAND_MIMIC_MULTIPLIERS, device=env.device)
        self.default_joints = robot.data.default_joint_pos[:1].clone()
        origin, identity = self.upper.new_zeros(1, 3), self.upper.new_tensor([[1., 0., 0., 0.]])
        pos, quat = model.link_poses(self.default_joints, origin, identity)
        paths = [p.rsplit('/', 1)[-1] for p in model.paths]
        palm = paths.index('inspire_palm_force_sensor')
        middle = paths.index('inspire_middle_force_sensor_3')
        wrist = paths.index('inspire_base_link')
        normal = quat_apply(quat[:, palm], self.upper.new_tensor([[0., 0., 1.]]))
        longitudinal = pos[:, middle]-pos[:, wrist]
        longitudinal -= (longitudinal*normal).sum(-1, keepdim=True)*normal
        longitudinal /= longitudinal.norm(dim=-1, keepdim=True)
        side = torch.cross(normal, longitudinal, dim=-1)
        hand_quat = quat_from_matrix(torch.stack((longitudinal, side, normal), -1))
        self.hand_in_eef = quat_mul(quat_conjugate(env._fast_default_eef[1][:1]), hand_quat)[0]
        # Fixed tactile links belong to their supporting rigid segment. Only
        # the same segment and an immediate physical joint connection are exempt.
        group, adjacent = [0], set()
        for index, frame in enumerate(model.frames, 1):
            parent, *_, axis, column = frame
            group.append(group[parent] if axis is None else index)
            if axis is not None:
                adjacent.add(tuple(sorted((group[parent], index))))
        # The thumb's two-axis base joint is split into thumb_1 (opposition)
        # and thumb_2 (flexion). Its mounting volume is inside the palm AABB
        # even in the authored open pose. Exempt only that base assembly;
        # thumb_3/4 versus palm and every thumb-versus-other-finger pair remain.
        adjacent.add(tuple(sorted((group[paths.index('inspire_base_link')],
                                   group[paths.index('inspire_left_thumb_2')]))))
        links = model.links.tolist()
        pairs = []
        for a, la in enumerate(links):
            for b in range(a+1, len(links)):
                lb = links[b]
                if not (paths[la].startswith(('inspire_', 'axia80_'))
                        and paths[lb].startswith(('inspire_', 'axia80_'))):
                    continue
                ga, gb = group[la], group[lb]
                if ga != gb and tuple(sorted((ga, gb))) not in adjacent:
                    pairs.append((a, b))
        self.pairs = torch.tensor(pairs, device=env.device, dtype=torch.long).reshape(-1, 2)

    def self_clear(self, joints, margin=.002):
        """Broadphase then bounded SAT batches; no PhysX self-collision change."""
        n = len(joints)
        origin = joints.new_zeros(n, 3)
        identity = joints.new_tensor([1., 0., 0., 0.]).expand(n, -1)
        pos, quat = self.model.link_poses(joints, origin, identity)
        rot = quat[:, self.model.links]
        center = pos[:, self.model.links]+quat_apply(rot, self.model.center.expand(n, -1, -1))
        extent = torch.matmul(matrix_from_quat(rot).abs(), self.model.half[..., None]).squeeze(-1)
        a, b = self.pairs.unbind(-1)
        overlap = ((center[:, a]-center[:, b]).abs() <= extent[:, a]+extent[:, b]+margin).all(-1)
        rows, pairs = overlap.nonzero(as_tuple=True)
        collisions = torch.zeros(n, device=joints.device, dtype=torch.long)
        for ids in torch.arange(len(rows), device=joints.device).split(16384):
            r = rows[ids]
            ia, ib = a[pairs[ids]], b[pairs[ids]]
            hit = ~separated_obbs(center[r, ia], rot[r, ia], self.model.half[ia],
                                 center[r, ib], rot[r, ib], self.model.half[ib], margin)
            collisions.index_add_(0, r, hit.long())
        return collisions == 0

    def sample(self, default_quat, base_quat, shelf_quat, generator, params):
        n = len(default_quat)
        draws = torch.rand((n, 9), device=default_quat.device, generator=generator)
        ranges = draws.new_tensor(params['orientation_bounds_deg'])*(torch.pi/180.)
        rpy = (2*draws[:, :3]-1)*ranges
        roll, pitch, yaw = rpy.unbind(-1)
        def rotation(angle, axis):
            q = draws.new_zeros(n, 4)
            q[:, 0], q[:, axis+1] = (angle/2).cos(), (angle/2).sin()
            return q
        local = quat_mul(rotation(pitch, 1), rotation(roll, 0))
        c = self.hand_in_eef.expand(n, -1)
        local = quat_mul(quat_mul(c, local), quat_conjugate(c))
        shelf_yaw = quat_mul(quat_mul(shelf_quat, rotation(yaw, 2)), quat_conjugate(shelf_quat))
        world = quat_mul(shelf_yaw, quat_mul(quat_mul(base_quat, default_quat), local))
        target = quat_mul(quat_conjugate(base_quat), world)
        bounds = draws.new_tensor(params['hand_drive_bounds'])
        u = bounds[:, 0]+draws[:, 3:]*(bounds[:, 1]-bounds[:, 0])
        joints = self.default_joints.expand(n, -1).clone()
        joints[:, self.hand_ids] = (u*self.upper)[:, self.mimic]*self.ratios
        tilt_valid = (roll.cos()*pitch.cos()) >= math.cos(math.radians(params['max_tilt_deg']))
        self_valid = self.self_clear(joints, params.get('hand_self_clearance_m', .002))
        return target, joints[:, self.hand_ids], u, rpy, tilt_valid, self_valid


def segment_distance(points, start, goal):
    delta = goal - start
    t = ((points - start[:, None]) * delta[:, None]).sum(-1)
    t = (t / delta.square().sum(-1).clamp_min(1e-12)[:, None]).clamp(0., 1.)
    return (points - start[:, None] - t[..., None] * delta[:, None]).norm(dim=-1)


def red_candidates(start, goal, radius, generator, count=16):
    """Sample the existing extruded RED polygon, excluding the swept target lane."""
    draw = torch.rand((len(start), count, 2), device=start.device, generator=generator)
    extension = (.2 - torch.minimum(start[:, 0], goal[:, 0])).clamp_min(0.)
    points = start[:, None] + draw[..., :1] * (goal-start)[:, None]
    points[..., 0] += draw[..., 1] * extension[:, None]
    # Same polygon as CircleResetLayout.red_distance, before its radius buffer.
    valid = (points.abs() < points.new_tensor([.195, .495])).all(-1)
    valid &= segment_distance(points, start, goal) >= radius[:, None]
    return points, valid


def front_candidates(start, goal, base_x, radius, generator, count=24, y_bounds=(-.5, .5)):
    """Continuous full-width approach area, from the target/goal to the base.

    The previous RED and outside IK areas are contained in this envelope.
    Reachability and actual robot geometry are checked after the batched IK.
    """
    draw = torch.rand((len(start), count, 2), device=start.device, generator=generator)
    lower = torch.minimum(start[:, 0], goal[:, 0])
    upper = torch.maximum(base_x, base_x.new_full(base_x.shape, .30))
    x = lower[:, None] + draw[..., 0]*(upper-lower)[:, None]
    y = y_bounds[0] + draw[..., 1]*(y_bounds[1]-y_bounds[0])
    points = torch.stack((x, y), -1)
    return points, segment_distance(points, start, goal) >= radius[:, None]


def shelf_candidates(start, goal, radius, generator, x_bounds=(-.2, .3), y_bounds=(-.25, .25), count=24):
    """Approved fixed EEF window, with 1mm reserve for the 0.5mm IK tolerance."""
    draw = torch.rand((len(start), count, 2), device=start.device, generator=generator)
    lower = start.new_tensor([x_bounds[0]+.001, y_bounds[0]+.001])
    upper = start.new_tensor([x_bounds[1]-.001, y_bounds[1]-.001])
    points = lower + draw*(upper-lower)
    return points, segment_distance(points, start, goal) >= radius[:, None]


def inside_red_polygon(points, start, goal):
    """Classify selected EEF points for the existing RED diagnostic buffer."""
    extension = torch.zeros_like(start)
    extension[:, 0] = (.2-torch.minimum(start[:, 0], goal[:, 0])).clamp_min(0.)
    vertices = torch.stack((start, goal, goal+extension, start+extension), 1)
    edges = vertices.roll(-1, 1)-vertices
    delta = points[:, None]-vertices
    cross = edges[..., 0]*delta[..., 1]-edges[..., 1]*delta[..., 0]
    inside = (cross >= -1e-7).all(-1) | (cross <= 1e-7).all(-1)
    area = (goal[:, 1]-start[:, 1]).abs()*extension[:, 0]
    line_low = vertices[..., 0].amin(-1)
    line_high = vertices[..., 0].amax(-1)
    degenerate = ((points[:, 1]-start[:, 1]).abs() <= 1e-7) & (points[:, 0] >= line_low) & (points[:, 0] <= line_high)
    return torch.where(area > 1e-10, inside, degenerate) & (points.abs() < points.new_tensor([.195, .495])).all(-1)


class RobotCollisionModel:
    """Cache USD joint frames and every enabled link collision box once.

    FK uses reset finger defaults and candidate arm joints. Rotated collision
    boxes are enclosed by shelf-frame AABBs: overlap rejects conservatively.
    """
    def __init__(self, env):
        from pxr import Usd, UsdPhysics
        from isaaclab.sim.utils import get_current_stage
        from .stabilize_sweep_geometry import collision_bounds
        stage = get_current_stage()
        path = env.scene.env_prim_paths[0] + '/Robot'
        robot = env.scene['robot']
        device = env.device
        def tensor(v):
            return torch.tensor(v, device=device, dtype=torch.float32)
        def rotation(q):
            return tensor((float(q.GetReal()), *map(float, q.GetImaginary())))
        pending = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(a) != 1 or len(b) != 1:
                continue
            axis = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis = 'XYZ'.index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError('Unsupported reset collision joint: ' + str(prim.GetPath()))
            name = prim.GetName()
            index = robot.joint_names.index(name) if axis is not None else None
            pending.append((str(a[0]), str(b[0]), tensor(tuple(joint.GetLocalPos0Attr().Get())),
                            rotation(joint.GetLocalRot0Attr().Get()), tensor(tuple(joint.GetLocalPos1Attr().Get())),
                            rotation(joint.GetLocalRot1Attr().Get()), axis, index))
        self.paths = [path+'/base_link']
        self.frames = []
        while pending:
            ready = [j for j in pending if j[0] in self.paths]
            if not ready:
                raise ValueError('Disconnected robot collision FK tree')
            for a, b, *frame in ready:
                if b in self.paths:
                    raise ValueError('Robot collision FK tree has duplicate child')
                self.frames.append((self.paths.index(a), *frame))
                self.paths.append(b)
            ready_children = {j[1] for j in ready}
            pending = [j for j in pending if j[1] not in ready_children]
        boxes, links = [], []
        for index, link in enumerate(self.paths):
            for low, high in collision_bounds(stage, link, link):
                boxes.append((low, high)); links.append(index)
        if not boxes:
            raise ValueError('Robot collision geometry is empty')
        bounds = tensor(boxes)
        self.center = bounds.mean(1)
        self.half = (bounds[:, 1]-bounds[:, 0])/2
        self.links = torch.tensor(links, device=device)
        self.hand_boxes = torch.tensor([('/axia80_link' in self.paths[i]
                                        or '/inspire_' in self.paths[i]) for i in links], device=device)
        shelf = tensor(collision_bounds(stage, env.scene.env_prim_paths[0]+'/Shelf',
                                        env.scene.env_prim_paths[0]+'/Shelf'))
        self.shelf_low, self.shelf_high = shelf[:, 0], shelf[:, 1]

    def link_poses(self, joints, base_pos, base_quat):
        pos, rot = [base_pos], [base_quat]
        for parent, p0, q0, p1, q1, axis, column in self.frames:
            p = pos[parent] + quat_apply(rot[parent], p0.expand_as(pos[parent]))
            q = quat_mul(rot[parent], q0.expand_as(rot[parent]))
            if axis is not None:
                angle = joints[:, column]/2
                motion = torch.zeros_like(q)
                motion[:, 0], motion[:, axis+1] = angle.cos(), angle.sin()
                q = quat_mul(q, motion)
            q = quat_mul(q, quat_conjugate(q1).expand_as(q))
            p = p - quat_apply(q, p1.expand_as(p))
            pos.append(p); rot.append(q)
        return torch.stack(pos, 1), torch.stack(rot, 1)

    def boxes(self, joints, base_pos, base_quat):
        p, q = self.link_poses(joints, base_pos, base_quat)
        q = q[:, self.links]
        center = p[:, self.links] + quat_apply(q, self.center.expand(len(joints), -1, -1))
        # Rotation matrix absolute values via transformed unit axes.
        extents = torch.zeros_like(center)
        for axis in range(3):
            direction = torch.zeros_like(center); direction[..., axis] = self.half[:, axis]
            extents += quat_apply(q, direction).abs()
        return center-extents, center+extents

    def clear(self, joints, base_pos, base_quat, object_low, object_high, active, margin=.003):
        low, high = self.boxes(joints, base_pos, base_quat)
        return self.clear_boxes(low, high, object_low, object_high, active, margin)

    def clear_boxes(self, low, high, object_low, object_high, active, margin=.003):
        object_overlap = ((high[:, :, None]+margin > object_low[:, None])
                          & (low[:, :, None]-margin < object_high[:, None])).all(-1)
        shelf_overlap = ((high[:, :, None]+margin > self.shelf_low)
                         & (low[:, :, None]-margin < self.shelf_high)).all(-1)
        return ~((object_overlap & active[:, None]).any(dim=(1, 2)) | shelf_overlap.any(dim=(1, 2)))

    def lane_clear(self, low, high, start, goal, target_radius):
        center = ((low+high)/2)[:, self.hand_boxes, :2]
        radius = ((high-low)/2)[:, self.hand_boxes, :2].norm(dim=-1)
        return (segment_distance(center, start, goal) >= target_radius[:, None]+radius+.003).all(-1)


@torch.no_grad()
def select_initial_pose(env, ids, local, yaw, active, sweep_local, default_local,
                        desired_local, base_pos, base_quat, shelf_pos, shelf_quat,
                        default_quat, ik_params):
    """Batched Warp selection with bounded recovery; preserve layout/RNG."""
    from .wrench_refactor_ik import sample_initial_eef
    if not hasattr(env, '_initial_pose_collision'):
        env._initial_pose_collision = RobotCollisionModel(env)
        env._initial_pose_rng = torch.Generator(device=env.device).manual_seed(int(env.cfg.seed or 0)+4319)
    rng = env._initial_pose_rng
    n = len(ids)
    model = env._initial_pose_collision
    randomized = 'orientation_bounds_deg' in ik_params
    rng_before = rng.get_state() if randomized else None
    if randomized and not hasattr(env, '_initial_hand_randomizer'):
        env._initial_hand_randomizer = InitialHandRandomizer(env, model)
    sampler = env._initial_hand_randomizer if randomized else None
    rejected = [0., 0., 0., 0.]
    tested = 0
    def posture(quat, base, shelf):
        nonlocal tested
        result = sampler.sample(quat, base, shelf, rng, ik_params)
        tested += len(quat)
        rejected[0] += (~result[-2]).sum()
        rejected[1] += (~result[-1]).sum()
        return result
    bounded = ik_params['initial_pose_mode'] == 'shelf_area_v1'
    # Separate generator leaves the parent's layout and IK draws untouched.
    legacy = sample_initial_eef(local[:, 0], sweep_local, default_local, rng,
        ik_params['front_clearance'], ik_params['lateral_offset'], ik_params['height_jitter'])
    bounds = env._revision_bounds_t[ids]
    offsets = bounds.mean(-2)
    rotations = torch.zeros((*yaw.shape, 4), device=env.device)
    rotations[..., 0], rotations[..., 3] = (yaw/2).cos(), (yaw/2).sin()
    centers = local + quat_apply(rotations, offsets)
    half = (bounds[..., 1, :]-bounds[..., 0, :])/2
    object_half = half.clone()
    object_half[..., 0] = yaw.cos().abs()*half[..., 0]+yaw.sin().abs()*half[..., 1]
    object_half[..., 1] = yaw.sin().abs()*half[..., 0]+yaw.cos().abs()*half[..., 1]
    start, goal = centers[:, 0, :2], centers[:, 0, :2]+sweep_local[:, :2]
    base_local = quat_apply_inverse(shelf_quat, base_pos-shelf_pos)
    # Exclude the target sweep plus the open hand's conservative 10cm radius.
    lane = half[:, 0, :2].norm(dim=-1)+.10
    if bounded:
        xy, allowed = shelf_candidates(start, goal, lane, rng, ik_params['initial_x_bounds'],
                                       ik_params['front_y_bounds'], count=24)
    else:
        xy, allowed = front_candidates(start, goal, base_local[:, 0], lane, rng,
                                       count=24, y_bounds=ik_params['front_y_bounds'])
    # Rank randomly sampled safe lane candidates, solve only four of them.
    order = allowed.float().argsort(dim=-1, descending=True, stable=True)[:, :4]
    front = desired_local[:, None].expand(-1, 4, -1).clone()
    front[..., :2] = xy.gather(1, order[..., None].expand(-1, -1, 2))
    allowed = allowed.gather(1, order)
    candidates = torch.cat((desired_local[:, None], legacy[:, None], front), 1)
    if bounded:
        lower = candidates.new_tensor([ik_params['initial_x_bounds'][0]+.001, ik_params['front_y_bounds'][0]+.001])
        upper = candidates.new_tensor([ik_params['initial_x_bounds'][1]-.001, ik_params['front_y_bounds'][1]-.001])
        candidates[..., :2] = candidates[..., :2].clamp(lower, upper)
    k = candidates.shape[1]
    def repeat(t):
        return t[:, None].expand(-1, k, *t.shape[1:]).reshape(n*k, *t.shape[1:]).contiguous()
    world = shelf_pos[:, None]+quat_apply(shelf_quat[:, None].expand(-1, k, -1), candidates)
    desired_base = quat_apply_inverse(repeat(base_quat), world.reshape(-1, 3)-repeat(base_pos))
    target_quat = repeat(default_quat)
    if randomized:
        target_quat, hand, drive, rpy, tilt_ok, self_ok = posture(target_quat, repeat(base_quat), repeat(shelf_quat))
    angles, valid, pe, re = env._fast_ik.solve(desired_base, target_quat,
                                             repeat(env._fast_initial[ids]), repeat(env._fast_limits[ids]))
    # FK/collision checks in the shelf frame, including all arm/hand links.
    joints = repeat(env.scene['robot'].data.default_joint_pos[ids]).clone()
    joints[:, env._goal_reset_arm_ids] = angles
    if randomized:
        joints[:, sampler.hand_ids] = hand
        rejected[2] += (~valid).sum()
        valid &= tilt_ok & self_ok
    rotation_local = quat_mul(quat_conjugate(shelf_quat), base_quat)
    low, high = model.boxes(joints, repeat(base_local), repeat(rotation_local))
    clear = model.clear_boxes(low, high, repeat(centers-object_half), repeat(centers+object_half), repeat(active))
    lane_clear = model.lane_clear(low, high, repeat(start), repeat(goal), repeat(half[:, 0, :2].norm(dim=-1)))
    valid = (valid & clear & lane_clear).view(n, k)
    if randomized:
        rejected[3] += (~(clear & lane_clear)).sum()
    valid[:, 2:] &= allowed
    missing = (~valid.any(-1)).nonzero().flatten()
    grid_rows = grid_candidates = grid_batches = 0
    grid_selected = torch.zeros(n, device=env.device, dtype=torch.bool) if bounded else None
    if len(missing):
        # A rare conservative-box rejection must not trigger scene resampling.
        # Check the eight corners of the ORIGINAL outside IK interval, once.
        corners = candidates.new_tensor([(x, y, z) for x in (0., 1.)
                                        for y in (0., 1.) for z in (-1., 1.)])
        fallback = default_local[missing, None].expand(-1, 8, -1).clone()
        front, lateral = ik_params['front_clearance'], ik_params['lateral_offset']
        side = torch.where(sweep_local[missing, 1] <= 0, 1., -1.)
        fallback[..., 0] = .20+front[0]+corners[:, 0]*(front[1]-front[0])
        fallback[..., 1] = local[missing, 0, 1, None]+side[:, None]*(lateral[0]+corners[:, 1]*(lateral[1]-lateral[0]))
        fallback[..., 2] += corners[:, 2]*ik_params['height_jitter']
        if bounded:
            fallback[..., :2] = fallback[..., :2].clamp(lower, upper)
        def extra(t):
            return t[missing, None].expand(-1, 8, *t.shape[1:]).reshape(len(missing)*8, *t.shape[1:]).contiguous()
        target_world = extra(shelf_pos)+quat_apply(extra(shelf_quat), fallback.flatten(0, 1))
        target_base = quat_apply_inverse(extra(base_quat), target_world-extra(base_pos))
        fq = extra(default_quat)
        if randomized:
            fq, fhand, fdrive, frpy, ftilt, fself = posture(fq, extra(base_quat), extra(shelf_quat))
        fa, fv, fp, fr = env._fast_ik.solve(target_base, fq,
                                           extra(env._fast_initial[ids]), extra(env._fast_limits[ids]))
        fj = extra(env.scene['robot'].data.default_joint_pos[ids]).clone()
        fj[:, env._goal_reset_arm_ids] = fa
        if randomized:
            fj[:, sampler.hand_ids] = fhand
            rejected[2] += (~fv).sum()
            fv &= ftilt & fself
        fl, fh = model.boxes(fj, extra(base_local), extra(rotation_local))
        fclear = model.clear_boxes(fl, fh, extra(centers-object_half), extra(centers+object_half), extra(active))
        fclear &= model.lane_clear(fl, fh, extra(start), extra(goal), extra(half[:, 0, :2].norm(dim=-1)))
        fv &= fclear
        if randomized:
            rejected[3] += (~fclear).sum()
        fv = fv.view(-1, 8)
        pending = (~fv.any(-1)).nonzero().flatten()
        if bounded and len(pending):
            # The legacy corners all lie on ONE side of the target. A narrow
            # window can block that whole interval while the other side is free.
            # Only exhausted rows scan both sides, across the unchanged window
            # and height interval. Three batches of 81 cap work and memory; no
            # scene redraw, tolerance relaxation, or physics step is involved.
            grid_rows = len(pending)
            grid = torch.cartesian_prod(
                torch.linspace(upper[0], lower[0], 9, device=env.device),
                torch.linspace(lower[1], upper[1], 9, device=env.device),
                candidates.new_tensor([-1., 0., 1.])*ik_params['height_jitter'])
            for chunk in grid.split(81):
                if not len(pending):
                    break
                rows = missing[pending]
                count = len(chunk)
                grid_candidates += len(rows)*count
                grid_batches += len(rows)
                def expand(t):
                    return t[rows, None].expand(-1, count, *t.shape[1:]).reshape(
                        len(rows)*count, *t.shape[1:]).contiguous()
                poses = chunk[None].expand(len(rows), -1, -1).clone()
                poses[..., 2] += default_local[rows, 2, None]
                world = expand(shelf_pos)+quat_apply(expand(shelf_quat), poses.flatten(0, 1))
                target = quat_apply_inverse(expand(base_quat), world-expand(base_pos))
                gq = expand(default_quat)
                if randomized:
                    gq, ghand, gdrive, grpy, gtilt, gself = posture(gq, expand(base_quat), expand(shelf_quat))
                ga, gv, gp, gr = env._fast_ik.solve(target, gq,
                    expand(env._fast_initial[ids]), expand(env._fast_limits[ids]))
                gj = expand(env.scene['robot'].data.default_joint_pos[ids]).clone()
                gj[:, env._goal_reset_arm_ids] = ga
                if randomized:
                    gj[:, sampler.hand_ids] = ghand
                    rejected[2] += (~gv).sum()
                    gv &= gtilt & gself
                gl, gh = model.boxes(gj, expand(base_local), expand(rotation_local))
                gclear = model.clear_boxes(gl, gh, expand(centers-object_half),
                                          expand(centers+object_half), expand(active))
                gclear &= model.lane_clear(gl, gh, expand(start), expand(goal),
                                          expand(half[:, 0, :2].norm(dim=-1)))
                gv &= gclear
                if randomized:
                    rejected[3] += (~gclear).sum()
                gv = gv.view(-1, count)
                found = gv.any(-1)
                first = gv.long().argmax(-1)[found]
                solved = pending[found]
                fallback[solved, 0] = poses[found, first]
                fa.view(-1, 8, 6)[solved, 0] = ga.view(-1, count, 6)[found, first]
                fp.view(-1, 8)[solved, 0] = gp.view(-1, count)[found, first]
                fr.view(-1, 8)[solved, 0] = gr.view(-1, count)[found, first]
                if randomized:
                    for dest, src in ((fq, gq), (fhand, ghand), (fdrive, gdrive), (frpy, grpy)):
                        dest.view(len(missing), 8, -1)[solved, 0] = src.view(len(rows), count, -1)[found, first]
                fv[solved, 0] = True
                grid_selected[rows[found]] = True
                pending = pending[~found]
        if not bool(fv.any(-1).all()):
            if randomized:
                directory = getattr(getattr(env, 'cfg', None), 'log_dir', None)
                if directory:
                    from pathlib import Path
                    from time import time_ns
                    path = Path(directory)/f'initial_pose_failure_{time_ns()}.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(env_ids=ids, failed_env_ids=ids[missing[~fv.any(-1)]],
                        local=local, yaw=yaw, active=active, sweep_local=sweep_local,
                        default_local=default_local, desired_local=desired_local,
                        base_pos=base_pos, base_quat=base_quat, shelf_pos=shelf_pos,
                        shelf_quat=shelf_quat, default_quat=default_quat, ik_params=ik_params,
                        rng_before=rng_before, bounds=bounds,
                        model={name: getattr(model, name) for name in
                               ('paths', 'frames', 'center', 'half', 'links', 'hand_boxes', 'shelf_low', 'shelf_high')},
                        initial=env._fast_initial[ids], limits=env._fast_limits[ids],
                        default_joint_pos=env.scene['robot'].data.default_joint_pos[ids],
                        arm_ids=env._goal_reset_arm_ids), path)
            raise RuntimeError('Collision-free initial pose budget exhausted: '+str(ids[missing[~fv.any(-1)]].tolist()))
        index = fv.long().argmax(-1)
        rows = torch.arange(len(missing), device=env.device)
        candidates[missing, 0] = fallback[rows, index]
        angles.view(n, k, 6)[missing, 0] = fa.view(-1, 8, 6)[rows, index]
        pe.view(n, k)[missing, 0] = fp.view(-1, 8)[rows, index]
        re.view(n, k)[missing, 0] = fr.view(-1, 8)[rows, index]
        if randomized:
            for dest, src in ((target_quat, fq), (hand, fhand), (drive, fdrive), (rpy, frpy)):
                dest.view(n, k, -1)[missing, 0] = src.view(len(missing), 8, -1)[rows, index]
        valid[missing, 0] = True
    # Prefer the broad continuous domain. Original outside poses are fallback.
    priority = valid.float()+candidates.new_tensor([0., 0., 1., 1., 1., 1.])
    priority = torch.where(valid, priority, -1.)
    choice = priority.argmax(-1)
    if not bool(valid.any(-1).all()):
        raise RuntimeError('Collision-free initial pose budget exhausted: '+str(ids[~valid.any(-1)].tolist()))
    row = torch.arange(n, device=env.device)
    selected_red = inside_red_polygon(candidates[row, choice, :2], start, goal)
    selected_front = (choice >= 2) if not bounded else (choice >= 2) | grid_selected
    env._initial_pose_metrics = {'Reset/front_selected_fraction': selected_front.float().mean(),
                                'Reset/legacy_fallback_fraction': (~selected_front).float().mean(),
                                'Reset/red_selected_fraction': selected_red.float().mean(),
                                'Reset/pose_candidates': float(k)+(8.*len(missing)+grid_candidates)/n,
                                'Reset/ik_attempts': 1.+float(len(missing)+grid_batches)/n,
                                'Reset/pose_recovery_fraction': float(len(missing))/n,
                                'Reset/pose_collision_rejected_fraction': (~clear).float().mean()}
    if bounded:
        env._initial_pose_metrics['Reset/pose_grid_recovery_fraction'] = float(grid_rows)/n
    if not hasattr(env, '_initial_pose_red_selected'):
        env._initial_pose_red_selected = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env._initial_pose_front_selected = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env._initial_pose_red_selected[ids] = selected_red
    env._initial_pose_front_selected[ids] = selected_front
    if randomized:
        for name, source in (('_initial_pose_selected_quat_base', target_quat),
                             ('_initial_pose_hand_targets', hand),
                             ('_initial_pose_hand_drive', drive), ('_initial_pose_rpy', rpy)):
            if not hasattr(env, name):
                setattr(env, name, source.new_zeros(env.num_envs, source.shape[-1]))
            getattr(env, name)[ids] = source.view(n, k, -1)[row, choice]
        for reason, count in zip(('tilt', 'hand_self_collision', 'ik', 'external_or_lane'), rejected):
            env._initial_pose_metrics[f'Reset/{reason}_rejected_fraction'] = count/tested
    return (candidates[row, choice], angles.view(n, k, 6)[row, choice],
            pe.view(n, k)[row, choice], re.view(n, k)[row, choice])
