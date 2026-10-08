"""One-object scene reset and six-joint, right-palm pre-push IK.

The reset uses live articulation FK, as the source Inspire reset does. PhysX
Jacobian buffers can be stale after tensor joint teleports; finite differences
at the actual control point avoid both that cache and COM/actor-origin offset
ambiguity without advancing physics or forwarding unrelated environments.
"""

from __future__ import annotations

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from hand_manipulation_rl.assets.robot import ARM_JOINT_NAMES, HAND_BASE_BODY_NAME, ROBOT_CONTACT_BODY_NAMES


RIGHT_PALM_QUAT_WXYZ = (math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0)
CONTROL_POINT_OFFSET_H = (0.0, 0.0, 0.10)


def sample_single_object_states(
    pose_array, env_origins: torch.Tensor, position_noise: float = 0.02
) -> torch.Tensor:
    """Independently choose an original Sweep-Policy slot and yaw per reset."""
    poses = torch.as_tensor(pose_array, dtype=env_origins.dtype, device=env_origins.device).reshape(-1, 7)
    if len(poses) == 0 or not torch.isfinite(poses).all():
        raise ValueError("pose_array must contain finite seven-component source poses")
    if not math.isfinite(position_noise) or position_noise < 0.0:
        raise ValueError("position_noise must be finite and non-negative")
    count = len(env_origins)
    slots = torch.randint(len(poses), (count,), device=env_origins.device)
    states = torch.zeros((count, 1, 13), dtype=env_origins.dtype, device=env_origins.device)
    states[:, 0, :3] = poses[slots, :3] + env_origins
    states[:, 0, :2] += position_noise * (2.0 * torch.rand((count, 2), device=env_origins.device) - 1.0)
    yaw = 2.0 * math.pi * torch.rand(count, device=env_origins.device) - math.pi
    states[:, 0, 3] = torch.cos(0.5 * yaw)
    states[:, 0, 6] = torch.sin(0.5 * yaw)
    return states


def spawn_single_sweep_object(
    env,
    env_ids: torch.Tensor,
    pose_array,
    object_width: float,
    target_name: str = "target",
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
    position_noise: float = 0.02,
) -> None:
    """Reset the sole instantiated object; command the original +Y 18 cm sweep."""
    collection: RigidObjectCollection = env.scene[object_collection_cfg.name]
    if collection.num_objects != 1 or tuple(collection.object_names) != (target_name,):
        raise ValueError("The Inspire sweep scene must instantiate exactly the named target object")
    if not math.isfinite(object_width) or object_width <= 0.0:
        raise ValueError("object_width must be finite and positive")
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    states = sample_single_object_states(pose_array, env.scene.env_origins[env_ids], position_noise)
    collection.write_object_link_state_to_sim(states, env_ids=env_ids)
    env.target_id[env_ids, 0] = 0
    env.target_width[env_ids, 0] = object_width
    env.sweep_dir[env_ids] = 0.0
    env.sweep_dir[env_ids, 1] = 0.18


def right_palm_pre_push_pose(
    target_pos_w: torch.Tensor,
    target_width: torch.Tensor,
    x_offset: float = -0.02,
    lateral_clearance: float = 0.04,
    z_offset: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C starts left of the target; H +Y faces right, +X up, +Z into shelf."""
    if lateral_clearance <= 0.0 or z_offset <= 0.0:
        raise ValueError("Pre-push lateral and vertical clearances must be positive")
    position = target_pos_w.clone()
    position[:, 0] += x_offset
    position[:, 1] -= target_width.reshape(-1) + lateral_clearance
    position[:, 2] += z_offset
    quaternion = target_pos_w.new_tensor(RIGHT_PALM_QUAT_WXYZ).expand(len(target_pos_w), -1).clone()
    return position, quaternion


def _rotation_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Column-vector rotation, kept torch-only for reset geometry validation."""
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


def collision_aabbs(
    body_positions: torch.Tensor, body_quaternions: torch.Tensor,
    local_centers: torch.Tensor, local_half_extents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enclose complete collider boxes using the current FK body transforms."""
    rotation = _rotation_from_quaternion(body_quaternions)
    centers = body_positions + (rotation @ local_centers.unsqueeze(-1)).squeeze(-1)
    half_extents = (rotation.abs() @ local_half_extents.unsqueeze(-1)).squeeze(-1)
    return centers - half_extents, centers + half_extents


def aabbs_overlap(
    minimum_a: torch.Tensor, maximum_a: torch.Tensor,
    minimum_b: torch.Tensor, maximum_b: torch.Tensor, margin: float = 0.0,
) -> torch.Tensor:
    """Touching boxes and gaps below the positive reset margin are rejected."""
    return torch.all((maximum_a + margin >= minimum_b) & (maximum_b + margin >= minimum_a), dim=-1)


def _collider_bounds(body, xform_cache) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Cache a rigid body's real collision geometry bounds, including Axia80."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    body_inverse = xform_cache.GetLocalToWorldTransform(body).GetInverse()
    shapes = {}
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        for shape in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if UsdGeom.Boundable(shape):
                shapes[str(shape.GetPath())] = shape
    points = []
    for shape in shapes.values():
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(shape), Usd.TimeCode.Default())
        if extent is None or len(extent) != 2:
            raise RuntimeError(f"No computable collision extent for {shape.GetPath()}")
        transform = xform_cache.GetLocalToWorldTransform(shape)
        for x in (float(extent[0][0]), float(extent[1][0])):
            for y in (float(extent[0][1]), float(extent[1][1])):
                for z in (float(extent[0][2]), float(extent[1][2])):
                    points.append(body_inverse.Transform(transform.Transform(Gf.Vec3d(x, y, z))))
    if not points:
        raise RuntimeError(f"Missing collision geometry for {body.GetPath()}")
    minimum = tuple(min(float(p[axis]) for p in points) for axis in range(3))
    maximum = tuple(max(float(p[axis]) for p in points) for axis in range(3))
    return (
        tuple(0.5 * (minimum[axis] + maximum[axis]) for axis in range(3)),
        tuple(0.5 * (maximum[axis] - minimum[axis]) for axis in range(3)),
    )


class RightPalmReachingPoseReset(ManagerTermBase):
    """Solve all six arm joints and fail closed on pose/clearance failures.

    Wrist-3 participates in full orientation IK. Clearance uses actual source
    robot/object collider extents, plus the reference active shelf board proxy.
    The proxy is deliberately distinct from physical shelf contact sensors:
    shelf uprights and other shelf geometry still require runtime contact checks.
    """

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params.get("robot_cfg", SceneEntityCfg("robot")).name]
        self.objects: RigidObjectCollection = env.scene[
            params.get("object_collection_cfg", SceneEntityCfg("object_collection")).name
        ]
        self.shelf = env.scene[params.get("shelf_cfg", SceneEntityCfg("shelf")).name]
        if not self.robot.is_fixed_base or self.objects.num_objects != 1:
            raise ValueError("Right-palm reset requires a fixed-base robot and one instantiated object")
        joint_names = tuple(params.get("arm_joint_names", ARM_JOINT_NAMES))
        self.arm_joint_ids, names = self.robot.find_joints(joint_names, preserve_order=True)
        if len(self.arm_joint_ids) != 6 or tuple(names) != joint_names:
            raise ValueError("Full-pose reset requires all six named arm joints in source order")
        self.wrist_index = names.index("wrist_3_joint")
        hand_name = params.get("eef_body_name", HAND_BASE_BODY_NAME)
        body_ids, body_names = self.robot.find_bodies(hand_name, preserve_order=True)
        if body_names != [hand_name]:
            raise ValueError(f"Expected one hand base {hand_name!r}, got {body_names}")
        self.hand_body_id = body_ids[0]
        self.offset = torch.tensor(params.get("eef_offset", CONTROL_POINT_OFFSET_H), device=env.device)
        self.safety_ids, _ = self.robot.find_bodies(list(ROBOT_CONTACT_BODY_NAMES), preserve_order=True)
        if len(self.safety_ids) != len(ROBOT_CONTACT_BODY_NAMES):
            raise ValueError("Robot body names do not match the source collision contract")
        self._initialize_collision_bounds(env)
        for name, shape, dtype in (
            ("target_init_pos_w", (env.num_envs, 3), torch.float32),
            ("desired_reaching_pose_w", (env.num_envs, 7), torch.float32),
            ("reaching_ik_success", (env.num_envs,), torch.bool),
            ("reaching_ik_attempt_count", (env.num_envs,), torch.long),
            ("reset_collision_free", (env.num_envs,), torch.bool),
        ):
            if not hasattr(env, name):
                setattr(env, name, torch.zeros(shape, dtype=dtype, device=env.device))

    def _initialize_collision_bounds(self, env) -> None:
        from isaaclab.sim.utils import find_first_matching_prim, get_all_matching_child_prims, get_current_stage
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = get_current_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        robot_root = f"{env.scene.env_prim_paths[0]}/Robot"
        bounds = [_collider_bounds(stage.GetPrimAtPath(f"{robot_root}/{name}"), cache) for name in ROBOT_CONTACT_BODY_NAMES]
        self.body_local_centers = torch.tensor([bound[0] for bound in bounds], device=env.device)
        self.body_half_extents = torch.tensor([bound[1] for bound in bounds], device=env.device)
        target_cfg = next(iter(self.objects.cfg.rigid_objects.values()))
        root = find_first_matching_prim(target_cfg.prim_path)
        bodies = get_all_matching_child_prims(
            str(root.GetPath()), predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI), traverse_instance_prims=False
        )
        if len(bodies) != 1:
            raise RuntimeError("The single source object must have exactly one rigid body")
        center, half_extent = _collider_bounds(bodies[0], cache)
        self.object_local_center = torch.tensor(center, device=env.device)
        self.object_half_extent = torch.tensor(half_extent, device=env.device)
        # Reward gates use the real upstream object surface. The source width
        # is a reset standoff, not necessarily the object's collision radius.
        env._target_collision_local_center = self.object_local_center
        env._target_collision_half_extent = self.object_half_extent
        tensors = (self.body_local_centers, self.body_half_extents, self.object_local_center, self.object_half_extent)
        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise RuntimeError("Non-finite source collision bounds")

    def __call__(
        self, env, env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
        object_collection_cfg: SceneEntityCfg = SceneEntityCfg("object_collection"),
        arm_joint_names=ARM_JOINT_NAMES, eef_body_name: str = HAND_BASE_BODY_NAME,
        eef_offset=CONTROL_POINT_OFFSET_H,
        reaching_x_offset: float = -0.02, reaching_z_offset: float = 0.12,
        side_clearance: float = 0.04, position_noise: float = 0.003,
        max_iterations: int = 100, damping: float = 0.045, step_size: float = 0.65,
        position_tolerance: float = 0.005, orientation_tolerance: float = 0.05,
        joint_limit_margin: float = 0.035, singular_value_min: float = 0.008,
        wrist_3_range=(-2.8, 2.8), clearance_margin: float = 0.004,
        shelf_surface_height: float = 1.05, shelf_xy_bounds=(-0.88, -0.52, -0.50, 0.50),
        joint_seed_offsets=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.3, -0.2, 0.2, 0.0, 0.0, -0.25),
            (-0.3, -0.2, 0.2, 0.0, 0.0, 0.25),
            (0.55, 0.15, -0.25, 0.15, 0.0, -0.35),
            (-0.55, 0.15, -0.25, -0.15, 0.0, 0.35),
        ),
    ) -> None:
        del robot_cfg, shelf_cfg, object_collection_cfg, arm_joint_names, eef_body_name, eef_offset
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        if max_iterations <= 0 or damping <= 0.0 or step_size <= 0.0:
            raise ValueError("IK iteration count, damping and step size must be positive")
        if position_tolerance <= 0.0 or orientation_tolerance <= 0.0 or clearance_margin <= 0.0:
            raise ValueError("IK tolerances and collision margin must be positive")
        if position_noise < 0.0 or joint_limit_margin < 0.0 or singular_value_min < 0.0:
            raise ValueError("Reset noise, limit margin and singular-value threshold must be non-negative")
        if not torch.all(env.sweep_dir[env_ids, 1] > 0.0):
            raise ValueError("Right-palm reset only supports the fixed rightward command")
        target = self.objects.data.object_link_state_w[env_ids, 0, :3].clone()
        desired_pos, desired_quat = right_palm_pre_push_pose(
            target, env.target_width[env_ids, 0], reaching_x_offset, side_clearance, reaching_z_offset
        )
        desired_pos += position_noise * (2.0 * torch.rand_like(desired_pos) - 1.0)
        env.target_init_pos_w[env_ids] = target
        env.desired_reaching_pose_w[env_ids] = torch.cat((desired_pos, desired_quat), dim=-1)
        env.reaching_ik_success[env_ids] = False
        env.reaching_ik_attempt_count[env_ids] = 0
        env.reset_collision_free[env_ids] = False
        seeds = torch.as_tensor(joint_seed_offsets, dtype=torch.float32, device=env.device)
        if seeds.ndim != 2 or seeds.shape[1] != 6 or len(seeds) == 0 or not torch.isfinite(seeds).all():
            raise ValueError("Every finite reset seed must contain six arm offsets")
        base_q = self.robot.data.joint_pos[env_ids].clone()
        position_errors = torch.full((len(env_ids),), float("inf"), device=env.device)
        rotation_errors = position_errors.clone()
        for seed in seeds:
            rows = torch.where(~env.reaching_ik_success[env_ids])[0]
            if rows.numel() == 0:
                break
            ids = env_ids[rows]
            env.reaching_ik_attempt_count[ids] += 1
            q = base_q[rows].clone()
            q[:, self.arm_joint_ids] = self.robot.data.default_joint_pos[ids][:, self.arm_joint_ids] + seed
            limits = self.robot.data.soft_joint_pos_limits[ids][:, self.arm_joint_ids]
            q[:, self.arm_joint_ids] = torch.clamp(q[:, self.arm_joint_ids], limits[..., 0] + joint_limit_margin, limits[..., 1] - joint_limit_margin)
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
            self._solve_pose(ids, desired_pos[rows], desired_quat[rows], max_iterations, damping, step_size,
                             position_tolerance, orientation_tolerance, joint_limit_margin)
            pos, quat = self._control_pose(ids)
            error_p, error_r = math_utils.compute_pose_error(pos, quat, desired_pos[rows], desired_quat[rows], rot_error_type="axis_angle")
            position_errors[rows] = torch.linalg.vector_norm(error_p, dim=-1)
            rotation_errors[rows] = torch.linalg.vector_norm(error_r, dim=-1)
            q_arm = self.robot.data.joint_pos[ids][:, self.arm_joint_ids]
            finite = torch.isfinite(q_arm).all(-1) & torch.isfinite(position_errors[rows]) & torch.isfinite(rotation_errors[rows])
            pose_ok = (position_errors[rows] <= position_tolerance) & (rotation_errors[rows] <= orientation_tolerance)
            wrist = q_arm[:, self.wrist_index]
            limits_ok = torch.all((q_arm >= limits[..., 0] + joint_limit_margin) & (q_arm <= limits[..., 1] - joint_limit_margin), dim=-1)
            wrist_ok = (wrist >= wrist_3_range[0]) & (wrist <= wrist_3_range[1])
            possible = finite & pose_ok & limits_ok & wrist_ok
            accepted = torch.zeros_like(possible)
            if possible.any():
                candidate_ids = ids[possible]
                jacobian = self._control_point_jacobian(candidate_ids)
                nonsingular = torch.linalg.svdvals(jacobian)[:, -1] >= singular_value_min
                clearance = self._clearance_mask(env, candidate_ids, clearance_margin, shelf_surface_height, shelf_xy_bounds)
                accepted[possible] = nonsingular & clearance
            accepted_ids = ids[accepted]
            env.reaching_ik_success[accepted_ids] = True
            env.reset_collision_free[accepted_ids] = True
        failed = ~env.reaching_ik_success[env_ids]
        if failed.any():
            failed_ids = env_ids[failed]
            # Never expose an unvalidated seed to PPO or keep a corrupt pose.
            restore_q = base_q[failed]
            self.robot.write_joint_state_to_sim(restore_q, torch.zeros_like(restore_q), env_ids=failed_ids)
            details = [
                {"env_id": int(env_ids[row]), "position_error_m": float(position_errors[row]),
                 "orientation_error_rad": float(rotation_errors[row]), "attempts": int(env.reaching_ik_attempt_count[env_ids[row]])}
                for row in torch.where(failed)[0][:8]
            ]
            raise RuntimeError(f"Unable to initialize collision-safe right-palm sweep pose: {details}")
        final_q = self.robot.data.joint_pos[env_ids].clone()
        self.robot.set_joint_position_target(final_q, env_ids=env_ids)
        self.robot.set_joint_velocity_target(torch.zeros_like(final_q), env_ids=env_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(final_q), env_ids=env_ids)

    def _control_pose(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position = self.robot.data.body_pos_w[env_ids, self.hand_body_id]
        quaternion = self.robot.data.body_quat_w[env_ids, self.hand_body_id]
        return math_utils.combine_frame_transforms(position, quaternion, self.offset.expand(len(env_ids), -1))

    def _control_point_jacobian(self, env_ids: torch.Tensor, epsilon: float = 1.0e-3) -> torch.Tensor:
        """World geometric Jacobian at C from live FK, including the TCP offset."""
        base_q = self.robot.data.joint_pos[env_ids].clone()
        position, quaternion = self._control_pose(env_ids)
        columns = []
        try:
            for joint_id in self.arm_joint_ids:
                perturbed_q = base_q.clone()
                perturbed_q[:, joint_id] += epsilon
                self.robot.write_joint_position_to_sim(perturbed_q, env_ids=env_ids)
                perturbed_pos, perturbed_quat = self._control_pose(env_ids)
                _, rotation = math_utils.compute_pose_error(position, quaternion, position, perturbed_quat, rot_error_type="axis_angle")
                columns.append(torch.cat(((perturbed_pos - position) / epsilon, rotation / epsilon), dim=-1))
        finally:
            self.robot.write_joint_position_to_sim(base_q, env_ids=env_ids)
        return torch.stack(columns, dim=-1)

    def _solve_pose(self, env_ids, desired_pos, desired_quat, iterations, damping, step_size,
                    position_tolerance, orientation_tolerance, joint_limit_margin):
        converged = torch.zeros(len(env_ids), dtype=torch.bool, device=env_ids.device)
        identity = torch.eye(6, device=env_ids.device).unsqueeze(0)
        for _ in range(iterations):
            rows = torch.where(~converged)[0]
            if rows.numel() == 0:
                break
            ids = env_ids[rows]
            pos, quat = self._control_pose(ids)
            position_error, rotation_error = math_utils.compute_pose_error(pos, quat, desired_pos[rows], desired_quat[rows], rot_error_type="axis_angle")
            reached = (torch.linalg.vector_norm(position_error, dim=-1) <= position_tolerance) & (torch.linalg.vector_norm(rotation_error, dim=-1) <= orientation_tolerance)
            converged[rows[reached]] = True
            solve_rows, solve_ids = rows[~reached], ids[~reached]
            if solve_rows.numel() == 0:
                break
            p_error, r_error = position_error[~reached], rotation_error[~reached]
            p_error *= torch.clamp(0.06 / torch.linalg.vector_norm(p_error, dim=-1, keepdim=True).clamp_min(1.0e-9), max=1.0)
            r_error *= torch.clamp(0.25 / torch.linalg.vector_norm(r_error, dim=-1, keepdim=True).clamp_min(1.0e-9), max=1.0)
            jacobian = self._control_point_jacobian(solve_ids)
            if not torch.isfinite(jacobian).all():
                break
            error = torch.cat((p_error, r_error), dim=-1)
            delta = jacobian.transpose(1, 2) @ torch.linalg.solve(jacobian @ jacobian.transpose(1, 2) + damping**2 * identity, error.unsqueeze(-1))
            q = self.robot.data.joint_pos[solve_ids].clone()
            q[:, self.arm_joint_ids] += (step_size * delta.squeeze(-1)).clamp(-0.18, 0.18)
            limits = self.robot.data.soft_joint_pos_limits[solve_ids][:, self.arm_joint_ids]
            q[:, self.arm_joint_ids] = torch.clamp(q[:, self.arm_joint_ids], limits[..., 0] + joint_limit_margin, limits[..., 1] - joint_limit_margin)
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=solve_ids)

    def _clearance_mask(self, env, env_ids, margin, surface_height, xy_bounds):
        body_min, body_max = collision_aabbs(
            self.robot.data.body_pos_w[env_ids][:, self.safety_ids],
            self.robot.data.body_quat_w[env_ids][:, self.safety_ids],
            self.body_local_centers.unsqueeze(0), self.body_half_extents.unsqueeze(0),
        )
        state = self.objects.data.object_link_state_w[env_ids, 0]
        object_min, object_max = collision_aabbs(
            state[:, :3], state[:, 3:7], self.object_local_center, self.object_half_extent
        )
        object_overlap = aabbs_overlap(body_min, body_max, object_min.unsqueeze(1), object_max.unsqueeze(1), margin)
        # Sweep-Policy's active board is depth X, width Y, with top z=1.05.
        # Its shelf root is at x=-0.7; bounds below are environment-local.
        origin = env.scene.env_origins[env_ids]
        x_min, x_max, y_min, y_max = xy_bounds
        xy_overlap = ((body_max[..., 0] + margin >= origin[:, 0:1] + x_min)
                      & (body_min[..., 0] - margin <= origin[:, 0:1] + x_max)
                      & (body_max[..., 1] + margin >= origin[:, 1:2] + y_min)
                      & (body_min[..., 1] - margin <= origin[:, 1:2] + y_max))
        top = self.shelf.data.root_pos_w[env_ids, 2:3] + surface_height
        board_risk = xy_overlap & (body_min[..., 2] <= top + margin)
        finite = torch.isfinite(body_min).all(dim=(1, 2)) & torch.isfinite(body_max).all(dim=(1, 2))
        return finite & ~object_overlap.any(-1) & ~board_risk.any(-1)


# Stable public names used by the new environment configuration.
randomize_single_target = spawn_single_sweep_object
initialize_right_palm_at_target = RightPalmReachingPoseReset
