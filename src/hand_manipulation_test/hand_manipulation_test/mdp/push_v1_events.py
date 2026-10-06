"""Push-v1 reset-only side pose; inherited bounded IK and collision checks."""

import torch
import isaaclab.utils.math as math_utils

from ..push_v1_math import push_v1_palm_rotation
from ..assets.robot import ARM_JOINT_NAMES
from .push_events import PushSafePoseReset
from .contact_events import _ids


class PushV1SafePoseReset(PushSafePoseReset):
    """Keep outward fingers and the physical wrist branch; retain bounded IK."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Cache vertices once. Resets select the exposed palmar face using
        # the actual sampled joints; normal steps only transform that point.
        from isaaclab.sim.utils import get_current_stage
        from ..collision_geometry import _points_in_frame

        name = getattr(env.cfg.task, "left_contact_reference_body", "inspire_thumb_force_sensor_4")
        self._thumb_contact_body_id = self.robot.body_names.index(name)
        body = get_current_stage().GetPrimAtPath(env.scene.env_prim_paths[0] + "/Robot/" + name)
        points = _points_in_frame(get_current_stage(), body, body, mesh_vertices=True)
        self._thumb_contact_vertices_b = torch.tensor(
            [[float(point[i]) for i in range(3)] for point in points], device=env.device
        )
        self._thumb_contact_face_b = torch.zeros((env.num_envs, 3), device=env.device)
        self._use_thumb_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, env_ids):
        index = _ids(env, env_ids)
        if not len(index):
            return
        defaults = self.robot.data.default_joint_pos
        original = defaults[index].clone()
        command = env.command_manager.get_term("target_position")
        left = command.angle_rad[index].cos() < 0.
        task = env.cfg.task
        right_seed = defaults.new_tensor(task.right_reset_arm_seed)
        left_seed = defaults.new_tensor(task.left_reset_arm_seed)
        seeded = original.clone()
        seeded[:, self.arm_joint_ids] = torch.where(left[:, None], left_seed, right_seed)
        # Keep the model's defaults intact even if bounded IK fails. Only
        # reset rows use these branch seeds, before any episode starts.
        defaults[index] = seeded
        try:
            super().__call__(env, index)
        finally:
            defaults[index] = original
        self._cache_contact_reference(index)

    def _cache_contact_reference(self, env_ids):
        _, hand_quaternion = self.palm_reference_pose_w(env_ids)
        thumb_up = math_utils.quat_apply(
            hand_quaternion, hand_quaternion.new_tensor((1., 0., 0.)).expand(len(env_ids), -1)
        )[:, 2]
        normal = math_utils.quat_apply(
            hand_quaternion, hand_quaternion.new_tensor((0., 1., 0.)).expand(len(env_ids), -1)
        )
        pad_quaternion = self.robot.data.body_link_quat_w[env_ids, self._thumb_contact_body_id]
        local_normal = math_utils.quat_apply_inverse(pad_quaternion, normal)
        projection = local_normal @ self._thumb_contact_vertices_b.mT
        mask = projection >= projection.amax(-1, keepdim=True) - .0005
        weights = mask.to(projection.dtype)
        self._thumb_contact_face_b[env_ids] = (
            weights @ self._thumb_contact_vertices_b / weights.sum(-1, keepdim=True).clamp_min(1.)
        )
        self._use_thumb_contact[env_ids] = thumb_up < 0.

    def contact_reference_pose_w(self, env_ids=None):
        """Actual central palm on the right and exposed palmar thumb on the left.

        H's contact normal remains horizontal on both sides. The left
        thumb-down pose cannot lower the central pad to a 6 cm Cube without
        a Table collision; its existing palmar pad is the reachable surface.
        """

        index = _ids(self._env, env_ids)
        central, hand_quaternion = self.palm_reference_pose_w(index)
        pad_position = self.robot.data.body_link_pos_w[index, self._thumb_contact_body_id]
        pad_quaternion = self.robot.data.body_link_quat_w[index, self._thumb_contact_body_id]
        thumb = pad_position + math_utils.quat_apply(pad_quaternion, self._thumb_contact_face_b[index])
        return torch.where(self._use_thumb_contact[index, None], thumb, central), hand_quaternion

    def contact_reference_offset_h(self, env_ids=None):
        """Live contact-point offset; track permitted hand-joint deflection."""

        index = _ids(self._env, env_ids)
        position, hand_quaternion = self.contact_reference_pose_w(index)
        hand_position = self.robot.data.body_link_pos_w[index, self.hand_body_id]
        return math_utils.quat_apply_inverse(hand_quaternion, position - hand_position)

    def _check_pose_and_limits(self, env_ids):
        valid = super()._check_pose_and_limits(env_ids)
        task = self._env.cfg.task
        wrist_joint_id = self.arm_joint_ids[ARM_JOINT_NAMES.index("wrist_2_joint")]
        wrist = self.robot.data.joint_pos[env_ids, wrist_joint_id]
        _, hand_quaternion = self.palm_reference_pose_w(env_ids)
        return (
            valid & torch.isfinite(wrist)
            & (wrist.sin() > task.wrist_2_branch_sin_margin)
            & (self._finger_outward_cos(env_ids, hand_quaternion) > task.minimum_finger_outward_cos)
        )

    def _sample_pose(self, env_ids):
        task = self._env.cfg.task
        cube = self._env.scene["target_object"].data.root_pos_w[env_ids]
        direction = self._env.command_manager.get_term("target_position").direction_w[env_ids]
        jitter = cube.new_tensor(task.reset_position_jitter_task)
        offset = cube.new_tensor(task.reset_position_offset_task) + (
            2 * torch.rand((len(env_ids), 3), device=self._env.device) - 1
        ) * jitter
        rotation = push_v1_palm_rotation(direction)
        tangent = torch.stack((-direction[:, 1], direction[:, 0], torch.zeros_like(direction[:, 0])), dim=-1)
        up = torch.zeros_like(direction)
        up[:, 2] = 1
        palm = cube - offset[:, :1] * direction + offset[:, 1:2] * tangent + offset[:, 2:] * up
        control = palm + (rotation @ (self.c_offset_h - self.palm_reference_h).unsqueeze(-1)).squeeze(-1)
        local_y = control[:, 1] - self._env.scene.env_origins[env_ids, 1]
        if not bool(((local_y >= task.reset_c_y_range[0]) & (local_y <= task.reset_c_y_range[1])).all()):
            raise RuntimeError("Push-v1 initial control point is outside reset_c_y_range")
        self.desired_palm_position_w[env_ids] = palm
        self.desired_c_pos_w[env_ids] = control
        self.direction_w[env_ids] = rotation[:, :, 1]
        hand_quaternion = math_utils.quat_from_matrix(rotation)
        self.desired_c_quat_w[env_ids] = math_utils.quat_unique(
            math_utils.quat_mul(hand_quaternion, self.c_quat_h.expand(len(env_ids), -1))
        )
