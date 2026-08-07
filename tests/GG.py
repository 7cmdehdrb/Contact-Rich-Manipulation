# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from .operational_space_cfg import OperationalSpaceControllerCfg


class OperationalSpaceController:
    """Operational-space controller.

    Reference:

    1. `A unified approach for motion and force control of robot manipulators: The operational space formulation <http://dx.doi.org/10.1109/JRA.1987.1087068>`_
       by Oussama Khatib (Stanford University)
    2. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    """

    def __init__(self, cfg: OperationalSpaceControllerCfg, num_envs: int, device: str):
        """Initialize operational-space controller.

        Args:
            cfg: The configuration for operational-space controller.
            num_envs: The number of environments.
            device: The device to use for computations.

        Raises:
            ValueError: When invalid control command is provided.
        """

        """  
        @configclass
        class OperationalSpaceControllerCfg:
        
            motion_control_axes_task: Sequence[int] = (1, 1, 1, 1, 1, 1)
            contact_wrench_control_axes_task: Sequence[int] = (0, 0, 0, 0, 0, 0)
            inertial_dynamics_decoupling: bool = False
            partial_inertial_dynamics_decoupling: bool = False
            gravity_compensation: bool = False
            impedance_mode: str = "fixed"
            motion_stiffness_task: float | Sequence[float] = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
            motion_damping_ratio_task: float | Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            motion_stiffness_limits_task: tuple[float, float] = (0, 1000)
            contact_wrench_stiffness_task: float | Sequence[float] | None = None
            nullspace_control: str = "none"
            nullspace_stiffness: float = 10.0
            nullspace_damping_ratio: float = 1.0
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device

        # resolve tasks-pace target dimensions
        self.target_list = list()
        for command_type in self.cfg.target_types:
            if command_type == "pose_rel":
                self.target_list.append(6)
            elif command_type == "pose_abs":
                self.target_list.append(7)
            elif command_type == "wrench_abs":
                self.target_list.append(6)
            else:
                raise ValueError(f"Invalid control command: {command_type}.")
        self.target_dim = sum(self.target_list)

        # create buffers
        # -- selection matrices, which might be defined in the task reference frame different from the root frame
        # Motion Task랑 Force Task에 대한 대각 행렬을 구축.
        """
        x = torch.tensor([1, 2, 3])
        y = torch.diag_embed(x)

        tensor(
            [[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]]
        )

        """

        self._selection_matrix_motion_task = torch.diag_embed(
            torch.tensor(
                self.cfg.motion_control_axes_task,
                dtype=torch.float,
                device=self._device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self._selection_matrix_force_task = torch.diag_embed(
            torch.tensor(
                self.cfg.contact_wrench_control_axes_task,
                dtype=torch.float,
                device=self._device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )

        # -- selection matrices in root frame
        # Task 행렬의 base 기준. 일단 0으로 초기화
        self._selection_matrix_motion_b = torch.zeros_like(
            self._selection_matrix_motion_task
        )
        self._selection_matrix_force_b = torch.zeros_like(
            self._selection_matrix_force_task
        )

        # -- commands
        # 타겟. 환경 수만큼 텐서로 확장
        self._task_space_target_task = torch.zeros(
            self.num_envs, self.target_dim, device=self._device
        )

        # -- Placeholders for motion/force control
        self.desired_ee_pose_task = None
        self.desired_ee_pose_b = None
        self.desired_ee_wrench_task = None
        self.desired_ee_wrench_b = None

        # -- buffer for operational space mass matrix
        # 질량 행렬. 6 * 6 * 환경 수 텐서로 0 초기화.
        self._os_mass_matrix_b = torch.zeros(self.num_envs, 6, 6, device=self._device)

        # -- Placeholder for the inverse of joint space mass matrix
        self._mass_matrix_inv = None

        # -- motion control gains
        # p gain을 설정하고, motion task 방향만 살림. force task 방향은 0으로 설정. (coupling 방지)
        self._motion_p_gains_task = torch.diag_embed(
            torch.ones(self.num_envs, 6, device=self._device)
            * torch.tensor(
                self.cfg.motion_stiffness_task, dtype=torch.float, device=self._device
            )
        )
        # -- -- zero out the axes that are not motion controlled, as keeping them non-zero will cause other axes
        # -- -- to act due to coupling
        self._motion_p_gains_task[:] = (
            self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
        )

        # d gain을 설정. p gain과 damping ratio를 이용하여 계산. (damping ratio는 1로 설정)
        # p gain 텐서에 곱했기 때문에, 커플링은 처리된 상태
        self._motion_d_gains_task = torch.diag_embed(
            2
            * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
            * torch.as_tensor(
                self.cfg.motion_damping_ratio_task,
                dtype=torch.float,
                device=self._device,
            ).reshape(1, -1)
        )

        # -- -- motion control gains in root frame
        # 루트 프레임용 텐서 초기화
        self._motion_p_gains_b = torch.zeros_like(self._motion_p_gains_task)
        self._motion_d_gains_b = torch.zeros_like(self._motion_d_gains_task)

        # -- force control gains
        # 1. force feedback 사용 여부에 의해 결정
        if self.cfg.contact_wrench_stiffness_task is not None:
            # self.cfg.contact_wrench_stiffness_task -> The proportional gain for determining operational space command forces for closed-loop contact force control. Sequence[float]

            # (N, 6, 6) 텐서 형성, fx,fy,fz,mx,my,mz
            self._contact_wrench_p_gains_task = torch.diag_embed(
                torch.ones(self.num_envs, 6, device=self._device)
                * torch.tensor(
                    self.cfg.contact_wrench_stiffness_task,
                    dtype=torch.float,
                    device=self._device,
                )
            )

            # Force Control 하지 않는 방향 제거
            self._contact_wrench_p_gains_task[:] = (
                self._selection_matrix_force_task @ self._contact_wrench_p_gains_task[:]
            )

            # -- -- force control gains in root frame
            # 루트 프레임 텐서 초기화
            self._contact_wrench_p_gains_b = torch.zeros_like(
                self._contact_wrench_p_gains_task
            )
        else:
            # feedback gain 사용하지 않음.
            self._contact_wrench_p_gains_task = None
            self._contact_wrench_p_gains_b = None

        # -- position gain limits
        self._motion_p_gains_limits = torch.zeros(
            self.num_envs, 6, 2, device=self._device
        )
        self._motion_p_gains_limits[..., 0], self._motion_p_gains_limits[..., 1] = (
            self.cfg.motion_stiffness_limits_task[0],
            self.cfg.motion_stiffness_limits_task[1],
        )

        # -- damping ratio limits
        self._motion_damping_ratio_limits = torch.zeros_like(
            self._motion_p_gains_limits
        )
        (
            self._motion_damping_ratio_limits[..., 0],
            self._motion_damping_ratio_limits[..., 1],
        ) = (
            self.cfg.motion_damping_ratio_limits_task[0],
            self.cfg.motion_damping_ratio_limits_task[1],
        )
        # -- end-effector contact wrench
        self._ee_contact_wrench_b = torch.zeros(self.num_envs, 6, device=self._device)

        # -- buffers for null-space control gains
        self._nullspace_p_gain = torch.tensor(
            self.cfg.nullspace_stiffness, dtype=torch.float, device=self._device
        )
        self._nullspace_d_gain = (
            2
            * torch.sqrt(self._nullspace_p_gain)
            * torch.tensor(
                self.cfg.nullspace_damping_ratio, dtype=torch.float, device=self._device
            )
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space of controller."""
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # task-space targets
            return self.target_dim
        elif self.cfg.impedance_mode == "variable_kp":
            # task-space targets + stiffness
            return self.target_dim + 6
        elif self.cfg.impedance_mode == "variable":
            # task-space targets + stiffness + damping
            return self.target_dim + 6 + 6
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

    """
    Operations.
    """

    def reset(self):
        """Reset the internals."""
        self.desired_ee_pose_b = None
        self.desired_ee_pose_task = None
        self.desired_ee_wrench_b = None
        self.desired_ee_wrench_task = None

    def set_command(
        self,
        command: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_task_frame_pose_b: torch.Tensor | None = None,
    ):
        """Set the task-space targets and impedance parameters.

        Args:
            command (torch.Tensor): A concatenated tensor of shape (``num_envs``, ``action_dim``) containing task-space
                targets (i.e., pose/wrench) and impedance parameters.
            current_ee_pose_b (torch.Tensor, optional): Current end-effector pose, in root frame, of shape
                (``num_envs``, 7), containing position and quaternion ``(w, x, y, z)``. Required for relative
                commands. Defaults to None.
            current_task_frame_pose_b: Current pose of the task frame, in root frame, in which the targets and the
                (motion/wrench) control axes are defined. It is a tensor of shape (``num_envs``, 7),
                containing position and the quaternion ``(w, x, y, z)``. Defaults to None.

        Format:
            Task-space targets, ordered according to 'command_types':

                Absolute pose: shape (``num_envs``, 7), containing position and quaternion ``(w, x, y, z)``.
                Relative pose: shape (``num_envs``, 6), containing delta position and rotation in axis-angle form.
                Absolute wrench: shape (``num_envs``, 6), containing force and torque.

            Impedance parameters: stiffness for ``variable_kp``, or stiffness, followed by damping ratio for
            ``variable``:

                Stiffness: shape (``num_envs``, 6)
                Damping ratio: shape (``num_envs``, 6)

        Raises:
            ValueError: When the command dimensions are invalid.
            ValueError: When an invalid impedance mode is provided.
            ValueError: When the current end-effector pose is not provided for the ``pose_rel`` command.
            ValueError: When an invalid control command is provided.
        """
        # Check the input dimensions
        if command.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_envs, self.action_dim)}'."
            )

        # Resolve the impedance parameters
        # 임피던스 모드가 fixed 면, k_p, k_d를 사용하지 않...나?
        if self.cfg.impedance_mode == "fixed":
            # task space targets (i.e., pose/wrench)
            # 타겟 = 커멘트로 바로 설정
            self._task_space_target_task[:] = command

        elif self.cfg.impedance_mode == "variable_kp":
            # split input command

            # 입력 커멘드와 stiffness 를 분리
            task_space_command, stiffness = torch.split(
                command, [self.target_dim, 6], dim=-1
            )

            # format command
            # stiffness에 클리핑
            stiffness = stiffness.clip_(
                min=self._motion_p_gains_limits[..., 0],
                max=self._motion_p_gains_limits[..., 1],
            )

            # task space targets + stiffness
            # 타겟 정보를 메모리에 저장
            self._task_space_target_task[:] = task_space_command.squeeze(dim=-1)

            # Stiffness 행렬을 대각 행렬로 변환
            self._motion_p_gains_task[:] = torch.diag_embed(stiffness)

            # Stiffness 행렬에서, Motion-Control 하지 않는 방향 제거
            self._motion_p_gains_task[:] = (
                self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
            )

            # Damping 행렬을 계산
            self._motion_d_gains_task = torch.diag_embed(
                2
                * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
                * torch.as_tensor(
                    self.cfg.motion_damping_ratio_task,
                    dtype=torch.float,
                    device=self._device,
                ).reshape(1, -1)
            )

        elif self.cfg.impedance_mode == "variable":
            # split input command
            # 입력 커멘드와 stiffness, damping ratio를 분리
            task_space_command, stiffness, damping_ratio = torch.split(
                command, [self.target_dim, 6, 6], dim=-1
            )

            # format command
            # stiffness와 damping ratio에 클리핑
            stiffness = stiffness.clip_(
                min=self._motion_p_gains_limits[..., 0],
                max=self._motion_p_gains_limits[..., 1],
            )
            damping_ratio = damping_ratio.clip_(
                min=self._motion_damping_ratio_limits[..., 0],
                max=self._motion_damping_ratio_limits[..., 1],
            )

            # task space targets + stiffness + damping
            # 타겟 정보를 메모리에 저장
            self._task_space_target_task[:] = task_space_command

            # Stiffness 행렬을 대각 행렬로 변환
            self._motion_p_gains_task[:] = torch.diag_embed(stiffness)

            # Stiffness 행렬에서, Motion-Control 하지 않는 방향 제거
            self._motion_p_gains_task[:] = (
                self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
            )

            # Damping 행렬을 계산
            self._motion_d_gains_task[:] = torch.diag_embed(
                2
                * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
                * damping_ratio
            )
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

        if current_task_frame_pose_b is None:
            current_task_frame_pose_b = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * self.num_envs,
                device=self._device,
            )

        # Resolve the target commands
        """
        for command_type in self.cfg.target_types:
        루프에 의하여, target_types에 따라, pose_rel, pose_abs, wrench_abs를 구분하여 처리
        """

        # pose_rel, pose_abs, wrench_abs
        target_groups = torch.split(
            self._task_space_target_task, self.target_list, dim=1
        )

        # Target 들에 대하여 순회
        for command_type, target in zip(self.cfg.target_types, target_groups):

            # 1. pose_rel: 현재 end-effector pose를 기준으로, delta pose를 적용하여, task frame에서의 desired pose를 계산
            if command_type == "pose_rel":
                # check input is provided
                if current_ee_pose_b is None:
                    raise ValueError("Current pose is required for 'pose_rel' command.")

                # Transform the current pose from base/root frame to task frame
                # 현재 pose에 변화량을 더해야 하므로 현재 end-effector pose가 반드시 필요 -> 현재 pose를 root frame에서 task frame으로 변환
                current_ee_pos_task, current_ee_rot_task = subtract_frame_transforms(
                    current_task_frame_pose_b[:, :3],
                    current_task_frame_pose_b[:, 3:],
                    current_ee_pose_b[:, :3],
                    current_ee_pose_b[:, 3:],
                )

                # compute targets in task frame
                # 상대 pose 명령 적용
                desired_ee_pos_task, desired_ee_rot_task = apply_delta_pose(
                    current_ee_pos_task, current_ee_rot_task, target
                )

                # Pos, Rot을 합쳐서 desired_ee_pose_task에 저장
                self.desired_ee_pose_task = torch.cat(
                    [desired_ee_pos_task, desired_ee_rot_task], dim=-1
                )

            elif command_type == "pose_abs":
                # compute targets
                # 절대 pose 명령 적용. 그대로 사용
                self.desired_ee_pose_task = target.clone()
            elif command_type == "wrench_abs":
                # compute targets
                # 절대 wrench 명령 적용. 그대로 사용
                self.desired_ee_wrench_task = target.clone()
            else:
                raise ValueError(f"Invalid control command: {command_type}.")

        # Rotation of task frame wrt root frame, converts a coordinate from task frame to root frame.
        # root frame에서 본 task frame의 quaternion
        R_task_b = matrix_from_quat(current_task_frame_pose_b[:, 3:])

        # Rotation of root frame wrt task frame, converts a coordinate from root frame to task frame.
        R_b_task = R_task_b.mT

        # Transform motion control stiffness gains from task frame to root frame
        # RKR^T
        # task frame에서 정의된 motion gain 행렬 -> root frame으로 변환
        self._motion_p_gains_b[:, 0:3, 0:3] = (
            R_task_b @ self._motion_p_gains_task[:, 0:3, 0:3] @ R_b_task
        )
        self._motion_p_gains_b[:, 3:6, 3:6] = (
            R_task_b @ self._motion_p_gains_task[:, 3:6, 3:6] @ R_b_task
        )

        # Transform motion control damping gains from task frame to root frame
        self._motion_d_gains_b[:, 0:3, 0:3] = (
            R_task_b @ self._motion_d_gains_task[:, 0:3, 0:3] @ R_b_task
        )
        self._motion_d_gains_b[:, 3:6, 3:6] = (
            R_task_b @ self._motion_d_gains_task[:, 3:6, 3:6] @ R_b_task
        )

        # Transform contact wrench gains from task frame to root frame (if applicable)
        if (
            self._contact_wrench_p_gains_task is not None
            and self._contact_wrench_p_gains_b is not None
        ):
            self._contact_wrench_p_gains_b[:, 0:3, 0:3] = (
                R_task_b @ self._contact_wrench_p_gains_task[:, 0:3, 0:3] @ R_b_task
            )
            self._contact_wrench_p_gains_b[:, 3:6, 3:6] = (
                R_task_b @ self._contact_wrench_p_gains_task[:, 3:6, 3:6] @ R_b_task
            )

        # Transform selection matrices from target frame to base frame
        self._selection_matrix_motion_b[:, 0:3, 0:3] = (
            R_task_b @ self._selection_matrix_motion_task[:, 0:3, 0:3] @ R_b_task
        )
        self._selection_matrix_motion_b[:, 3:6, 3:6] = (
            R_task_b @ self._selection_matrix_motion_task[:, 3:6, 3:6] @ R_b_task
        )
        self._selection_matrix_force_b[:, 0:3, 0:3] = (
            R_task_b @ self._selection_matrix_force_task[:, 0:3, 0:3] @ R_b_task
        )
        self._selection_matrix_force_b[:, 3:6, 3:6] = (
            R_task_b @ self._selection_matrix_force_task[:, 3:6, 3:6] @ R_b_task
        )

        # Transform desired pose from task frame to root frame
        if self.desired_ee_pose_task is not None:
            self.desired_ee_pose_b = torch.zeros_like(self.desired_ee_pose_task)
            self.desired_ee_pose_b[:, :3], self.desired_ee_pose_b[:, 3:] = (
                combine_frame_transforms(
                    current_task_frame_pose_b[:, :3],
                    current_task_frame_pose_b[:, 3:],
                    self.desired_ee_pose_task[:, :3],
                    self.desired_ee_pose_task[:, 3:],
                )
            )

        # Transform desired wrenches to root frame
        if self.desired_ee_wrench_task is not None:
            self.desired_ee_wrench_b = torch.zeros_like(self.desired_ee_wrench_task)
            self.desired_ee_wrench_b[:, :3] = (
                R_task_b @ self.desired_ee_wrench_task[:, :3].unsqueeze(-1)
            ).squeeze(-1)
            self.desired_ee_wrench_b[:, 3:] = (
                R_task_b @ self.desired_ee_wrench_task[:, 3:].unsqueeze(-1)
            ).squeeze(-1) + torch.cross(
                current_task_frame_pose_b[:, :3],
                self.desired_ee_wrench_b[:, :3],
                dim=-1,
            )

    def compute(
        self,
        jacobian_b: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_ee_vel_b: torch.Tensor | None = None,
        current_ee_force_b: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
        current_joint_pos: torch.Tensor | None = None,
        current_joint_vel: torch.Tensor | None = None,
        nullspace_joint_pos_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            jacobian_b: The Jacobian matrix of the end-effector in root frame. It is a tensor of shape
                (``num_envs``, 6, ``num_DoF``).
            current_ee_pose_b: The current end-effector pose in root frame. It is a tensor of shape
                (``num_envs``, 7), which contains the position and quaternion ``(w, x, y, z)``. Defaults to ``None``.
            current_ee_vel_b: The current end-effector velocity in root frame. It is a tensor of shape
                (``num_envs``, 6), which contains the linear and angular velocities. Defaults to None.
            current_ee_force_b: The current external force on the end-effector in root frame. It is a tensor of
                shape (``num_envs``, 3), which contains the linear force. Defaults to ``None``.
            mass_matrix: The joint-space mass/inertia matrix. It is a tensor of shape (``num_envs``, ``num_DoF``,
                ``num_DoF``). Defaults to ``None``.
            gravity: The joint-space gravity vector. It is a tensor of shape (``num_envs``, ``num_DoF``). Defaults
                to ``None``.
            current_joint_pos: The current joint positions. It is a tensor of shape (``num_envs``, ``num_DoF``).
                Defaults to ``None``.
            current_joint_vel: The current joint velocities. It is a tensor of shape (``num_envs``, ``num_DoF``).
                Defaults to ``None``.
            nullspace_joint_pos_target: The target joint positions the null space controller is trying to enforce, when
                possible. It is a tensor of shape (``num_envs``, ``num_DoF``).

        Raises:
            ValueError: When motion-control is enabled but the current end-effector pose or velocity is not provided.
            ValueError: When inertial dynamics decoupling is enabled but the mass matrix is not provided.
            ValueError: When the current end-effector pose is not provided for the ``pose_rel`` command.
            ValueError: When closed-loop force control is enabled but the current end-effector force is not provided.
            ValueError: When gravity compensation is enabled but the gravity vector is not provided.
            ValueError: When null-space control is enabled but the system is not redundant.
            ValueError: When dynamically consistent pseudo-inverse is enabled but the mass matrix inverse is not
                provided.
            ValueError: When null-space control is enabled but the current joint positions and velocities are not
                provided.
            ValueError: When target joint positions are provided for null-space control but their dimensions do not
                match the current joint positions.
            ValueError: When an invalid null-space control method is provided.

        Returns:
            Tensor: The joint efforts computed by the controller. It is a tensor of shape (``num_envs``, ``num_DoF``).
        """

        # deduce number of DoF
        num_DoF = jacobian_b.shape[2]
        # create joint effort vector
        joint_efforts = torch.zeros(self.num_envs, num_DoF, device=self._device)

        # compute joint efforts for motion-control
        # 목표 Pose가 설정되어 있는 경우,
        if self.desired_ee_pose_b is not None:

            # check input is provided
            if current_ee_pose_b is None or current_ee_vel_b is None:
                raise ValueError(
                    "Current end-effector pose and velocity are required for motion control."
                )

            # -- end-effector tracking error
            # Pose Error 계산 (N, 6) 텐서. position error + rotation error(axis-angle)
            pose_error_b = torch.cat(
                compute_pose_error(
                    current_ee_pose_b[:, :3],
                    current_ee_pose_b[:, 3:],
                    self.desired_ee_pose_b[:, :3],
                    self.desired_ee_pose_b[:, 3:],
                    rot_error_type="axis_angle",
                ),
                dim=-1,
            )

            # Velocity Error 계산. 목표 속도는 0으로 가정. (N, 6) 텐서 -> 항상 Current Velocity를 -1 곱한 값
            velocity_error_b = (
                -current_ee_vel_b
            )  # zero target velocity. The target is assumed to be stationary.

            # -- desired end-effector acceleration (spring-damper system)
            # "e" 계산. p gain * pose error + d gain * velocity error. (N, 6, 1) 텐서
            des_ee_acc_b = self._motion_p_gains_b @ pose_error_b.unsqueeze(
                -1
            ) + self._motion_d_gains_b @ velocity_error_b.unsqueeze(-1)

            # -- Inertial dynamics decoupling
            # 관성 동역학 분리 여부에 따른 연산 차이
            if self.cfg.inertial_dynamics_decoupling:

                # check input is provided
                # 반드시 Mass Matrix가 제공되어야 함. (관성 동역학 분리 시)
                if mass_matrix is None:
                    raise ValueError("Mass matrix is required for inertial decoupling.")

                # Compute operational space mass matrix
                self._mass_matrix_inv = torch.inverse(mass_matrix)

                # Partial inertial dynamics decoupling 상태라면,
                if self.cfg.partial_inertial_dynamics_decoupling:
                    # Fill in the translational and rotational parts of the inertia separately, ignoring their coupling

                    # 선형 운동용 OS Inertia, J M^(-1) J^T, 역행렬 계산
                    self._os_mass_matrix_b[:, 0:3, 0:3] = torch.inverse(
                        jacobian_b[:, 0:3]
                        @ self._mass_matrix_inv
                        @ jacobian_b[:, 0:3].mT
                    )
                    # 회전 운동용 OS Inertia, J M^(-1) J^T, 역행렬 계산
                    self._os_mass_matrix_b[:, 3:6, 3:6] = torch.inverse(
                        jacobian_b[:, 3:6]
                        @ self._mass_matrix_inv
                        @ jacobian_b[:, 3:6].mT
                    )

                    # 즉, 선형/회전에 의한 커플링 효과를 무시함

                else:
                    # Calculate the operational space mass matrix fully accounting for the couplings
                    # Full inertial dynamics decoupling
                    # J M^(-1) J^T, 역행렬 계산
                    self._os_mass_matrix_b[:] = torch.inverse(
                        jacobian_b @ self._mass_matrix_inv @ jacobian_b.mT
                    )

                # (Generalized) operational space command forces
                # F = (J M^(-1) J^T)^(-1) * \ddot(x_des) = M_task * \ddot(x_des)
                # Operational-space command force 계산
                os_command_forces_b = self._os_mass_matrix_b @ des_ee_acc_b

            else:
                # Task-space impedance control: command forces = \ddot(x_des).
                # Please note that the definition of task-space impedance control varies in literature.
                # This implementation ignores the inertial term. For inertial decoupling,
                # use inertial_dynamics_decoupling=True.

                # Inertial decoupling을 사용하지 않는 경우
                # F_cmd = \ddot(x_des)
                os_command_forces_b = des_ee_acc_b

            # -- joint-space commands
            # Joint-Space Effortf로 변환. J^T * S(Selection Matrix) * F_cmd
            joint_efforts += (
                jacobian_b.mT @ self._selection_matrix_motion_b @ os_command_forces_b
            ).squeeze(-1)

        # compute joint efforts for contact wrench/force control
        # Desired End-Effector Wrench가 설정되어 있는 경우,
        if self.desired_ee_wrench_b is not None:
            # -- task-space contact wrench

            # Closed-loop force control 사용시,
            if self.cfg.contact_wrench_stiffness_task is not None:

                # check input is provided
                if current_ee_force_b is None:
                    raise ValueError(
                        "Current end-effector force is required for closed-loop force control."
                    )

                # We can only measure the force component at the contact, so only apply the feedback for only the force
                # component, keep the control of moment components open loop

                # 현재 EEF Wrench 구성. [current_ee_force_b, desired_moment_b]
                self._ee_contact_wrench_b[:, 0:3] = current_ee_force_b
                self._ee_contact_wrench_b[:, 3:6] = self.desired_ee_wrench_b[:, 3:6]

                # closed-loop control with feedforward term
                # Wrench Command 계산. wrench_command = desired_wrench + Kp * (desired_wrench - current_wrench)
                os_contact_wrench_command_b = self.desired_ee_wrench_b.unsqueeze(
                    -1
                ) + self._contact_wrench_p_gains_b @ (
                    self.desired_ee_wrench_b - self._ee_contact_wrench_b
                ).unsqueeze(
                    -1
                )

            else:
                # open-loop control
                # 측정 Wrench를 사용하지 않고, Desired Wrench를 그대로 사용
                os_contact_wrench_command_b = self.desired_ee_wrench_b.unsqueeze(-1)

            # -- joint-space commands
            # Joint-Space Effort로 변환. J^T * S(Selection Matrix) * Wrench_command
            joint_efforts += (
                jacobian_b.mT
                @ self._selection_matrix_force_b
                @ os_contact_wrench_command_b
            ).squeeze(-1)

        # add gravity compensation (bias correction)
        # Gravity Compensation 사용시,
        if self.cfg.gravity_compensation:
            # check input is provided
            if gravity is None:
                raise ValueError("Gravity vector is required for gravity compensation.")

            # add gravity compensation
            # Gravity Compensation Torque를 Joint Effort에 추가. 여기서 Gravity Torque는 Joint-Space에서 계산된 Torque임. (Joint-Space Mass Matrix를 사용하지 않고, Gravity Vector만 사용)
            joint_efforts += gravity

        # Add null-space control
        # -- Free null-space control
        if self.cfg.nullspace_control == "none":
            # No additional control is applied in the null space.
            # Null-Space Control을 사용하지 않음. (즉, Null-Space에 대한 추가 제어 없음)
            pass

        else:
            # Check if the system is redundant
            if num_DoF <= 6:
                # 7 DoF 이상이면 redundant manipulator. 6 DoF 이하이면 redundant 아님.
                # 따라서, UR5e 같은 6 DoF 로봇은 Null-Space Control을 사용할 수 없음.
                raise ValueError(
                    "Null-space control is only applicable for redundant manipulators."
                )

            # Calculate the pseudo-inverse of the Jacobian
            if (
                self.cfg.inertial_dynamics_decoupling
                and not self.cfg.partial_inertial_dynamics_decoupling
            ):
                # inertial_dynamics_decoupling=True, partial_inertial_dynamics_decoupling=False
                # 위 2개 조건을 만족해야, Dynamically consistent pseudo-inverse를 계산할 수 있음. (즉, Full Inertial Decoupling 상태에서만 가능)

                # Dynamically consistent pseudo-inverse allows decoupling of null space and task space
                if self._mass_matrix_inv is None or mass_matrix is None:
                    raise ValueError(
                        "Mass matrix inverse is required for dynamically consistent pseudo-inverse"
                    )

                # Dynamically consistent pseudo-inverse 계산.
                # _os_mass_matrix_b @ jacobian_b @ _mass_matrix_inv
                # _os_mass_matrix_b = (J M^(-1) J^T)^(-1)
                jacobian_pinv_transpose = (
                    self._os_mass_matrix_b @ jacobian_b @ self._mass_matrix_inv
                )
            else:
                # Moore-Penrose pseudo-inverse if full inertia matrix is not available (e.g., no/partial decoupling)
                # 위 조건에 안맞으면, Moore–Penrose pseudo-inverse 대신 사용
                jacobian_pinv_transpose = torch.pinverse(jacobian_b).mT

            # Calculate the null-space projector
            # Null-space projector 계산
            # P = I - J^T * J^+T
            nullspace_jacobian_transpose = (
                torch.eye(n=num_DoF, device=self._device)
                - jacobian_b.mT @ jacobian_pinv_transpose
            )

            # Null space position control
            if self.cfg.nullspace_control == "position":
                # 유일하게 제공하는 Null-Space Control 방법. (즉, Null-Space에 대한 Position Control)

                # Check if the current joint positions and velocities are provided
                if current_joint_pos is None or current_joint_vel is None:
                    raise ValueError(
                        "Current joint positions and velocities are required for null-space control."
                    )

                # Calculate the joint errors for nullspace position control
                if nullspace_joint_pos_target is None:
                    nullspace_joint_pos_target = torch.zeros_like(current_joint_pos)
                # Check if the dimensions of the target nullspace joint positions match the current joint positions
                elif nullspace_joint_pos_target.shape != current_joint_pos.shape:
                    raise ValueError(
                        f"The target nullspace joint positions shape '{nullspace_joint_pos_target.shape}' does not"
                        f"match the current joint positions shape '{current_joint_pos.shape}'."
                    )

                # 관절 position 및 velocity error 계산
                joint_pos_error_nullspace = (
                    nullspace_joint_pos_target - current_joint_pos
                )
                joint_vel_error_nullspace = -current_joint_vel

                # Calculate the desired joint accelerations
                # Null-space 목표 관절 가속도 계산
                joint_acc_nullspace = (
                    self._nullspace_p_gain * joint_pos_error_nullspace
                    + self._nullspace_d_gain * joint_vel_error_nullspace
                ).unsqueeze(-1)

                # Calculate the projected torques in null-space
                if mass_matrix is not None:
                    # Mass matrix가 있는 경우
                    # Null-space torque 계산. tau_null = P^T * M * joint_acc_nullspace
                    tau_null = (
                        nullspace_jacobian_transpose @ mass_matrix @ joint_acc_nullspace
                    ).squeeze(-1)

                else:
                    # Mass matrix가 없는 경우
                    # Null-space torque 계산. tau_null = P^T * joint_acc_nullspace
                    tau_null = nullspace_jacobian_transpose @ joint_acc_nullspace

                # Add the null-space joint efforts to the total joint efforts
                # 전체 effort에 추가
                joint_efforts += tau_null

            else:
                raise ValueError(
                    f"Invalid null-space control method: {self.cfg.nullspace_control}."
                )

        # 최종 joint_efforts 반환
        return joint_efforts
