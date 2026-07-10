"""Newton Physics backend for UR5e reaching."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from src.learning.controllers import CartesianController, JointController
from src.learning.envs.base_reach_env import ReachEnvConfig, RewardTerms, VecEnv
from src.learning.scripts.generate_ur5e_urdf import generate_ur5e_urdf


class NewtonReachEnv(VecEnv):
    """rsl_rl-compatible UR5e reaching env backed by Newton simulation."""

    num_actions: int = 0

    def __init__(self, cfg: ReachEnvConfig) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.max_episode_length = cfg.episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)

        self.frame_dt = 1.0 / cfg.sim_fps
        self.sim_substeps = cfg.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.global_step = 0

        self._build_newton_model()
        self.joint_pos = torch.zeros((self.num_envs, 6), dtype=torch.float32, device=self.device)
        self.prev_joint_pos = torch.zeros_like(self.joint_pos)
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.eef_pos = torch.zeros_like(self.target_pos)
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32, device=self.device)
        self.reset()

    def seed(self, seed: int = -1) -> int:
        if seed < 0:
            seed = self.cfg.seed
        self.generator.manual_seed(seed)
        return seed

    def reset(self, env_ids: torch.Tensor | None = None) -> tuple[TensorDict, dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = env_ids.to(device=self.device, dtype=torch.long)

        home = self._home_joint_pos()
        noise = 0.015 * torch.randn((env_ids.numel(), 6), generator=self.generator, device=self.device)
        self.joint_pos[env_ids] = home + noise
        self.prev_joint_pos[env_ids] = self.joint_pos[env_ids]
        self.joint_vel[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.target_pos[env_ids] = self._sample_targets(env_ids.numel())
        self.episode_length_buf[env_ids] = 0

        self._write_newton_joint_state(self.joint_pos)
        self.eef_pos = self._read_eef_pos()
        return self.get_observations(), {}

    def get_observations(self) -> TensorDict:
        obs = torch.cat(
            (
                self.joint_pos,
                self.joint_vel,
                self.eef_pos,
                self.target_pos,
                self.target_pos - self.eef_pos,
            ),
            dim=-1,
        )
        return TensorDict({"policy": obs}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(device=self.device, dtype=self.joint_pos.dtype)
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected action shape {(self.num_envs, self.num_actions)}, got {tuple(actions.shape)}.")

        self.prev_joint_pos.copy_(self.joint_pos)
        joint_delta = self._action_to_joint_delta(actions)
        target_joint_pos = self._clamp_joint_limits(self.joint_pos + joint_delta)
        self._write_newton_joint_targets(target_joint_pos)
        self._simulate()

        self.joint_pos = self._read_joint_pos()
        self.joint_vel = (self.joint_pos - self.prev_joint_pos) / self.frame_dt
        self.eef_pos = self._read_eef_pos()
        self.episode_length_buf += 1
        self.global_step += 1

        rewards, terms = self._compute_rewards(actions)
        reached = terms.success > 0.0
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = timeout | (reached if self.cfg.terminate_on_success else torch.zeros_like(reached))
        self.last_actions.copy_(actions)

        if torch.any(dones):
            self.reset(torch.nonzero(dones, as_tuple=False).squeeze(-1))

        extras = {
            "time_outs": timeout,
            "log": {
                "/reach/mean_distance": terms.distance.mean(),
                "/reach/position_tracking": terms.position_tracking.mean(),
                "/reach/fine_tracking_reward": terms.fine_tracking_reward.mean(),
                "/reach/success_rate": reached.float().mean(),
                "/reach/action_penalty": terms.action_penalty.mean(),
                "/reach/action_rate_penalty": terms.action_rate_penalty.mean(),
                "/reach/joint_velocity_penalty": terms.joint_velocity_penalty.mean(),
            },
        }
        self._render_if_needed()
        return self.get_observations(), rewards, dones.long(), extras

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()

    def forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Return EEF positions for Cartesian controller Jacobian estimates."""
        old_pos = self.joint_pos.clone()
        self._write_newton_joint_state(joint_pos)
        eef_pos = self._read_eef_pos()
        self._write_newton_joint_state(old_pos)
        return eef_pos

    def _action_to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _build_newton_model(self) -> None:
        import warp as wp

        if self.cfg.warp_cache_dir:
            wp.config.kernel_cache_dir = self.cfg.warp_cache_dir

        import newton
        import newton.viewer
        from newton import JointTargetMode
        from newton.selection import ArticulationView

        if hasattr(newton, "use_coord_layout_targets"):
            newton.use_coord_layout_targets = True

        urdf_path = generate_ur5e_urdf()

        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.add_urdf(
            str(Path(urdf_path).resolve()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            hide_visuals=False,
            ignore_inertial_definitions=True,
            force_position_velocity_actuation=False,
        )
        target_body = robot.add_body(
            xform=wp.transform(wp.vec3(0.45, 0.0, 0.35), wp.quat_identity()),
            mass=0.01,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-4),
            label="target_marker",
            is_kinematic=True,
        )
        robot.add_shape_sphere(
            target_body,
            radius=0.025,
            as_site=True,
            label="target_marker_shape",
        )
        for dof_id in range(len(robot.joint_target_ke)):
            robot.joint_target_ke[dof_id] = self.cfg.joint_stiffness
            robot.joint_target_kd[dof_id] = self.cfg.joint_damping
            robot.joint_target_mode[dof_id] = int(JointTargetMode.POSITION)

        builder = newton.ModelBuilder()
        builder.replicate(robot, self.num_envs, spacing=(0.0, 0.0, 0.0))
        builder.add_ground_plane(height=-1.05, label="ground")
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.55, 0.0, -0.04), wp.quat_identity()),
            hx=0.55,
            hy=0.35,
            hz=0.04,
            label="reach_table",
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.04), wp.quat_identity()),
            hx=0.12,
            hy=0.12,
            hz=0.04,
            label="robot_base_block",
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None
        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)
        self.articulation_view = ArticulationView(
            self.model,
            "*ur5e*",
            exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE],
        )
        if self.articulation_view.count != self.num_envs:
            raise RuntimeError(
                f"Expected {self.num_envs} UR5e articulations, got {self.articulation_view.count}."
            )
        if self.articulation_view.joint_dof_count != 6:
            raise RuntimeError(f"Expected 6 UR5e dofs, got {self.articulation_view.joint_dof_count}.")

        self._newton = newton
        self._wp = wp
        self._target_attr = "joint_target_q" if hasattr(self.control, "joint_target_q") else "joint_target_pos"
        self._tool_body_indices = [i for i, label in enumerate(self.model.body_label) if str(label).endswith("/tool0")]
        if len(self._tool_body_indices) != self.num_envs:
            raise RuntimeError(f"Expected {self.num_envs} tool0 bodies, got {len(self._tool_body_indices)}.")
        self._target_body_indices = [
            i for i, label in enumerate(self.model.body_label) if str(label) == "target_marker"
        ]
        if len(self._target_body_indices) != self.num_envs:
            raise RuntimeError(f"Expected {self.num_envs} target marker bodies, got {len(self._target_body_indices)}.")
        self._joint_lower = self._read_limit("joint_limit_lower")
        self._joint_upper = self._read_limit("joint_limit_upper")
        self.viewer = self._create_viewer(newton, wp)
        if self.viewer is not None:
            self.viewer.set_model(self.model)
            if hasattr(self.viewer, "set_camera"):
                self.viewer.set_camera(
                    pos=wp.vec3(1.6, -2.1, 1.2),
                    pitch=-22.0,
                    yaw=-140.0,
                )
            for _ in range(3):
                self.viewer.begin_frame(0.0)
                self.viewer.log_state(self.state_0)
                self.viewer.end_frame()

    def _simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def _create_viewer(self, newton, wp):
        if self.cfg.viewer == "null":
            return None
        if self.cfg.viewer == "gl":
            return newton.viewer.ViewerGL(headless=self.cfg.headless)
        if self.cfg.viewer == "usd":
            return newton.viewer.ViewerUSD(output_path="logs/reach_scene.usd", num_frames=None)
        if self.cfg.viewer == "viser":
            return newton.viewer.ViewerViser()
        raise ValueError(f"Unsupported viewer for training env: {self.cfg.viewer}")

    def _render_if_needed(self) -> None:
        if self.viewer is None:
            return
        if self.cfg.render_every <= 0 or self.global_step % self.cfg.render_every != 0:
            return
        if hasattr(self.viewer, "is_running") and not self.viewer.is_running():
            return

        frame_start_time = time.perf_counter()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None and hasattr(self.viewer, "log_contacts"):
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()
        if self.cfg.render_fps:
            sleep_time = (1.0 / self.cfg.render_fps) - (time.perf_counter() - frame_start_time)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    def _read_joint_pos(self) -> torch.Tensor:
        joint_np = self.articulation_view.get_dof_positions(self.state_0).numpy().reshape(self.num_envs, 6)
        return torch.as_tensor(joint_np, dtype=torch.float32, device=self.device)

    def _read_eef_pos(self) -> torch.Tensor:
        self._newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        body_q = self.state_0.body_q.numpy().reshape(-1, 7)
        eef_np = body_q[self._tool_body_indices, :3].copy()
        return torch.as_tensor(eef_np, dtype=torch.float32, device=self.device)

    def _read_limit(self, name: str) -> torch.Tensor:
        limit_np = self.articulation_view.get_attribute(name, self.model).numpy().reshape(self.num_envs, 6)
        limit = torch.as_tensor(limit_np, dtype=torch.float32, device=self.device)
        return torch.where(torch.isfinite(limit), limit, torch.full_like(limit, 10.0))

    def _write_newton_joint_state(self, joint_pos: torch.Tensor) -> None:
        joint_np = joint_pos.detach().to("cpu", dtype=torch.float32).numpy().reshape(self.num_envs, 1, 6)
        zero_np = np.zeros_like(joint_np)
        joint_wp = self._wp.array(joint_np, dtype=float, device=self.model.device)
        zero_wp = self._wp.array(zero_np, dtype=float, device=self.model.device)
        self.articulation_view.set_dof_positions(self.state_0, joint_wp)
        self.articulation_view.set_dof_positions(self.state_1, joint_wp)
        self.articulation_view.set_dof_velocities(self.state_0, zero_wp)
        self.articulation_view.set_dof_velocities(self.state_1, zero_wp)
        self.articulation_view.set_attribute(self._target_attr, self.control, joint_wp)
        self._newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        self._sync_target_markers()

    def _write_newton_joint_targets(self, joint_pos: torch.Tensor) -> None:
        joint_np = joint_pos.detach().to("cpu", dtype=torch.float32).numpy().reshape(self.num_envs, 1, 6)
        joint_wp = self._wp.array(joint_np, dtype=float, device=self.model.device)
        self.articulation_view.set_attribute(self._target_attr, self.control, joint_wp)
        self._sync_target_markers()

    def _sync_target_markers(self) -> None:
        if not hasattr(self, "_target_body_indices"):
            return
        target_np = self.target_pos.detach().to("cpu", dtype=torch.float32).numpy()
        for state in (self.state_0, self.state_1):
            body_q = state.body_q.numpy()
            for env_id, body_id in enumerate(self._target_body_indices):
                body_q[body_id, :3] = target_np[env_id]
                body_q[body_id, 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=body_q.dtype)
            state.body_q.assign(body_q)

    def _clamp_joint_limits(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(joint_pos, self._joint_lower), self._joint_upper)

    def _compute_rewards(self, actions: torch.Tensor) -> tuple[torch.Tensor, RewardTerms]:
        distance = torch.linalg.norm(self.target_pos - self.eef_pos, dim=-1)
        position_tracking = self.cfg.position_tracking_weight * distance
        fine_tracking_reward = self.cfg.fine_tracking_weight * (
            1.0 - torch.tanh(distance / self.cfg.fine_tracking_std)
        )
        success = (distance < self.cfg.success_radius).float()
        action_penalty = torch.sum(actions.square(), dim=-1) * self.cfg.action_penalty
        action_rate_penalty = torch.sum((actions - self.last_actions).square(), dim=-1) * self.cfg.action_rate_penalty
        joint_velocity_penalty = torch.sum(self.joint_vel.square(), dim=-1) * self.cfg.joint_velocity_penalty
        rewards = (
            position_tracking
            + fine_tracking_reward
            + success * self.cfg.success_reward
            - action_penalty
            - action_rate_penalty
            - joint_velocity_penalty
        )
        return rewards, RewardTerms(
            distance,
            position_tracking,
            fine_tracking_reward,
            success,
            action_penalty,
            action_rate_penalty,
            joint_velocity_penalty,
        )

    def _sample_targets(self, count: int) -> torch.Tensor:
        low = torch.tensor(self.cfg.target_low, dtype=torch.float32, device=self.device)
        high = torch.tensor(self.cfg.target_high, dtype=torch.float32, device=self.device)
        rand = torch.rand((count, 3), generator=self.generator, device=self.device)
        return low + (high - low) * rand

    def _home_joint_pos(self) -> torch.Tensor:
        return torch.tensor(self.cfg.home_joint_pos, dtype=torch.float32, device=self.device)


class NewtonReachJointEnv(NewtonReachEnv):
    """Newton UR5e reaching task with 6D joint-space policy actions."""

    num_actions = 6

    def __init__(self, cfg: ReachEnvConfig) -> None:
        self.controller = JointController(cfg.action_scale, cfg.max_joint_delta)
        super().__init__(cfg)

    def _action_to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.controller.to_joint_delta(actions), -self.cfg.max_joint_delta, self.cfg.max_joint_delta)


class NewtonReachCartesianEnv(NewtonReachEnv):
    """Newton UR5e reaching task with 3D Cartesian EEF policy actions."""

    num_actions = 3

    def __init__(self, cfg: ReachEnvConfig, damping: float = 0.05) -> None:
        self.controller = CartesianController(cfg.action_scale, damping=damping, max_delta=cfg.max_joint_delta)
        super().__init__(cfg)

    def _action_to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        return self.controller.to_joint_delta(actions, self.joint_pos, self.forward_kinematics)
