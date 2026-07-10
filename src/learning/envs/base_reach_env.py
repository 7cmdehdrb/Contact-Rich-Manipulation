"""Shared reaching environment contract.

The first backend is intentionally lightweight: it uses UR5e forward kinematics
to validate the rsl_rl vector-env contract before the Newton URDF adapter is
wired in. The public observation/action/reward contract is the same one the
Newton backend should preserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict

try:
    from rsl_rl.env import VecEnv
except ImportError:  # pragma: no cover - keeps static tooling usable outside newton env.
    class VecEnv:  # type: ignore[no-redef]
        pass


@dataclass
class ReachEnvConfig:
    """Configuration shared by the joint and Cartesian reach envs."""

    num_envs: int = 128
    episode_length: int = 160
    device: str = "cpu"
    backend: str = "newton"
    warp_cache_dir: str | None = None
    sim_fps: int = 50
    sim_substeps: int = 4
    joint_stiffness: float = 500.0
    joint_damping: float = 50.0
    seed: int = 7
    action_scale: float = 0.04
    max_joint_delta: float = 0.08
    success_radius: float = 0.035
    success_reward: float = 2.0
    action_penalty: float = 0.01
    joint_velocity_penalty: float = 0.002
    target_low: tuple[float, float, float] = (0.28, -0.28, 0.10)
    target_high: tuple[float, float, float] = (0.68, 0.28, 0.55)
    home_joint_pos: tuple[float, float, float, float, float, float] = (
        0.0,
        -1.5708,
        1.5708,
        -1.5708,
        -1.5708,
        0.0,
    )

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "ReachEnvConfig":
        env_cfg = cfg.get("env", cfg)
        return cls(**{key: value for key, value in env_cfg.items() if key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RewardTerms:
    distance: torch.Tensor
    success: torch.Tensor
    action_penalty: torch.Tensor
    joint_velocity_penalty: torch.Tensor


class BaseReachEnv(VecEnv):
    """rsl_rl-compatible vector environment for UR5e point reaching."""

    num_actions: int = 0

    def __init__(self, cfg: ReachEnvConfig) -> None:
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.device = torch.device(cfg.device)
        self.max_episode_length = cfg.episode_length
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.generator = torch.Generator(device=self.device).manual_seed(cfg.seed)

        self.joint_pos = self._home_joint_pos().repeat(self.num_envs, 1)
        self.prev_joint_pos = self.joint_pos.clone()
        self.joint_vel = torch.zeros_like(self.joint_pos)
        self.target_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.eef_pos = self.forward_kinematics(self.joint_pos)
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
        self.eef_pos[env_ids] = self.forward_kinematics(self.joint_pos[env_ids])
        self.target_pos[env_ids] = self._sample_targets(env_ids.numel())
        self.episode_length_buf[env_ids] = 0
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
        self.joint_pos = self.joint_pos + joint_delta
        self.joint_vel = self.joint_pos - self.prev_joint_pos
        self.eef_pos = self.forward_kinematics(self.joint_pos)
        self.episode_length_buf += 1

        rewards, terms = self._compute_rewards(actions)
        reached = terms.success > 0.0
        timeout = self.episode_length_buf >= self.max_episode_length
        dones = reached | timeout

        if torch.any(dones):
            self.reset(torch.nonzero(dones, as_tuple=False).squeeze(-1))

        extras = {
            "time_outs": timeout,
            "log": {
                "/reach/mean_distance": terms.distance.mean(),
                "/reach/success_rate": reached.float().mean(),
                "/reach/action_penalty": terms.action_penalty.mean(),
                "/reach/joint_velocity_penalty": terms.joint_velocity_penalty.mean(),
            },
        }
        return self.get_observations(), rewards, dones.long(), extras

    def close(self) -> None:
        return None

    def forward_kinematics(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """Compute UR5e tool position using standard DH parameters."""
        dtype = joint_pos.dtype
        device = joint_pos.device
        a = torch.tensor([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0], dtype=dtype, device=device)
        d = torch.tensor([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996], dtype=dtype, device=device)
        alpha = torch.tensor([1.57079632679, 0.0, 0.0, 1.57079632679, -1.57079632679, 0.0], dtype=dtype, device=device)

        transform = torch.eye(4, dtype=dtype, device=device).expand(joint_pos.shape[0], 4, 4).clone()
        for i in range(6):
            transform = transform @ self._dh_transform(a[i], alpha[i], d[i], joint_pos[:, i])
        return transform[:, :3, 3]

    def _action_to_joint_delta(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _compute_rewards(self, actions: torch.Tensor) -> tuple[torch.Tensor, RewardTerms]:
        distance = torch.linalg.norm(self.target_pos - self.eef_pos, dim=-1)
        success = (distance < self.cfg.success_radius).float()
        action_penalty = torch.sum(actions.square(), dim=-1) * self.cfg.action_penalty
        joint_velocity_penalty = torch.sum(self.joint_vel.square(), dim=-1) * self.cfg.joint_velocity_penalty
        rewards = -distance + success * self.cfg.success_reward - action_penalty - joint_velocity_penalty
        return rewards, RewardTerms(distance, success, action_penalty, joint_velocity_penalty)

    def _sample_targets(self, count: int) -> torch.Tensor:
        low = torch.tensor(self.cfg.target_low, dtype=torch.float32, device=self.device)
        high = torch.tensor(self.cfg.target_high, dtype=torch.float32, device=self.device)
        rand = torch.rand((count, 3), generator=self.generator, device=self.device)
        return low + (high - low) * rand

    def _home_joint_pos(self) -> torch.Tensor:
        return torch.tensor(self.cfg.home_joint_pos, dtype=torch.float32, device=self.device)

    @staticmethod
    def _dh_transform(a: torch.Tensor, alpha: torch.Tensor, d: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        batch = theta.shape[0]
        transform = torch.zeros((batch, 4, 4), dtype=theta.dtype, device=theta.device)
        ct = torch.cos(theta)
        st = torch.sin(theta)
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        transform[:, 0, 0] = ct
        transform[:, 0, 1] = -st * ca
        transform[:, 0, 2] = st * sa
        transform[:, 0, 3] = a * ct
        transform[:, 1, 0] = st
        transform[:, 1, 1] = ct * ca
        transform[:, 1, 2] = -ct * sa
        transform[:, 1, 3] = a * st
        transform[:, 2, 1] = sa
        transform[:, 2, 2] = ca
        transform[:, 2, 3] = d
        transform[:, 3, 3] = 1.0
        return transform
