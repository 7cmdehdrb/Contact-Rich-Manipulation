"""Trainable obstacle compression for RSL-RL 5 actor and critic models.

The environment supplies the original 57D ``policy`` group and a separate 40D
``obstacles`` group. Keeping raw observations in rollout storage lets PPO
recompute the embedding and train its encoder on every optimization pass.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict
from torch import nn


class MaskedObstacleEncoder(nn.Module):
    """Compress ten ``[present, x, y, z]`` rows into a 16D embedding."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] | list[int] = (32,),
        embedding_dim: int = 16,
        activation: str = "relu",
        observation_slots: int = 10,
    ) -> None:
        super().__init__()
        self.observation_slots = observation_slots
        self.mlp = MLP(4 * observation_slots, embedding_dim, hidden_dims, activation)

    def forward(self, obstacles: torch.Tensor) -> torch.Tensor:
        rows = obstacles.unflatten(-1, (self.observation_slots, 4))
        # where also masks NaN/Inf dummy coordinates; multiplication by zero
        # would propagate them. Real rows retain their supplied coordinates.
        rows = torch.where(rows[..., :1] > 0.5, rows, torch.zeros_like(rows))
        return self.mlp(rows.flatten(start_dim=-2))


class ObstacleEncoderModel(MLPModel):
    """Concatenate the original observation with a learned obstacle embedding."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        obstacle_group: str = "obstacles",
        encoder_hidden_dims: tuple[int, ...] | list[int] = (32,),
        embedding_dim: int = 16,
        encoder_activation: str = "relu",
    ) -> None:
        self.obstacle_group = obstacle_group
        self.embedding_dim = embedding_dim
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg,
        )
        self.obstacle_encoder = MaskedObstacleEncoder(
            encoder_hidden_dims, embedding_dim, encoder_activation,
        )

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
    ) -> tuple[list[str], int]:
        selected = obs_groups[obs_set]
        if selected.count(self.obstacle_group) != 1:
            raise ValueError(f"{obs_set} must include the obstacle group exactly once.")
        obstacles = obs[self.obstacle_group]
        if obstacles.ndim != 2 or obstacles.shape[-1] != 40:
            raise ValueError("Obstacle observations must have shape (num_envs, 40).")
        direct_groups = [name for name in selected if name != self.obstacle_group]
        groups, dimension = super()._get_obs_dim(obs, {obs_set: direct_groups}, obs_set)
        if dimension != 57:
            raise ValueError("Contact Sweep requires the original 57D policy observations.")
        return groups, dimension

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.embedding_dim

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        direct = super().get_latent(obs, masks, hidden_state)
        obstacles = self.obstacle_encoder(obs[self.obstacle_group])
        return torch.cat((direct, obstacles), dim=-1)

    def as_jit(self) -> nn.Module:
        return _ExportObstacleEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportObstacleEncoderModel(self, verbose=verbose)


class _ExportObstacleEncoderModel(nn.Module):
    """Export the complete actor, consuming ``[policy57, obstacles40]``."""

    is_recurrent: bool = False

    def __init__(self, model: ObstacleEncoderModel, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.direct_dim = model.obs_dim
        self.input_size = model.obs_dim + 40
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.obstacle_encoder = copy.deepcopy(model.obstacle_encoder)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None else nn.Identity()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        direct = self.obs_normalizer(obs[..., :self.direct_dim])
        obstacles = self.obstacle_encoder(obs[..., self.direct_dim:])
        return self.deterministic_output(self.mlp(torch.cat((direct, obstacles), dim=-1)))

    @torch.jit.export
    def reset(self) -> None:
        """Match the stateless RSL-RL policy export interface."""

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
