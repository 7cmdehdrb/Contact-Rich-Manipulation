"""Trainable masked obstacle encoding for the fusion task's 55D policy input."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict
from torch import nn

from ..contact_sweep_clutter.models import MaskedObstacleEncoder


class ContactSweepClutterFusionModel(MLPModel):
    """Use initial geometry for both models and live target XYZ for Critic only."""

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
        if selected.count(self.obstacle_group) != 1 or selected.count("policy") != 1:
            raise ValueError(f"{obs_set} must include policy and obstacle groups exactly once.")
        if obs["policy"].ndim != 2 or obs["policy"].shape[-1] != 55:
            raise ValueError("Clutter Fusion requires 55D policy observations.")
        obstacles = obs[self.obstacle_group]
        if obstacles.ndim != 2 or obstacles.shape[-1] != 40:
            raise ValueError("Obstacle observations must have shape (num_envs, 40).")
        if obs_set not in ("actor", "critic"):
            raise ValueError("Clutter Fusion observation set must be actor or critic.")
        expected = ["policy", self.obstacle_group]
        if obs_set == "critic":
            expected.insert(1, "privileged")
        if selected != expected:
            raise ValueError(f"{obs_set} observation groups must be {expected} exactly once in order.")
        if obs_set == "critic":
            privileged = obs["privileged"]
            if privileged.ndim != 2 or privileged.shape[-1] != 3:
                raise ValueError("Privileged target position must have shape (num_envs, 3).")
        direct_groups = [name for name in selected if name != self.obstacle_group]
        return super()._get_obs_dim(obs, {obs_set: direct_groups}, obs_set)

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
        return _ExportClutterFusionModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportClutterFusionModel(self, verbose=verbose)


class _ExportClutterFusionModel(nn.Module):
    """Export the full actor, consuming the 95D [policy55, obstacles40] input."""

    is_recurrent: bool = False

    def __init__(self, model: ContactSweepClutterFusionModel, verbose: bool = False) -> None:
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
        """Match the stateless RSL-RL export interface."""

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
