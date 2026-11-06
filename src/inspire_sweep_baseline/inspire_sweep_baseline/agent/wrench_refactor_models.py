"""Independent Refactor models with separate initial/live obstacle encoders."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState, MLP
from tensordict import TensorDict
from torch import nn


class WrenchRefactorObstacleEncoder(nn.Module):
    """Trainable 64D to 16D compression; masked coordinates never enter the MLP."""

    def __init__(self, hidden_dims=(32,), embedding_dim=16, activation="relu", observation_slots=16):
        super().__init__()
        self.observation_slots = observation_slots
        self.mlp = MLP(4 * observation_slots, embedding_dim, hidden_dims, activation)

    def forward(self, obstacles: torch.Tensor) -> torch.Tensor:
        rows = obstacles.unflatten(-1, (self.observation_slots, 4))
        rows = torch.where(rows[..., :1] > 0.5, rows, torch.zeros_like(rows))
        return self.mlp(rows.flatten(start_dim=-2))


class WrenchRefactorModel(MLPModel):
    """Actor: policy74 + initial64. Critic: current-target raw74 + live64.

    Obstacle embeddings are recomputed from rollout observations during PPO,
    so each model's independent encoder participates in its own gradients.
    No Final/Holding configuration or model class appears in the inheritance.
    """

    raw_observation_dim = 74
    observation_groups = {'actor': ('policy', 'obstacles'), 'critic': ('critic', 'critic_obstacles')}

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims=(256, 128, 64), activation="elu",
        obs_normalization=False, distribution_cfg=None, obstacle_group=None,
        encoder_hidden_dims=(32,), embedding_dim=16, encoder_activation="relu",
        obstacle_observation_slots=16,
    ):
        if obs_set not in ("actor", "critic"):
            raise ValueError("Refactor observation set must be actor or critic")
        required_obstacle_group = self.observation_groups[obs_set][1]
        if obstacle_group is not None and obstacle_group != required_obstacle_group:
            raise ValueError(f"{obs_set} must use {required_obstacle_group}")
        if obstacle_observation_slots != 16 or embedding_dim != 16:
            raise ValueError("Refactor requires 16 obstacle slots and a 16D embedding")
        self.obstacle_group = required_obstacle_group
        self.obstacle_observation_slots = obstacle_observation_slots
        self.embedding_dim = embedding_dim
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg,
        )
        self.obstacle_encoder = WrenchRefactorObstacleEncoder(
            encoder_hidden_dims, embedding_dim, encoder_activation, obstacle_observation_slots,
        )

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        raw_group = self.observation_groups[obs_set][0]
        expected = [raw_group, self.obstacle_group]
        if obs_groups[obs_set] != expected:
            raise ValueError(f"{obs_set} observation groups must be {expected} in order")
        if obs[raw_group].ndim != 2 or obs[raw_group].shape[-1] != self.raw_observation_dim:
            raise ValueError(f"Refactor requires {self.raw_observation_dim}D policy observations")
        obstacles = obs[self.obstacle_group]
        if obstacles.ndim != 2 or obstacles.shape[-1] != 64:
            raise ValueError("Refactor requires 64D obstacle observations")
        direct = [name for name in expected if name != self.obstacle_group]
        return super()._get_obs_dim(obs, {obs_set: direct}, obs_set)

    def _get_latent_dim(self):
        return self.obs_dim + self.embedding_dim

    def get_latent(self, obs, masks: torch.Tensor | None = None, hidden_state: HiddenState = None):
        direct = super().get_latent(obs, masks, hidden_state)
        obstacles = self.obstacle_encoder(obs[self.obstacle_group])
        return torch.cat((direct, obstacles), dim=-1)

    def as_jit(self):
        return _ExportWrenchRefactorModel(self)

    def as_onnx(self, verbose=False):
        return _ExportWrenchRefactorModel(self, verbose=verbose)


class WrenchRefactorRelativeModel(WrenchRefactorModel):
    """Independent Actor/Critic MLPs trained from the same stored input tensors."""

    raw_observation_dim = 58
    observation_groups = {'actor': ('policy', 'obstacles'), 'critic': ('policy', 'obstacles')}


class WrenchRefactorRelativeV2Model(WrenchRefactorRelativeModel):
    """Same networks and EEF obstacle input, separate 58D direct groups."""
    observation_groups = {'actor': ('policy', 'obstacles'), 'critic': ('critic', 'obstacles')}


class _ExportWrenchRefactorModel(nn.Module):
    """Export policy + initial obstacles: 138D for Refactor, 122D for Relative."""

    is_recurrent: bool = False

    def __init__(self, model, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.direct_dim = model.obs_dim
        self.input_size = model.obs_dim + 4 * model.obstacle_observation_slots
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
        """Stateless RSL-RL export interface."""

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
