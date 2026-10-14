"""V2 robot self-collision detection, penalties, and diagnostics."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ManagerTermBase


_STATE_ATTR = "_sweep_v2_self_collision_state"


class SelfCollisionTermination(ManagerTermBase):
    """Terminate on hard or sustained non-adjacent robot self-contact."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self.sensor_names = tuple(params["sensor_names"])
        excluded_pairs = tuple(tuple(pair) for pair in params["excluded_pairs"])
        self.command_name = params["command_name"]

        if len(set(self.sensor_names)) != len(self.sensor_names):
            raise ValueError("Self-collision sensor names must be unique.")
        sensor_index = {name: index for index, name in enumerate(self.sensor_names)}
        excluded_indices: set[tuple[int, int]] = set()
        for first_name, second_name in excluded_pairs:
            if first_name not in sensor_index or second_name not in sensor_index:
                raise ValueError(
                    "Unknown self-collision exclusion pair: "
                    f"{first_name}, {second_name}."
                )
            first_index = sensor_index[first_name]
            second_index = sensor_index[second_name]
            if first_index == second_index:
                raise ValueError("A self-collision exclusion pair cannot repeat one sensor.")
            excluded_indices.add(tuple(sorted((first_index, second_index))))

        self._allowed_filter_indices: list[tuple[int, ...]] = []
        for sensor_id in range(len(self.sensor_names)):
            allowed = tuple(
                filter_id
                for filter_id in range(sensor_id + 1, len(self.sensor_names))
                if (sensor_id, filter_id) not in excluded_indices
            )
            self._allowed_filter_indices.append(allowed)
        if not any(self._allowed_filter_indices):
            raise ValueError("Self-collision configuration contains no detectable link pairs.")

        self.current_max_force = torch.zeros(self.num_envs, device=self.device)
        self.episode_max_force = torch.zeros_like(self.current_max_force)
        self.current_pair_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.last_episode_pair_id = torch.full_like(self.current_pair_id, -1)
        self.last_episode_max_force = torch.zeros_like(self.current_max_force)
        self.consecutive_contact_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )

        command = env.command_manager.get_term(self.command_name)
        required_metrics = (
            "self_collision_force_max",
            "self_collision_termination_rate",
        )
        missing_metrics = [name for name in required_metrics if name not in command.metrics]
        if missing_metrics:
            raise ValueError(
                f"Command '{self.command_name}' is missing self-collision metrics: "
                f"{missing_metrics}."
            )
        if hasattr(env, _STATE_ATTR):
            raise RuntimeError("Only one v2 self-collision termination term is supported.")
        setattr(env, _STATE_ATTR, self)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.last_episode_pair_id[env_ids] = self.current_pair_id[env_ids]
        self.last_episode_max_force[env_ids] = self.episode_max_force[env_ids]
        self.current_max_force[env_ids] = 0.0
        self.episode_max_force[env_ids] = 0.0
        self.current_pair_id[env_ids] = -1
        self.consecutive_contact_steps[env_ids] = 0

    def _compute_max_force(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        max_force = torch.zeros(self.num_envs, device=self.device)
        max_pair_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        expected_filter_count = len(self.sensor_names)
        for sensor_id, sensor_name in enumerate(self.sensor_names):
            allowed_indices = self._allowed_filter_indices[sensor_id]
            if not allowed_indices:
                continue
            force_matrix_w = env.scene[sensor_name].data.force_matrix_w
            if force_matrix_w is None:
                raise RuntimeError(
                    f"Contact sensor '{sensor_name}' has no filtered force matrix."
                )
            if force_matrix_w.shape[1] != 1:
                raise RuntimeError(
                    f"Contact sensor '{sensor_name}' must resolve to exactly one rigid body."
                )
            if force_matrix_w.shape[2] != expected_filter_count:
                raise RuntimeError(
                    f"Contact sensor '{sensor_name}' resolved "
                    f"{force_matrix_w.shape[2]} filters; expected "
                    f"{expected_filter_count}."
                )
            pair_forces = torch.linalg.vector_norm(
                force_matrix_w[:, :, allowed_indices, :], dim=-1
            )
            pair_forces = torch.nan_to_num(
                pair_forces, nan=0.0, posinf=float("inf"), neginf=0.0
            ).amax(dim=1)
            sensor_max_force, local_pair_index = pair_forces.max(dim=1)
            allowed_tensor = torch.as_tensor(
                allowed_indices, dtype=torch.long, device=self.device
            )
            sensor_pair_id = (
                sensor_id * expected_filter_count
                + allowed_tensor[local_pair_index]
            )
            update = sensor_max_force > max_force
            max_force = torch.where(update, sensor_max_force, max_force)
            max_pair_id = torch.where(update, sensor_pair_id, max_pair_id)
        return max_force, max_pair_id

    def __call__(
        self,
        env,
        sensor_names: tuple[str, ...],
        excluded_pairs: tuple[tuple[str, str], ...],
        command_name: str,
        persistent_force_threshold: float,
        hard_force_threshold: float,
        persistence_steps: int,
    ) -> torch.Tensor:
        del sensor_names, excluded_pairs, command_name
        if not 0.0 < persistent_force_threshold < hard_force_threshold:
            raise ValueError(
                "Persistent force threshold must be positive and below the hard threshold."
            )
        if persistence_steps < 1:
            raise ValueError("Self-collision persistence must be at least one step.")

        max_force, pair_id = self._compute_max_force(env)
        self.current_max_force[:] = max_force
        self.current_pair_id[:] = pair_id
        self.episode_max_force[:] = torch.maximum(
            self.episode_max_force, self.current_max_force
        )
        persistent_contact = self.current_max_force > persistent_force_threshold
        self.consecutive_contact_steps[:] = torch.where(
            persistent_contact,
            self.consecutive_contact_steps + 1,
            torch.zeros_like(self.consecutive_contact_steps),
        )
        terminated = (self.current_max_force > hard_force_threshold) | (
            self.consecutive_contact_steps >= persistence_steps
        )

        command = env.command_manager.get_term(self.command_name)
        command.metrics["self_collision_force_max"][:] = self.episode_max_force
        command.metrics["self_collision_termination_rate"][:] = terminated.float()
        return terminated


def _state(env) -> SelfCollisionTermination:
    state = getattr(env, _STATE_ATTR, None)
    if not isinstance(state, SelfCollisionTermination):
        raise RuntimeError(
            "The v2 self-collision termination term must run before its reward terms."
        )
    return state


def self_collision_force_penalty(
    env,
    soft_force_threshold: float,
    hard_force_threshold: float,
) -> torch.Tensor:
    """Return a smooth [0, 1] penalty for current non-adjacent self-contact."""
    if not 0.0 <= soft_force_threshold < hard_force_threshold:
        raise ValueError("Soft force threshold must be below the hard threshold.")
    return torch.clamp(
        (_state(env).current_max_force - soft_force_threshold)
        / (hard_force_threshold - soft_force_threshold),
        min=0.0,
        max=1.0,
    )


def self_collision_terminal_penalty(
    env,
    termination_term_name: str,
    base_cost: float,
    early_termination_cost: float,
) -> torch.Tensor:
    """Return a dt-corrected terminal cost, stronger for early collision escape."""
    if base_cost < 0.0 or early_termination_cost < 0.0:
        raise ValueError("Self-collision terminal costs must be non-negative.")
    if env.step_dt <= 0.0 or env.max_episode_length <= 0:
        raise ValueError("Environment timing must be positive.")
    terminated = env.termination_manager.get_term(termination_term_name).float()
    remaining_fraction = torch.clamp(
        (env.max_episode_length - env.episode_length_buf).float()
        / float(env.max_episode_length),
        min=0.0,
        max=1.0,
    )
    cost = base_cost + early_termination_cost * remaining_fraction
    # RewardManager multiplies every term by step_dt.  Divide here so the configured
    # cost is the actual one-shot episode return contribution.
    return terminated * cost / env.step_dt
