"""Clutter v1 Critic with live target XYZ and an inherited obstacle encoder."""

from tensordict import TensorDict

from ..contact_sweep_clutter.models import ObstacleEncoderModel


class PrivilegedObstacleEncoderModel(ObstacleEncoderModel):
    """Append live target XYZ to the Critic's direct policy observations.

    The Actor keeps the parent's model and its 57D policy plus 40D initial
    obstacles contract. Only this Critic model accepts the privileged group;
    both models retain their own trainable 40D-to-16D obstacle encoder.
    """

    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
    ) -> tuple[list[str], int]:
        selected = obs_groups[obs_set]
        if selected.count("privileged") != 1:
            raise ValueError(f"{obs_set} must include the privileged group exactly once.")
        privileged = obs["privileged"]
        if privileged.ndim != 2 or privileged.shape[-1] != 3:
            raise ValueError("Privileged target position must have shape (num_envs, 3).")
        # Preserve the parent's 57D policy and 40D obstacle checks. Its encoder,
        # optimizer participation, checkpoint format and export remain inherited.
        parent_groups = [name for name in selected if name != "privileged"]
        groups, dimension = super()._get_obs_dim(obs, {obs_set: parent_groups}, obs_set)
        return [*groups, "privileged"], dimension + 3
