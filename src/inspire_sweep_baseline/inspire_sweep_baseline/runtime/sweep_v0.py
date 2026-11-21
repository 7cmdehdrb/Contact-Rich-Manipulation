"""Separate Sweep-family entry point using the shared, unchanged Sweep lifecycle."""
from .stabilize_sweep import SweepOnlyEnv
from isaaclab.utils.math import quat_apply_inverse
from ..mdp.common.stabilize_sweep_contact import centroid
from ..mdp.common.sweep_v0_state import SingleContactSweepState, SingleContactV1SweepState
from ..mdp.reward_functions.sweep_contact_preference import sweep_region


class SweepEnv(SweepOnlyEnv):
    """Sweep v0; reset, contact sampling and telemetry use the shared runtime."""


class SweepSingleContactEnv(SweepEnv):
    """Keep the Sweep lifecycle; replace only accumulated contact credit."""

    _state_type = SingleContactSweepState


class SweepSingleContactV1Env(SweepSingleContactEnv):
    """Preserve physics capture; initialize reach only after the parent reset."""

    _state_type = SingleContactV1SweepState

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        shelf_quat = self.scene['shelf'].data.root_quat_w[env_ids]
        direction = quat_apply_inverse(shelf_quat, self.sweep_dir[env_ids])
        self._stabilize_state.reset_reach(
            env_ids, centroid(self)[env_ids], self.scene['ee_frame'].data.target_pos_w[env_ids, 0],
            self._revision_bounds_t[env_ids, 0], direction, sweep_region(direction), shelf_quat,
        )
