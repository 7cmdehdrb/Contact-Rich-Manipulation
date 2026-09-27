"""Corrective inherited environment for reliable pre-contact exploration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .assets.robot import HAND_CONTACT_BODY_NAMES, PALM_SENSOR_BODY_NAMES
from .env_approach import BlindSweepApproachEnv
from .mdp.v1_rewards import SelectedSurfaceApproachProgress
from .sensors import DORSAL_PARENT_BY_CHANNEL

if TYPE_CHECKING:
    from .env_v1_cfg import BlindSweepApproachV1EnvCfg


_PALM_TARGET_FILTER_IDS = tuple(
    HAND_CONTACT_BODY_NAMES.index(name) for name in PALM_SENSOR_BODY_NAMES
)
_DORSAL_TARGET_CHANNEL_FILTER_IDS = tuple(
    HAND_CONTACT_BODY_NAMES.index(name) for name in DORSAL_PARENT_BY_CHANNEL
)


class BlindSweepApproachV1Env(BlindSweepApproachEnv):
    """Approach-v0 plus reset references and TargetObject contact filtering."""

    cfg: "BlindSweepApproachV1EnvCfg"

    def __init__(self, cfg: "BlindSweepApproachV1EnvCfg", render_mode=None, **kwargs):
        self.initial_selected_surface_position_w = torch.zeros(
            cfg.scene.num_envs,
            3,
            dtype=torch.float32,
            device=cfg.sim.device,
        )
        self._palm_target_filter_ids = torch.as_tensor(
            _PALM_TARGET_FILTER_IDS, dtype=torch.long, device=cfg.sim.device
        )
        self._dorsal_target_channel_filter_ids = torch.as_tensor(
            _DORSAL_TARGET_CHANNEL_FILTER_IDS,
            dtype=torch.long,
            device=cfg.sim.device,
        )
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def _finalize_reset(self, env_ids: torch.Tensor) -> None:
        super()._finalize_reset(env_ids)
        self.initial_selected_surface_position_w[env_ids] = (
            self.selected_surface_control_point_w()[env_ids]
        )
        term = self.reward_manager.get_term_cfg(
            "selected_surface_approach_progress"
        ).func
        if not isinstance(term, SelectedSurfaceApproachProgress):
            raise RuntimeError("v1 approach progress reward was not instantiated correctly")
        term.capture_reference(env_ids)

    def selected_surface_approach_target_w(self) -> torch.Tensor:
        """Return a fixed endpoint beyond contact, on the commanded push path.

        The previous upstream-face target was reached before any physical hand
        collider touched the Cube.  This target uses the calibrated free
        approach frontier instead.  Reward credit beyond it is unlocked only
        by actual Cube motion; see ``contact_path_potential``.  A small
        mode/orientation-conditioned drop follows the board-safe diagonal
        trajectory measured by the contact probe.
        """

        initial = self.initial_selected_surface_position_w
        direction = self.command_direction_w
        target = initial + self.approach_free_distance_m().unsqueeze(-1) * direction
        target[:, 2] = initial[:, 2] - self.approach_contact_drop_m()
        return target

    def approach_free_distance_m(self) -> torch.Tensor:
        """Return the collision-calibrated dense-credit frontier per branch."""

        hand_x_up = self.hand_x_points_up()
        palm = self.surface_mode == 0
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        palm_distance = torch.where(
            hand_x_up,
            zeros + self.cfg.task.approach_free_distance_palm_hand_x_up_m,
            zeros + self.cfg.task.approach_free_distance_palm_hand_x_down_m,
        )
        dorsal_distance = torch.where(
            hand_x_up,
            zeros + self.cfg.task.approach_free_distance_dorsal_hand_x_up_m,
            zeros + self.cfg.task.approach_free_distance_dorsal_hand_x_down_m,
        )
        return torch.where(palm, palm_distance, dorsal_distance)

    def approach_contact_drop_m(self) -> torch.Tensor:
        """Return the calibrated descent from reset to the safe contact path."""

        hand_x_up = self.hand_x_points_up()
        palm = self.surface_mode == 0
        zeros = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        palm_drop = torch.where(
            hand_x_up,
            zeros + self.cfg.task.approach_contact_drop_palm_hand_x_up_m,
            zeros + self.cfg.task.approach_contact_drop_palm_hand_x_down_m,
        )
        dorsal_drop = torch.where(
            hand_x_up,
            zeros + self.cfg.task.approach_contact_drop_dorsal_hand_x_up_m,
            zeros + self.cfg.task.approach_contact_drop_dorsal_hand_x_down_m,
        )
        return torch.where(palm, palm_drop, dorsal_drop)

    def selected_surface_approach_height_w(self) -> torch.Tensor:
        """Return the safe path height at the current forward progress."""

        current = self.selected_surface_control_point_w()
        initial = self.initial_selected_surface_position_w
        direction_xy = self.command_direction_w[:, :2]
        direction_xy = direction_xy / torch.clamp_min(
            torch.linalg.vector_norm(direction_xy, dim=-1, keepdim=True), 1.0e-8
        )
        forward = torch.sum((current[:, :2] - initial[:, :2]) * direction_xy, dim=-1)
        alpha = torch.clamp(forward / self.approach_free_distance_m(), min=0.0, max=1.0)
        return initial[:, 2] - alpha * self.approach_contact_drop_m()

    def hand_object_contact(self) -> torch.Tensor:
        """Detect meaningful TargetObject contact on any physical hand body.

        This is intentionally separate from the tactile reward.  In palm
        episodes the source asset first touches with ``inspire_base_link`` or
        thumb carrier links; its 17 physical pad bodies remained untouched in
        calibration even though real Robot--Cube contact existed.
        """

        return (
            self.hand_object_contact_force_n()
            >= self.cfg.task.approach_target_contact_threshold_n
        )

    def hand_object_contact_force_n(self) -> torch.Tensor:
        """Return the largest TargetObject force on any physical hand body."""

        matrix = self.scene["target_hand_contacts"].data.force_matrix_w
        if matrix is None or matrix.shape[1] != 1:
            raise RuntimeError("Target hand-contact sensor has an invalid force matrix")
        magnitudes = torch.linalg.vector_norm(matrix[:, 0], dim=-1)
        return torch.amax(magnitudes, dim=-1)

    def selected_surface_object_contact(self) -> torch.Tensor:
        """Detect TargetObject contact on the selected palm/dorsal body set."""

        return torch.any(self.selected_surface_object_tactile_bits().bool(), dim=-1)

    def selected_surface_object_tactile_bits(self) -> torch.Tensor:
        """Return strict 17-channel tactile bits filtered to TargetObject.

        Do not silently call parent-link contact a physical palm-pad reading.
        The broader contact acquisition signal is exposed separately by
        :meth:`hand_object_contact`.
        """

        matrix = self.scene["target_hand_contacts"].data.force_matrix_w
        if matrix is None or matrix.shape[1] != 1:
            raise RuntimeError("Target hand-contact sensor has an invalid force matrix")
        forces = matrix[:, 0]
        palm_magnitude = torch.linalg.vector_norm(
            torch.index_select(forces, 1, self._palm_target_filter_ids), dim=-1
        )
        dorsal_magnitude = torch.linalg.vector_norm(
            torch.index_select(
                forces,
                1,
                self._dorsal_target_channel_filter_ids,
            ),
            dim=-1,
        )
        palm_bits = palm_magnitude >= self.cfg.task.tactile_threshold_n
        dorsal_bits = dorsal_magnitude >= self.cfg.task.dorsal_tactile_threshold_n
        return torch.where(self.surface_mode.unsqueeze(-1).bool(), dorsal_bits, palm_bits)


__all__ = ["BlindSweepApproachV1Env"]
