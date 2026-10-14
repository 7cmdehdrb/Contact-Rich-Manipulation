"""Cube-palm and table-robot contact predicates, independent of Isaac imports."""

from __future__ import annotations

import math

import torch


def _require_threshold(threshold_n: float) -> None:
    if not math.isfinite(threshold_n) or threshold_n <= 0.0:
        raise ValueError("Contact threshold must be finite and positive")


def table_contact_from_history(force_history_w: torch.Tensor, threshold_n: float) -> torch.Tensor:
    """Detect any individual robot-body pair crossing the threshold in history.

    Input is ``(N, physics_history, one_table_body, robot_filters, 3)``.
    Pair forces are never added, so opposing loads cannot cancel and several
    subthreshold pairs cannot combine into a false contact.
    """

    _require_threshold(threshold_n)
    if (
        force_history_w.ndim != 5
        or force_history_w.shape[2] != 1
        or force_history_w.shape[-1] != 3
        or force_history_w.shape[1] == 0
        or force_history_w.shape[3] == 0
    ):
        raise ValueError("Table force history must have shape (N, history, 1, robot_filters, 3)")
    magnitudes = torch.linalg.vector_norm(force_history_w, dim=-1)
    return magnitudes.flatten(start_dim=1).amax(dim=-1) >= threshold_n


def cube_palm_contact_from_forces(force_matrix_w: torch.Tensor, threshold_n: float) -> torch.Tensor:
    """Detect Cube contact on any of the seventeen physical palmar pads.

    Input is ``(N, one_cube_body, 17_palm_filters, 3)``. Carrier-link and
    unfiltered tactile forces are intentionally absent from this predicate.
    """

    _require_threshold(threshold_n)
    if force_matrix_w.ndim != 4 or tuple(force_matrix_w.shape[1:]) != (1, 17, 3):
        raise ValueError("Cube-palm force matrix must have shape (N, 1, 17, 3)")
    magnitudes = torch.linalg.vector_norm(force_matrix_w, dim=-1)
    return magnitudes.flatten(start_dim=1).amax(dim=-1) >= threshold_n


def table_contact_mask(env) -> torch.Tensor:
    history = env.scene["table_contacts"].data.force_matrix_w_history
    if history is None:
        raise RuntimeError("Table contact sensor requires robot-filtered force history")
    if history.ndim == 5 and history.shape[1] < env.cfg.decimation:
        raise RuntimeError("Table contact history must cover every substep in the policy interval")
    return table_contact_from_history(history, env.cfg.task.contact_threshold_n)


def cube_palm_contact_mask(env) -> torch.Tensor:
    matrix = env.scene["cube_palm_contacts"].data.force_matrix_w
    if matrix is None:
        raise RuntimeError("Cube contact sensor requires seventeen physical palm filter bodies")
    return cube_palm_contact_from_forces(matrix, env.cfg.task.contact_threshold_n)


__all__ = [
    "table_contact_from_history", "cube_palm_contact_from_forces", "table_contact_mask", "cube_palm_contact_mask",
]
