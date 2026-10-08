"""Manager-based sweep with an explicit mixed arm/hand action space."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from sweeping_policy.env import ShelfSweepEnv


class InspireShelfSweepEnv(ShelfSweepEnv):
    """Keep Sweep's vectorized target state and expose normalized hand inputs."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Joint teleports update FK without a new contact/wrench sample. Keep
        # each reset row masked until its first actual physics step.
        self.sensor_data_fresh = torch.zeros(
            cfg.scene.num_envs, dtype=torch.bool, device=cfg.sim.device
        )
        self._sweep_metric_steps = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        self._sweep_metric_sums = {}
        self._sweep_metric_maxima = {}
        self._sweep_final_progress = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        if self.action_manager.total_action_dim != 8:
            raise RuntimeError("Expected six OSC and two Inspire synergy action values")
        if self.scene["object_collection"].num_objects != 1:
            raise RuntimeError("The Inspire sweep must instantiate exactly one object per environment")
        self.single_action_space = gym.spaces.Box(
            low=np.array([-np.inf] * 6 + [0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf] * 6 + [1.0, 1.0], dtype=np.float32),
        )
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids):
        # Capture completed episodes before reset overwrites poses and commands.
        metric_log = {}
        valid_ids = env_ids[self._sweep_metric_steps[env_ids] > 0]
        if len(valid_ids) > 0:
            steps = self._sweep_metric_steps[valid_ids]
            for name, total in self._sweep_metric_sums.items():
                metric_log[f"Sweep/{name}"] = (total[valid_ids] / steps).mean()
            for name, maximum in self._sweep_metric_maxima.items():
                metric_log[f"Sweep/{name}"] = maximum[valid_ids].mean()
            sums = self._sweep_metric_sums
            metric_log["Sweep/wrist_blocked_given_near_hand_rate"] = (
                sums["hand_only_blocked_wrist_rate"][valid_ids]
                / sums["near_hand_rate"][valid_ids].clamp_min(1.0)
            ).mean()
            metric_log["Sweep/final_progress_m"] = self._sweep_final_progress[valid_ids].mean()
        super()._reset_idx(env_ids)
        self.extras["log"].update(metric_log)
        self._sweep_metric_steps[env_ids] = 0
        for buffer in (*self._sweep_metric_sums.values(), *self._sweep_metric_maxima.values()):
            buffer[env_ids] = 0
        self._sweep_final_progress[env_ids] = 0
        self.sensor_data_fresh[env_ids] = False

    def _record_sweep_metrics(self, values):
        """Accumulate unweighted step metrics; log episode means at reset."""
        metrics = {
            "near_hand_rate": values["near_hand"],
            "near_wrist_rate": values["near_wrist"],
            "near_wrist_uncompensated_rate": values["wrist_y_distance_uncompensated_m"] < 0.04,
            "gate_rate": values["gate"],
            "hand_only_blocked_wrist_rate": values["near_hand"] & ~values["near_wrist"],
            "eef_gate_distance_m": values["reaching_distance_m"],
            "wrist_y_distance_m": values["wrist_y_distance_m"],
            "wrist_y_distance_uncompensated_m": values["wrist_y_distance_uncompensated_m"],
            "eef_wrist_y_separation_m": values["eef_wrist_y_separation_m"],
            "palm_contact_rate": values["palm_contact"],
            "any_pad_contact_rate": values["palm_contact"] | values["other_pad_contact"],
            "gate_without_pad_contact_rate": values["gate"] & ~(values["palm_contact"] | values["other_pad_contact"]),
            "forward_velocity_m_s": values["object_forward_velocity_m_s"],
            "forward_motion_rate": values["object_forward_velocity_m_s"] > 0.005,
            "backward_motion_rate": values["object_forward_velocity_m_s"] < -0.005,
            "goal_distance_m": values["goal_distance_m"],
            "height_abs_error_m": values["eef_height_error_m"].abs(),
            "object_tilt_deg": values["object_tilt_deg"],
            "hand_up_error_deg": values["hand_up_error_deg"],
        }
        self._sweep_metric_steps += 1
        for name, value in metrics.items():
            if name not in self._sweep_metric_sums:
                self._sweep_metric_sums[name] = torch.zeros_like(self._sweep_metric_steps)
            self._sweep_metric_sums[name] += value.detach().float()
        maxima = {
            "gate_reached_rate": values["gate"],
            "goal_region_reached_rate": values["goal_region"],
            "max_progress_m": values["object_progress_m"],
            "max_object_tilt_deg": values["object_tilt_deg"],
        }
        for name, value in maxima.items():
            if name not in self._sweep_metric_maxima:
                self._sweep_metric_maxima[name] = torch.zeros_like(self._sweep_metric_steps)
            torch.maximum(self._sweep_metric_maxima[name], value.detach().float(), out=self._sweep_metric_maxima[name])
        self._sweep_final_progress.copy_(values["object_progress_m"].detach())

    def step(self, action):
        self.sensor_data_fresh[:] = True
        return super().step(action)
