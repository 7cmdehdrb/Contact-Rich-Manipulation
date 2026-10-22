"""Reset-safe playback measurements, separate from observations and rewards."""

from __future__ import annotations

import csv
from pathlib import Path

import torch


WRENCH_AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
VELOCITY_AXES = ("vx", "vy", "vz")
EPISODE_FIELDS = ("episode_id", "episode_step", "terminated", "time_out", "delta_valid", "target_id")


def csv_columns(num_envs, object_labels):
    """Wide format: one synchronized row per control step, env indices start at 0."""
    wrench = ["time"] + [f"{axis}_{i}" for i in range(num_envs) for axis in WRENCH_AXES]
    return {
        "wrench_world.csv": wrench,
        "wrench_eef.csv": wrench.copy(),
        "object_velocity.csv": ["time"] + [
            f"{axis}_{label}_{i}" for i in range(num_envs)
            for label in object_labels for axis in VELOCITY_AXES
        ],
        "episodes.csv": ["time"] + [
            f"{field}_{i}" for i in range(num_envs) for field in EPISODE_FIELDS
        ],
    }


class PlaybackTelemetry:
    """Capture actual final-substep values, including pre-reset terminal states."""

    def __init__(self, env):
        self.num_envs = env.num_envs
        self.object_names = tuple(env.scene["object_collection"].object_names)
        self.target_index = self.object_names.index("mass_cylinder")
        self.object_labels = tuple("Target" if name == "mass_cylinder" else name for name in self.object_names)
        self.episode_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.captured = torch.zeros_like(self.previous_valid)
        self.values = {
            "wrench_world.csv": torch.empty((env.num_envs, 6), device=env.device),
            "wrench_eef.csv": torch.empty((env.num_envs, 6), device=env.device),
            "object_velocity.csv": torch.empty((env.num_envs, len(self.object_names) * 3), device=env.device),
            "episodes.csv": torch.empty((env.num_envs, len(EPISODE_FIELDS)), dtype=torch.float64, device=env.device),
        }

    def reset(self, env_ids):
        self.episode_ids[env_ids] += 1
        self.previous_valid[env_ids] = False

    def begin_step(self):
        self.captured.zero_()

    def capture(self, env, env_ids):
        if len(env_ids) == 0:
            return
        if bool(self.captured[env_ids].any()):
            raise RuntimeError("Playback transition captured twice")
        if not bool(env.sensor_data_fresh[env_ids].all()):
            raise RuntimeError("Stale playback sensor sample")
        if bool((self.episode_ids[env_ids] < 0).any()):
            raise RuntimeError("Reset the environment before collecting playback")
        if bool((env.target_id[env_ids, 0] != self.target_index).any()):
            raise RuntimeError("Fusion target mapping changed during collection")

        # Reuse the physical sample cached by PureWristWrench, exactly as the
        # existing policy and world-axis wrench penalty do. No noisy observation
        # or stateful reward is evaluated a second time.
        sample = env.observation_manager.cfg.policy.wrist_wrench.func._read(env)
        data = {
            "wrench_world.csv": sample.pure_world,
            "wrench_eef.csv": sample.pure_eef,
            "object_velocity.csv": env.scene["object_collection"].data.object_lin_vel_w.flatten(1),
            "episodes.csv": torch.stack((
                self.episode_ids, env.episode_length_buf,
                env.reset_terminated, env.reset_time_outs, self.previous_valid,
                env.target_id[:, 0],
            ), dim=-1).to(torch.float64),
        }
        for name, value in data.items():
            if not bool(torch.isfinite(value[env_ids]).all()):
                raise RuntimeError(f"Non-finite playback telemetry: {name}")
            self.values[name][env_ids] = value[env_ids]
        self.previous_valid[env_ids] = True
        self.captured[env_ids] = True

    def finish_step(self):
        if not bool(self.captured.all()):
            raise RuntimeError("Missing playback telemetry samples")


class PlaybackCSVWriter:
    """Write all environments on each row; never overwrite existing files."""

    def __init__(self, directory, telemetry):
        self.columns = csv_columns(telemetry.num_envs, telemetry.object_labels)
        self.files = {}
        self.writers = {}
        self.rows = 0
        try:
            for name, columns in self.columns.items():
                stream = (Path(directory) / name).open("x", newline="")
                self.files[name] = stream
                self.writers[name] = csv.writer(stream)
                self.writers[name].writerow(columns)
        except BaseException:
            self.close()
            raise

    def write(self, telemetry, time_s):
        telemetry.finish_step()
        for name, value in telemetry.values.items():
            self.writers[name].writerow([time_s, *value.detach().cpu().flatten().tolist()])
            self.files[name].flush()
        self.rows += 1

    def close(self):
        for stream in self.files.values():
            stream.close()
