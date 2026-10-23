"""Physical F/T samples and reward context, independent of policy inputs.

One row per environment and control transition, including terminal transitions.
v1 raw uses the exact legacy policy transform. v2 raw/support/pure use the v2
measurement-joint frame and EEF moment origin. Do not subtract v2 support from
v1 raw: the original implementation uses a different raw-frame transform."""

from __future__ import annotations

from ..mdp.common.compensated_wrench import PureWrenchReader
from ..mdp.observation_functions.sweep import wrist_wrench_eef
from pathlib import Path
import csv as csv
import torch as torch

AXES = ("fx_N", "fy_N", "fz_N", "mx_Nm", "my_Nm", "mz_Nm")


class WrenchTelemetry:
    def __init__(self, env):
        self._reader = None
        self.episode_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.previous = torch.zeros((env.num_envs, 6), device=env.device)
        self.previous_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.captured = torch.zeros_like(self.previous_valid)
        self.columns = None
        self.values = None

    def reset(self, env_ids):
        self.episode_ids[env_ids] += 1
        self.previous_valid[env_ids] = False
        self.previous[env_ids] = 0.0

    def begin_step(self):
        self.captured.zero_()

    def capture(self, env, env_ids):
        if len(env_ids) == 0:
            return
        if bool(self.captured[env_ids].any()):
            raise RuntimeError("F/T transition captured twice")
        if self._reader is None:
            self._reader = PureWrenchReader(env.scene["robot"], gravity_w=env.cfg.sim.gravity)
        frame = env.scene["ee_frame"].data
        sample = self._reader.read(
            frame.target_pos_w[:, 0], frame.target_quat_w[:, 0], env.sensor_data_fresh,
        )
        legacy_raw = wrist_wrench_eef(env)
        valid = env.sensor_data_fresh
        delta_valid = valid & self.previous_valid
        delta = torch.where(delta_valid[:, None], sample.pure_world - self.previous, 0.0)
        data = {}

        def scalar(name, value):
            if not isinstance(value, torch.Tensor):
                value = torch.full((env.num_envs,), value, device=env.device)
            data[name] = value

        def vector(prefix, value, labels):
            for i, label in enumerate(labels):
                scalar(f"{prefix}_{label}", value[:, i])

        scalar("control_step", env.common_step_counter)
        scalar("physics_step", env._sim_step_counter)
        scalar("sim_time_s", env.common_step_counter * env.step_dt)
        scalar("env_id", torch.arange(env.num_envs, device=env.device))
        scalar("episode_id", self.episode_ids)
        scalar("episode_step", env.episode_length_buf)
        scalar("episode_time_s", env.episode_length_buf * env.step_dt)
        scalar("sample_valid", valid)
        scalar("delta_valid", delta_valid)
        scalar("terminated", env.reset_terminated)
        scalar("time_out", env.reset_time_outs)
        for name in env.termination_manager.active_terms:
            scalar(f"done_{name}", env.termination_manager.get_term(name))
        scalar("target_id", env.target_id[:, 0])
        rows = torch.arange(env.num_envs, device=env.device)
        target = env.scene["object_collection"].data.object_pos_w[rows, env.target_id[:, 0].long()]
        goal = env.command_manager.get_command("target_goal_pos")
        error = torch.linalg.vector_norm(target[:, :2] - goal[:, :2], dim=-1)
        scalar("goal_error_xy_m", error)
        scalar("inside_goal_1cm", error <= 0.01)
        # Shared physical tactile reader, without observation noise/RNG calls.
        term = env.observation_manager.cfg.policy.tactile
        bits = term.func(env, **term.params)
        scalar("tactile_count", (bits > 0.5).sum(-1))
        vector("tactile", bits, tuple(str(i) for i in range(17)))
        vector("eef_pos_world", frame.target_pos_w[:, 0], ("x_m", "y_m", "z_m"))
        vector("eef_quat_world", frame.target_quat_w[:, 0], ("w", "x", "y", "z"))
        vector("v1_raw_eef", legacy_raw, AXES)
        for name in ("raw_eef", "support_eef", "pure_eef", "raw_world", "support_world", "pure_world"):
            vector(f"v2_{name}", getattr(sample, name), AXES)
        vector("delta_pure_world", delta, AXES)
        scalar("delta_force_norm_N", torch.linalg.vector_norm(delta[:, :3], dim=-1))
        scalar("delta_moment_norm_Nm", torch.linalg.vector_norm(delta[:, 3:], dim=-1))
        vector("action", env.action_manager.action, tuple(str(i) for i in range(8)))
        # This buffer is already weighted; multiply dt once to recover actual
        # transition rewards. Never re-evaluate stateful progress reward terms.
        rewards = env.reward_manager._step_reward * env.step_dt
        for i, name in enumerate(env.reward_manager.active_terms):
            scalar(f"reward_{name}", rewards[:, i])
        scalar("reward_positive", rewards.clamp_min(0.0).sum(-1))
        scalar("reward_net", env.reward_buf)
        scalar("learning_iteration", env.learning_iteration)
        for name in ("arm_action_rate", "arm_joint_velocity"):
            scalar(f"weight_{name}", env.reward_manager.get_term_cfg(name).weight)
        for name in ("pushing_speed", "arm_speed_excess"):
            scalar(f"free_speed_{name}", env.reward_manager.get_term_cfg(name).params["free_speed"])

        columns = tuple(data)
        # Keep counters precise during long recordings; convert one packed
        # tensor to CPU per completed control step in the writer.
        values = torch.stack([data[k].to(torch.float64) for k in columns], dim=-1)
        if not torch.isfinite(values[env_ids]).all():
            raise RuntimeError("Non-finite F/T telemetry; refusing to write a corrupt sample")
        if self.columns is None:
            self.columns = columns
            self.values = torch.empty_like(values)
        if columns != self.columns:
            raise RuntimeError("Telemetry schema changed during recording")
        self.values[env_ids] = values[env_ids]
        self.previous[env_ids] = sample.pure_world[env_ids]
        self.previous_valid[env_ids] = valid[env_ids]
        self.captured[env_ids] = True

    def finish_step(self):
        if not bool(self.captured.all()):
            raise RuntimeError("Missing F/T telemetry rows after control step")


class WrenchCSVWriter:
    """Streaming CSV; exclusive creation prevents accidental result overwrite."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = self.path.open("x", newline="")
        self._writer = csv.writer(self._file)
        self._columns = None
        self.rows = 0

    def write(self, telemetry):
        telemetry.finish_step()
        if self._columns is None:
            self._columns = telemetry.columns
            self._writer.writerow(self._columns)
        if self._columns != telemetry.columns:
            raise RuntimeError("CSV columns changed")
        values = telemetry.values.detach().cpu().tolist()
        self._writer.writerows(values)
        self.rows += len(values)
        self._file.flush()

    def close(self):
        self._file.close()
