"""Playback-only measurements at the same pre-reset state as the reward inputs."""
from __future__ import annotations

import csv
from pathlib import Path
import torch

from ..contact_sweep_clutter_fusion.telemetry import PlaybackTelemetry
from .contacts import object_hand_sensor_name, hand_obstacle_contact_force_w
from .env import ContactSweepClutterFusionV1_0Env


class RewardPlaybackTelemetry(PlaybackTelemetry):
    def __init__(self, env):
        super().__init__(env)
        n, k = env.num_envs, len(self.object_names)
        self.termination_names = tuple(env.termination_manager.active_terms)
        self._previous_wrench = torch.zeros((n, 6), device=env.device)
        self._previous_active = torch.zeros((n, k), dtype=torch.bool, device=env.device)
        self.extra_columns = {
            'eef_speed': ['eef_linear_speed_m_s'],
            'object_contact': [f'contact_force_N_obj{i}' for i in range(k)],
            'obstacle_net': ['obstacle_net_force_N'],
            'object_acceleration': [f'acceleration_m_s2_obj{i}' for i in range(k)],
            'active': [f'active_obj{i}' for i in range(k)],
            'acceleration_valid': [f'acceleration_valid_obj{i}' for i in range(k)],
            'raw_joint': [f'raw_joint_{axis}' for axis in ('Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm')],
            'raw_eef': [f'raw_eef_{axis}' for axis in ('Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm')],
            'wrench_change': ['force_magnitude_change_N', 'moment_magnitude_change_Nm'],
            'termination_terms': [f'done_{name}' for name in self.termination_names],
        }
        for name, columns in self.extra_columns.items():
            self.values[name] = torch.empty((n, len(columns)), device=env.device)

    def reset(self, env_ids):
        super().reset(env_ids)
        self._previous_wrench[env_ids] = 0
        self._previous_active[env_ids] = False

    def capture(self, env, env_ids):
        if len(env_ids) == 0:
            return
        # Match EEF cost's origin-to-EEF transport exactly; no COM substitution.
        params = env.reward_manager.get_term_cfg('eef_linear_speed').params
        robot = env.scene[params.get('asset_name', 'robot')]
        body = robot.body_names.index(params.get('body_name', 'axia80_link'))
        offset = env.scene[params.get('frame_name', 'ee_frame')].data.target_pos_w[:, 0] - robot.data.body_link_pos_w[:, body]
        velocity = robot.data.body_link_lin_vel_w[:, body] + torch.cross(robot.data.body_link_ang_vel_w[:, body], offset, dim=-1)
        objects = env.scene['object_collection']
        # Keep each object's resultant separately, including target, before masks.
        forces = torch.stack([
            -env.scene[object_hand_sensor_name(name)].data.force_matrix_w.sum(dim=(1, 2))
            for name in self.object_names
        ], dim=1)
        term = env.observation_manager.cfg.policy.wrist_wrench.func
        sample = term._read(env)
        active = env._active_object_mask
        valid = self.previous_valid
        change = torch.stack((
            (sample.pure_eef[:, :3].norm(dim=-1) - self._previous_wrench[:, :3].norm(dim=-1)).abs(),
            (sample.pure_eef[:, 3:].norm(dim=-1) - self._previous_wrench[:, 3:].norm(dim=-1)).abs(),
        ), dim=-1)
        data = {
            'eef_speed': velocity.norm(dim=-1, keepdim=True),
            'object_contact': forces.norm(dim=-1),
            'obstacle_net': hand_obstacle_contact_force_w(env).norm(dim=-1, keepdim=True),
            'object_acceleration': objects.data.object_com_lin_acc_w.norm(dim=-1),
            'active': active,
            'acceleration_valid': active & self._previous_active,
            'raw_joint': term._reader.raw_reader.read_raw_f(env.sensor_data_fresh),
            'raw_eef': sample.raw_eef,
            'wrench_change': torch.where(valid[:, None], change, 0.),
            'termination_terms': torch.stack([env.termination_manager.get_term(name) for name in self.termination_names], dim=-1),
        }
        # Check exact wrench used by the stateful penalty without invoking it again.
        actual = env.reward_manager.get_term_cfg('wrench_change').func._previous_wrench
        if not torch.equal(actual[env_ids], sample.pure_eef[env_ids]):
            raise RuntimeError('Telemetry wrench differs from reward input')
        for name, value in data.items():
            if not bool(torch.isfinite(value[env_ids]).all()):
                raise RuntimeError(f'Nonfinite reward input: {name}')
            self.values[name][env_ids] = value[env_ids].to(self.values[name].dtype)
        self._previous_wrench[env_ids] = sample.pure_eef[env_ids]
        self._previous_active[env_ids] = active[env_ids]
        super().capture(env, env_ids)


class RewardTelemetryEnv(ContactSweepClutterFusionV1_0Env):
    def __init__(self, cfg, render_mode=None, **kwargs):
        self.playback_telemetry = None
        self._telemetry_in_step = False
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.playback_telemetry = RewardPlaybackTelemetry(self)

    def _reset_idx(self, env_ids):
        telemetry = self.playback_telemetry
        if telemetry is not None and self._telemetry_in_step:
            telemetry.capture(self, env_ids)
        super()._reset_idx(env_ids)
        if telemetry is not None:
            telemetry.reset(env_ids)

    def step(self, action):
        self.playback_telemetry.begin_step()
        self._telemetry_in_step = True
        try:
            result = super().step(action)
        finally:
            self._telemetry_in_step = False
        self.playback_telemetry.capture(self, self.sensor_data_fresh.nonzero().flatten())
        self.playback_telemetry.finish_step()
        return result


class RewardCSVWriter:
    """One file per environment, plus explicit terminal/reset events."""
    def __init__(self, directory, telemetry):
        from ..contact_sweep_clutter_fusion.telemetry import EPISODE_FIELDS
        self.streams = []
        self.writers = []
        self.rows = 0
        self.names = ['episodes.csv', 'wrench_eef.csv', 'wrench_world.csv', 'object_velocity.csv', *telemetry.extra_columns]
        self.columns = ['sim_time_s', 'env_id', *EPISODE_FIELDS]
        self.columns += [f'penalty_eef_{a}' for a in ('Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm')]
        self.columns += [f'penalty_world_{a}' for a in ('Fx_N','Fy_N','Fz_N','Mx_Nm','My_Nm','Mz_Nm')]
        self.columns += [f'{a}_m_s_obj{i}' for i in range(len(telemetry.object_names)) for a in ('vx','vy','vz')]
        self.columns += [c for columns in telemetry.extra_columns.values() for c in columns]
        try:
            for i in range(telemetry.num_envs):
                stream = (Path(directory) / f'env_{i:02d}.csv').open('x', newline='')
                self.streams.append(stream)
                writer = csv.writer(stream)
                writer.writerow(self.columns)
                self.writers.append(writer)
            self.events = (Path(directory) / 'resets.csv').open('x', newline='')
            self.streams.append(self.events)
            self.event_writer = csv.writer(self.events)
            self.event_writer.writerow(['sim_time_s','env_id','ended_episode_id','terminal_episode_step','next_episode_id','terminated','time_out','causes'])
        except BaseException:
            self.close()
            raise

    def write(self, telemetry, time_s):
        telemetry.finish_step()
        values = torch.cat([telemetry.values[name].to(torch.float64) for name in self.names], dim=-1).cpu().tolist()
        for i, value in enumerate(values):
            self.writers[i].writerow([time_s, i, *value])
            episode, episode_step, terminated, timeout = value[:4]
            if terminated or timeout:
                flags = telemetry.values['termination_terms'][i].tolist()
                causes = '|'.join(name for name, flag in zip(telemetry.termination_names, flags) if flag)
                self.event_writer.writerow([time_s, i, int(episode), int(episode_step), int(episode)+1, int(terminated), int(timeout), causes])
        self.rows += 1
        if self.rows % 100 == 0:
            for stream in self.streams:
                stream.flush()

    def close(self):
        for stream in self.streams:
            stream.close()
