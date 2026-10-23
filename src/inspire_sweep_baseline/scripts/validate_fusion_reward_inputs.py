#!/usr/bin/env python3
"""Validate per-env acquisition and export explicit episode intervals and reset counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def validate(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'metadata.json').read_text())
    assert metadata['status'] in ('complete', 'stopped_by_user'), 'Acquisition incomplete'
    count = metadata['csv_rows_per_file']
    if metadata['status'] == 'complete':
        assert count == metadata['arguments']['steps'], 'Acquisition incomplete'
    else:
        assert 150 <= count * metadata['control_dt_s'] <= 200, 'Outside revised user duration'
    n = metadata['arguments']['num_envs']
    dt = metadata['control_dt_s']
    object_ids = range(len(metadata['object_names']))
    expected_times = np.arange(1,count+1) * dt
    resets = pd.read_csv(directory/'resets.csv')
    results, episodes = [], []
    for env_id in range(n):
        path = directory/f'env_{env_id:02d}.csv'
        frame = pd.read_csv(path)
        assert len(frame) == count, f'Incomplete env {env_id}'
        assert list(frame.columns) == metadata['columns'], 'Schema mismatch'
        assert np.isfinite(frame.to_numpy(dtype=float)).all(), 'Nonfinite telemetry'
        assert np.allclose(frame.sim_time_s, expected_times, rtol=0, atol=1e-10)
        assert (frame.env_id == env_id).all()
        episode = frame.episode_id.to_numpy()
        step = frame.episode_step.to_numpy()
        done = (frame.terminated.to_numpy().astype(bool) | frame.time_out.to_numpy().astype(bool))
        valid = frame.delta_valid.to_numpy().astype(bool)
        assert step[0] == 1 and not valid[0]
        assert np.array_equal(np.diff(episode), done[:-1].astype(int)), 'Episode boundary mismatch'
        assert np.array_equal(step[1:], np.where(done[:-1],1,step[:-1]+1)), 'Episode step mismatch'
        assert np.array_equal(valid[1:], ~done[:-1]), 'Delta crosses reset'
        flags = frame[[f'done_{name}' for name in metadata['termination_names']]].to_numpy().astype(bool)
        assert np.array_equal(flags.any(axis=1),done), 'Termination cause mismatch'
        assert np.array_equal(frame.done_time_out.astype(bool),frame.time_out.astype(bool))
        assert np.array_equal(frame[[f'done_{name}' for name in metadata['termination_names'] if name!='time_out']].any(axis=1),frame.terminated.astype(bool))
        assert (frame.eef_linear_speed_m_s >= 0).all()
        for i in object_ids:
            active = frame[f'active_obj{i}'].to_numpy().astype(bool)
            accel_valid = frame[f'acceleration_valid_obj{i}'].to_numpy().astype(bool)
            expected = np.zeros(count,dtype=bool)
            expected[1:] = active[1:] & active[:-1] & valid[1:]
            assert np.array_equal(accel_valid,expected), 'Acceleration validity crosses reset'
            assert (frame[f'acceleration_m_s2_obj{i}'] >= 0).all()
            assert (frame[f'contact_force_N_obj{i}'] >= 0).all()
        obstacle_force_sum = sum(frame[f'contact_force_N_obj{i}'] * frame[f'active_obj{i}'] * (frame.target_id != i) for i in object_ids)
        assert (frame.obstacle_net_force_N <= obstacle_force_sum + 1e-4*(1+obstacle_force_sum)).all(), 'Force triangle bound'
        for quantity, axes, change_name in [('force',('Fx_N','Fy_N','Fz_N'),'force_magnitude_change_N'), ('moment',('Mx_Nm','My_Nm','Mz_Nm'),'moment_magnitude_change_Nm')]:
            magnitude = np.linalg.norm(frame[[f'penalty_eef_{a}' for a in axes]],axis=1)
            world = np.linalg.norm(frame[[f'penalty_world_{a}' for a in axes]],axis=1)
            assert np.allclose(magnitude,world,rtol=2e-5,atol=1e-4), 'Wrench frame norm mismatch'
            differences = np.zeros(count)
            differences[1:] = np.abs(np.diff(magnitude))
            differences[~valid] = 0
            assert np.allclose(frame[change_name], differences,rtol=1e-3,atol=2e-3), 'Wrench change mismatch'
        events = resets[resets.env_id == env_id]
        assert len(events) == int(done.sum()), 'Missing reset event'
        assert np.allclose(events.sim_time_s,frame.loc[done,'sim_time_s'],rtol=0,atol=1e-10)
        assert np.array_equal(events.ended_episode_id,episode[done])
        assert np.array_equal(events.next_episode_id,episode[done]+1)
        for event in events.itertuples():
            row = int(round(event.sim_time_s/dt))-1
            causes = '|'.join(name for name, flag in zip(metadata['termination_names'],flags[row]) if flag)
            assert event.causes == causes, 'Reset cause mismatch'
        for eid, group in frame.groupby('episode_id',sort=False):
            end = group.iloc[-1]
            causes = '|'.join(name for name in metadata['termination_names'] if end[f'done_{name}'])
            episodes.append(dict(env_id=env_id,episode_id=int(eid),start_sim_time_s=float(group.sim_time_s.iloc[0]-dt),end_sim_time_s=float(end.sim_time_s),samples=len(group),duration_s=len(group)*dt,completed=bool(end.terminated or end.time_out),causes=causes))
        results.append(dict(env_id=env_id,rows=count,simulation_seconds=count*dt,resets=int(done.sum()),episodes_observed=frame.episode_id.nunique(),termination_counts={name:int(frame[f'done_{name}'].sum()) for name in metadata['termination_names']},eef_speed_max=float(frame.eef_linear_speed_m_s.max()),obstacle_net_force_max_N=float(frame.obstacle_net_force_N.max())))
    pd.DataFrame(episodes).to_csv(directory/'episode_index.csv',index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k!='termination_counts'} for row in results]).to_csv(directory/'environment_summary.csv',index=False)
    result = dict(status='passed',num_envs=n,rows_per_env=count,total_samples=n*count,simulation_seconds_per_env=count*dt,total_resets=len(resets),total_episodes_observed=len(episodes),environments=results)
    (directory/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.directory),indent=2))
