"""Reduce raw evaluation chunks to episode CSVs and direction summaries (no Sim)."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SECTORS = ('11-1', '1-3', '3-5', '5-7', '7-9', '9-11')


def goal_sector(initial, goal):
    """Clockwise, 12=-shelf X, 3=+shelf Y; lower boundary included."""
    dx, dy = np.asarray(goal)[:2] - np.asarray(initial)[:2]
    if not np.isfinite([dx, dy]).all() or math.hypot(dx, dy) <= 1e-12:
        return 'undefined', None
    angle = math.degrees(math.atan2(dy, -dx)) % 360
    # Remove roundoff at exact trigonometric sector boundaries (e.g. 30 deg).
    return SECTORS[int((((angle + 30) % 360) + 1e-10) // 60) % 6], angle


def write_csv(path, rows):
    # Keep headers even for an empty timeout population.
    fields = list(rows[0]) if rows else EPISODE_FIELDS
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


EPISODE_FIELDS = [
    'env_id', 'episode_id', 'sector', 'goal_clock_angle_deg', 'completed', 'steps', 'duration_s',
    'time_out', 'horizon_reached', 'terminated', 'clean_timeout', 'termination_terms',
    'initial_x_m', 'initial_y_m', 'initial_z_m', 'goal_x_m', 'goal_y_m', 'goal_z_m',
    'final_x_m', 'final_y_m', 'final_z_m', 'error_x_m', 'error_y_m', 'error_z_m',
    'error_xy_m', 'within_goal_radius', 'valid_speed_steps', 'invalid_speed_steps',
    'moving_steps', 'moving_contact_steps', 'moving_instant_contact_steps',
    'moving_duration_s', 'moving_contact_duration_s', 'moving_contact_ratio',
]


def extract(input_dir, output_dir=None, speed_threshold=.001, speed_dimension='3d'):
    input_dir = Path(input_dir).resolve()
    metadata = json.loads((input_dir / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['schema_version'] != 1:
        raise ValueError('Unsupported evaluation schema')
    dt = metadata['control_dt_s']
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(speed_threshold) or speed_threshold < 0:
        raise ValueError('dt must be positive; speed threshold must be finite and nonnegative')
    if speed_dimension not in ('3d', 'xy'):
        raise ValueError('speed_dimension must be 3d or xy')
    output = Path(output_dir).resolve() if output_dir else input_dir / 'analysis'
    raw_dir = input_dir / 'raw'
    if output == raw_dir or raw_dir in output.parents or output == input_dir:
        raise ValueError('Analysis output must be separate from the run root and raw data')
    chunks = sorted(raw_dir.glob('steps_*.npz'))
    if not chunks:
        raise ValueError('No raw chunks found')
    episodes = {}
    previous_key = {}
    sample_count = 0
    for chunk in chunks:
        with np.load(chunk, allow_pickle=False) as archive:
            # NpzFile decompresses on each __getitem__; materialize once per chunk.
            data = {key: archive[key] for key in archive.files}
            term_keys = sorted(k for k in data if k.startswith('term__'))
            velocity = data['velocity_w'][:, :2] if speed_dimension == 'xy' else data['velocity_w']
            speeds = np.linalg.norm(velocity, axis=-1)
            count = len(data['env_id'])
            if any(len(data[k]) != count for k in data):
                raise ValueError(f'Column length mismatch in {chunk}')
            for i in range(count):
                key = int(data['env_id'][i]), int(data['episode_id'][i])
                if not 0 <= key[0] < metadata['num_envs']:
                    raise ValueError(f'Invalid env ID: {key}')
                step = int(data['episode_step'][i])
                if key not in episodes:
                    if step != 1 or key[1] != previous_key.get(key[0], -1) + 1:
                        raise ValueError(f'Missing episode start/chunk: {key}, step {step}')
                    previous = episodes.get((key[0], key[1] - 1))
                    if previous is not None and not previous['complete']:
                        raise ValueError(f'New episode before preceding termination: {key}')
                    previous_key[key[0]] = key[1]
                    episodes[key] = dict(initial=data['initial_shelf'][i].copy(), goal=data['goal_shelf'][i].copy(),
                                         steps=0, complete=False, valid=0, invalid=0, moving=0, contact=0, instant=0)
                episode = episodes[key]
                if episode['complete'] or step != episode['steps'] + 1:
                    raise ValueError(f'Duplicate/missing sample or sample after termination: {key}, step {step}')
                if not np.allclose(episode['goal'], data['goal_shelf'][i], atol=1e-6, rtol=0, equal_nan=True):
                    raise ValueError(f'Goal changed within episode {key}; this extractor expects episode-fixed goals')
                episode['steps'] = step
                valid = bool(np.isfinite(speeds[i]))
                moving = valid and bool(speeds[i] >= speed_threshold)
                episode['valid'] += valid
                episode['invalid'] += not valid
                episode['moving'] += moving
                episode['contact'] += moving and bool(data['contact_accepted'][i])
                episode['instant'] += moving and bool(data['contact_instant'][i])
                episode['final'] = data['object_shelf'][i].copy()
                episode['timeout'] = bool(data['time_out'][i])
                episode['terminated'] = bool(data['terminated'][i])
                episode['horizon'] = bool(data['horizon_reached'][i])
                episode['complete'] = episode['timeout'] or episode['terminated']
                episode['terms'] = '|'.join(k[6:] for k in term_keys if data[k][i])
            sample_count += count

    rows = []
    radius = metadata.get('goal_radius_m')
    for (env_id, episode_id), e in sorted(episodes.items()):
        sector, angle = goal_sector(e['initial'], e['goal'])
        error = e['final'] - e['goal']
        distance = float(np.linalg.norm(error[:2]))
        row = dict(env_id=env_id, episode_id=episode_id, sector=sector,
                   goal_clock_angle_deg=angle, completed=int(e['complete']), steps=e['steps'],
                   duration_s=e['steps'] * dt, time_out=int(e['timeout']), horizon_reached=int(e['horizon']),
                   terminated=int(e['terminated']), clean_timeout=int(e['timeout'] and e['horizon'] and not e['terminated']),
                   termination_terms=e['terms'])
        for name, vector in [('initial', e['initial']), ('goal', e['goal']), ('final', e['final']), ('error', error)]:
            row.update({f'{name}_{axis}_m': float(value) if np.isfinite(value) else None
                        for axis, value in zip('xyz', vector)})
        row.update(error_xy_m=distance if math.isfinite(distance) else None,
                   within_goal_radius=int(distance <= radius) if radius is not None and math.isfinite(distance) else None,
                   valid_speed_steps=e['valid'], invalid_speed_steps=e['invalid'], moving_steps=e['moving'],
                   moving_contact_steps=e['contact'], moving_instant_contact_steps=e['instant'],
                   moving_duration_s=e['moving'] * dt, moving_contact_duration_s=e['contact'] * dt,
                   moving_contact_ratio=e['contact'] / e['moving'] if e['moving'] else None)
        rows.append(row)
    complete = [r for r in rows if r['completed']]
    timeouts = [r for r in complete if r['time_out'] and r['horizon_reached']]
    if metadata['status'] == 'complete':
        quota = metadata['episodes_per_env']
        if len(complete) != metadata['num_envs'] * quota or len(complete) != len(rows):
            raise ValueError('Complete run does not match episode quotas')
        if sample_count != metadata['raw_samples']:
            raise ValueError('Complete run sample count differs from manifest')
    groups = []
    for sector in ('all', *SECTORS, 'undefined'):
        selected = complete if sector == 'all' else [r for r in complete if r['sector'] == sector]
        timed = [r for r in selected if r['time_out'] and r['horizon_reached']]
        errors = [r['error_xy_m'] for r in timed if r['error_xy_m'] is not None]
        moving = sum(r['moving_steps'] for r in selected)
        contact = sum(r['moving_contact_steps'] for r in selected)
        groups.append(dict(sector=sector, completed_episodes=len(selected), timeout_episodes=len(timed),
                           clean_timeout_episodes=sum(r['clean_timeout'] for r in timed),
                           early_ended_episodes=len(selected) - len(timed), valid_timeout_positions=len(errors),
                           moving_steps=moving, moving_contact_steps=contact,
                           moving_instant_contact_steps=sum(r['moving_instant_contact_steps'] for r in selected),
                           moving_duration_s=moving * dt, moving_contact_duration_s=contact * dt,
                           moving_contact_ratio=contact / moving if moving else None,
                           invalid_speed_steps=sum(r['invalid_speed_steps'] for r in selected),
                           median_error_xy_m=float(np.median(errors)) if errors else None,
                           p90_error_xy_m=float(np.percentile(errors, 90)) if errors else None,
                           rmse_xy_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                           within_goal_radius_fraction=(sum(r['within_goal_radius'] or 0 for r in timed
                                                            if r['error_xy_m'] is not None) / len(errors)
                                                        if errors and radius is not None else None)))
    summary = dict(schema_version=1, source_run=str(input_dir), source_status=metadata['status'],
                   task=metadata['task'], checkpoint=metadata['checkpoint'],
                   checkpoint_sha256=metadata['checkpoint_sha256'], seed=metadata['seed'],
                   contact_definition=metadata['contact_definition'], goal_radius_m=radius,
                   speed_threshold_m_s=speed_threshold, speed_dimension=speed_dimension,
                   clock='12=-shelf X, 3=+shelf Y; direction=goal-initial; half-open 60 degree sectors',
                   timeout_cohort='horizon_reached AND time_out, including simultaneous failures',
                   contact_cohort='all completed episodes, including early failures; excludes partial episodes',
                   contact_aggregation='sum moving_contact_steps / sum moving_steps, never mean of episode ratios',
                   raw_samples=sample_count, raw_chunks=len(chunks), completed_episodes=len(complete),
                   partial_episodes=len(rows) - len(complete),
                   partial_steps_excluded=sum(r['steps'] for r in rows if not r['completed']), groups=groups)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'episodes.csv', rows)
    write_csv(output / 'timeout_episodes.csv', timeouts)
    write_csv(output / 'direction_summary.csv', groups)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True, help='Run directory containing metadata.json and raw/')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--speed-threshold', type=float, default=.001)
    parser.add_argument('--speed-dimension', choices=('3d', 'xy'), default='3d')
    args = parser.parse_args()
    print(extract(args.input_dir, args.output_dir, args.speed_threshold, args.speed_dimension))


if __name__ == '__main__':
    main()
