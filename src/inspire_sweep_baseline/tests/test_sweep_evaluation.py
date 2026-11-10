"""Evaluation correctness: pre-reset samples, direction boundaries, weighted rates."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from extract_sweep_results import SECTORS, extract, goal_sector


@pytest.mark.parametrize('angle,sector', [(0, '11-1'), (29.99, '11-1'), (30, '1-3'),
    (90, '3-5'), (150, '5-7'), (210, '7-9'), (270, '9-11'), (330, '11-1'), (359, '11-1')])
def test_direction_boundaries(angle, sector):
    radians = math.radians(angle)
    initial = np.array([.01, -.02, 1.05])
    goal = initial + np.array([-.1 * math.cos(radians), .1 * math.sin(radians), 0])
    assert goal_sector(initial, goal)[0] == sector
    assert goal_sector(initial, initial)[0] == 'undefined'


def raw_fixture(root, partial=False):
    root.mkdir()
    (root / 'raw').mkdir()
    metadata = dict(schema_version=1, status='partial' if partial else 'complete', control_dt_s=.02,
        num_envs=2, episodes_per_env=1, raw_samples=5, task='TEST_FIXTURE', checkpoint='synthetic.pt',
        checkpoint_sha256='fixture', seed=42, contact_definition='fixture_2_of_3', goal_radius_m=.03)
    (root / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    # Env 0: three moving steps with one accepted; env 1: one moving step accepted.
    # Mean episode ratios would be 2/3; correct pooled ratio is 2/4.
    data = dict(env_id=np.array([0, 1, 0, 1, 0]), episode_id=np.zeros(5, dtype=int),
        episode_step=np.array([1, 1, 2, 2, 3]),
        initial_shelf=np.tile([0, 0, 1.05], (5, 1)),
        goal_shelf=np.array([[-.1, 0, 1.05], [0, .1, 1.05], [-.1, 0, 1.05], [0, .1, 1.05], [-.1, 0, 1.05]]),
        object_shelf=np.array([[0, 0, 1.05], [0, 0, 1.05], [0, 0, 1.05], [0, .08, 1.05], [-.07, 0, 1.05]]),
        velocity_w=np.array([[.001, 0, 0], [0, 0, .002], [.002, 0, 0], [0, 0, 0], [.003, 0, 0]]),
        contact_accepted=np.array([0, 1, 0, 0, 1], dtype=bool), contact_instant=np.array([0, 0, 0, 0, 1], dtype=bool),
        terminated=np.array([0, 0, 0, 1, 0], dtype=bool), time_out=np.array([0, 0, 0, 0, 1], dtype=bool),
        horizon_reached=np.array([0, 0, 0, 0, 1], dtype=bool), term__timeout=np.array([0, 0, 0, 0, 1], dtype=bool))
    if partial:
        data['time_out'][-1] = False
        data['horizon_reached'][-1] = False
    # Split one episode across chunks to check streaming state continuity.
    for number, selection in enumerate([slice(0, 3), slice(3, None)]):
        np.savez_compressed(root / 'raw' / f'steps_{number:06d}.npz', **{k: v[selection] for k, v in data.items()})
    return data


def test_extract_weighted_contact_and_timeout_population(tmp_path):
    run = tmp_path / 'run'
    raw_fixture(run)
    output = extract(run)
    report = json.loads((output / 'summary.json').read_text())
    total = report['groups'][0]
    assert (total['completed_episodes'], total['timeout_episodes'], total['early_ended_episodes']) == (2, 1, 1)
    assert total['moving_steps'] == 4  # inclusive >= .001; 3D includes vertical movement
    assert total['moving_contact_ratio'] == .5
    assert report['groups'][2]['moving_contact_ratio'] is None  # empty sector, never fake zero
    rows = list(csv.DictReader((output / 'timeout_episodes.csv').open()))
    assert len(rows) == 1 and float(rows[0]['error_x_m']) == pytest.approx(.03)
    assert rows[0]['sector'] == '11-1'
    xy = extract(run, run / 'xy_analysis', speed_dimension='xy')
    assert json.loads((xy / 'summary.json').read_text())['groups'][0]['moving_steps'] == 3
    higher = extract(run, run / 'higher_threshold', speed_threshold=.01)
    assert json.loads((higher / 'summary.json').read_text())['groups'][0]['moving_contact_ratio'] is None


def test_partial_episode_excluded_and_missing_chunk_rejected(tmp_path):
    run = tmp_path / 'partial'
    raw_fixture(run, partial=True)
    report = json.loads((extract(run) / 'summary.json').read_text())
    assert report['partial_episodes'] == 1 and report['partial_steps_excluded'] == 3
    assert report['groups'][0]['moving_contact_ratio'] == 1
    (run / 'raw' / 'steps_000000.npz').unlink()
    with pytest.raises(ValueError, match='Missing episode start'):
        extract(run)


def test_plots_empty_sectors_and_zero_movement(tmp_path):
    from plot_sweep_results import plot
    run = tmp_path / 'run'
    raw_fixture(run)
    analysis = extract(run, speed_threshold=1)
    output = plot(analysis, group_by='direction')
    assert {p.name for p in output.glob('*.png')} == {
        'timeout_scatter.png', 'generated_goals.png', 'goal_directions.png', 'moving_contact_ratio.png', 'error_statistics.png', 'error_boxplot.png'}
    metadata = json.loads((output / 'plot_metadata.json').read_text())
    assert metadata['displayed_panels']['timeout_scatter'] == ['all', '11-1']
    # Observed episodes with no moving steps remain visible as N/A.
    assert metadata['displayed_panels']['moving_contact_ratio'] == ['all', '11-1', '3-5']


def test_individual_plots_include_observed_directions_only(tmp_path):
    from plot_sweep_results import plot
    run = tmp_path / 'run'
    raw_fixture(run)
    output = plot(extract(run), scatter_limit_cm=1, individual=True, group_by='direction')
    metadata = json.loads((output / 'plot_metadata.json').read_text())
    assert len(metadata['individual_files']) == 15
    assert len(list((output / 'individual').rglob('*.png'))) == 15
    assert (output / 'individual/timeout_scatter/all.png').is_file()
    assert (output / 'individual/timeout_scatter/11-1.png').is_file()
    assert not (output / 'individual/timeout_scatter/3-5.png').exists()
    assert (output / 'individual/generated_goals/3-5.png').is_file()
    assert not any('5-7' in path for path in metadata['individual_files'])


def test_plot_zoom_retains_full_range_and_reports_outside_counts(tmp_path):
    from plot_sweep_results import plot
    run = tmp_path / 'run'
    raw_fixture(run)
    output = plot(extract(run), scatter_limit_cm=1)
    assert (output / 'timeout_scatter.png').is_file()
    assert (output / 'timeout_scatter_zoom.png').is_file()
    metadata = json.loads((output / 'plot_metadata.json').read_text())
    assert metadata['zoom_counts']['all'] == dict(total=1, visible=0, outside=1)
    with pytest.raises(ValueError, match='finite and positive'):
        plot(run / 'analysis', scatter_limit_cm=0)


@pytest.mark.parametrize('x,y,group', [(1, 1, 'Q1'), (-1, 1, 'Q2'),
    (-1, -1, 'Q3'), (1, -1, 'Q4'), (0, 1, 'Q1'), (-1, 0, 'Q2'),
    (0, -1, 'Q4'), (0, 0, 'undefined'), (float('nan'), 1, 'undefined')])
def test_gt_quadrant_signs_and_axes(x, y, group):
    from plot_sweep_results import goal_quadrant
    assert goal_quadrant(x, y) == group


def test_gt_groups_and_error_statistics_use_unclipped_episode_data(tmp_path):
    from plot_sweep_results import plot
    run = tmp_path / 'run'
    raw_fixture(run)
    output = plot(extract(run), scatter_limit_cm=1)
    metadata = json.loads((output / 'plot_metadata.json').read_text())
    assert metadata['group_by'] == 'goal-quadrant'
    assert metadata['displayed_panels']['generated_goals'] == ['all', 'Q1', 'Q2']
    assert metadata['displayed_panels']['timeout_scatter'] == ['all', 'Q2']
    assert metadata['displayed_panels']['moving_contact_ratio'] == ['all', 'Q1', 'Q2']
    groups = list(csv.DictReader((output / 'plot_episode_groups.csv').open()))
    assert [(r['sector'], r['plot_group']) for r in groups] == [('11-1', 'Q2'), ('3-5', 'Q1')]
    stats = list(csv.DictReader((output / 'error_statistics.csv').open()))
    assert [r['group'] for r in stats] == ['all', 'Q2']
    # The only timeout error is outside the 1 cm zoom, yet all statistics retain it.
    for row in stats:
        assert int(row['n']) == 1
        for name in ('median_cm', 'mean_cm', 'min_cm', 'max_cm'):
            assert float(row[name]) == pytest.approx(3)
    assert metadata['zoom_counts']['all']['outside'] == 1


def test_error_statistics_distinguish_median_mean_and_extremes():
    from plot_sweep_results import grouped_error_statistics
    rows = [dict(plot_group='Q1', error_xy_m=str(x / 100)) for x in (1, 2, 10, 20)]
    rows.append(dict(plot_group='Q2', error_xy_m='0.03'))
    result = grouped_error_statistics(rows, ('Q1', 'Q2'))
    assert result[1] == dict(group='Q1', n=4, median_cm=6, mean_cm=8.25, min_cm=1, max_cm=20)
    assert result[0]['n'] == 5 and result[2]['n'] == 1


def test_boxplot_quartiles_and_actual_min_max_whiskers():
    from plot_sweep_results import error_box_statistics
    rows = [dict(plot_group='Q1', error_xy_m=str(x / 100)) for x in (1, 2, 10, 20)]
    result = error_box_statistics(rows, ('Q1',))
    assert result[1] == dict(label='Q1', n=4, med=6, q1=1.75, q3=12.5, mean=8.25, whislo=1, whishi=20)


def test_large_error_filter_uses_one_upper_fence_and_retains_small_errors():
    from plot_sweep_results import filter_large_errors
    rows = [dict(error_xy_m=str(x / 100), plot_group='Q1' if x < 3 else 'Q2') for x in (0, 1, 2, 3, 100)]
    kept, excluded, policy = filter_large_errors(rows)
    assert [float(r['error_xy_m']) * 100 for r in kept] == [0, 1, 2, 3]
    assert len(excluded) == 1
    assert policy['upper_fence_cm'] == 6
    assert policy['retained_n'] + policy['excluded_n'] == policy['source_n'] == 5


def test_boxplot_break_only_when_retained_data_exceeds_main_axis(tmp_path):
    from plot_sweep_results import box_axis_ranges, plot
    assert box_axis_ranges([0, 6.2, 10]) == dict(main=[0., 10.], upper=None)
    result = box_axis_ranges([1, 5, 11, 20])
    assert result == dict(main=[0., 10.], upper=[10.5, 20.5])
    run = tmp_path / 'run'
    raw_fixture(run)
    output = plot(extract(run), error_axis_max_cm=1)
    metadata = json.loads((output / 'plot_metadata.json').read_text())
    assert metadata['error_axis_ranges']['main'] == [0., 1.]
    assert metadata['error_axis_ranges']['upper'] is not None
    assert (output / 'error_boxplot.png').is_file()


@pytest.mark.parametrize('asynchronous', [False, True])
def test_probe_copies_terminal_state_without_reward_reexecution(tmp_path, monkeypatch, tensor_math, asynchronous):
    # Use the real checked-in quaternion math. No simulator is launched in this CPU test.
    monkeypatch.setitem(sys.modules, 'isaaclab.utils.math', tensor_math)
    constants = types.ModuleType('inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion')
    constants.SHELF_SURFACE_HEIGHT = 1.05
    contacts = types.ModuleType('inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts')
    contacts.object_hand_sensor_name = lambda name: 'hand_contact_' + name
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setitem(sys.modules, contacts.__name__, contacts)
    path = SCRIPTS.parent / 'inspire_sweep_baseline/runtime/sweep_evaluation.py'
    spec = importlib.util.spec_from_file_location('inspire_sweep_baseline.runtime.sweep_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    term_class = type('GoalContactMaintenance', (), {'__module__': 'inspire_sweep_baseline.mdp.reward_functions.sweep_simple'})
    term = term_class()
    term._history = torch.zeros((2, 3), dtype=torch.bool)
    term._cached_step = None
    term._cached_reward = torch.full((2,), 20.)  # goal credit must not fabricate contact
    cfg = NS(func=term, params={'threshold_n': .01, 'eef_height_limit': .15})
    objects = NS(object_names=['target'], data=NS(object_pos_w=torch.tensor([[[3., 4., 1.05]], [[5., 6., 1.05]]]),
                                                object_lin_vel_w=torch.ones((2, 1, 3))))
    shelf = NS(data=NS(root_pos_w=torch.tensor([[3., 4., 0.], [5., 6., 0.]]),
                       root_quat_w=torch.tensor([[1., 0., 0., 0.], [math.sqrt(.5), 0., 0., math.sqrt(.5)]])))
    scene = {'object_collection': objects, 'shelf': shelf,
             'ee_frame': NS(data=NS(target_pos_w=torch.tensor([[[0., 0., 1.1]], [[0., 0., 1.1]]]))),
             'hand_contact_target': NS(data=NS(force_matrix_w=torch.zeros((2, 1, 2, 3))))}
    termination = NS(active_terms=['timeout'], terminated=torch.zeros(2, dtype=torch.bool),
                     time_outs=torch.zeros(2, dtype=torch.bool), dones=torch.zeros(2, dtype=torch.bool))
    termination.get_term = lambda name: termination.time_outs
    env = NS(num_envs=2, device='cpu', scene=scene, target_id=torch.zeros((2, 1), dtype=torch.long),
             episode_length_buf=torch.zeros(2, dtype=torch.long), common_step_counter=0,
             max_episode_length=2 if asynchronous else 1,
             sensor_data_fresh=torch.ones(2, dtype=torch.bool), _revision_initial_shelf_pos=shelf.data.root_pos_w,
             termination_manager=termination,
             command_manager=NS(get_command=lambda name: torch.tensor([[3., 4., 1.05], [5., 6., 1.05]])))
    calls = []

    def compute(dt):
        calls.append(dt)
        term._cached_step = env.common_step_counter
        return torch.tensor([7., 8.])

    env.reward_manager = NS(active_terms=['hand_contact'], get_term_cfg=lambda name: cfg, compute=compute)
    probe = module.SweepEvaluationProbe(env, tmp_path, episodes_per_env=1, chunk_steps=2)
    probe.prepare_step()
    objects.data.object_pos_w[:, 0, 0] += .02
    env.episode_length_buf[:] = 1
    env.common_step_counter = 1
    termination.time_outs[:] = torch.tensor([True, not asynchronous])
    termination.dones[:] = termination.time_outs
    result = env.reward_manager.compute(dt=.02)
    objects.data.object_pos_w[:] = 999  # mimic auto-reset after capture
    term._history[:] = True
    termination.time_outs[:] = False
    assert calls == [.02] and result.tolist() == [7, 8]
    if asynchronous:
        assert not probe.finished
        env.episode_length_buf[:] = torch.tensor([0, 1])
        probe.prepare_step()
        env.episode_length_buf[:] = torch.tensor([1, 2])
        env.common_step_counter = 2
        termination.time_outs[:] = termination.dones[:] = True
        env.reward_manager.compute(dt=.02)
        assert calls == [.02, .02]
    assert probe.finished
    probe.close()
    assert env.reward_manager.compute is compute
    with np.load(tmp_path / 'steps_000000.npz') as data:
        assert data['object_shelf'][:2, 0] == pytest.approx([.02, 0], abs=1e-6)
        assert data['object_shelf'][:2, 1] == pytest.approx([0, -.02], abs=1e-6)
        assert data['time_out'][0]
        assert not data['contact_accepted'][:2].any()
        assert (data['contact_reward_credit'] == 20).all()
        assert data['env_id'].tolist() == ([0, 1, 1] if asynchronous else [0, 1])
