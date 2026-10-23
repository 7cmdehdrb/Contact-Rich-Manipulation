"""Synthetic checks for physical quantities and pre-reset episode boundaries."""
import csv
from types import SimpleNamespace
import torch

from conftest import MODULE_ROOT
from test_fusion_telemetry import harness  # noqa: F401: reused CPU fixture
from test_omni_configuration import _execute_definitions


def test_reward_input_capture_and_per_environment_csv(harness, tmp_path):
    h = harness
    env = h.env
    env._active_object_mask = torch.tensor([[True, True, False, True], [True, True, True, True]])
    sample = h.sample
    sample.raw_eef = sample.pure_eef + 10
    env.scene['robot'] = SimpleNamespace(body_names=['axia80_link'], data=SimpleNamespace(
        body_link_pos_w=torch.zeros(2,1,3),
        body_link_lin_vel_w=torch.tensor([[[3.,0,0]], [[3.,0,0]]]),
        body_link_ang_vel_w=torch.tensor([[[0.,0,2.]], [[0.,0,2.]]]),
    ))
    env.scene['ee_frame'] = SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.tensor([[[2.,0,0]], [[2.,0,0]]])))
    env.scene['object_collection'].data.object_com_lin_acc_w = torch.tensor([[[0.,3.,4.]]*4]*2)
    for i, name in enumerate(env.scene['object_collection'].object_names):
        force = torch.zeros(2,1,2,3)
        force[:,:,0,0] = i + 1
        force[:,:,1,0] = -(i + 1)/2
        env.scene['hand_contact_' + name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=force))
    term = env.observation_manager.cfg.policy.wrist_wrench.func
    term._reader = SimpleNamespace(raw_reader=SimpleNamespace(read_raw_f=lambda mask: sample.raw_eef + 1))
    env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(
        params={}, func=SimpleNamespace(_previous_wrench=sample.pure_eef.clone())))
    env.termination_manager = SimpleNamespace(active_terms=['object_drop'], get_term=lambda name: torch.tensor([False,True]))
    module = _execute_definitions(MODULE_ROOT/'contact_sweep_clutter_fusion_v1_0/reward_telemetry.py', {
        '__name__': h.module.__name__, '__package__': h.module.__package__,
        'PlaybackTelemetry': h.module.PlaybackTelemetry, 'ContactSweepClutterFusionV1_0Env': object,
        'torch':torch, 'csv':csv, 'Path':type(tmp_path),
        'object_hand_sensor_name':lambda name:'hand_contact_'+name,
        'hand_obstacle_contact_force_w':lambda env:torch.tensor([[2.,0,0],[4.5,0,0]]),
    })
    telemetry = module.RewardPlaybackTelemetry(env)
    telemetry.reset(torch.arange(2))
    telemetry.begin_step()
    telemetry.capture(env,torch.tensor([1]))
    terminal = {k: v[1].clone() for k,v in telemetry.values.items()}
    telemetry.reset(torch.tensor([1]))
    env.sensor_data_fresh[1] = False
    sample.pure_eef[1] = 999
    telemetry.capture(env,torch.tensor([0]))
    telemetry.finish_step()
    for key,value in terminal.items():
        assert torch.equal(telemetry.values[key][1],value)
    assert telemetry.values['eef_speed'][:,0].tolist() == [5.,5.]
    assert telemetry.values['object_contact'][0].tolist() == [.5,1.,1.5,2.]
    assert telemetry.values['object_acceleration'][0].tolist() == [5.]*4
    assert not telemetry.values['acceleration_valid'].any()
    assert telemetry.values['wrench_change'].count_nonzero() == 0
    # Writer runs normal imports inside __init__, so use production import machinery
    # already installed by the package's test fixtures.
    assert telemetry.values['episodes.csv'][1,:5].tolist() == [0,1,1,0,0]
