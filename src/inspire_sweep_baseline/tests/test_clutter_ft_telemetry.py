"""CPU lifecycle and CSV tests; fixtures are synthetic, not physical samples."""

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PACKAGE_ROOT, PROJECT_ROOT
from test_omni_configuration import _execute_definitions
from test_clutter_v1_models import runner_configs, v1_models, _privileged_observations


@pytest.fixture
def telemetry_harness():
    count = 2
    pure = torch.tensor([[1., 2., 3., .1, .2, .3], [4., 5., 6., .4, .5, .6]])
    support = torch.ones_like(pure) * 10
    sample = SimpleNamespace(
        pure_world=pure, pure_eef=pure + 1, support_world=support,
        support_eef=support, raw_world=pure + support, raw_eef=pure + 1 + support,
    )

    class Reader:
        def __init__(self, *args, **kwargs):
            pass

        def read(self, *args, **kwargs):
            return sample

    module = _execute_definitions(
        MODULE_ROOT / 'contact_sweep_clutter_ft_test/telemetry.py',
        {'torch': torch, 'csv': csv, 'Path': Path, 'PureWrenchReader': Reader,
         'wrist_wrench_eef': lambda env: torch.full((2, 6), 123.)},
    )
    rewards = {'reaching': SimpleNamespace(weight=30.),
               'arm_action_rate': SimpleNamespace(weight=-.05),
               'arm_joint_velocity': SimpleNamespace(weight=-.05),
               'pushing_speed': SimpleNamespace(params={'free_speed':.06}),
               'arm_speed_excess': SimpleNamespace(params={'free_speed':.9})}
    frame = SimpleNamespace(target_pos_w=torch.zeros(2, 1, 3),
                            target_quat_w=torch.tensor([[[1.,0,0,0]],[[1.,0,0,0]]]))
    term_flags = {'time_out': torch.tensor([False, False]), 'collision': torch.tensor([False, True])}
    env = SimpleNamespace(
        num_envs=count, device='cpu', cfg=SimpleNamespace(sim=SimpleNamespace(gravity=(0.,0.,-9.81))),
        common_step_counter=1, _sim_step_counter=2, step_dt=.02, learning_iteration=12800,
        episode_length_buf=torch.tensor([4, 9]), sensor_data_fresh=torch.ones(2, dtype=torch.bool),
        reset_terminated=torch.tensor([False, True]), reset_time_outs=torch.zeros(2, dtype=torch.bool),
        target_id=torch.zeros(2, 1, dtype=torch.long),
        scene={'robot': object(), 'ee_frame':SimpleNamespace(data=frame),
               'object_collection':SimpleNamespace(data=SimpleNamespace(object_pos_w=torch.zeros(2,1,3)))},
        command_manager=SimpleNamespace(get_command=lambda name:torch.zeros(2,3)),
        termination_manager=SimpleNamespace(active_terms=list(term_flags), get_term=lambda name:term_flags[name]),
        observation_manager=SimpleNamespace(cfg=SimpleNamespace(policy=SimpleNamespace(
            tactile=SimpleNamespace(func=lambda env:torch.zeros(2,17),params={})))),
        action_manager=SimpleNamespace(action=torch.zeros(2,8)),
        reward_manager=SimpleNamespace(active_terms=list(rewards),
            get_term_cfg=lambda name:rewards[name],
            _step_reward=torch.tensor([[2.,-.1,-.2,-.3,-.4],[3.,-.5,-.6,-.7,-.8]])),
        reward_buf=torch.tensor([.02,.008]),
    )
    telemetry = module.WrenchTelemetry(env)
    telemetry.reset(torch.tensor([0,1]))
    return SimpleNamespace(env=env, term=telemetry, module=module, sample=sample)


def row(telemetry, index):
    return dict(zip(telemetry.columns, telemetry.values[index].tolist()))


def test_terminal_sample_survives_partial_reset_and_delta_does_not_cross_episodes(telemetry_harness):
    h = telemetry_harness; t=h.term; env=h.env
    t.begin_step()
    t.capture(env, torch.tensor([1]))  # terminal, before reset
    terminal = row(t,1)
    t.reset(torch.tensor([1]))
    env.sensor_data_fresh[1] = False
    env.episode_length_buf[1] = 0
    env.reset_terminated[1] = False
    h.sample.pure_world[1] = 999  # reset-side/stale value must never replace terminal
    t.capture(env, torch.tensor([0]))
    t.finish_step()
    assert row(t,1) == terminal
    assert terminal['v2_pure_world_fx_N']==4
    assert terminal['episode_step']==9 and terminal['terminated']==1
    assert terminal['episode_id']==0 and terminal['delta_valid']==0
    assert terminal['v1_raw_eef_fx_N']==123  # actual legacy path, NOT v2 raw

    env.common_step_counter+=1
    env.sensor_data_fresh[:]=True
    env.episode_length_buf[:]=torch.tensor([5,1])
    h.sample.pure_world[0,0]+=2
    t.begin_step(); t.capture(env,torch.tensor([0,1])); t.finish_step()
    assert row(t,0)['delta_valid']==1 and row(t,0)['delta_pure_world_fx_N']==2
    assert row(t,1)['delta_valid']==0 and row(t,1)['delta_force_norm_N']==0
    assert row(t,1)['episode_id']==1 and row(t,1)['episode_step']==1


def test_missing_duplicate_and_nonfinite_rows_are_rejected(telemetry_harness):
    h=telemetry_harness; t=h.term
    t.begin_step();t.capture(h.env,torch.tensor([0]))
    with pytest.raises(RuntimeError, match='Missing'):
        t.finish_step()
    with pytest.raises(RuntimeError, match='twice'):
        t.capture(h.env,torch.tensor([0]))
    h.sample.pure_eef[1,0]=float('nan')
    with pytest.raises(RuntimeError, match='Non-finite'):
        t.capture(h.env,torch.tensor([1]))


def test_csv_preserves_units_reward_weights_and_exclusive_creation(telemetry_harness,tmp_path):
    h=telemetry_harness; t=h.term
    t.begin_step();t.capture(h.env,torch.tensor([0,1]));t.finish_step()
    path=tmp_path/'wrench.csv'; writer=h.module.WrenchCSVWriter(path)
    writer.write(t);writer.close()
    rows=list(csv.DictReader(path.open()))
    assert writer.rows==2 and len(rows)==2
    assert float(rows[0]['reward_reaching'])==pytest.approx(.04)
    assert float(rows[0]['reward_positive'])==pytest.approx(.04)
    assert float(rows[0]['weight_arm_action_rate'])==pytest.approx(-.05)
    assert 'v2_pure_eef_mz_Nm' in rows[0]
    with pytest.raises(FileExistsError):h.module.WrenchCSVWriter(path)


def test_runtime_captures_before_parent_reset_without_extra_observations():
    events=[]
    class Parent:
        def __init__(self,cfg,render_mode=None,**kwargs):
            self.sensor_data_fresh=torch.ones(2,dtype=torch.bool)
        def _reset_idx(self,ids):
            events.append(('parent_reset',ids.tolist()))
            self.sensor_data_fresh[ids]=False
        def step(self,action):
            events.append(('physics',None))
            self._reset_idx(torch.tensor([1]))
            return 'unchanged'
    class Telemetry:
        def __init__(self,env):pass
        def begin_step(self):events.append(('begin',None))
        def capture(self,env,ids):events.append(('capture',ids.tolist()))
        def reset(self,ids):events.append(('reset_history',ids.tolist()))
        def finish_step(self):events.append(('finish',None))
    cls=_execute_definitions(MODULE_ROOT/'contact_sweep_clutter_ft_test/env.py',
        {'ContactSweepClutterV1CurriculumEnv':Parent,'WrenchTelemetry':Telemetry}).ContactSweepClutterFTTestEnv
    env=cls(None)
    assert env.step(None)=='unchanged'
    assert events==[('begin',None),('physics',None),('capture',[1]),('parent_reset',[1]),
                    ('reset_history',[1]),('capture',[0]),('finish',None)]
    events.clear();env._reset_idx(torch.tensor([0]))
    assert events==[('parent_reset',[0]),('reset_history',[0])]


def test_registration_inherits_curriculum_input_and_runner():
    regs={};gym=SimpleNamespace(registry=regs)
    gym.register=lambda **kw:regs.update({kw['id']:kw})
    _execute_definitions(MODULE_ROOT/'__init__.py',{'gym':gym})
    spec=regs['Isaac-Inspire-Contact-Sweep-Clutter-FT-Test-v0']
    assert spec['entry_point'].endswith('runtime.contact_sweep_clutter:ContactSweepClutterFTTestEnv')
    assert spec['kwargs']['rsl_rl_cfg_entry_point']==regs['Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1']['kwargs']['rsl_rl_cfg_entry_point']
    class Parent: pass
    cls=_execute_definitions(MODULE_ROOT/'contact_sweep_clutter_ft_test/env_cfg.py',
        {'configclass':lambda x:x,'ContactSweepClutterV1CurriculumEnvCfg':Parent}).ContactSweepClutterFTTestEnvCfg
    assert issubclass(cls,Parent)
    # No override of observations, reward configuration, physics, or curricula.
    assert not any(k in cls.__dict__ for k in ('observations','rewards','scene','curriculum','__post_init__'))


def test_requested_checkpoint_loads_strictly_and_runs_cpu_inference(v1_models,runner_configs,monkeypatch):
    """Real trained weights and RSL runner, synthetic inputs; no physics claim."""
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
    checkpoint=PROJECT_ROOT/'logs/backup/Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1/2026-10-20_22-49-30/model_12800.pt'
    assert checkpoint.is_file(), 'Requested training result is missing'
    obs=_privileged_observations(count=8)
    class Env:
        num_envs=8;num_actions=8;device='cpu';cfg={}
        def __init__(self):self.unwrapped=self
        def get_observations(self):return obs
        def set_learning_iteration(self,i):self.learning_iteration=i
    env=Env(); cfg=runner_configs[1].to_dict()
    runner=CurriculumOnPolicyRunner(env,cfg,device='cpu')
    runner.load(str(checkpoint),strict=True,map_location='cpu')
    assert runner.current_learning_iteration==12800 and env.learning_iteration==12800
    with torch.inference_mode():
        actions=runner.get_inference_policy(device='cpu')(obs)
    assert actions.shape==(8,8) and torch.isfinite(actions).all()


def test_checkpoint_selection_uses_numeric_iteration(tmp_path):
    spec=importlib.util.spec_from_file_location('_ft_collect_cli',PACKAGE_ROOT/'scripts/collect_clutter_ft.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for n in (950,1000,10000):(tmp_path/f'model_{n}.pt').touch()
    (tmp_path/'model_bad.pt').touch()
    assert module.checkpoint_path(tmp_path).name=='model_10000.pt'
