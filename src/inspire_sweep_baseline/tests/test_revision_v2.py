"""Revision-v2 boundaries, one-shot events, and inherited reward contract."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from test_revision_v1 import functions, configs, omni_config, osc


def height_env(heights, origins=None):
    h = torch.tensor(heights, dtype=torch.float64)
    n = len(h)
    origins = torch.arange(n, dtype=h.dtype) if origins is None else torch.tensor(origins, dtype=h.dtype)
    shelf = torch.zeros(n, 3, dtype=h.dtype)
    shelf[:, 2] = origins
    pos = torch.zeros(n, 1, 3, dtype=h.dtype)
    pos[:, 0, 2] = origins + 1.05 + h
    return NS(cfg=NS(goal_reward_height_limit=.095, contact_height_limit=.095),
              scene={'ee_frame': NS(data=NS(target_pos_w=pos))},
              _revision_initial_shelf_pos=shelf)


def test_height_distance_contact_match_at_world_ceiling(functions, monkeypatch):
    env = height_env([.0949, .095, .0951, float('nan'), float('inf'), -.01])
    expected = [True, False, False, False, False, True]
    assert functions.goal_reward_height_valid(env).tolist() == expected
    monkeypatch.setattr(functions, 'exponential_planar_goal_proximity', lambda *a, **k: torch.ones(6))
    assert functions.goal_distance_v2(env).tolist() == [1,0,0,0,0,1]
    monkeypatch.setattr(functions, 'target_measurements', lambda *a: (
        torch.zeros(6), torch.zeros(6), torch.ones(6, dtype=torch.bool)))
    assert functions.goal_hold_v2(env).tolist() == [1,0,0,0,0,1]
    env.sensor_data_fresh = torch.ones(6, dtype=torch.bool)
    force = torch.ones(6,3)
    monkeypatch.setattr(functions, 'target_hand_contact_force_w', lambda _: force)
    assert functions.target_contact_below_height(env).tolist() == [1,0,0,0,0,1]
    force[0] = 0
    env.sensor_data_fresh[5] = False
    assert not functions.target_contact_below_height(env).any()
    force[0] = float('nan')
    assert not functions.target_contact_below_height(env).any()
    env._revision_initial_shelf_pos[0,2] = float('nan')
    assert not functions.goal_reward_height_valid(env)[0]


def test_smoothstep_boundaries_physical_mask_and_no_speed_attenuation(functions, monkeypatch):
    d = torch.tensor([0,.025,.025+1e-7,.03,.035-1e-7,.035,.04,float('nan')], dtype=torch.float64)
    speed = torch.zeros_like(d)
    valid = torch.isfinite(d)
    env = height_env([.09]*len(d))
    monkeypatch.setattr(functions, 'target_measurements', lambda *a: (d,speed,valid))
    before = functions.goal_hold_v2(env)
    assert before.tolist() == pytest.approx([1,1,1,.5,0,0,0,0], abs=1e-8)
    speed[:] = 100
    torch.testing.assert_close(functions.goal_hold_v2(env), before)
    valid[0] = False  # production target_measurements owns finite-speed and physical validity
    assert functions.goal_hold_v2(env)[0] == 0
    for radius in (0, .025, float('nan')):
        with pytest.raises(ValueError):
            functions.goal_hold_v2(env, outer_radius=radius)
    # Endpoint slopes tend to zero; target success boundary remains in the smooth region.
    assert abs((before[2]-before[1])/1e-7) < .01
    assert abs((before[5]-before[4])/1e-7) < .01


def test_hold_retains_real_physical_and_finite_velocity_checks(functions):
    n = 11
    env = height_env([.09]*n, origins=[0]*n)
    env.num_envs, env.device = n, 'cpu'
    env.target_id = torch.zeros(n, 1, dtype=torch.long)
    env._active_object_mask = torch.ones(n, 1, dtype=torch.bool)
    pos, vel = torch.zeros(n, 1, 3), torch.zeros(n, 1, 3)
    pos[..., 2] = 1.05
    quat = torch.zeros(n, 1, 4)
    quat[..., 0] = 1
    env._revision_bounds_t = torch.tensor([[[-.02,-.02,0], [.02,.02,.1]]]).repeat(n,1,1,1)
    env.scene['object_collection'] = NS(data=NS(object_pos_w=pos, object_lin_vel_w=vel, object_quat_w=quat))
    env.scene['shelf'] = NS(data=NS(root_pos_w=torch.zeros(n,3), root_quat_w=quat[:,0].clone()))
    goal = torch.zeros(n,3)
    env.command_manager = NS(get_command=lambda _: goal)
    vel[1,0] = torch.tensor([3.,4.,0.])  # finite speed does not attenuate Hold
    vel[2,0,0] = float('nan')
    vel[3,0,0] = float('inf')
    env._active_object_mask[4] = False
    pos[5,0,0] = .17
    pos[6,0,1] = .47
    pos[7,0,2] = 1.02
    pos[8,0,2] = 1.08
    quat[9,0] = torch.tensor([math.cos(math.pi/4), math.sin(math.pi/4),0.,0.])
    goal[10,0] = float('nan')
    assert functions.goal_hold_v2(env).tolist() == [1,1]+[0]*9


@pytest.mark.parametrize('dt', [.01,.02,.1])
def test_entry_speed_cache_latch_partial_reset_and_timeout(functions, dt):
    speeds = torch.tensor([0,.01,.02,.03,.05,.07,float('nan'),float('inf'),.01], dtype=torch.float64)
    n = len(speeds)
    pos = torch.zeros(n,1,3,dtype=torch.float64)
    pos[:,0,0] = .02
    pos[-1,0,0] = .03
    vel = torch.zeros_like(pos)
    vel[:,0,2] = speeds  # verify 3D, not XY speed
    env = NS(num_envs=n,device='cpu',cfg=NS(goal_success_radius=.03),step_dt=dt,
             common_step_counter=0,_sim_step_counter=0,target_id=torch.zeros(n,1,dtype=torch.long),
             scene={'object_collection':NS(data=NS(object_pos_w=pos,object_lin_vel_w=vel))},
             command_manager=NS(get_command=lambda _: torch.zeros(n,3,dtype=torch.float64)))
    entry = functions.FirstGoalEntryBonusV2(NS(params={}), env)
    paid = entry(env)*dt*25
    expected = [25*math.exp(-(max(0,v-.01)/.02)**2) for v in [0,.01,.02,.03,.05,.07]] + [0,0,0]
    assert paid.tolist() == pytest.approx(expected)
    vel.zero_()
    torch.testing.assert_close(entry(env)*dt*25, paid)  # freeze payout within this step
    entry.reset(torch.tensor([0]))
    assert entry(env)[0] == 0
    assert entry(env)[1] > 0  # other env cache survives partial reset
    env.common_step_counter += 1
    assert (entry(env)*dt*25).tolist() == pytest.approx([25]+[0]*(n-1))
    for x in (.04,.01):
        env.common_step_counter += 1
        pos[:,0,0] = x
        result = entry(env)*dt*25
    assert result.tolist() == pytest.approx([0]*(n-1)+[25])
    env.termination_manager = NS(get_term=lambda _: torch.ones(n,dtype=torch.bool))
    assert (result+functions.successful_timeout_bonus(env)*dt*75).tolist() == pytest.approx([75]*(n-1)+[100])
    entry.reset()
    env.common_step_counter += 1
    assert (entry(env)*dt*25).tolist() == pytest.approx([25]*n)
    for kwargs in ({'free_speed':-1},{'speed_scale':0},{'speed_scale':float('inf')}):
        with pytest.raises(ValueError):
            entry(env, **kwargs)


def test_only_requested_reward_changes_and_order(configs):
    module = configs('rewards','revision')
    parent = module.RevisionV1RewardsCfg()
    before = parent.to_dict()
    child = module.RevisionV2RewardsCfg()
    names = [k for k,v in vars(parent).items() if hasattr(v,'weight')]
    assert names == [k for k,v in vars(child).items() if hasattr(v,'weight')]
    changed = {k for k in names if getattr(parent,k).to_dict() != getattr(child,k).to_dict()}
    assert changed == {'target_goal_distance','target_goal_hold','target_goal_first_entry_bonus',
                       'target_goal_timeout_bonus','object_acceleration_penalty'}
    assert child.target_goal_hold.weight == 10
    assert child.target_goal_hold.params == {'full_reward_radius':.025,'outer_radius':.035}
    assert child.target_goal_first_entry_bonus.weight == 25
    assert child.target_goal_timeout_bonus.weight == 75
    assert child.object_acceleration_penalty.params == {'free_acceleration':15.,'max_cost':100.}
    assert module.RevisionV1RewardsCfg().to_dict() == before


def test_acceleration_15_threshold_cap_and_first_sample(load_mdp):
    import importlib
    module = importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.contact_sweep_clutter_fusion')
    a = torch.zeros(1,5,3,dtype=torch.float64)
    a[0,:,0] = torch.tensor([14.9,15,15.1,20,1000],dtype=a.dtype)
    env = NS(num_envs=1,device='cpu',common_step_counter=0,_sim_step_counter=0,
             scene={'object_collection':NS(data=NS(object_com_lin_acc_w=a))},
             _active_object_mask=torch.ones(1,5,dtype=torch.bool))
    cost = module.ObjectAccelerationPenalty(NS(params={}),env)
    assert cost(env,free_acceleration=15).item() == 0
    env.common_step_counter += 1
    assert cost(env,free_acceleration=15).item() == pytest.approx(200+.5*math.expm1(.2),abs=1e-5)
    cost.reset(torch.tensor([0]))
    env.common_step_counter += 1
    assert cost(env,free_acceleration=15).item() == 0
