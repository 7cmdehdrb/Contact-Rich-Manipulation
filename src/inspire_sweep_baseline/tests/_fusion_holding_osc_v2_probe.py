"""Real simulator contract and bounded-effort probe for Holding OSC v2."""
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
try:
    import torch
    import gymnasium as gym
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, axis_angle_from_quat
    from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.ur10_osc_reach_env_cfg import UR10OSCReachEnvCfg
    from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_v1.env_cfg import FusionHoldingV1EnvCfg
    from inspire_sweep_baseline.contact_sweep_clutter_fusion_holding_osc_v2.env_cfg import FusionHoldingOSCV2EnvCfg
    task = 'Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-v2'
    output = Path('outputs/fusion_holding_osc_v2_validation')
    output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(task, device='cuda:0', num_envs=2)
    assert FusionHoldingOSCV2EnvCfg.__bases__ == (FusionHoldingV1EnvCfg,)
    parent = FusionHoldingV1EnvCfg()
    child_dict = FusionHoldingOSCV2EnvCfg().to_dict()
    parent_dict = parent.to_dict()
    child_dict['actions']['arm_action'] = parent_dict['actions']['arm_action']
    for key in ('stiffness', 'damping'):
        assert child_dict['scene']['robot']['actuators']['arm'][key] == 0.
        child_dict['scene']['robot']['actuators']['arm'][key] = parent_dict['scene']['robot']['actuators']['arm'][key]
    assert child_dict == parent_dict
    action = cfg.actions.arm_action.to_dict()
    reference = UR10OSCReachEnvCfg().actions.arm_action.to_dict()
    for key in ('class_type', 'joint_names', 'body_name', 'body_offset'):
        action.pop(key); reference.pop(key)
    assert action == reference
    eef = cfg.scene.ee_frame.target_frames[0]
    assert cfg.actions.arm_action.body_name == eef.prim_path.rsplit('/', 1)[-1]
    assert cfg.actions.arm_action.body_offset.pos == eef.offset.pos
    assert cfg.actions.arm_action.body_offset.rot == eef.offset.rot
    runner = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    assert runner.actor.obstacle_observation_slots == runner.critic.obstacle_observation_slots == 16
    cfg.seed = 42
    cfg.log_dir = str(output)
    env = gym.make(task, cfg=cfg)
    raw = env.unwrapped
    obs, _ = env.reset(seed=42)
    assert obs['obstacles'].shape == (2, 64)
    assert raw.action_manager.total_action_dim == 8
    term = raw.action_manager.get_term('arm_action')
    assert len(term._joint_ids) == 6
    original_pose = term._compute_ee_pose
    poses = torch.tensor([[0.,0.,0.,1.,0.,0.,0.], [0.,0.,0.,2**-.5,0.,0.,2**-.5]],device=raw.device)
    term._compute_ee_pose = lambda: term._ee_pose_b.copy_(poses)
    commands = torch.tensor([[1.,-.5,.25,.4,-.3,.2]]*2,device=raw.device)
    term.process_actions(commands)
    torch.testing.assert_close(term.raw_actions, commands)
    torch.testing.assert_close(term._processed_actions[:,:3], commands[:,:3]*.03)
    angles = commands[:,3:]*.15
    delta = quat_from_euler_xyz(*angles.unbind(-1))
    torch.testing.assert_close(term._processed_actions[:,3:],axis_angle_from_quat(delta))
    torch.testing.assert_close(term._osc.desired_ee_pose_b[:,:3],commands[:,:3]*.03)
    torch.testing.assert_close(term._osc.desired_ee_pose_b[:,3:],quat_mul(delta,poses[:,3:]))
    term._compute_ee_pose = original_pose
    limits = raw.scene['robot'].data.joint_effort_limits[:,term._joint_ids]
    original_compute = term._osc.compute
    supplied = limits*2
    supplied[0,0] = float('nan')
    term._osc.compute = lambda **kwargs: supplied
    term.apply_actions()
    assert term.joint_efforts[0].count_nonzero() == 0
    torch.testing.assert_close(term.joint_efforts[1],limits[1])
    assert term.torque_saturated.all()
    term.reset(torch.tensor([0],device=raw.device))
    assert not term.torque_saturated[0] and term.torque_saturated[1]
    term._osc.compute = original_compute
    env.reset(seed=42)
    done = 0
    saturated = 0
    terminations = []
    with torch.inference_mode():
        for step in range(32):
            commands = torch.zeros(2,8,device=raw.device)
            commands[:,6:] = 1.
            if step >= 16: commands[:,0] = .03
            obs, reward, terminated, truncated, _ = env.step(commands)
            assert torch.isfinite(reward).all()
            assert all(torch.isfinite(v).all() for v in obs.values())
            assert torch.isfinite(raw.scene['robot'].data.joint_pos).all()
            assert (term.joint_efforts.abs() <= limits + 1e-5).all()
            if (terminated | truncated).any():
                terminations.append(dict(step=step, terms={
                    name: raw.termination_manager.get_term(name).tolist()
                    for name in raw.termination_manager.active_terms
                    if raw.termination_manager.get_term(name).any()
                }))
            done += int((terminated|truncated).sum())
            saturated += int(term.torque_saturated.sum())
    result = dict(task=task,steps=32,num_envs=2,seed=42,done_count=done,saturated_env_steps=saturated,
                  config_matches_reference=True,parent_only_arm_changed=True,terminations=terminations,
                  mixed_rpy_base_frame_checked=True,effort_clamp_nonfinite_and_partial_reset_checked=True,
                  body_name=cfg.actions.arm_action.body_name,body_offset=cfg.actions.arm_action.body_offset.to_dict())
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    env.close()
    print('HOLDING OSC V2 PROBE PASSED',flush=True)
except BaseException:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
    app.close()
