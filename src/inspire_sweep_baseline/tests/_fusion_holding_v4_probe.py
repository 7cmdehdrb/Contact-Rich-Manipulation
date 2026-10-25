"""Real registry/config contracts and optional reset/step for the v4 family."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", choices=("Holding-v4", "Holding-OSC-v4", "Holding-OSC-ObjectRandom-v3"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs import FusionHoldingV3EnvCfg, FusionHoldingV4EnvCfg
    from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs import FusionHoldingOSCV3EnvCfg, FusionHoldingOSCV4EnvCfg
    from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs import FusionHoldingOSCObjectRandomV2EnvCfg, FusionHoldingOSCObjectRandomV3EnvCfg
    from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion_holding_scheduled import ReachingOrGoal, goal_settling_centered
    from inspire_sweep_baseline.mdp.reward_functions.goal_rewards import action_rate_l2, joint_vel_l2

    assert FusionHoldingV4EnvCfg.__bases__ == (FusionHoldingV3EnvCfg,)
    assert FusionHoldingOSCV4EnvCfg.__bases__ == (FusionHoldingV4EnvCfg,)
    assert FusionHoldingOSCObjectRandomV3EnvCfg.__bases__ == (FusionHoldingOSCV4EnvCfg,)
    parent, child = FusionHoldingV3EnvCfg(), FusionHoldingV4EnvCfg()
    assert child.terminations.success is None
    assert "max_hold_time" not in child.rewards.goal_hold.params
    assert child.rewards.goal_hold.func is goal_settling_centered
    assert child.rewards.reaching.func is ReachingOrGoal
    assert child.rewards.goal_hold.weight == parent.rewards.goal_hold.weight == 4.
    assert child.rewards.reaching.weight == parent.rewards.reaching.weight == 30.
    child.rewards.goal_hold = parent.rewards.goal_hold.copy()
    child.rewards.reaching = parent.rewards.reaching.copy()
    child.terminations.success = parent.terminations.success.copy()
    assert child.to_dict() == parent.to_dict()
    osc, old_osc = FusionHoldingOSCV4EnvCfg(), FusionHoldingOSCV3EnvCfg()
    assert osc.actions.to_dict() == old_osc.actions.to_dict()
    assert osc.scene.robot.to_dict() == old_osc.scene.robot.to_dict()
    osc.rewards, osc.terminations = old_osc.rewards.copy(), old_osc.terminations.copy()
    assert osc.to_dict() == old_osc.to_dict()
    random, old_random = FusionHoldingOSCObjectRandomV3EnvCfg(), FusionHoldingOSCObjectRandomV2EnvCfg()
    random.rewards, random.terminations = old_random.rewards.copy(), old_random.terminations.copy()
    assert random.to_dict() == old_random.to_dict()

    report = {"config_comparisons": "passed", "tasks": []}
    out = Path("outputs/fusion_holding_v4_validation")
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"{args.smoke}.json" if args.smoke else "config.json")
    for suffix, cls in (("Holding-v4", FusionHoldingV4EnvCfg), ("Holding-OSC-v4", FusionHoldingOSCV4EnvCfg),
                        ("Holding-OSC-ObjectRandom-v3", FusionHoldingOSCObjectRandomV3EnvCfg)):
        task = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-" + suffix
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        assert type(cfg) is cls
        assert cfg.terminations.success is None
        assert cfg.terminations.time_out.time_out
        assert agent.actor.obstacle_observation_slots == agent.critic.obstacle_observation_slots == 16
        assert agent.experiment_name.endswith(suffix.lower().replace("-", "_").replace("objectrandom", "object_random"))
        for name, fn, weight in (("arm_action_rate", action_rate_l2, -.03), ("hand_action_rate", action_rate_l2, -.03),
                                 ("arm_joint_velocity", joint_vel_l2, -.03), ("hand_joint_velocity", joint_vel_l2, -.01)):
            term = getattr(cfg.rewards, name)
            assert term.func is fn and term.weight == weight
        record = {"task": task, "experiment": agent.experiment_name}
        report["tasks"].append(record)
        if args.smoke == suffix:
            cfg.scene.num_envs, cfg.sim.device, cfg.seed = 2, args.device, 42
            cfg.log_dir = str(out)
            env = gym.make(task, cfg=cfg)
            raw = env.unwrapped
            obs, _ = env.reset(seed=42)
            assert "success" not in raw.termination_manager.active_terms
            for _ in range(10):
                obs, reward, terminated, truncated, _ = env.step(torch.zeros((2, 8), device=raw.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(v).all() for v in obs.values())
            record["reset_step"] = "10 steps passed"
            record["termination_terms"] = raw.termination_manager.active_terms
            # Force the inherited timeout on env=0 without changing failure rules.
            raw.episode_length_buf[0] = raw.max_episode_length - 1
            _, _, _, truncated, _ = env.step(torch.zeros((2, 8), device=raw.device))
            assert truncated[0].item()
            record["timeout"] = "passed"
            env.close()
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
finally:
    app.close()
