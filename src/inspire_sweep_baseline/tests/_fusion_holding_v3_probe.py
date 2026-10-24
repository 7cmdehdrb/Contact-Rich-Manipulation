"""Real launcher imports/config comparisons; optional asset-dependent reset/step."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--task", choices=("Holding-v3", "Holding-OSC-v3", "Holding-OSC-ObjectRandom-v2"))
parser.add_argument("--curriculum-iterations", type=int, nargs="+", default=[6000, 9000, 12000, 16000, 0])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.smoke and not args.task:
    parser.error("--smoke requires --task; use a fresh simulator process for each environment")
apply_renderer_compatibility(args)
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch
    import inspire_sweep_baseline
    from isaaclab_tasks.utils import load_cfg_from_registry
    from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_envs import FusionHoldingV2EnvCfg, FusionHoldingV3EnvCfg
    from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_envs import FusionHoldingOSCV2EnvCfg, FusionHoldingOSCV3EnvCfg
    from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_holding_osc_object_random_envs import FusionHoldingOSCObjectRandomV1EnvCfg, FusionHoldingOSCObjectRandomV2EnvCfg
    from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion_holding_scheduled import CappedGoalSettling
    from inspire_sweep_baseline.mdp.reward_functions.goal_rewards import action_rate_l2, joint_vel_l2

    assert FusionHoldingV3EnvCfg.__bases__ == (FusionHoldingV2EnvCfg,)
    assert FusionHoldingOSCV3EnvCfg.__bases__ == (FusionHoldingV3EnvCfg,)
    assert FusionHoldingOSCObjectRandomV2EnvCfg.__bases__ == (FusionHoldingOSCV3EnvCfg,)
    parent, child = FusionHoldingV2EnvCfg(), FusionHoldingV3EnvCfg()
    assert child.rewards.goal_hold.func is CappedGoalSettling
    assert child.rewards.goal_hold.params["max_hold_time"] == 3.
    child.rewards.goal_hold = parent.rewards.goal_hold.copy()
    assert child.to_dict() == parent.to_dict()
    old_osc, new_osc = FusionHoldingOSCV2EnvCfg(), FusionHoldingOSCV3EnvCfg()
    assert old_osc.actions.to_dict() == new_osc.actions.to_dict()
    new_osc.rewards = old_osc.rewards.copy()
    assert new_osc.to_dict() == old_osc.to_dict()
    old, new = FusionHoldingOSCObjectRandomV1EnvCfg(), FusionHoldingOSCObjectRandomV2EnvCfg()
    assert new.curriculum.obstacle_population.params == dict(
        initial_gray=1, initial_white=1, final_gray=6, final_white=6,
        white_start_iteration=3000, gray_start_iteration=4000, interval=2000,
    )
    new.rewards, new.curriculum = old.rewards.copy(), old.curriculum.copy()
    assert new.to_dict() == old.to_dict()
    report = {"config_comparisons": "passed", "tasks": []}
    output = Path("outputs/fusion_holding_v3_validation")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / (f"probe_{args.task}.json" if args.task else "probe.json")

    def save_report():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for suffix, cls in (("Holding-v3", FusionHoldingV3EnvCfg), ("Holding-OSC-v3", FusionHoldingOSCV3EnvCfg),
                        ("Holding-OSC-ObjectRandom-v2", FusionHoldingOSCObjectRandomV2EnvCfg)):
        if args.task and suffix != args.task:
            continue
        task = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-" + suffix
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        assert isinstance(cfg, cls)
        assert agent.actor.obstacle_observation_slots == agent.critic.obstacle_observation_slots == 16
        assert agent.experiment_name.endswith(suffix.lower().replace("-", "_").replace("objectrandom", "object_random"))
        assert cfg.rewards.arm_action_rate.func is action_rate_l2
        assert cfg.rewards.hand_action_rate.func is action_rate_l2
        assert cfg.rewards.arm_joint_velocity.func is joint_vel_l2
        assert cfg.rewards.hand_joint_velocity.func is joint_vel_l2
        assert [cfg.rewards.arm_action_rate.weight, cfg.rewards.hand_action_rate.weight,
                cfg.rewards.arm_joint_velocity.weight, cfg.rewards.hand_joint_velocity.weight] == [-.03, -.03, -.03, -.01]
        record = {"task": task, "experiment": agent.experiment_name}
        report["tasks"].append(record)
        if args.smoke:
            cfg.scene.num_envs, cfg.sim.device = 2, args.device
            cfg.seed = 42
            cfg.log_dir = "outputs/fusion_holding_v3_validation"
            env = gym.make(task, cfg=cfg)
            raw = env.unwrapped
            obs, _ = env.reset(seed=42)
            for _ in range(10):
                obs, reward, terminated, truncated, _ = env.step(torch.zeros((2, 8), device=raw.device))
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
            record["reset_step"] = "passed"
            save_report()
            if suffix.endswith("ObjectRandom-v2"):
                record["curriculum_resets"] = []
                for iteration in args.curriculum_iterations:
                    raw.set_learning_iteration(iteration)
                    try:
                        env.reset()
                    except Exception as exc:
                        record["failed_iteration"] = iteration
                        record["error"] = str(exc)
                        save_report()
                        env.close()
                        raise
                    spawn = raw.event_manager.get_term_cfg("object_spawn").params
                    expected_count = 1 + spawn["gray_obstacles"] + spawn["white_obstacles"]
                    assert (raw._active_object_mask.sum(-1) == expected_count).all()
                    record["curriculum_resets"].append(iteration)
                    save_report()
                raw.set_learning_iteration(9000)
                env.reset(seed=42)
                positions = raw.scene["object_collection"].data.object_pos_w.clone()
                env.reset(seed=42)
                torch.testing.assert_close(positions, raw.scene["object_collection"].data.object_pos_w)
                record["seeded_padding_reset"] = "passed"
            env.close()
    save_report()
    print(json.dumps(report, indent=2))
finally:
    app.close()
