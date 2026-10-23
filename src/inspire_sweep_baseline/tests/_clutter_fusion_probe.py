"""Short real Isaac/PhysX fusion-contract and RSL-RL PPO integration probe."""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--output", type=Path, default=Path("outputs/clutter_fusion_validation/probe.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
from inspire_sweep_baseline.mdp.event_functions.contact_sweep_clutter_fusion import classify_visibility_torch
from inspire_sweep_baseline.mdp.observation_functions.contact_sweep_clutter_fusion import initial_goal_polar_in_root
from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion import WrenchRatePenalty
from inspire_sweep_baseline.mdp.observation_functions.contact_sweep_clutter import PureWristWrench


TASK = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v0"
report = {"task": TASK, "checks": [], "scope": "Short contract/PPO smoke; no trained-policy performance claim."}


def stage_values(env):
    rewards = env.reward_manager
    return [
        rewards.get_term_cfg("pushing_speed").params["free_speed"],
        rewards.get_term_cfg("arm_speed_excess").params["free_speed"],
        rewards.get_term_cfg("action_rate").weight,
        rewards.get_term_cfg("eef_linear_speed").params["free_speed"],
        math.degrees(env.event_manager.get_term_cfg("object_spawn").params["excluded_x_half_angle"]),
    ]


def check_goal(env):
    command = env.command_manager.get_term("target_goal_pos")
    delta = command.command[:, :3] - command.target_init_state_w[:, :3]
    root_delta = quat_apply_inverse(env.scene["robot"].data.root_quat_w, delta)
    polar = initial_goal_polar_in_root(env)
    torch.testing.assert_close(polar[:, 0], root_delta[:, :2].norm(dim=-1))
    expected_xy = torch.stack((polar[:, 1].cos(), -polar[:, 1].sin()), dim=-1) * polar[:, :1]
    torch.testing.assert_close(expected_xy, root_delta[:, :2], atol=1e-6, rtol=1e-5)
    assert ((polar[:, 1] >= 0) & (polar[:, 1] < 2 * math.pi)).all()
    shelf_delta = quat_apply_inverse(env.scene["shelf"].data.root_quat_w, delta)
    half_angle = env.event_manager.get_term_cfg("object_spawn").params["excluded_x_half_angle"]
    axial_cosine = shelf_delta[:, 0].abs() / shelf_delta[:, :2].norm(dim=-1)
    assert (axial_cosine <= math.cos(half_angle) + 2e-5).all()
    return polar.detach().cpu().tolist()


def check_eef_speed_math(env):
    """Use COM transport as an independent equivalent of the link-origin code."""
    robot = env.scene["robot"]
    index = robot.body_names.index("axia80_link")
    data = robot.data
    offset = env.scene["ee_frame"].data.target_pos_w[:, 0] - data.body_com_pos_w[:, index]
    velocity = data.body_com_lin_vel_w[:, index] + torch.cross(
        data.body_com_ang_vel_w[:, index], offset, dim=-1,
    )
    term = env.reward_manager.get_term_cfg("eef_linear_speed")
    speed = velocity.norm(dim=-1)
    excess = (speed - term.params["free_speed"]).clamp_min(0)
    expected = (0.5 * torch.expm1(2 * excess.clamp_max(0.5 * math.log1p(2 * term.params["max_cost"]))))
    expected = expected.clamp_max(term.params["max_cost"])
    torch.testing.assert_close(term.func(env, **term.params), expected, atol=1e-5, rtol=1e-4)
    return float(speed.max())


def check_wrench_step(env, obs, terminated, truncated):
    manager = env.reward_manager
    index = manager.active_terms.index("wrench_change")
    rates = manager._step_reward[:, index]
    assert (rates <= 1e-7).all() and (rates >= -0.100001).all()
    earned = sum(
        manager._episode_sums[name].clamp_min(0)
        for name in manager.active_terms if manager.get_term_cfg(name).weight > 0
    )
    assert (-manager._episode_sums["wrench_change"] <= 0.05 * earned + 1e-6).all()
    assert torch.count_nonzero(obs["policy"][terminated | truncated, -6:]) == 0
    reader = env.observation_manager.cfg.policy.wrist_wrench.func
    torch.testing.assert_close(obs["policy"][:, -6:], reader(env))
    return float((-rates).max())


def benchmark_visibility(env):
    if not str(env.device).startswith("cuda"):
        return {"status": "skipped_cpu"}
    positions = env._initial_object_positions_w
    shelf = env.scene["shelf"].data
    local = quat_apply_inverse(
        shelf.root_quat_w[:, None].expand(-1, positions.shape[1], -1),
        positions - shelf.root_pos_w[:, None],
    )
    centers = torch.stack((local[..., 1], -local[..., 0]), dim=-1)
    timings = {}
    for count in (4096, 10000):
        batch = centers.repeat((math.ceil(count / env.num_envs), 1, 1))[:count].contiguous()
        for _ in range(3):
            classify_visibility_torch(batch, radii=.04, threshold=.35, check_inputs=False)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            result = classify_visibility_torch(batch, radii=.04, threshold=.35, check_inputs=False)
        end.record()
        end.synchronize()
        assert result.visible.shape == (count, positions.shape[1])
        timings[str(count)] = {"mean_ms": start.elapsed_time(end) / 20, "repetitions": 20}
    return timings


env = None
try:
    assert args.num_envs >= 4
    cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=args.num_envs)
    former = {
        task: parse_env_cfg(task, device=cfg.sim.device, num_envs=args.num_envs)
        for task in ("Isaac-Inspire-Contact-Sweep-Clutter-v2", "Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1")
    }
    former_values = {task: value.to_dict() for task, value in former.items()}
    assert type(cfg).__bases__ == (ManagerBasedRLEnvCfg,)
    assert type(cfg.scene).__bases__ == (InteractiveSceneCfg,)
    for name in ("observations", "actions", "commands", "rewards", "terminations", "events", "curriculum"):
        assert type(getattr(cfg, name)).__bases__ == (object,), name
    for name in ("scene", "actions", "commands"):
        assert getattr(cfg, name).to_dict() == getattr(former[next(iter(former))], name).to_dict(), name
    # These thresholds already differ in the pre-migration registered configs.
    fusion_terminations = cfg.terminations.to_dict()
    clutter_terminations = former[next(iter(former))].terminations.to_dict()
    assert fusion_terminations["push_fast"]["params"] == {"speed_condition": 1.0}
    assert clutter_terminations["push_fast"]["params"] == {"speed_condition": 0.8}
    fusion_terminations["push_fast"]["params"] = clutter_terminations["push_fast"]["params"]
    assert fusion_terminations == clutter_terminations
    report["checks"].append("independent_cfg_bases_and_existing_scene_action_command_termination_contract")
    cfg.seed = 42
    cfg.log_dir = str(args.output.parent)
    env = gym.make(TASK, cfg=cfg).unwrapped
    assert type(env).__bases__ == (ManagerBasedRLEnv,)
    # Keep physics buffers ordinary tensors for the subsequent training setup.
    # Entering inference_mode here would create lazy scene buffers that cannot
    # be updated by RslRlVecEnvWrapper's later reset outside inference_mode.
    with torch.no_grad():
        obs, _ = env.reset(seed=42)
        report["observation_shapes"] = {name: list(value.shape) for name, value in obs.items()}
        assert report["observation_shapes"] == {
            "policy": [env.num_envs, 55], "obstacles": [env.num_envs, 40], "privileged": [env.num_envs, 3],
        }
        assert torch.count_nonzero(obs["policy"][:, -6:]) == 0
        assert type(env.observation_manager.cfg.policy.wrist_wrench.func) is PureWristWrench
        assert type(env.reward_manager.get_term_cfg("wrench_change").func) is WrenchRatePenalty
        expected_weights = {
            "reaching": 30., "goal_distance": 100., "goal_hold": .9,
            "contact_any": 0.4, "contact_count": 0.4,
            "action_rate": -.03, "joint_velocity": -.03, "object": -30.,
            "shelf_collision": -.5, "pushing_speed": -5., "arm_speed_excess": -1.,
            "shelf_failure": -3., "eef_linear_speed": -1., "eef_orientation": -.02, "wrench_change": -.1,
        }
        assert set(env.reward_manager.active_terms) == set(expected_weights)
        for name, weight in expected_weights.items():
            assert env.reward_manager.get_term_cfg(name).weight == weight, name
        report["reward_weights_at_iteration0"] = expected_weights
        report["checks"].extend(["real_resolved_reward_terms_and_weights", "v2_compensated_reader_reused", "reset_6axis_zero"])
        action = torch.zeros((env.num_envs, 8), device=env.device)
        action[:, -2:] = 1.
        stages = [
            (0, [.1, 1., -.03, 1.5, 60.]),
            (299, [.1, 1., -.03, 1.5, 60.]),
            (300, [.09, .98, -.035, 1.48, 60.]),
            (500, [.09, .98, -.035, 1.48, 60.]),
            (600, [.08, .96, -.04, 1.46, 60.]),
            (699, [.08, .96, -.04, 1.46, 60.]),
            (700, [.08, .96, -.04, 1.46, 59.]),
            (1000, [.07, .94, -.045, 1.44, 58.]),
            (1200, [.06, .92, -.05, 1.42, 57.]),
            (2500, [.05, .84, -.06, 1.34, 50.]),
            (3000, [.05, .8, -.06, 1.3, 48.]),
            (5000, [.05, .8, -.06, 1.18, 38.]),
            (13000, [.05, .8, -.06, .7, 0.]),
            (20000, [.05, .8, -.06, .7, 0.]),
        ]
        report["stages"] = []
        maximum_speed = maximum_wrench_rate_cost = 0.
        for iteration, expected in stages:
            old_goal = env.command_manager.get_command("target_goal_pos").clone()
            env.set_learning_iteration(iteration)
            assert torch.equal(old_goal, env.command_manager.get_command("target_goal_pos"))
            actual = stage_values(env)
            assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(actual, expected)), (iteration, actual)
            assert env.reward_manager.get_term_cfg("joint_velocity").weight == -.03
            obs, _ = env.reset()
            polar = check_goal(env)
            for step in range(2):
                obs, reward, terminated, truncated, _ = env.step(action)
                assert torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in obs.values())
                maximum_speed = max(maximum_speed, check_eef_speed_math(env))
                maximum_wrench_rate_cost = max(maximum_wrench_rate_cost, check_wrench_step(env, obs, terminated, truncated))
                if step == 0:
                    index = env.reward_manager.active_terms.index("wrench_change")
                    assert torch.count_nonzero(env.reward_manager._step_reward[:, index]) == 0
            report["stages"].append({"iteration": iteration, "values": actual, "reset_polar_goals": polar})
        report["checks"].extend(["schedule_boundaries", "iteration_updates_preserve_live_goals", "reset_goal_arcs_and_root_polar", "finite_physics_steps", "eef_speed_COM_transport_equivalence", "wrench_history_caps_and_budget"])
        report["maximum_eef_speed_m_s"] = maximum_speed
        report["maximum_wrench_penalty_rate"] = maximum_wrench_rate_cost
        obs, _ = env.reset(seed=43)
        untouched = torch.tensor([0, 2], device=env.device)
        ids = torch.tensor([1, 3], device=env.device)
        old_obstacles = obs["obstacles"].clone()
        old_goal = env.command_manager.get_command("target_goal_pos").clone()
        old_initial = env._initial_object_positions_w.clone()
        obs, _ = env.reset(env_ids=ids)
        assert torch.equal(obs["obstacles"][untouched], old_obstacles[untouched])
        assert torch.equal(env._initial_object_positions_w[untouched], old_initial[untouched])
        assert torch.equal(env.command_manager.get_command("target_goal_pos")[untouched], old_goal[untouched])
        assert torch.count_nonzero(obs["policy"][ids, -6:]) == 0
        assert not env.reward_manager.get_term_cfg("wrench_change").func._previous_valid[ids].any()
        env.episode_length_buf[ids] = env.max_episode_length - 1
        obs, reward, terminated, truncated, _ = env.step(action)
        assert truncated[ids].all()
        assert torch.count_nonzero(obs["policy"][ids, -6:]) == 0
        report["checks"].extend(["partial_reset_preserves_other_snapshots_goals", "partial_and_automatic_reset_sensor_history"])
        report["visibility_gpu_benchmark"] = benchmark_visibility(env)

    # Match train.py's real compatibility pass, then update a tiny fresh policy.
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    agent_cfg.device = str(env.device)
    agent_cfg.num_steps_per_env = 4
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 1
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = CurriculumOnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(env.device))
    runner.current_learning_iteration = 299
    before = [p.detach().clone() for p in runner.alg.actor.obstacle_encoder.parameters()]
    runner.learn(2, init_at_random_ep_len=False)
    assert env.learning_iteration == 300
    assert all(
        math.isclose(a, b, abs_tol=1e-10) for a, b in zip(stage_values(env), [.09, .98, -.035, 1.48, 60.])
    )
    assert any(not torch.equal(a, b) for a, b in zip(before, runner.alg.actor.obstacle_encoder.parameters()))
    assert all(torch.isfinite(p).all() for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.parent / "probe_model_300.pt"
    runner.save(str(checkpoint))
    report["ppo"] = {"iterations": [299, 300], "steps_per_env_per_iteration": 4, "checkpoint": str(checkpoint), "logging_directory": None}
    report["checks"].append("real_PPO_rollout_update_and_curriculum_boundary")
    for task, old in former.items():
        assert old.to_dict() == former_values[task]
        assert parse_env_cfg(task, device=cfg.sim.device, num_envs=args.num_envs).to_dict() == former_values[task]
    report["checks"].append("former_env_cfgs_unchanged")
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
except BaseException:
    report["status"] = "failed"
    report["failure"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # Preserve nonzero failure status with this installed Kit fast-shutdown behavior.
    os._exit(1)
finally:
    if env is not None:
        env.close()
    launcher.app.close()
