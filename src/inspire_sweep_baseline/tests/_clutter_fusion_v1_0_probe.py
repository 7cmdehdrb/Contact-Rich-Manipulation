"""Real PhysX contact, lifecycle, curriculum and tiny PPO probe for Fusion v1.0."""

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
parser.add_argument("--output", type=Path, default=Path("outputs/fusion_v1_0_revision_validation/probe.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

import inspire_sweep_baseline  # noqa: F401
from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner
from inspire_sweep_baseline.assets.robot import HAND_CONTACT_BODY_NAMES
from inspire_sweep_baseline.assets.tactile_regions import TACTILE_REGIONS
from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import hand_obstacle_contact_force_w, object_hand_sensor_name, whole_hand_contact_force_w
from inspire_sweep_baseline.mdp.reward_functions.contact_sweep_clutter_fusion import hand_contact_any
from inspire_sweep_baseline.envs.contact_sweep_clutter_fusion_envs import ContactSweepClutterFusionV3EnvCfg

TASK = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v1_0"
PARENT_TASK = "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v3"
report = {"task": TASK, "checks": [], "scope": "Small GPU integration probe; no trained-policy performance claim."}
env = None


def stages(env):
    rewards = env.reward_manager
    return [
        math.degrees(env.event_manager.get_term_cfg("object_spawn").params["excluded_x_half_angle"]),
        rewards.get_term_cfg("goal_hold").params["position_tolerance"],
        rewards.get_term_cfg("wrench_change").params["free_force_change"],
        rewards.get_term_cfg("wrench_change").params["free_moment_change"],
        rewards.get_term_cfg("object_acceleration").params["free_acceleration"],
    ]


def check_physical_contacts(env):
    """Place a real obstacle into the calibrated palm surface, without task resets."""
    env.reset(seed=43)
    robot, objects = env.scene["robot"], env.scene["object_collection"]
    region = TACTILE_REGIONS[0]
    pad_id = robot.body_names.index(region["name"])
    fixed_joints = robot.data.joint_pos.clone()
    tactile = env.observation_manager.cfg.policy.tactile
    offsets = [(0., 0.), (.02, 0.), (-.02, 0.), (.04, 0.), (-.04, 0.), (0., .02), (0., -.02)]
    diagnostics = []
    for offset_x, offset_y in offsets:
        robot.write_joint_state_to_sim(fixed_joints, torch.zeros_like(robot.data.joint_vel))
        states = objects.data.object_link_state_w.clone()
        states[..., :3] = env.scene.env_origins[:, None] + states.new_tensor([0., 2., 4.])
        states[..., 0] += torch.arange(objects.num_objects, device=env.device)[None] * .15
        states[..., 3:7] = states.new_tensor([1., 0., 0., 0.])
        states[..., 7:] = 0.
        pad_pos = robot.data.body_link_pos_w[0, pad_id]
        pad_quat = robot.data.body_link_quat_w[0, pad_id]
        local_point = states.new_tensor(region["verification"]["parent_surface_point_local_m"])
        local_normal = states.new_tensor(region["verification"]["parent_surface_outward_normal_local"])
        point = pad_pos + quat_apply(pad_quat, local_point)
        normal = quat_apply(pad_quat, local_normal)
        offset = quat_apply(pad_quat, states.new_tensor([offset_x, offset_y, 0.]))
        states[0, 1, :3] = point + offset - .002 * normal
        up = states.new_tensor([0., 0., 1.])
        orientation = torch.cat(((1. + normal[2]).reshape(1), torch.linalg.cross(up, normal)))
        states[0, 1, 3:7] = orientation / orientation.norm()
        objects.write_object_link_state_to_sim(states)
        env.scene.write_data_to_sim()
        env._sim_step_counter += 1
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        env.common_step_counter += 1
        env.sensor_data_fresh[:] = True
        env.target_id[:] = 0
        matrix = env.scene[object_hand_sensor_name(objects.object_names[1])].data.force_matrix_w
        direct = -matrix[0].sum(dim=(0, 1))
        filtered = hand_obstacle_contact_force_w(env)[0]
        net = whole_hand_contact_force_w(env)[0]
        bits = tactile.func(env, **tactile.params)
        diagnostics.append({"offset": [offset_x, offset_y], "filtered_norm_N": float(filtered.norm()),
                            "whole_hand_norm_N": float(net.norm()), "tactile_channels": bits[0].nonzero().flatten().tolist()})
        if direct.norm() > .01 and net.norm() > .01 and bits[0, 0] > .5:
            torch.testing.assert_close(filtered, direct, atol=1e-4, rtol=1e-4)
            # get_command() concatenates a copy; mutate the command term's
            # live position buffer when exercising reward distance boundaries.
            goal = env.command_manager.get_term("target_goal_pos").pos_command_w
            goal[0, :2] = objects.data.object_pos_w[0, 0, :2] + .2
            assert hand_contact_any(env)[0] == 1
            env.target_id[0] = 1
            assert hand_obstacle_contact_force_w(env)[0].norm() < 1e-5
            goal[0, :2] = objects.data.object_pos_w[0, 1, :2] + .2
            assert hand_contact_any(env)[0] == 1
            env.target_id[0] = 0
            # With unchanged physical contact, the live goal-hold boundary
            # controls the fade before and after curriculum changes/rewinds.
            fades = []
            for iteration in (0, 1000, 2500, 7500, 0):
                env.set_learning_iteration(iteration)
                tolerance = env.reward_manager.get_term_cfg("goal_hold").params["position_tolerance"]
                position = objects.data.object_pos_w[0, 0, :2]
                contact_params = env.reward_manager.get_term_cfg("hand_contact").params
                assert contact_params["goal_contact_fade_width"] == 0.
                # Exercise both the configured disabled fade and the function's
                # optional fade; pass parameters as RewardManager actually does.
                for width, samples in (
                    (0., ((-.005, 1.), (.005, 1.), (.015, 1.))),
                    (.015, ((-.005, 0.), (0., 0.), (.0075, .5), (.015, 1.), (.025, 1.))),
                ):
                    for margin, expected in samples:
                        goal[0, :2] = position + position.new_tensor([tolerance + margin, 0.])
                        actual = float(hand_contact_any(env, **{**contact_params, "goal_contact_fade_width": width})[0])
                        assert math.isclose(actual, expected, abs_tol=3e-6), (iteration, width, margin, actual)
                        fades.append([iteration, tolerance, width, margin, actual])
            return {"fixture": "real obstacle_0 against calibrated palm; other objects isolated",
                    "matrix_shape": list(matrix.shape), "diagnostics": diagnostics,
                    "goal_contact_fade": fades,
                    "same_contact_excluded_after_target_id_change": True,
                    "normal_force_only": True}
    raise AssertionError(f"Failed real obstacle/palm contact and tactile fixture: {diagnostics}")


try:
    assert args.num_envs == 4, "Partial-reset fixture uses four environments."
    cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=args.num_envs)
    parent = parse_env_cfg(PARENT_TASK, device=cfg.sim.device, num_envs=args.num_envs)
    parent_before = parent.to_dict()
    assert type(cfg).__bases__ == (ContactSweepClutterFusionV3EnvCfg,)
    for name in ("rewards", "terminations", "curriculum"):
        assert type(getattr(cfg, name)).__bases__ == (object,), name
    for name in ("observations", "actions", "commands"):
        assert getattr(cfg, name).to_dict() == getattr(parent, name).to_dict(), name
    assert cfg.scene.hand_tactile.to_dict() == parent.scene.hand_tactile.to_dict()
    report["checks"].append("v3_inheritance_standalone_manager_cfgs_and_preserved_observations_actions")
    cfg.seed = 42
    cfg.log_dir = str(args.output.parent)
    env = gym.make(TASK, cfg=cfg).unwrapped
    with torch.no_grad():
        obs, _ = env.reset(seed=42)
        shapes = {name: list(value.shape) for name, value in obs.items()}
        assert shapes == {"policy": [4, 55], "obstacles": [4, 40], "privileged": [4, 3]}, shapes
        assert env.action_manager.total_action_dim == 8
        assert torch.count_nonzero(obs["policy"][:, -6:]) == 0
        report["observation_shapes"] = shapes
        report["action_dimension"] = 8
        report["reward_terms"] = list(env.reward_manager.active_terms)
        report["curriculum_terms"] = list(env.curriculum_manager.active_terms)
        report["termination_terms"] = list(env.termination_manager.active_terms)
        assert len(report["reward_terms"]) == 17
        assert set(report["curriculum_terms"]) == {
            "obstacle_population", "pushing_free_speed", "arm_free_speed", "eef_free_speed",
            "goal_deadzone", "wrench_free_force_change", "wrench_free_moment_change", "object_free_acceleration",
        }
        assert len(report["termination_terms"]) == 6
        sensor = env.scene["whole_hand_contact"]
        assert set(sensor.body_names) == set(HAND_CONTACT_BODY_NAMES)
        assert len(sensor.body_names) == 30
        assert env.scene["object_collection"].num_objects == 7
        assert env._active_object_mask.sum(-1).tolist() == [3] * env.num_envs
        report["whole_hand_sensor_bodies"] = sensor.body_names
        report["checks"].append("actual_manager_counts_sensor_bodies_observation_and_action_shapes")
        action = torch.zeros((env.num_envs, 8), device=env.device)
        action[:, -2:] = 1.
        boundaries = [
            (0, [30.0, 0.01, 5000.0, 500.0, 100.0]),
            (999, [30.0, 0.01, 5000.0, 500.0, 100.0]),
            (1000, [30.0, 0.01, 4999.0, 499.9, 99.98]),
            (1249, [30.0, 0.01, 4999.0, 499.9, 99.98]),
            (1250, [30.0, 0.01, 4998.0, 499.8, 99.96]),
            (1499, [30.0, 0.01, 4998.0, 499.8, 99.96]),
            (1500, [30.0, 0.01, 4997.0, 499.7, 99.94]),
            (2500, [30.0, 0.01, 4993.0, 499.3, 99.86]),
            (3000, [30.0, 0.01, 4991.0, 499.1, 99.82]),
            (3499, [30.0, 0.01, 4990.0, 499.0, 99.8]),
            (3500, [25.0, 0.01, 4989.0, 498.9, 99.78]),
            (6000, [0.0, 0.01, 4979.0, 497.9, 99.58]),
            (7000, [0.0, 0.01, 4975.0, 497.5, 99.5]),
            (10750, [0.0, 0.01, 4960.0, 496.0, 99.2]),
            (11999, [0.0, 0.01, 4956.0, 495.6, 99.12]),
            (12000, [0.0, 0.01, 4955.0, 495.5, 99.1]),
            (20000, [0.0, 0.01, 4923.0, 492.3, 98.46]),
            (1250000, [0.0, 0.01, 10.0, 1, 0.1]),
            (0, [30.0, 0.01, 5000.0, 500.0, 100.0]),
        ]
        report["schedule_boundaries"] = []
        for iteration, expected in boundaries:
            old_goal = env.command_manager.get_command("target_goal_pos").clone()
            env.set_learning_iteration(iteration)
            assert torch.equal(old_goal, env.command_manager.get_command("target_goal_pos"))
            actual = stages(env)
            assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(actual, expected)), (iteration, actual)
            report["schedule_boundaries"].append({"iteration": iteration, "values": actual})
        report["checks"].append("all_new_curriculum_boundaries_clamps_rewind_and_live_goal_preservation")
        populations = []
        for iteration, gray, white in ((0, 1, 1), (2999, 1, 1), (3000, 1, 2),
                                        (4499, 1, 2), (4500, 2, 2), (5999, 2, 2),
                                        (6000, 2, 3), (7499, 2, 3), (7500, 3, 3),
                                        (20000, 3, 3), (2500, 1, 1)):
            old_mask = env._active_object_mask.clone()
            env.set_learning_iteration(iteration)
            assert torch.equal(env._active_object_mask, old_mask)
            params = env.event_manager.get_term_cfg("object_spawn").params
            assert (params["gray_obstacles"], params["white_obstacles"]) == (gray, white)
            obs, _ = env.reset(seed=42)
            assert (env._active_object_mask.sum(-1) == 1 + gray + white).all()
            gravity_flags = env.scene["object_collection"].root_physx_view.get_disable_gravities()
            assert torch.equal(
                gravity_flags.bool(),
                (~env._active_object_mask).T.reshape(-1, 1).to(gravity_flags.device),
            )
            slots = obs["obstacles"].reshape(env.num_envs, 10, 4)
            present = slots[..., 0].bool()
            assert (slots[:, 1:, 0] <= slots[:, :-1, 0]).all()
            assert (present.sum(-1) <= gray + white).all()
            assert torch.count_nonzero(slots[~present]) == 0
            assert torch.count_nonzero(slots[:, 6:]) == 0
            # Inactive parked bodies do not drift or trigger drop termination.
            before_positions = env.scene["object_collection"].data.object_pos_w.clone()
            _, reward, terminated, timed_out, _ = env.step(action)
            assert torch.isfinite(reward).all()
            after_positions = env.scene["object_collection"].data.object_pos_w
            if not torch.equal(after_positions[~env._active_object_mask], before_positions[~env._active_object_mask]):
                report["population_pose_mismatch"] = {
                    "iteration": iteration, "terminated": terminated.tolist(), "timeout": timed_out.tolist(),
                    "mask": env._active_object_mask.tolist(), "before": before_positions.tolist(),
                    "after": after_positions.tolist(),
                    "disabled_view_flags": env.scene["object_collection"].root_physx_view.get_disable_simulations().tolist(),
                }
            torch.testing.assert_close(after_positions[~env._active_object_mask],
                                       before_positions[~env._active_object_mask], rtol=0, atol=0)
            assert not env.termination_manager.get_term("object_drop").any()
            if iteration == 3000:
                # WHITE2 has just changed from parked to active. Raise it above
                # the board and verify real gravity/wake response after reset.
                objects = env.scene["object_collection"]
                body_id = 1 + env.cfg.max_gray_obstacles + 1
                raised = objects.data.object_link_state_w[0:1, body_id:body_id + 1].clone()
                raised[..., 2] += 0.2
                raised[..., 7:] = 0.0
                objects.write_object_link_state_to_sim(
                    raised, env_ids=torch.tensor([0], device=env.device),
                    object_ids=torch.tensor([body_id], device=env.device),
                )
                env.scene.write_data_to_sim()
                env._sim_step_counter += 1
                env.sim.step(render=False)
                env.scene.update(env.physics_dt)
                assert objects.data.object_pos_w[0, body_id, 2] < raised[0, 0, 2]
                report["newly_active_white2_gravity_response"] = True
            populations.append({"iteration": iteration, "gray": gray, "white": white,
                                "active_objects": env._active_object_mask.sum(-1).tolist()})
        report["population_stages"] = populations
        report["checks"].append("population_3000_4500_6000_7500_rewind_and_inactive_physics_observations")
        env.set_learning_iteration(0)
        env.reset(seed=42)
        wrench = env.reward_manager.get_term_cfg("wrench_change").func
        acceleration = env.reward_manager.get_term_cfg("object_acceleration").func
        for step in range(12):
            obs, reward, terminated, timeout, _ = env.step(action)
            assert torch.isfinite(reward).all()
            native_acc = objects.data.object_com_lin_acc_w
            physx_acc = objects.data._reshape_view_to_data(objects.root_physx_view.get_accelerations())[..., :3]
            torch.testing.assert_close(native_acc, physx_acc)
            assert torch.isfinite(native_acc).all()
            assert (acceleration._cached_cost <= 100.0 * env._active_object_mask.sum(-1)).all()
            if step == 1:
                magnitude = torch.linalg.vector_norm(native_acc, dim=-1)
                excess = (magnitude - 100.).clamp(0., .5 * math.log(201.))
                expected = (.5 * torch.expm1(2. * excess)).clamp(max=100.)
                expected = torch.where(env._active_object_mask, expected, 0.).sum(-1)
                ongoing = ~(terminated | timeout)
                torch.testing.assert_close(acceleration._cached_cost[ongoing], expected[ongoing])
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.count_nonzero(obs["policy"][terminated | timeout, -6:]) == 0
            reader = env.observation_manager.cfg.policy.wrist_wrench.func
            torch.testing.assert_close(reader(env), obs["policy"][:, -6:])
            earned = sum(env.reward_manager._episode_sums[name].clamp_min(0.)
                         for name in env.reward_manager.active_terms
                         if env.reward_manager.get_term_cfg(name).weight > 0.)
            assert (-env.reward_manager._episode_sums["wrench_change"] <= .05 * earned + 1e-6).all()
            if step == 0:
                for name in ("wrench_change", "object_acceleration"):
                    index = env.reward_manager.active_terms.index(name)
                    assert torch.count_nonzero(env.reward_manager._step_reward[:, index]) == 0
        report["checks"].append("12_finite_physical_steps_EEF_sample_budget_and_first_sample_history")
        report["checks"].append("native_PhysX_COM_acceleration_matches_reward_and_active_count_cap")
        ids = torch.tensor([1, 3], device=env.device)
        others = torch.tensor([0, 2], device=env.device)
        old_obstacles = obs["obstacles"].clone()
        old_goal = env.command_manager.get_command("target_goal_pos").clone()
        old_valid = acceleration._previous_valid.clone()
        old_wrench = wrench._previous_wrench.clone()
        obs, _ = env.reset(env_ids=ids)
        assert torch.equal(obs["obstacles"][others], old_obstacles[others])
        assert torch.equal(env.command_manager.get_command("target_goal_pos")[others], old_goal[others])
        assert torch.equal(acceleration._previous_valid[others], old_valid[others])
        assert torch.equal(wrench._previous_wrench[others], old_wrench[others])
        assert not acceleration._previous_valid[ids].any() and not wrench._previous_valid[ids].any()
        assert torch.count_nonzero(obs["policy"][ids, -6:]) == 0
        env.episode_length_buf[ids] = env.max_episode_length - 1
        obs, reward, terminated, timeout, _ = env.step(action)
        assert timeout[ids].all()
        assert not acceleration._previous_valid[ids].any() and not wrench._previous_valid[ids].any()
        report["checks"].append("partial_and_timeout_resets_clear_only_relevant_reward_history")
        report["physical_contact"] = check_physical_contacts(env)
        report["checks"].append("real_whole_hand_contact_target_identity_mask_and_existing_palm_tactile")
        env.reset(seed=42)

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    agent_cfg.device = str(env.device)
    agent_cfg.num_steps_per_env = 4
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 1
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = CurriculumOnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(env.device))
    runner.current_learning_iteration = 2499
    before = [parameter.detach().clone() for parameter in runner.alg.actor.obstacle_encoder.parameters()]
    runner.learn(2, init_at_random_ep_len=False)
    assert env.learning_iteration == 2500
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(stages(env), [30., .01, 4993., 499.3, 99.86]))
    assert any(not torch.equal(a, b) for a, b in zip(before, runner.alg.actor.obstacle_encoder.parameters()))
    assert all(torch.isfinite(parameter).all() for model in (runner.alg.actor, runner.alg.critic)
               for parameter in model.parameters())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.parent / "model_2500.pt"
    runner.save(str(checkpoint))
    report["ppo"] = {"iterations": [2499, 2500], "steps_per_env_per_iteration": 4,
                     "checkpoint": str(checkpoint), "logging_directory": None}
    report["checks"].append("real_PPO_update_and_iteration_2500_curriculum_boundary")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from training_recovery import restore_for_training
    # RSL-RL materialized lazy scene buffers during inference-mode rollouts.
    # Reuse that mode while resetting this same live test environment. The
    # recovery launcher itself starts a fresh process/environment each attempt.
    with torch.inference_mode():
        env.set_learning_iteration(7500)
        env.reset(seed=42)
        assert (env._active_object_mask.sum(-1) == 7).all()
        restored = restore_for_training(runner, checkpoint)
    assert restored == env.learning_iteration == 2500
    assert (env._active_object_mask.sum(-1) == 3).all()
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(stages(env), [30., .01, 4993., 499.3, 99.86]))
    report["checks"].append("checkpoint_2500_restores_reward_and_population_before_reset_and_rollout")
    assert parent.to_dict() == parent_before
    assert parse_env_cfg(PARENT_TASK, device=cfg.sim.device, num_envs=args.num_envs).to_dict() == parent_before
    report["checks"].append("v3_cfg_unchanged")
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
    os._exit(1)
finally:
    if env is not None:
        env.close()
    launcher.app.close()
