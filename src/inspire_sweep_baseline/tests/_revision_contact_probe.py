"""Actual PhysX target/hand contacts at low and high EEF heights.

Controlled validation fixture, not a normal-spawn or learned-policy test. The
arm base stays fixed; a Jacobian IK reconfiguration moves its joints and normal
OSC torque control holds the resulting pose during physical contact steps.
The target is then placed a measured few millimetres into the palm envelope.
High contacts use a gravity-disabled floating target to isolate the height gate.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from isaaclab.app import AppLauncher
from cli_args import apply_renderer_compatibility

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--output", type=Path, default=Path("outputs/revision_validation/contact_physics.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
apply_renderer_compatibility(args)
app = AppLauncher(args).app
env = None
report = {"scope": "Controlled real contact fixture; not normal spawn or learned policy validation", "samples": []}

try:
    import numpy as np
    import torch
    import gymnasium as gym
    import isaaclab.sim as sim
    from isaaclab.utils.math import axis_angle_from_quat, quat_apply, quat_conjugate, quat_mul, skew_symmetric_matrix
    import inspire_sweep_baseline
    from inspire_sweep_baseline.assets.robot import ARM_JOINT_NAMES
    from inspire_sweep_baseline.envs.revision_envs import RevisionEnvCfg
    from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import target_hand_contact_force_w, whole_hand_contact_force_w
    from inspire_sweep_baseline.mdp.reward_functions.revision import target_contact_below_height
    from revision_contact_geometry_probe import _body_collider_points, _rotation, measure_contact_height_geometry

    cfg = RevisionEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = 42
    cfg.log_dir = str(args.output.parent)
    wrapped = gym.make("Isaac-Inspire-Revision-v0", cfg=cfg)
    env = wrapped.unwrapped
    env.reset(seed=42)
    robot, objects = env.scene["robot"], env.scene["object_collection"]
    joint_ids, _ = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
    body_index = robot.body_names.index("axia80_link")
    jacobian_index = body_index - int(robot.is_fixed_base)
    initial_root = robot.data.root_pos_w.clone()
    initial_joint_pos = robot.data.joint_pos.clone()
    initial_eef_quat = env.scene["ee_frame"].data.target_quat_w[:, 0].clone()
    surface = env._revision_initial_shelf_pos[:, 2] + 1.05
    all_rows = torch.arange(args.num_envs * 10, dtype=torch.long)
    stage = sim.get_current_stage()
    palm_local, _, _ = _body_collider_points(stage, "/World/envs/env_0/Robot/inspire_base_link")
    palm_index = robot.body_names.index("inspire_base_link")
    palm_sensor_index = robot.body_names.index("inspire_palm_force_sensor")
    report["initial_geometry"] = measure_contact_height_geometry(env)
    report["target_kinds"] = np.asarray(env._revision_object_kinds)[:, 0].tolist()
    report["target_scales"] = np.asarray(env._final_object_scales)[:, 0].tolist()
    selected_envs = [i for i, kind in enumerate(report["target_kinds"]) if kind == 0]
    if not selected_envs:
        raise RuntimeError("This contact fixture needs a sampled cylinder target; use --num_envs 4 or 8 with seed42")
    report["fixture_selected_envs"] = selected_envs
    report["fixture_target_kind"] = "cylinder"
    print("CONTACT_PROBE initialized", flush=True)

    def refresh():
        env.sim.forward()
        env.scene.update(env.physics_dt)

    def park_objects():
        states = objects.data.object_link_state_w.clone()
        states[..., :3] = env.scene.env_origins[:, None]
        states[..., 2] += 50. + 2. * torch.arange(10, device=env.device)
        states[..., 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
        states[..., 7:] = 0.
        objects.write_object_link_state_to_sim(states)
        objects.root_physx_view.set_disable_gravities(torch.ones((args.num_envs*10, 1), dtype=torch.uint8), all_rows)
        env._active_object_mask[:] = False
        env._active_object_mask[:, 0] = True
        refresh()

    def solve_pose(desired_position, desired_quaternion):
        """Damped least-squares IK on actual PhysX Jacobians, no root edits."""
        for iteration in range(120):
            current = env.scene["ee_frame"].data
            current_pos, current_quat = current.target_pos_w[:, 0], current.target_quat_w[:, 0]
            rotation_error = axis_angle_from_quat(quat_mul(desired_quaternion, quat_conjugate(current_quat)))
            error = torch.cat((desired_position - current_pos, rotation_error), dim=-1)
            if float(error[:, :3].norm(dim=-1).max()) < 2.e-4 and float(error[:, 3:].norm(dim=-1).max()) < 2.e-3:
                break
            jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index, :, joint_ids].clone()
            offset = current_pos - robot.data.body_link_pos_w[:, body_index]
            jacobian[:, :3] -= torch.bmm(skew_symmetric_matrix(offset), jacobian[:, 3:])
            regularized = jacobian @ jacobian.transpose(1, 2) + .015**2 * torch.eye(6, device=env.device)[None]
            delta = (jacobian.transpose(1, 2) @ torch.linalg.solve(regularized, error[..., None])).squeeze(-1)
            delta *= (.06 / delta.abs().amax(-1, keepdim=True).clamp_min(.06))
            positions = robot.data.joint_pos[:, joint_ids] + delta
            limits = robot.data.soft_joint_pos_limits[:, joint_ids]
            positions = positions.clamp(limits[..., 0], limits[..., 1])
            robot.write_joint_state_to_sim(positions, torch.zeros_like(positions), joint_ids=joint_ids)
            refresh()
        measured = env.scene["ee_frame"].data.target_pos_w[:, 0].clone()
        result = {"iterations": iteration + 1, "max_position_error_m": float((measured-desired_position).norm(dim=-1).max()),
                  "eef_heights_m": (measured[:, 2]-surface).tolist()}
        print("CONTACT_IK " + json.dumps(result), flush=True)
        return result

    def place_target(depth, airborne):
        """Align a target bounding-box face to the actual palm's front surface."""
        states = objects.data.object_link_state_w.clone()
        states[:, 0, 3:7] = torch.tensor([1., 0., 0., 0.], device=env.device)
        states[:, 0, 7:] = 0.
        for row in selected_envs:
            palm_pos = robot.data.body_link_pos_w[row, palm_index].cpu().numpy()
            palm_rot = _rotation(robot.data.body_link_quat_w[row, palm_index].cpu().numpy())
            points = palm_local @ palm_rot.T + palm_pos
            normal = _rotation(robot.data.body_link_quat_w[row, palm_sensor_index].cpu().numpy())[:, 2]
            normal_xy = normal[:2] / np.linalg.norm(normal[:2])
            projection = points @ normal
            front = points[projection >= projection.max() - .003].mean(0)
            bounds = env._final_object_bounds[row, 0]
            center, half = bounds.mean(0), (bounds[1] - bounds[0]) / 2
            target_center_xy = front[:2] + normal_xy * (float(np.abs(normal_xy) @ half[:2]) - depth)
            states[row, 0, :2] = torch.as_tensor(target_center_xy - center[:2], device=env.device)
            states[row, 0, 2] = (float(front[2] - center[2]) if airborne
                                 else float(surface[row]) - float(bounds[0, 2]))
        objects.write_object_link_state_to_sim(states)
        flags = torch.ones((args.num_envs * 10, 1), dtype=torch.uint8)
        if not airborne:
            flags[selected_envs] = 0
        objects.root_physx_view.set_disable_gravities(flags, all_rows)
        objects.root_physx_view.wake_up(torch.arange(args.num_envs, dtype=torch.long))
        refresh()

    def physics_step():
        actions = torch.zeros((args.num_envs, 8), device=env.device)
        actions[:, -2:] = 1.
        env.action_manager.process_action(actions)
        # Read every physical substep: a floating target can receive a real
        # impulse on the first step and separate before the policy-step end.
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        env.sensor_data_fresh[:] = True

    with torch.inference_mode():
        for mode, height, airborne in (("low", .095, False), ("high", .15, True)):
            env.reset(seed=42)
            park_objects()
            desired = env._revision_initial_shelf_pos.clone()
            desired[:, 2] = surface + height
            ik = solve_pose(desired, initial_eef_quat)
            report.setdefault("ik", {})[mode] = ik
            report.setdefault("ik_geometry", {})[mode] = measure_contact_height_geometry(env, env_id=selected_envs[0])
            if ik["max_position_error_m"] > .01:
                raise AssertionError(f"IK failed for {mode}: {ik}")
            # Each attempt begins from the same IK pose. A contact impulse can
            # otherwise move the target away and obscure the height-gate test.
            hold_joint_pos = robot.data.joint_pos.clone()
            for depth in (0., .001):
                robot.write_joint_state_to_sim(hold_joint_pos, torch.zeros_like(hold_joint_pos))
                refresh()
                place_target(depth, airborne)
                for step in range(5):
                    physics_step()
                    force = target_hand_contact_force_w(env)
                    other = whole_hand_contact_force_w(env) - force
                    reward = target_contact_below_height(env)
                    heights = env.scene["ee_frame"].data.target_pos_w[:, 0, 2] - surface
                    row = {"mode": mode, "fixture_penetration_m": depth, "step": step,
                           "force_norm_n": force.norm(dim=-1).tolist(), "eef_heights_m": heights.tolist(),
                           "reward": reward.tolist(), "other_hand_force_norm_n": other.norm(dim=-1).tolist(),
                           "target_speed_m_s": objects.data.object_lin_vel_w[:, 0].norm(dim=-1).tolist()}
                    report["samples"].append(row)
                print("CONTACT_ATTEMPT " + json.dumps(report["samples"][-5:]), flush=True)
            report.setdefault("final_geometry", {})[mode] = measure_contact_height_geometry(env, env_id=selected_envs[0])
        torch.testing.assert_close(robot.data.root_pos_w, initial_root, atol=1.e-6, rtol=0.)
        low = [row for row in report["samples"] if row["mode"] == "low"]
        high = [row for row in report["samples"] if row["mode"] == "high"]
        low_hits = sum(force >= .01 and h < .12 and value == 1.
                       for row in low for force, h, value in zip(row["force_norm_n"], row["eef_heights_m"], row["reward"]))
        high_hits = sum(force >= .01 and h >= .12 and value == 0.
                        for row in high for force, h, value in zip(row["force_norm_n"], row["eef_heights_m"], row["reward"]))
        wrong = [(row["mode"], force, h, value) for row in report["samples"]
                 for force, h, value in zip(row["force_norm_n"], row["eef_heights_m"], row["reward"])
                 if value != float(force >= .01 and h < .12)]
        report["result"] = {"low_contact_reward_one_samples": low_hits,
                            "high_contact_reward_zero_samples": high_hits,
                            "incorrect_gate_samples": wrong, "robot_root_unchanged": True,
                            "passed": low_hits > 0 and high_hits > 0 and not wrong}
        if not report["result"]["passed"]:
            raise AssertionError(report["result"])
except BaseException as error:
    report["error"] = repr(error)
    traceback.print_exc()
    raise
finally:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("CONTACT_RESULT " + json.dumps(report.get("result", {"error": report.get("error")})), flush=True)
    if env is not None:
        env.close()
    app.close()
