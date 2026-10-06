#!/usr/bin/env python3
"""Verify Push v1 managers and an optional scripted OSC physics fixture."""

from __future__ import annotations

import argparse
from dataclasses import fields
from itertools import product
import math
from pathlib import Path
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
PUSH_V1_TASK_ID = "Isaac-Hand-Manipulation-Push-v1"


def build_parser(app_launcher):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=(PUSH_V1_TASK_ID,), default=PUSH_V1_TASK_ID)
    parser.add_argument("--num_envs", "--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-samples", type=int, default=2)
    parser.add_argument("--check-boundaries", action="store_true")
    parser.add_argument("--check-timeout", action="store_true")
    parser.add_argument("--debug-vis", action="store_true")
    parser.add_argument("--disable-markers", action="store_true")
    parser.add_argument("--skip-physical-fixture", action="store_true")
    parser.add_argument("--physical-only", action="store_true")
    parser.add_argument("--physical-steps", type=int, default=350)
    parser.add_argument("--physical-goal", action="store_true", help="Push to the full command goal and require the real success termination")
    parser.add_argument("--physical-goal-distance", type=float, default=.20)
    parser.add_argument("--physical-angle-offset-deg", type=float, default=0., help="Offset both right/left fixture commands within [-10,10] degrees")
    parser.add_argument("--fixture-verbose", action="store_true", help="Print fixture diagnostics without enabling Kit verbose logging")
    app_launcher.add_app_launcher_args(parser)
    return parser


def validate_args(args):
    if args.num_envs <= 0 or args.reset_samples < 0:
        raise ValueError("--num-envs must be positive and --reset-samples nonnegative")
    if not 1 <= args.steps < 500 or not 1 <= args.physical_steps <= 450:
        raise ValueError("--steps must be in [1,499] and --physical-steps in [1,450]")
    if args.physical_only and args.skip_physical_fixture:
        raise ValueError("--physical-only cannot be combined with --skip-physical-fixture")
    if args.physical_goal and args.skip_physical_fixture:
        raise ValueError("--physical-goal cannot be combined with --skip-physical-fixture")
    if not math.isfinite(args.physical_goal_distance) or not .20 <= args.physical_goal_distance <= .30:
        raise ValueError("--physical-goal-distance must be within [.20,.30]")
    if not math.isfinite(args.physical_angle_offset_deg) or not -10. <= args.physical_angle_offset_deg <= 10.:
        raise ValueError("--physical-angle-offset-deg must be within [-10,10]")


def valid_start_bounds(task, direction_w, distance_m):
    """Intersect the configured Cube box with safe starts and safe goals."""
    padding = math.sqrt(3)*task.cube_size/2 + task.table_path_margin_m
    safe_low = tuple(task.table_center[i]-task.table_size[i]/2+padding for i in range(2))
    safe_high = tuple(task.table_center[i]+task.table_size[i]/2-padding for i in range(2))
    low = tuple(max(task.object_xy_range_low[i], safe_low[i], safe_low[i]-distance_m*direction_w[i]) for i in range(2))
    high = tuple(min(task.object_xy_range_high[i], safe_high[i], safe_high[i]-distance_m*direction_w[i]) for i in range(2))
    if any(low[i] > high[i] for i in range(2)):
        raise ValueError("The command has no complete Table path inside the Cube spawn box")
    return low, high


def sampled_start_bounds(task, direction_w, distance_m):
    """Return the actual conditional sampler domain for one angle/length."""
    path_low, path_high = valid_start_bounds(task, direction_w, distance_m)
    if not task.command_centered_path:
        return path_low, path_high
    midpoint_x = task.table_center[0]+task.command_midpoint_x_offset_m
    center_x = midpoint_x-.5*distance_m*direction_w[0]
    jitter = task.command_initial_x_jitter_m
    low = (center_x-jitter, task.object_xy_range_low[1])
    high = (center_x+jitter, task.object_xy_range_high[1])
    if any(low[i] < path_low[i]-1e-7 or high[i] > path_high[i]+1e-7 for i in range(2)):
        raise ValueError("The conditional sampler domain leaves the spawn box or complete Table path")
    return low, high


class PushV1Smoke:
    """Keep runtime checks separate from the task's normal manager behavior."""

    def __init__(self, env, args, torch, math_utils):
        from hand_manipulation_test import constants, geometry
        from hand_manipulation_test.action_math import inspire_synergy_to_joint_positions
        from hand_manipulation_test.assets.robot import HAND_CLOSED_TARGETS, HAND_JOINT_NAMES, HAND_OPEN_TARGETS, ROBOT_CONTACT_BODY_NAMES
        self.env, self.args, self.torch, self.math = env, args, torch, math_utils
        self.cfg, self.constants, self.geometry = env.cfg, constants, geometry
        self.robot, self.cube = env.scene["robot"], env.scene["target_object"]
        self.arm, self.hand = (env.action_manager.get_term(name) for name in ("arm_action", "hand_action"))
        self.command = env.command_manager.get_term("target_position")
        self.safe = env.event_manager.get_term_cfg("safe_hand").func
        self.ids = torch.arange(env.num_envs, device=env.device)
        self.reset_seconds = []
        self.map_hand = inspire_synergy_to_joint_positions
        self.hand_names, self.open_hand, self.closed_hand = HAND_JOINT_NAMES, HAND_OPEN_TARGETS, HAND_CLOSED_TARGETS
        self.robot_body_names = ROBOT_CONTACT_BODY_NAMES

    def part(self, observation, name):
        return observation["policy"][:, self.constants.PUSH_V1_OBSERVATION_SLICES[name]]

    def close(self, actual, expected, atol=2e-6):
        self.torch.testing.assert_close(actual, expected, rtol=0., atol=atol)

    def counters(self):
        return {name: getattr(self.safe, name).clone() for name in ("attempts", "accepted_seed_index", "ik_iterations", "obb_checks")}

    def unchanged_counters(self, before, ids=None):
        ids = self.ids if ids is None else ids
        for name, current in self.counters().items():
            self.close(current[ids], before[name][ids], atol=0.)

    def hold_action(self):
        action = self.torch.zeros(self.env.num_envs, 8, device=self.env.device)
        # Hold the requested reset state rather than feeding load-induced
        # joint deflection back as a new target on every policy step.
        action[:, 6:] = self.hand.synergy_to_action(self.safe.initial_hand_synergy)
        return action

    def cube_midpoint(self):
        task = self.cfg.task
        return tuple((task.object_xy_range_low[i] + task.object_xy_range_high[i])/2 for i in range(2)) + (task.cube_center_height_m,)

    def select_both_sides(self, distance_m=None, angle_offset_rad=0.):
        torch, task = self.torch, self.cfg.task
        if abs(angle_offset_rad) > task.command_angle_jitter_rad+1e-7:
            raise ValueError("Physical fixture angle exceeds the configured command band")
        angle = torch.tensor([angle_offset_rad+(0. if i % 2 == 0 else math.pi)
                              for i in range(self.env.num_envs)], device=self.env.device)
        distance = task.command_distance_range[0] if distance_m is None else distance_m
        position = angle.new_tensor(self.cube_midpoint()).expand(self.env.num_envs, 3).clone()
        if task.command_centered_path:
            base_quaternion = self.geometry.base_link_pose_w(self.env)[1]
            direction_base = torch.stack((angle.sin(), -angle.cos(), torch.zeros_like(angle)), -1)
            direction_w = self.math.quat_apply(base_quaternion, direction_base)
            position[:, 0] = task.table_center[0]+task.command_midpoint_x_offset_m-.5*distance*direction_w[:, 0]
        self.command.set_reset_specs(self.ids, position, angle, distance)

    def check_command(self):
        task, command, torch = self.cfg.task, self.command, self.torch
        local = command.target_pos_w - self.env.scene.env_origins
        low, high = local.new_tensor(task.object_xy_range_low), local.new_tensor(task.object_xy_range_high)
        assert bool(((local[:, :2] >= low-5e-6) & (local[:, :2] <= high+5e-6)).all())
        self.close(local[:, 2], local.new_full((self.env.num_envs,), task.cube_center_height_m), atol=5e-6)
        self.close(command.goal_pos_w, command.target_pos_w + command.distance_m[:, None]*command.direction_w, atol=5e-6)
        angle = torch.remainder(command.angle_rad + math.pi/2, math.pi) - math.pi/2
        assert bool((angle.abs() <= task.command_angle_jitter_rad+2e-6).all())
        lower, upper = task.command_distance_range
        assert bool(((command.distance_m >= lower-1e-6) & (command.distance_m <= upper+1e-6)).all())
        if task.command_centered_path:
            centered_x = task.table_center[0]+task.command_midpoint_x_offset_m-.5*command.distance_m*command.direction_w[:, 0]
            assert bool(((local[:, 0]-centered_x).abs() <= task.command_initial_x_jitter_m+5e-6).all()), "Cube X leaves the actual conditional sampling domain"
        radius = math.sqrt(3)*task.cube_size/2 + task.table_path_margin_m
        extent = local.new_tensor(task.table_size[:2])/2 - radius
        for position in (command.target_pos_w, command.goal_pos_w):
            relative = position[:, :2] - self.env.scene.env_origins[:, :2] - local.new_tensor(task.table_center[:2])
            assert bool((relative.abs() <= extent+5e-6).all()), "Cube path leaves the orientation-safe Table rectangle"

    def expected_target_error(self):
        position, quaternion = self.geometry.control_point_pose_w(self.env)
        desired_w = self.robot.data.root_link_pos_w + self.math.quat_apply(
            self.robot.data.root_link_quat_w, self.arm.desired_c_pose_b[:, :3])
        return self.math.quat_apply_inverse(quaternion, desired_w-position)

    def check_policy(self, observation, reset_ids=None):
        torch, env = self.torch, self.env
        assert observation["policy"].shape == (env.num_envs, 64)
        assert bool(torch.isfinite(observation["policy"]).all())
        assert tuple(env.observation_manager.active_terms["policy"]) == tuple(name for name, _ in self.constants.PUSH_V1_OBSERVATION_LAYOUT)
        self.close(self.part(observation, "surface_header_and_tactile")[:, 0], torch.zeros(env.num_envs, device=env.device), atol=0.)
        frame = env.scene["ee_frame"].data
        for name, position in (("initial_target_relative_position", self.command.target_pos_w), ("current_cube_base_position", self.cube.data.root_pos_w)):
            self.close(self.part(observation, name), self.math.quat_apply_inverse(frame.source_quat_w, position-frame.source_pos_w), atol=5e-6)
        encoded = torch.stack((self.command.angle_rad.sin(), self.command.angle_rad.cos(), self.command.distance_m), -1)
        self.close(self.part(observation, "push_command"), encoded)
        self.close(self.part(observation, "accumulated_translation_error"), self.expected_target_error(), atol=5e-6)
        if reset_ids is not None:
            for name in ("surface_header_and_tactile", "wrist_wrench_c", "last_action", "accumulated_translation_error"):
                value = self.part(observation, name)[reset_ids]
                self.close(value, torch.zeros_like(value), atol=5e-6 if name == "accumulated_translation_error" else 0.)
        else:
            live = (env.episode_length_buf > 0)[:, None]
            bits = self.geometry.palm_tactile_bits(env)
            self.close(self.part(observation, "surface_header_and_tactile")[:, 1:], torch.where(live, bits, torch.zeros_like(bits)), atol=0.)
            measured = self.geometry.wrist_wrench_c(env).measured_c
            expected = torch.cat(((measured[:, :3]/self.cfg.task.wrench_force_observation_scale_n).clamp(-1, 1),
                                  (measured[:, 3:]/self.cfg.task.wrench_moment_observation_scale_nm).clamp(-1, 1)), -1)
            self.close(self.part(observation, "wrist_wrench_c"), torch.where(live, expected, torch.zeros_like(expected)))

    def check_initial(self, observation, ids):
        torch, task = self.torch, self.cfg.task
        self.check_policy(observation, reset_ids=ids)
        self.check_command()
        assert bool(self.safe.check_final(ids).all()), "Reset did not certify actual FK/limits/whole-robot clearance"
        assert abs(task.reset_clearance_margin_m-.004) < 1e-8
        assert not bool(self.safe.last_cube_overlap[ids].any() or self.safe.last_table_overlap[ids].any())
        assert bool(((self.safe.attempts[ids] >= 1) & (self.safe.attempts[ids] <= len(task.reset_joint_seed_offsets))).all())
        self.close(self.robot.data.joint_vel[ids], torch.zeros_like(self.robot.data.joint_vel[ids]))
        self.close(self.cube.data.root_pos_w[ids], self.command.target_pos_w[ids], atol=5e-6)
        self.close(self.cube.data.root_vel_w[ids], torch.zeros_like(self.cube.data.root_vel_w[ids]))
        table_local = self.env.scene["table"].data.root_pos_w[ids] - self.env.scene.env_origins[ids]
        self.close(table_local, table_local.new_tensor(task.table_center).expand(len(ids), -1), atol=5e-6)
        synergy = self.part(observation, "hand_state")[ids]
        low, high = task.initial_hand_open_range
        assert bool(((synergy >= low-1e-6) & (synergy <= high+1e-6)).all())
        joint_ids, _ = self.robot.find_joints(list(self.hand_names), preserve_order=True)
        expected = self.map_hand(synergy, synergy.new_tensor([self.open_hand[name] for name in self.hand_names]), synergy.new_tensor([self.closed_hand[name] for name in self.hand_names]))
        self.close(self.robot.data.joint_pos[ids][:, joint_ids], expected)
        palm, hand_quaternion = self.safe.palm_reference_pose_w(ids)
        expected_h = self.math.quat_mul(self.safe.desired_c_quat_w[ids], self.math.quat_conjugate(self.safe.c_quat_h.expand(len(ids), -1)))
        assert bool(((hand_quaternion*expected_h).sum(-1).abs() >= math.cos(task.reset_orientation_tolerance_rad/2)-1e-6).all())
        normal = self.math.quat_apply(hand_quaternion, palm.new_tensor((0., 1., 0.)).expand(len(ids), -1))
        tolerance = task.reset_orientation_tolerance_rad
        outward_cos = self.safe._finger_outward_cos(ids, hand_quaternion)
        assert bool((outward_cos > task.minimum_finger_outward_cos).all()), "Push v1 fingers must face away from the actual robot base"
        wrist_ids, _ = self.robot.find_joints(["wrist_2_joint"], preserve_order=True)
        wrist = self.robot.data.joint_pos[ids][:, wrist_ids[0]]
        assert bool((wrist.sin() > task.wrist_2_branch_sin_margin).all()), "Push v1 wrist_2 must use the physical positive-sine branch"
        assert bool(((normal*self.command.direction_w[ids]).sum(-1) >= math.cos(tolerance)-1e-6).all()), "Palm normal does not face the command direction"
        state = self.command.state()
        for name in ("contact_seen", "push_seen", "valid_push_contact", "success", "failure"):
            assert not bool(getattr(state, name)[ids].any()), name
        for name in ("raw_contact_time_s", "valid_push_contact_time_s"):
            self.close(self.command.metrics[name][ids], torch.zeros(len(ids), device=self.env.device), atol=0.)

    def reset(self, ids=None, seed=None):
        ids = self.ids if ids is None else ids
        before = self.env._sim_step_counter
        started = time.perf_counter()
        observation, _ = self.env.reset(env_ids=ids, seed=seed)
        self.reset_seconds.append(time.perf_counter()-started)
        assert self.env._sim_step_counter == before, "Reset advanced the physics clock"
        self.check_initial(observation, ids)
        return observation

    def step(self, action, allow_done=False):
        before = self.env._sim_step_counter
        observation, reward, terminated, truncated, _ = self.env.step(action)
        assert self.env._sim_step_counter-before == self.cfg.decimation
        if not allow_done:
            assert not bool((terminated | truncated).any()), f"Unexpected end: terminated={terminated.tolist()}, truncated={truncated.tolist()}"
        self.check_policy(observation)
        assert bool(self.torch.isfinite(reward).all())
        for name in ("table_contacts", "cube_palm_contacts", "cube_table_contacts"):
            history = self.env.scene[name].data.force_matrix_w_history
            assert history.shape[1] == self.cfg.decimation and bool(self.torch.isfinite(history).all()), name
        return observation, terminated, truncated

    def check_partial_reset(self):
        torch = self.torch
        # Build an observable persistent target without stepping physics.
        # This isolates partial-reset history preservation from plant motion.
        delta = torch.zeros(self.env.num_envs, 6, device=self.env.device)
        delta[:, 0] = .5
        self.arm.process_actions_for_envs(delta, self.ids)
        before_error = self.arm.position_target_error_c_m.clone()
        assert bool((before_error.norm(dim=-1) > 1e-4).all())
        before_pose = self.arm.desired_c_pose_b.clone()
        target, goal = self.command.target_pos_w.clone(), self.command.goal_pos_w.clone()
        before_state = {item.name: getattr(self.command.state(), item.name).clone() for item in fields(self.command.state())}
        before_counters = self.counters()
        observation = self.reset(ids=self.ids[:1])
        if self.env.num_envs > 1:
            kept = self.ids[1:]
            self.close(self.arm.desired_c_pose_b[kept], before_pose[kept], atol=0.)
            self.close(self.part(observation, "accumulated_translation_error")[kept], before_error[kept], atol=5e-6)
            self.close(self.command.target_pos_w[kept], target[kept], atol=0.)
            self.close(self.command.goal_pos_w[kept], goal[kept], atol=0.)
            self.unchanged_counters(before_counters, kept)
            after = self.command.state()
            for name, value in before_state.items():
                self.close(getattr(after, name)[kept], value[kept], atol=0.)

    def check_markers(self):
        from pxr import UsdPhysics
        if self.cfg.commands.target_position.debug_vis:
            self.command._debug_vis_callback(None)
            assert self.command._target_marker.count == self.env.num_envs
            assert self.command._goal_marker.count == self.env.num_envs
        if self.cfg.scene.ee_frame.debug_vis:
            frame = self.env.scene["ee_frame"]
            frame._debug_vis_callback(None)
            assert frame.frame_visualizer.count == 3*self.env.num_envs
        for prim in self.env.sim.stage.Traverse():
            if str(prim.GetPath()).startswith("/Visuals/HandManipulation"):
                assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)

    def contracts(self):
        assert self.env.action_manager.total_action_dim == 8
        assert self.env.max_episode_length == 500 and abs(self.env.step_dt-.02) < 1e-8
        assert set(self.env.termination_manager.active_terms) == {"failure", "success", "time_out"}
        assert set(self.env.reward_manager.active_terms) == {"approach", "progress", "backslide", "first_contact", "contact", "alignment", "action_rate", "success", "failure", "roll", "action_excess", "contact_distance", "eef_height"}
        assert set(self.env.scene.sensors) == {"palm_tactile", "ee_frame", "table_contacts", "cube_palm_contacts", "cube_table_contacts"}
        self.select_both_sides()
        observation = self.reset(seed=self.args.seed)
        if self.args.check_boundaries:
            task = self.cfg.task
            base_quaternion = self.geometry.base_link_pose_w(self.env)[1][:1]
            specs = []
            for angle_deg in (-10., 0., 10., 170., 180., 190.):
                angle = math.radians(angle_deg)
                direction_base = base_quaternion.new_tensor([[math.sin(angle), -math.cos(angle), 0.]])
                direction_w = self.math.quat_apply(base_quaternion, direction_base)[0].tolist()
                low, high = sampled_start_bounds(task, direction_w, task.command_distance_range[1])
                print("PUSH_V1_PATH_BOUNDARIES", {"angle_deg": angle_deg, "distance_m": task.command_distance_range[1], "low_xy": low, "high_xy": high}, flush=True)
                specs.extend((x, y, angle) for x, y in product((low[0], high[0]), (low[1], high[1])))
            for offset in range(0, len(specs), self.env.num_envs):
                batch = [specs[(offset+i) % len(specs)] for i in range(self.env.num_envs)]
                self.command.set_reset_specs(self.ids, [(x, y, task.cube_center_height_m) for x, y, _ in batch], [angle for _, _, angle in batch], task.command_distance_range[1])
                observation = self.reset()
        for _ in range(self.args.reset_samples):
            observation = self.reset()
        self.check_markers()
        initial, goal, command = self.command.target_pos_w.clone(), self.command.goal_pos_w.clone(), self.command.command.clone()
        counters = self.counters()
        action = self.hold_action()
        peak_error = 0.
        for _ in range(self.args.steps):
            observation, _, _ = self.step(action)
            peak_error = max(peak_error, float(self.part(observation, "accumulated_translation_error").norm(dim=-1).max()))
            self.close(self.command.target_pos_w, initial, atol=0.)
            self.close(self.command.goal_pos_w, goal, atol=0.)
            self.close(self.command.command, command, atol=0.)
        self.unchanged_counters(counters)
        measured = self.geometry.wrist_wrench_c(self.env).measured_c
        assert bool((measured[:, :3].norm(dim=-1) > 1e-4).all()), "Live F/T unexpectedly removed the Hand load"
        # Move the actual dynamic Cube and independently re-read observations.
        fixed_observation = self.part(observation, "initial_target_relative_position").clone()
        pose = self.cube.data.root_pose_w.clone()
        pose[:, :3] += .02*self.command.direction_w
        self.cube.write_root_pose_to_sim(pose)
        self.cube.write_root_velocity_to_sim(self.torch.zeros(self.env.num_envs, 6, device=self.env.device))
        observation = self.env.observation_manager.compute()
        self.check_policy(observation)
        self.close(self.part(observation, "initial_target_relative_position"), fixed_observation, atol=0.)
        self.close(self.command.target_pos_w, initial, atol=0.)
        self.check_partial_reset()
        if self.args.check_timeout:
            self.reset()
            action, counters = self.hold_action(), self.counters()
            for index in range(500):
                _, terminated, truncated = self.step(action, allow_done=True)
                if index < 499:
                    assert not bool((terminated | truncated).any())
                    self.unchanged_counters(counters)
            assert bool(truncated.all()) and not bool(terminated.any()), "Timeout must truncate at the 500th step"
        print("[PASS] Push v1 64D/8D manager/reset/sensor contracts", {"hold_steps": self.args.steps, "peak_target_error_m": peak_error, "reset_seconds": self.reset_seconds}, flush=True)


class GoalPushPlanner:
    """Plan palm targets without writing robot, Cube, or contact state."""

    def __init__(self, torch, initial_cube, goal, direction, start_palm):
        self.torch = torch
        self.initial, self.goal, self.direction, self.start = (value.clone() for value in (initial_cube, goal, direction, start_palm))
        self.advance = initial_cube.new_zeros(initial_cube.shape[0])
        self.lowered = self.advance.bool()
        self.released = self.advance.bool()
        self.release_target = start_palm.clone()

    def target(self, actual_palm, cube_position, active):
        torch = self.torch
        self.lowered |= (actual_palm[:, 2]-self.start[:, 2]).abs() < .005
        release = active & ~self.released & ((cube_position-self.goal).norm(dim=-1) <= .008)
        self.release_target[release] = actual_palm[release]-.006*self.direction[release]
        self.released |= release
        self.advance += .0012*(active & self.lowered & ~self.released).to(self.advance.dtype)
        delta = cube_position-self.initial
        cross_track = delta-(delta*self.direction).sum(-1, keepdim=True)*self.direction
        cross_track = cross_track.clone()
        cross_track[:, 2] = 0.
        # Follow the measured Cube tangentially to keep the selected palmar
        # surface on its side. Subtracting drift would move the hand farther
        # away from a drifting Cube and could lose contact.
        cross_track *= (.02/cross_track.norm(dim=-1, keepdim=True).clamp_min(.02)).clamp_max(1.)
        desired = self.start+self.advance[:, None]*self.direction+cross_track
        return torch.where(self.released[:, None], self.release_target, desired)


class TerminalPhysicalRecorder:
    """Copy real terminal geometry while Manager still owns the old episode."""

    def __init__(self, smoke):
        self.smoke = smoke
        self.records = [None]*smoke.env.num_envs

    def __enter__(self):
        self.original = self.smoke.command._record_metrics

        def record(state):
            self.original(state)
            s = self.smoke
            terminal = state.success | state.failure
            for index in terminal.nonzero(as_tuple=False).flatten().tolist():
                if self.records[index] is not None:
                    continue
                self.records[index] = {
                    "state": {item.name: getattr(state, item.name)[index].clone() for item in fields(state)},
                    "cube_position_w": s.cube.data.root_pos_w[index].clone(),
                    "cube_velocity_w": s.cube.data.root_lin_vel_w[index].clone(),
                    "initial_cube_w": s.command.target_pos_w[index].clone(),
                    "goal_w": s.command.goal_pos_w[index].clone(),
                    "counter": {name: value[index].clone() for name, value in s.counters().items()},
                }
        self.smoke.command._record_metrics = record
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.smoke.command._record_metrics = self.original


class OscPushFixture:
    """Script the actual OSC; Cube dynamics and sensor histories stay unforced."""

    def __init__(self, smoke):
        self.smoke = smoke

    def contact_pose(self):
        safe = self.smoke.safe
        reference = getattr(safe, "contact_reference_pose_w", None)
        return reference() if reference is not None else safe.palm_reference_pose_w()

    def approach_start(self, initial, direction):
        """Center the selected contact pad on the Cube's actual side.

        A palmar thumb is offset from the central palm in the tangent axis.
        Advancing its reset X unchanged only grazes the Cube's corner.
        The production approach observation/reward uses this same Cube side.
        """
        start = self.contact_pose()[0].clone()
        delta = initial-start
        tangent = delta-(delta*direction).sum(-1, keepdim=True)*direction
        tangent[:, 2] = 0.
        start += tangent
        start[:, 2] = initial[:, 2]+self.smoke.cfg.task.contact_palm_height_m
        return start

    def action_toward(self, desired_palm, desired_c_quaternion):
        s = self.smoke
        position, quaternion = s.geometry.control_point_pose_w(s.env)
        hand_quaternion = s.math.quat_mul(desired_c_quaternion, s.math.quat_conjugate(s.safe.c_quat_h.expand(s.env.num_envs, -1)))
        reference_offset = getattr(s.safe, "contact_reference_offset_h", None)
        offset = s.safe.c_offset_h - (
            reference_offset() if reference_offset is not None else s.safe.palm_reference_h
        )
        desired_c = desired_palm + s.math.quat_apply(hand_quaternion, offset.expand(s.env.num_envs, -1))
        persistent_w = s.robot.data.root_link_pos_w + s.math.quat_apply(s.robot.data.root_link_quat_w, s.arm.desired_c_pose_b[:, :3])
        action = s.hold_action()
        action[:, :3] = (s.math.quat_apply_inverse(quaternion, desired_c-persistent_w)/action.new_tensor(s.cfg.actions.arm_action.translation_scale)).clamp(-1, 1)
        reference_w = s.math.quat_mul(s.robot.data.root_link_quat_w, s.arm.reference_c_quat_b)
        error = s.math.quat_mul(s.math.quat_conjugate(reference_w), desired_c_quaternion)
        action[:, 3:6] = (s.math.axis_angle_from_quat(error)/action.new_tensor(s.cfg.actions.arm_action.rotation_scale)).clamp(-1, 1)
        return action

    def run(self):
        s, torch = self.smoke, self.smoke.torch
        s.select_both_sides(angle_offset_rad=math.radians(s.args.physical_angle_offset_deg))
        s.reset()
        counters, initial = s.counters(), s.cube.data.root_pos_w.clone()
        direction = s.command.direction_w.clone()
        preferred = s.safe.desired_c_quat_w.clone()
        start_palm = self.approach_start(initial, direction)
        peak = torch.zeros(s.env.num_envs, device=s.env.device)
        contact = torch.zeros(s.env.num_envs, dtype=torch.bool, device=s.env.device)
        advance = torch.zeros_like(peak)
        lowered = torch.zeros_like(contact)
        for index in range(s.args.physical_steps):
            actual_palm = self.contact_pose()[0]
            lowered |= (actual_palm[:, 2]-start_palm[:, 2]).abs() < .005
            moving = lowered & ~(contact & (peak >= .015))
            advance += .0008*moving.float()
            desired = start_palm + advance[:, None]*direction
            s.step(self.action_toward(desired, preferred))
            state = s.command.state()
            contact |= state.valid_push_contact
            peak = torch.maximum(peak, ((s.cube.data.root_pos_w-initial)*direction).sum(-1))
            if (s.args.fixture_verbose or s.args.verbose) and index % 25 == 0:
                print("PUSH_V1_PHYSICS", {"step": index, "lowered": lowered.tolist(), "contact": contact.tolist(), "forward_m": peak.tolist(), "metrics": {name: values.tolist() for name, values in s.command.metrics.items()}}, flush=True)
            if bool((contact & (peak >= .015)).all()):
                break
        assert bool(contact.all()), "Scripted OSC produced no supported side contact"
        assert bool((peak >= .015).all()), f"Actual OSC Cube displacement stayed below 1.5cm: {peak.tolist()}"
        desired = start_palm + (advance-.006).clamp_min(0)[:, None]*direction
        settled = torch.zeros(s.env.num_envs, dtype=torch.long, device=s.env.device)
        for _ in range(35):
            s.step(self.action_toward(desired, preferred))
            state = s.command.state()
            stable = state.grounded & (s.cube.data.root_lin_vel_w.norm(dim=-1) <= s.cfg.task.success_speed_m_s)
            settled = torch.where(stable, settled+1, torch.zeros_like(settled))
        assert bool((settled >= math.ceil(s.cfg.task.success_hold_time_s/s.env.step_dt)).all()), settled.tolist()
        s.unchanged_counters(counters)
        print("[PASS] Scripted Push v1 OSC side contact/displacement/settling", {"steps": index+1, "forward_m": peak.tolist(), "settled_steps": settled.tolist()}, flush=True)

    def run_goal(self):
        s, torch = self.smoke, self.smoke.torch
        distance = s.args.physical_goal_distance
        assert s.cfg.task.command_distance_range[0] <= distance <= s.cfg.task.command_distance_range[1]
        s.select_both_sides(distance, math.radians(s.args.physical_angle_offset_deg))
        s.reset()
        initial, goal = s.command.target_pos_w.clone(), s.command.goal_pos_w.clone()
        direction, preferred = s.command.direction_w.clone(), s.safe.desired_c_quat_w.clone()
        counters = s.counters()
        start = self.approach_start(initial, direction)
        planner = GoalPushPlanner(torch, initial, goal, direction, start)
        succeeded = torch.zeros(s.env.num_envs, device=s.env.device, dtype=torch.bool)
        with TerminalPhysicalRecorder(s) as recorder:
            for index in range(s.args.physical_steps):
                active = ~succeeded
                actual_palm = self.contact_pose()[0]
                desired = planner.target(actual_palm, s.cube.data.root_pos_w, active)
                action = self.action_toward(desired, preferred)
                # Manager has already reset completed rows. Hold their fresh
                # episode while the other original episodes continue.
                action[succeeded] = s.hold_action()[succeeded]
                _, terminated, truncated = s.step(action, allow_done=True)
                success = s.env.termination_manager.get_term("success").clone() & active
                failure = s.env.termination_manager.get_term("failure").clone() & active
                timeout = s.env.termination_manager.get_term("time_out").clone() & active
                assert not bool((failure | timeout | (truncated & active)).any()), f"Full-goal fixture failed: failure={failure.tolist()}, timeout={timeout.tolist()}"
                assert not bool((terminated & active & ~success).any()), "An original episode terminated without success"
                succeeded |= success
                remaining = s.ids[~succeeded]
                s.unchanged_counters(counters, remaining)
                s.close(s.command.target_pos_w[remaining], initial[remaining], atol=0.)
                s.close(s.command.goal_pos_w[remaining], goal[remaining], atol=0.)
                if s.args.fixture_verbose and index % 25 == 0:
                    print("PUSH_V1_FULL_GOAL", {"step": index, "success": succeeded.tolist(), "released": planner.released.tolist(),
                        "goal_error_m": (s.cube.data.root_pos_w-goal).norm(dim=-1).tolist(),
                        "metrics": {name: value.tolist() for name, value in s.command.metrics.items()}}, flush=True)
                if bool(succeeded.all()):
                    break
            assert bool(succeeded.all()), f"Full command goal did not terminate successfully within {s.args.physical_steps} steps: {succeeded.tolist()}"
            summaries = []
            for env_id, record in enumerate(recorder.records):
                assert record is not None, f"Missing pre-reset physical record for env {env_id}"
                state = record["state"]
                assert bool(state["success"] & ~state["failure"] & state["grounded"] & state["contact_seen"])
                assert float(state["cube_goal_distance_m"]) <= s.cfg.task.success_distance_m+1e-6
                assert float(record["cube_velocity_w"].norm()) <= s.cfg.task.success_speed_m_s+1e-6
                assert float(state["settled_time_s"]) >= s.cfg.task.success_hold_time_s-1e-6
                s.close(record["initial_cube_w"], initial[env_id], atol=0.)
                s.close(record["goal_w"], goal[env_id], atol=0.)
                for name, value in record["counter"].items():
                    s.close(value, counters[name][env_id], atol=0.)
                summaries.append({"env_id": env_id, "goal_error_m": float(state["cube_goal_distance_m"]),
                    "speed_m_s": float(record["cube_velocity_w"].norm()), "settled_time_s": float(state["settled_time_s"]),
                    "cube_position_w": record["cube_position_w"].tolist(), "goal_w": record["goal_w"].tolist()})
        print("[PASS] Scripted Push v1 OSC full-goal physical success", {"steps": index+1, "distance_m": distance, "episodes": summaries}, flush=True)


def run(args):
    import gymnasium as gym
    import torch
    from isaaclab.utils import math as math_utils
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import hand_manipulation_test  # noqa: F401
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs, cfg.seed = args.num_envs, args.seed
    if args.device is not None:
        cfg.sim.device = args.device
    cfg.commands.target_position.debug_vis = args.debug_vis and not args.disable_markers
    cfg.scene.ee_frame.debug_vis = args.debug_vis and not args.disable_markers
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        smoke = PushV1Smoke(env, args, torch, math_utils)
        if not args.physical_only:
            smoke.contracts()
        if not args.skip_physical_fixture:
            fixture = OscPushFixture(smoke)
            fixture.run_goal() if args.physical_goal else fixture.run()
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher
    args = build_parser(AppLauncher).parse_args()
    validate_args(args)
    simulation_app = AppLauncher(args, fast_shutdown=True).app
    exit_code = 0
    try:
        run(args)
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        import omni.kit.app
        sys.stdout.flush()
        sys.stderr.flush()
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
