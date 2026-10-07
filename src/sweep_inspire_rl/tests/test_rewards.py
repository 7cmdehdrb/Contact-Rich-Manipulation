"""Behavioral checks for the right-only reward adaptation without Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "sweep_inspire_rl"


class Scene(dict):
    num_envs = 1


def _functions(path, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ImportFrom))
             and (isinstance(n, ast.FunctionDef) or n.module == "__future__")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


geometry = _functions(PACKAGE / "mdp/events.py", {"torch": torch, "math": __import__("math"), "SceneEntityCfg": lambda x: None})
source = _functions(PACKAGE.parents[2] / "example/Sweep-Policy/sweeping_policy/mdp/reward_random_sweep.py", {
    "torch": torch, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
})
reward = _functions(PACKAGE / "mdp/rewards.py", {
    "torch": torch, "matrix_from_quat": geometry["_rotation_from_quaternion"],
    "collision_aabbs": geometry["collision_aabbs"],
    "PALM_SURFACE_POINT_H": (0.0004, 0.0127, 0.1196),
    "CONTROL_POINT_OFFSET_H": (0.0, 0.05, 0.10),
    "palm_tactile_bits": lambda env: env.palm_bits,
    "source_pushing_target": source["pushing_target"],
})


def _env(velocity=0.075):
    target = torch.tensor([[-0.6, 0.0, 1.05]])
    state = torch.zeros(1, 1, 13)
    state[:, 0, :3] = target
    state[:, 0, 3] = 1.0
    scene = {
        "robot": SimpleNamespace(),
        "wrist_frame": SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.tensor([[[-0.6, -0.06, 1.17]]]))),
        "object_collection": SimpleNamespace(data=SimpleNamespace(
            object_pos_w=target[:, None], object_link_state_w=state,
            object_lin_vel_w=torch.tensor([[[0.0, velocity, 0.0]]])
        )),
        "ee_frame": SimpleNamespace(data=SimpleNamespace(
            # C is ahead of the physical palmar pad and can pass the target
            # root in Y while the actual pad still touches its upstream face.
            target_pos_w=torch.tensor([[[-0.5804, 0.0073, 1.1696]]]),
            target_quat_w=torch.tensor([[[2.0**-0.5, 0.0, -2.0**-0.5, 0.0]]]),
        )),
    }
    return SimpleNamespace(
        scene=Scene(scene), num_envs=1, target_id=torch.zeros(1, 1, dtype=torch.long),
        sweep_dir=torch.tensor([[0.0, 0.18, 0.0]]), target_width=torch.tensor([[0.06]]),
        _target_collision_local_center=torch.tensor([0.0, 0.0, 0.08]),
        _target_collision_half_extent=torch.tensor([0.03, 0.03, 0.08]),
        palm_bits=torch.tensor([[1.0] + [0.0] * 16]),
        command_manager=SimpleNamespace(get_command=lambda _: target + torch.tensor([[0.0, 0.18, 0.0]])),
    )


def test_reaching_peak_matches_raised_pre_push_pose():
    env = _env()
    torch.testing.assert_close(reward["hand_reaching"](env), torch.ones(1))
    env.scene["ee_frame"].data.target_pos_w[:, :, 2] += 0.10
    assert reward["hand_reaching"](env).item() < 0.4


def _source_gate_pose(env):
    env.scene["ee_frame"].data.target_pos_w[:] = torch.tensor([[[-0.62, -0.06, 1.14]]])
    return env


@pytest.mark.parametrize("velocity", [-0.11, -0.075, -0.05, 0.0, 0.05, 0.075, 0.1, 0.11])
@pytest.mark.parametrize("goal_distance", [0.0, 0.02, 0.18])
def test_push_reward_matches_source_for_velocity_and_goal_branches(velocity, goal_distance):
    env = _source_gate_pose(_env(velocity))
    env.palm_bits.zero_()
    env.command_manager.get_command = lambda _: env.scene["object_collection"].data.object_pos_w[:, 0] + torch.tensor([[0.0, goal_distance, 0.0]])
    torch.testing.assert_close(reward["pushing_target"](env), source["pushing_target"](env, "target_goal_pos"), rtol=0, atol=0)


def test_source_gate_does_not_require_contact_or_alignment():
    env = _source_gate_pose(_env())
    env.palm_bits.zero_()
    env.scene["ee_frame"].data.target_quat_w[:] = torch.tensor([[[0.0, 2.0**-0.5, 0.0, 2.0**-0.5]]])
    assert reward["palm_alignment"](env).item() < 0.1
    torch.testing.assert_close(reward["pushing_target"](env), torch.tensor([0.5]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("frame", ["ee_frame", "wrist_frame"])
def test_source_gate_requires_both_eef_and_wrist_near_offset(frame):
    env = _source_gate_pose(_env())
    assert reward["pushing_target"](env).item() > 0
    env.scene[frame].data.target_pos_w[:, :, 1] += 0.06
    assert reward["pushing_target"](env).item() == 0


def test_goal_region_bypasses_source_gate_and_contact():
    env = _env()
    env.palm_bits.zero_()
    env.scene["ee_frame"].data.target_pos_w += 1.0
    env.scene["wrist_frame"].data.target_pos_w += 1.0
    env.command_manager.get_command = lambda _: env.scene["object_collection"].data.object_pos_w[:, 0].clone()
    torch.testing.assert_close(reward["pushing_target"](env), torch.tensor([2.0]))


def test_source_speed_shaping_is_symmetric_in_y():
    right = reward["pushing_target"](_source_gate_pose(_env(0.075)))
    left = reward["pushing_target"](_source_gate_pose(_env(-0.075)))
    torch.testing.assert_close(right, left)
    assert right.item() > 0


@pytest.mark.parametrize("distance,enabled", [(0.04, True), (0.0519, True), (0.0521, False)])
def test_inspire_eef_gate_uses_5_2_cm(distance, enabled):
    env = _source_gate_pose(_env())
    env.scene["ee_frame"].data.target_pos_w[:, :, 2] += distance
    assert (reward["pushing_target"](env).item() > 0) == enabled


def test_old_x_z_mismatch_fits_new_gate_when_y_matches():
    env = _source_gate_pose(_env())
    env.scene["ee_frame"].data.target_pos_w[:, :, 0] += 0.0396
    env.scene["ee_frame"].data.target_pos_w[:, :, 2] += 0.0296
    assert source["pushing_target"](env, "target_goal_pos").item() == 0
    assert reward["pushing_target"](env).item() > 0


def test_inspire_wrist_gate_keeps_original_4_cm_limit():
    env = _source_gate_pose(_env())
    env.scene["wrist_frame"].data.target_pos_w[:, :, 1] += 0.045
    assert reward["pushing_target"](env).item() == 0


def test_upstream_surface_uses_yawed_actual_collider_for_reaching():
    env = _env()
    env._target_collision_half_extent[0] = 0.06
    env.scene["object_collection"].data.object_link_state_w[:, 0, 3:7] = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    torch.testing.assert_close(reward["upstream_surface_position"](env), torch.tensor([-0.06]))


def test_arm_velocity_limit_does_not_depend_on_articulation_order():
    env = _env()
    env.scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.tensor([[99.0, 0.0, 0.5, 0.0]])))
    env.action_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(_joint_ids=[3, 1, 2]))
    assert not reward["hand_velocity_limit"](env).item()
    env.scene["robot"].data.joint_vel[0, 2] = 1.01
    assert reward["hand_velocity_limit"](env).item()


def test_gate_diagnostics_capture_source_gate_and_survive_reset_mutation():
    env = _source_gate_pose(_env())
    env.sweep_gate_diagnostic_env = 0
    env.episode_length_buf = torch.tensor([50])
    env.sensor_data_fresh = torch.tensor([True])
    done = torch.tensor([True])
    env.termination_manager = SimpleNamespace(active_terms=["time_out"], get_term=lambda _: done)
    env.scene["palm_tactile"] = SimpleNamespace(
        body_names=["inspire_palm_force_sensor"],
        data=SimpleNamespace(net_forces_w=torch.zeros(1, 1, 3)),
    )
    env.palm_bits.zero_()
    expected = source["pushing_target"](env, "target_goal_pos")
    torch.testing.assert_close(reward["pushing_target"](env), expected)
    snapshot = env.sweep_gate_diagnostics
    assert snapshot["near_hand"].item()
    assert snapshot["near_wrist"].item()
    assert not snapshot["palm_contact"].item()
    assert snapshot["gate"].item()
    assert snapshot["done_time_out"].item()
    env.episode_length_buf.zero_()
    env.scene["object_collection"].data.object_pos_w.zero_()
    done.zero_()
    assert snapshot["episode_step"].item() == 50
    assert snapshot["object_z_w_m"].item() > 1.0
    assert snapshot["done_time_out"].item()
