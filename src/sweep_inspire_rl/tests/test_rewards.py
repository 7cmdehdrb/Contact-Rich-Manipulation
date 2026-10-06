"""Behavioral checks for the right-only reward adaptation without Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


PACKAGE = Path(__file__).resolve().parents[1] / "sweep_inspire_rl"


def _functions(path, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ImportFrom))
             and (isinstance(n, ast.FunctionDef) or n.module == "__future__")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


geometry = _functions(PACKAGE / "mdp/events.py", {"torch": torch, "math": __import__("math"), "SceneEntityCfg": lambda x: None})
reward = _functions(PACKAGE / "mdp/rewards.py", {
    "torch": torch, "matrix_from_quat": geometry["_rotation_from_quaternion"],
    "collision_aabbs": geometry["collision_aabbs"],
    "PALM_SURFACE_POINT_H": (0.0004, 0.0127, 0.1196),
    "CONTROL_POINT_OFFSET_H": (0.0, 0.05, 0.10),
    "palm_tactile_bits": lambda env: env.palm_bits,
})


def _env(velocity=0.075):
    target = torch.tensor([[-0.6, 0.0, 1.05]])
    state = torch.zeros(1, 1, 13)
    state[:, 0, :3] = target
    state[:, 0, 3] = 1.0
    scene = {
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
        scene=scene, num_envs=1, target_width=torch.tensor([[0.06]]),
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


def test_dorsal_orientation_is_penalized_and_cannot_receive_progress():
    env = _env()
    torch.testing.assert_close(reward["palm_alignment"](env), torch.ones(1))
    assert reward["pushing_target"](env).item() > 0.0
    # A 180-degree local X rotation reverses the palmar normal and fingers.
    env.scene["ee_frame"].data.target_quat_w[:] = torch.tensor([[[0.0, 2.0**-0.5, 0.0, 2.0**-0.5]]])
    assert reward["palm_alignment"](env).item() < 0.1
    assert reward["pushing_target"](env).item() == 0.0


def test_reverse_object_velocity_is_penalized():
    right = reward["pushing_target"](_env(velocity=0.075))
    left = reward["pushing_target"](_env(velocity=-0.075))
    assert right.item() > 0.0
    assert left.item() < 0.0


def test_virtual_control_point_can_cross_center_while_palm_is_upstream():
    env = _env()
    assert env.scene["ee_frame"].data.target_pos_w[0, 0, 1] > 0.0
    palm = reward["palm_surface_position"](env)
    torch.testing.assert_close(palm, torch.tensor([[-0.6, -0.03, 1.17]]))
    assert reward["pushing_target"](env).item() > 0.0


def test_upstream_surface_uses_yawed_actual_collider_not_nominal_width():
    env = _env()
    env._target_collision_half_extent[0] = 0.06
    env.scene["object_collection"].data.object_link_state_w[:, 0, 3:7] = torch.tensor(
        [[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]]
    )
    torch.testing.assert_close(reward["upstream_surface_position"](env), torch.tensor([-0.06]))


def test_downstream_hand_cannot_receive_progress_reward():
    env = _env()
    env.scene["ee_frame"].data.target_pos_w[:, :, 1] = 0.08
    assert reward["pushing_target"](env).item() == 0.0


def test_goal_kernel_keeps_source_terminal_region_reward():
    env = _env()
    target = env.scene["object_collection"].data.object_pos_w[:, 0]
    env.command_manager.get_command = lambda _: target.clone()
    torch.testing.assert_close(reward["pushing_target"](env), torch.tensor([2.0]))


def test_thumb_or_finger_contact_without_palm_pad_gets_no_push_reward():
    env = _env()
    env.palm_bits[:, 0] = 0.0
    env.palm_bits[:, 1:] = 1.0
    assert reward["pushing_target"](env).item() == 0.0


def test_goal_reached_through_carrier_cannot_bypass_palm_contact_gate():
    env = _env()
    env.palm_bits.zero_()
    env.command_manager.get_command = lambda _: env.scene["object_collection"].data.object_pos_w[:, 0].clone()
    assert reward["pushing_target"](env).item() == 0.0


def test_arm_velocity_limit_does_not_depend_on_articulation_order():
    env = _env()
    env.scene["robot"] = SimpleNamespace(data=SimpleNamespace(joint_vel=torch.tensor([[99.0, 0.0, 0.5, 0.0]])))
    env.action_manager = SimpleNamespace(get_term=lambda _: SimpleNamespace(_joint_ids=[3, 1, 2]))
    assert not reward["hand_velocity_limit"](env).item()
    env.scene["robot"].data.joint_vel[0, 2] = 1.01
    assert reward["hand_velocity_limit"](env).item()
