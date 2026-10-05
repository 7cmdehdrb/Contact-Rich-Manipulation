"""Actual quaternion/base/table geometry gates inward rolls in both reset terms."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hand_manipulation_test.action_math import quaternion_from_rotation_vector, quaternion_multiply, quaternion_rotate


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "hand_manipulation_test"


def _gate_class(kind):
    namespace = {"torch": torch, "math": math, "quaternion_rotate": quaternion_rotate}
    for relative, name, methods in (
        ("mdp/contact_events.py", "ContactSafePoseReset", ("_finger_outward_cos", "_check_pose_and_limits")),
        ("mdp/push_events.py", "PushSafePoseReset", ("_check_pose_and_limits",)),
    ):
        tree = ast.parse((PACKAGE_ROOT / relative).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        cls.decorator_list = []
        cls.bases = [] if name == "ContactSafePoseReset" else [ast.Name(id="ContactSafePoseReset", ctx=ast.Load())]
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), relative, "exec"), namespace)
    return namespace["ContactSafePoseReset" if kind == "contact" else "PushSafePoseReset"]


def _fixture(kind):
    fixture = _gate_class(kind)()
    ids = torch.arange(4)
    # Right and left outward orientations have opposite thumb directions.
    angle = torch.tensor([[0., -math.pi/2, 0.], [math.pi/math.sqrt(2), 0., -math.pi/math.sqrt(2)]], dtype=torch.float64)
    outward_quaternion = quaternion_from_rotation_vector(angle)
    if kind == "contact":
        tilt = quaternion_from_rotation_vector(torch.tensor([[0., 0., -.55]], dtype=angle.dtype))
        outward_quaternion = quaternion_multiply(outward_quaternion, tilt)
    roll = quaternion_from_rotation_vector(torch.tensor([[0., math.pi, 0.]], dtype=angle.dtype))
    quaternion = torch.cat((outward_quaternion, quaternion_multiply(outward_quaternion, roll)))
    normal = quaternion_rotate(quaternion, angle.new_tensor([0., 1., 0.]).expand(4, -1))
    translation = angle.new_tensor([[0., 0., 0.], [5., -2.5, 0.], [-48.75, 37.5, 0.], [50., 50., 0.]])
    base = translation + angle.new_tensor([.02, -.03, .79505])
    table = base + angle.new_tensor([-.75, 0., .23495])
    palm = translation + angle.new_tensor([-.70, 0., 1.11])
    scene = {"table": SimpleNamespace(data=SimpleNamespace(root_pos_w=table)),
             "target_object": SimpleNamespace(data=SimpleNamespace(root_pos_w=palm + .2 * normal))}
    task = SimpleNamespace(reset_joint_limit_margin_rad=.035, reset_wrist_3_range_rad=(-2.8, 2.8),
                           reset_position_tolerance_m=.003, reset_orientation_tolerance_rad=.05)
    fixture._env = SimpleNamespace(scene=scene, cfg=SimpleNamespace(task=task))
    fixture.robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=base,
        joint_pos=torch.zeros(4, 6, dtype=angle.dtype),
        soft_joint_pos_limits=angle.new_tensor([-3., 3.]).expand(4, 6, 2)))
    fixture.arm_joint_ids, fixture.wrist_local_index = list(range(6)), 5
    fixture.position_error_m = torch.zeros(4, dtype=angle.dtype)
    fixture.orientation_error_rad = torch.zeros_like(fixture.position_error_m)
    fixture.palm_alignment_cos = torch.zeros_like(fixture.position_error_m)
    fixture.palm_reference_pose_w = lambda index: (palm[index], quaternion[index])
    fixture._pose_errors = lambda index: (torch.zeros(len(index), 3, dtype=angle.dtype), torch.zeros(len(index), 3, dtype=angle.dtype))
    fixture.direction_w = normal
    return fixture, ids, normal


@pytest.mark.parametrize("kind", ("contact", "push"))
def test_final_pose_gate_rejects_inward_fingers_even_when_palm_normal_and_ik_are_valid(kind):
    fixture, ids, normal = _fixture(kind)
    if kind == "contact":
        assert bool((normal[:, 2].abs() > .5).all()), "Contact must exercise strict 3D facing"
    else:
        torch.testing.assert_close(normal[:, 2], torch.zeros(4, dtype=normal.dtype), atol=1e-14, rtol=0)
    valid = fixture._check_pose_and_limits(ids)
    assert valid.tolist() == [True, True, False, False]
    torch.testing.assert_close(fixture.palm_alignment_cos, torch.ones(4, dtype=normal.dtype))
    assert fixture._finger_outward_cos(torch.tensor([1, 3])).tolist() == pytest.approx([1., -1.])


def test_outward_gate_uses_actual_table_and_base_positions_instead_of_assuming_world_minus_x():
    fixture, ids, _ = _fixture("push")
    fixture._env.scene["table"].data.root_pos_w[:, 0] = fixture.robot.data.root_link_pos_w[:, 0] + .75
    cosine = fixture._finger_outward_cos(ids)
    torch.testing.assert_close(cosine, torch.tensor([-1., -1., 1., 1.], dtype=cosine.dtype))
