"""CPU boundary checks for Revision's success, contact and physical-bottom contract."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch


PACKAGE = Path(__file__).resolve().parents[1] / "inspire_sweep_baseline"


def definitions(relative, **bindings):
    path = PACKAGE / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {"torch": torch, "math": math, "ManagerTermBase": object, **bindings}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def rewards(**bindings):
    return definitions("mdp/reward_functions/revision.py", GOAL_SUCCESS_RADIUS=0.03,
                       HOLD_OUTER_RADIUS=0.08, SHELF_SURFACE_HEIGHT=1.05, **bindings)


def test_target_contact_requires_fresh_force_and_strict_fixed_eef_height():
    force = torch.tensor([[0.01, 0, 0]] * 7 + [[0, 0, 0]], dtype=torch.float64)
    force[5, 0] = float("nan")
    ceiling = 1.05 + 0.12
    z = torch.tensor([ceiling - 1e-6, ceiling, ceiling + 1e-6, float("nan"),
                      ceiling - 1e-6, ceiling - 1e-6, ceiling - 1e-6, 0.1], dtype=torch.float64)
    frame = torch.zeros(8, 1, 3, dtype=torch.float64)
    frame[:, 0, 2] = z
    env = NS(
        cfg=NS(contact_height_limit=0.12),
        scene={"ee_frame": NS(data=NS(target_pos_w=frame))},
        sensor_data_fresh=torch.tensor([True, True, True, True, False, True, True, True]),
        _revision_initial_shelf_pos=torch.zeros(8, 3, dtype=torch.float64),
    )
    func = rewards(target_hand_contact_force_w=lambda env: force)["target_contact_below_height"]
    assert func(env).tolist() == [1, 0, 0, 0, 0, 0, 1, 0]
    # The limit is fixed: neither target shape nor the current shelf pose is read.
    env.scene["shelf"] = NS(data=NS(root_pos_w=torch.full((8, 3), 100.0)))
    env._revision_bounds_t = torch.full((8, 10, 2, 3), 100.0)
    assert func(env).tolist() == [1, 0, 0, 0, 0, 0, 1, 0]


def test_hold_is_continuous_at_three_cm_and_zero_at_eight_cm():
    distances = torch.tensor([0.0, 0.03 - 1e-8, 0.03, 0.03 + 1e-8, 0.08 - 1e-8, 0.08, 0.08 + 1e-8], dtype=torch.float64)
    func = rewards(target_measurements=lambda env, name: (
        distances, torch.zeros_like(distances), torch.ones(7, dtype=torch.bool),
    ))["goal_hold"]
    result = func(NS(cfg=NS(goal_success_radius=0.03)))
    assert result[0] == 2.0
    assert result[2] == pytest.approx(0.5)
    assert result[1:4].tolist() == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)
    assert result[4] > 0
    assert result[5:].tolist() == [0, 0]
    with pytest.raises(ValueError, match="synchronized"):
        func(NS(cfg=NS(goal_success_radius=0.02)))


def test_timeout_partition_and_bonus_include_exact_success_boundary():
    positions = torch.tensor([[0.03 - 1e-8, 0, 1.05], [0.03, 0, 1.05],
                              [0.03 + 1e-8, 0, 1.05], [float("nan"), 0, 1.05]], dtype=torch.float64)
    env = NS(command_manager=NS(get_command=lambda _: torch.zeros(4, 7, dtype=torch.float64)),
             step_dt=0.02, record_goal_accuracy_metrics=lambda: None)
    timeouts = torch.ones(4, dtype=torch.bool)
    functions = definitions("mdp/termination_functions/revision.py", time_out=lambda env: timeouts,
                            goal_radius=lambda env: 0.03, _object_position=lambda env: positions)
    success = functions["success_at_timeout"](env)
    miss = functions["unsuccessful_timeout"](env)
    assert success.tolist() == [True, True, False, False]
    assert torch.equal(success ^ miss, timeouts)
    env.termination_manager = NS(get_term=lambda _: success)
    bonus = rewards()["successful_timeout_bonus"](env)
    assert (bonus * env.step_dt * 30.0).tolist() == [30.0, 30.0, 0.0, 0.0]
    timeouts[:] = False
    assert not functions["success_at_timeout"](env).any()
    assert not functions["unsuccessful_timeout"](env).any()


def _quat_apply(q, v):
    t = 2.0 * torch.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.cross(q[..., 1:], t, dim=-1)


def _quat_conjugate(q):
    return q * torch.tensor([1, -1, -1, -1], dtype=q.dtype)


def _quat_mul(a, b):
    return torch.cat((a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdim=True),
                      a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:]
                      + torch.cross(a[..., 1:], b[..., 1:], dim=-1)), dim=-1)


def test_bottom_is_invariant_to_asset_root_offset_and_uses_actual_orientation():
    q = torch.tensor([[[1., 0, 0, 0], [1., 0, 0, 0],
                       [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0]]], dtype=torch.float64)
    bounds = torch.tensor([[
        [[-.03, -.04, 0], [.03, .04, .10]],
        [[-.03, -.04, -.05], [.03, .04, .05]],
        [[-.03, -.04, -.05], [.03, .04, .05]],
    ]], dtype=torch.float64)
    position = torch.tensor([[[0, 0, 1.05], [0, 0, 1.10], [0, 0, 1.08]]], dtype=torch.float64)
    shelf = NS(root_pos_w=torch.zeros(1, 3, dtype=torch.float64),
               root_quat_w=torch.tensor([[1., 0, 0, 0]], dtype=torch.float64))
    env = NS(scene={"object_collection": NS(data=NS(object_pos_w=position, object_quat_w=q)),
                    "shelf": NS(data=shelf)}, _revision_bounds_t=bounds)
    functions = definitions("mdp/common/revision_goal.py", SHELF_SURFACE_HEIGHT=1.05,
        quat_apply=_quat_apply, quat_apply_inverse=lambda q, v: _quat_apply(_quat_conjugate(q), v),
        quat_conjugate=_quat_conjugate, quat_mul=_quat_mul)
    _, bottom = functions["object_geometry_in_shelf"](env)
    assert bottom.tolist()[0] == pytest.approx([1.05, 1.05, 1.05])


def test_revision_cfgs_never_inherit_or_instantiate_other_environment_cfgs():
    paths = [PACKAGE / "envs/revision_envs.py", PACKAGE / "agent/revision_ppo_cfg.py"]
    paths += [PACKAGE / f"mdp/{kind}/revision.py" for kind in
              ("actions", "commands", "rewards", "terminations", "curricula")]
    forbidden = ("Final", "Holding", "ContactSweep")
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert not any(ast.unparse(base).startswith(forbidden) for base in node.bases), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert not (node.func.id.endswith("Cfg") and node.func.id.startswith(forbidden)), path
