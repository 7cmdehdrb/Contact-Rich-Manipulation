"""Exercise the real RL step/reward loop: shelf cost survives terminal reset."""

from _source_units import source_text

import ast
from types import MethodType, SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT, PROJECT_ROOT
from test_clutter_v1_rewards import v1_mdp  # noqa: F401
from test_omni_configuration import _execute_definitions


def test_shelf_failure_config_uses_event_weight_minus_three(v1_mdp):
    namespace = {
        "configclass": lambda cls: cls,
        "RewardTermCfg": lambda **kwargs: SimpleNamespace(**kwargs),
        "ContactSweepClutterRewardsCfg": object,
        **vars(v1_mdp.mdp),
    }
    cfg = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_v1/rewards_cfg.py", namespace,
    ).ContactSweepClutterV1RewardsCfg
    assert cfg.shelf_failure.func is v1_mdp.mdp.shelf_failure
    assert cfg.shelf_failure.weight == -3.0


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_terminal_step_contains_minus_three_before_partial_reset(v1_mdp, dt):
    trace = []
    noop = lambda *args, **kwargs: None
    env = SimpleNamespace(
        num_envs=3, device="cpu", step_dt=dt, physics_dt=dt / 2,
        cfg=SimpleNamespace(decimation=2, sim=SimpleNamespace(render_interval=2)),
        _sim_step_counter=0, common_step_counter=0,
        episode_length_buf=torch.zeros(3, dtype=torch.long), extras={},
        action_manager=SimpleNamespace(process_action=noop, apply_action=noop),
        sim=SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: False, step=noop),
        scene=SimpleNamespace(write_data_to_sim=noop, update=noop),
        recorder_manager=SimpleNamespace(
            active_terms=[], record_pre_step=noop,
            record_post_physics_decimation_step=noop, record_pre_reset=noop, record_post_reset=noop,
        ),
        command_manager=SimpleNamespace(compute=noop),
        event_manager=SimpleNamespace(available_modes=[]),
        observation_manager=SimpleNamespace(compute=lambda **kwargs: {}),
        shelf_failed=torch.tensor([True, False, False]),
        timeout=torch.tensor([False, True, False]),
        other_failed=torch.tensor([False, False, True]),
    )

    def compute_termination():
        trace.append("termination")
        env.termination_manager.terminated = (env.shelf_failed | env.other_failed).clone()
        env.termination_manager.time_outs = env.timeout.clone()
        return env.shelf_failed | env.other_failed | env.timeout

    env.termination_manager = SimpleNamespace(
        compute=compute_termination, get_term=lambda name: env.shelf_failed if name == "shelf_collision" else None,
    )

    # Execute production RewardManager.compute rather than duplicating dt math.
    manager_class = _execute_definitions(
        PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py",
        {"torch": torch, "ManagerBase": object}, selected_classes={"RewardManager"},
    ).RewardManager
    manager = manager_class.__new__(manager_class)
    manager._env = env
    manager._term_names = ["shelf_failure"]
    manager._term_cfgs = [SimpleNamespace(func=v1_mdp.mdp.shelf_failure, weight=-3.0, params={})]
    manager._episode_sums = {"shelf_failure": torch.zeros(3)}
    manager._reward_buf = torch.zeros(3)
    manager._step_reward = torch.zeros(3, 1)

    def compute_reward(dt):
        trace.append("reward")
        return manager.compute(dt)

    env.reward_manager = SimpleNamespace(compute=compute_reward)

    def reset(ids):
        trace.append("reset")
        # The cost must already exist before flags/history can be cleared.
        torch.testing.assert_close(manager._episode_sums["shelf_failure"], torch.tensor([-3., 0., 0.]))
        assert ids.tolist() == [0, 1, 2]
        env.shelf_failed[ids] = False
        env.timeout[ids] = False
        env.other_failed[ids] = False
        manager._episode_sums["shelf_failure"][ids] = 0.

    env._reset_idx = reset

    # Keep the actual framework step body, replacing only simulator dependencies.
    source = PROJECT_ROOT / "IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py"
    tree = ast.parse(source_text(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRLEnv")
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), step,
    ], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    env.step = MethodType(namespace["step"], env)

    _, reward, terminated, truncated, _ = env.step(torch.zeros(3, 8))
    torch.testing.assert_close(reward, torch.tensor([-3., 0., 0.]))
    assert terminated.tolist() == [True, False, True]
    assert truncated.tolist() == [False, True, False]
    assert trace == ["termination", "reward", "reset"]

    # After auto-reset a normal step must not inherit or repeat the old event.
    trace.clear()
    _, reward, terminated, truncated, _ = env.step(torch.zeros(3, 8))
    assert reward.tolist() == [0., 0., 0.]
    assert not terminated.any() and not truncated.any()
    assert trace == ["termination", "reward"]
