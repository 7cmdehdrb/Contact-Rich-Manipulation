"""Shared compensated observation lifecycle, without launching Isaac Sim."""

from types import SimpleNamespace

import pytest
import torch

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions


@pytest.fixture
def observation_harness(make_env):
    env = make_env(3)
    env.common_step_counter = 10
    env._sim_step_counter = 20
    env.cfg = SimpleNamespace(sim=SimpleNamespace(gravity=(0.0, 0.0, -9.81)))
    values_eef = torch.arange(18, dtype=torch.float32).reshape(3, 6)
    values_world = values_eef + 100.0
    calls = []
    constructors = []

    class TermBase:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env

    class Reader:
        def __init__(self, robot, gravity_w):
            constructors.append((robot, gravity_w))

        def read(self, pos, quat, valid_env_mask):
            calls.append((env.common_step_counter, env._sim_step_counter, valid_env_mask.clone()))
            return SimpleNamespace(
                pure_eef=torch.where(valid_env_mask[:, None], values_eef, 0.0),
                pure_world=torch.where(valid_env_mask[:, None], values_world, 0.0),
            )

    cls = _execute_definitions(
        MODULE_ROOT / "contact_sweep_clutter_v2/observations_cfg.py",
        {"torch": torch, "ManagerTermBase": TermBase, "PureWrenchReader": Reader},
        selected_classes={"PureWristWrench"},
    ).PureWristWrench
    return SimpleNamespace(
        env=env, term=cls(SimpleNamespace(params={}), env),
        values_eef=values_eef, values_world=values_world,
        calls=calls, constructors=constructors,
    )


def test_constructor_shape_probe_needs_no_live_scene_or_step_counters(observation_harness):
    h = observation_harness
    h.env.sensor_data_fresh.zero_()
    # ObservationManager probes dimensions before reading sensors is safe.
    del h.env.common_step_counter
    del h.env._sim_step_counter
    h.env.scene.clear()
    assert torch.equal(h.term(h.env), torch.zeros(3, 6))
    assert torch.equal(h.term.world_wrench(h.env), torch.zeros(3, 6))
    assert h.calls == []
    assert h.constructors == []


def test_observation_and_reward_share_one_physical_sample_and_return_copies(observation_harness):
    h = observation_harness
    observed = h.term(h.env)
    world = h.term.world_wrench(h.env)
    assert torch.equal(observed, h.values_eef)
    assert torch.equal(world, h.values_world)
    observed.zero_()
    world.fill_(-999.0)
    assert torch.equal(h.term(h.env), h.values_eef)
    assert torch.equal(h.term.world_wrench(h.env), h.values_world)
    assert len(h.calls) == 1
    assert len(h.constructors) == 1
    # A physics tick within the same control step must invalidate the sample.
    h.env._sim_step_counter += 1
    h.term.world_wrench(h.env)
    assert len(h.calls) == 2
    h.env.common_step_counter += 1
    h.term(h.env)
    assert len(h.calls) == 3
    assert len(h.constructors) == 1


def test_partial_reset_masks_only_reset_rows_and_keeps_other_readings(observation_harness):
    h = observation_harness
    before = h.term.world_wrench(h.env)
    h.env.sensor_data_fresh[1] = False
    # Stale sensor values can even be invalid after a pose reset.
    h.values_eef[1] = float("nan")
    h.values_world[1] = float("nan")
    h.term.reset(torch.tensor([1]))
    observed = h.term(h.env)
    world = h.term.world_wrench(h.env)
    assert torch.equal(observed[1], torch.zeros(6))
    assert torch.equal(world[1], torch.zeros(6))
    assert torch.equal(world[[0, 2]], before[[0, 2]])
    assert len(h.calls) == 2
    assert h.calls[-1][2].tolist() == [True, False, True]

    h.env._sim_step_counter += 2
    h.env.common_step_counter += 1
    h.env.sensor_data_fresh[:] = True
    h.values_eef[1].fill_(3.0)
    h.values_world[1].fill_(7.0)
    assert torch.equal(h.term(h.env)[1], torch.full((6,), 3.0))
    assert torch.equal(h.term.world_wrench(h.env)[1], torch.full((6,), 7.0))
    assert len(h.calls) == 3


def test_full_reset_skips_physics_read_and_cannot_reuse_prior_episode_cache(observation_harness):
    h = observation_harness
    h.term(h.env)
    h.env.sensor_data_fresh.zero_()
    h.term.reset()
    assert torch.equal(h.term(h.env), torch.zeros(3, 6))
    assert torch.equal(h.term.world_wrench(h.env), torch.zeros(3, 6))
    assert len(h.calls) == 1
    # Even the same counter pair cannot reuse the old episode sample.
    h.env.sensor_data_fresh[:] = True
    h.values_eef.add_(500.0)
    h.values_world.add_(500.0)
    assert torch.equal(h.term(h.env), h.values_eef)
    assert torch.equal(h.term.world_wrench(h.env), h.values_world)
    assert len(h.calls) == 2
