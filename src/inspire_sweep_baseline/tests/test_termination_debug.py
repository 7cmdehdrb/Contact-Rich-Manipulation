"""Commented termination diagnostics must leave reset silent and intact."""

from types import SimpleNamespace

import torch

from conftest import MODULE_ROOT
from test_omni_configuration import _execute_definitions


def test_termination_diagnostics_are_silent_even_when_all_terms_fire(capsys):
    names = ("time_out", "object_drop", "push_fast", "shelf_collision", "hand_velocity", "shelf_displacement")
    flags = {name: torch.tensor([True, False, True]) for name in names}

    class Base:
        def _reset_idx(self, env_ids):
            for flag in flags.values():
                flag[env_ids] = False
            self.episode_length_buf[env_ids] = 0

    module = _execute_definitions(
        MODULE_ROOT / "env.py", {"ManagerBasedRLEnv": Base, "torch": torch},
    )
    env = module.InspireSweepEnv.__new__(module.InspireSweepEnv)
    env.sensor_data_fresh = torch.ones(3, dtype=torch.bool)
    env.action_manager = SimpleNamespace(_action=torch.zeros(3, 8), _prev_action=torch.zeros(3, 8))
    env.episode_length_buf = torch.tensor([1, 2, 500])
    env.termination_manager = SimpleNamespace(
        active_terms=names, get_term=lambda name: flags[name],
        get_term_cfg=lambda name: SimpleNamespace(func=lambda env: None, time_out=name == "time_out"),
    )
    env._reset_idx(torch.tensor([0, 2]))
    assert capsys.readouterr().out == ""
    assert env.episode_length_buf.tolist() == [0, 2, 0]
    env._reset_idx(torch.tensor([0, 2]))
    assert capsys.readouterr().out == ""


def test_tasks_use_their_registered_runtime_family():
    registrations = {}
    gym = SimpleNamespace(registry=registrations)
    gym.register = lambda **kwargs: registrations.update({kwargs["id"]: kwargs})
    _execute_definitions(MODULE_ROOT / "__init__.py", {"gym": gym})
    assert registrations
    contact_runtimes = {
        "Isaac-Inspire-Contact-Sweep-v0": "contact_sweep.env:ContactSweepEnv",
        "Isaac-Inspire-Contact-Sweep-Clutter-v0": "contact_sweep_clutter.env:ContactSweepClutterEnv",
        "Isaac-Inspire-Contact-Sweep-Clutter-v1": "contact_sweep_clutter_v1.env:ContactSweepClutterV1Env",
        "Isaac-Inspire-Contact-Sweep-Clutter-v2": "contact_sweep_clutter_v2.env:ContactSweepClutterV2Env",
        "Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1": "contact_sweep_clutter_v1_curriculum.env:ContactSweepClutterV1CurriculumEnv",
        "Isaac-Inspire-Contact-Sweep-Clutter-FT-Test-v0": "contact_sweep_clutter_ft_test.env:ContactSweepClutterFTTestEnv",
        "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v0": "contact_sweep_clutter_fusion.env:ContactSweepClutterFusionEnv",
        "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v3": "contact_sweep_clutter_fusion.env:ContactSweepClutterFusionEnv",
        "Isaac-Inspire-Contact-Sweep-Clutter-Fusion-v2": "contact_sweep_clutter_fusion.env:ContactSweepClutterFusionEnv",
    }
    for task, registration in registrations.items():
        runtime = contact_runtimes.get(task, "env:InspireSweepEnv")
        assert registration["entry_point"] == f"inspire_sweep_baseline.{runtime}"
