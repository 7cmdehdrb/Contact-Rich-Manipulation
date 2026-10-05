"""Check install, registration and public manager contracts without launching Kit."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "hand_manipulation_test"
TASK_ID = "Isaac-Hand-Manipulation-Test-v0"
CONTACT_TASK_ID = "Isaac-Hand-Manipulation-Contact-v0"
PUSH_TASK_ID = "Isaac-Hand-Manipulation-Push-v0"
ENTRY_POINTS = {
    "entry_point": "hand_manipulation_test.env:HandManipulationTestEnv",
    "env_cfg_entry_point": "hand_manipulation_test.config.ur5e.reach_env_cfg:UR5eInspireReachEnvCfg",
    "rsl_rl_cfg_entry_point": "hand_manipulation_test.agents.rsl_rl_ppo_cfg:HandManipulationTestPPORunnerCfg",
}
CONTACT_ENTRY_POINTS = {
    "entry_point": "hand_manipulation_test.contact_env:HandManipulationContactEnv",
    "env_cfg_entry_point": "hand_manipulation_test.config.ur5e.contact_env_cfg:UR5eInspireContactEnvCfg",
    "rsl_rl_cfg_entry_point": "hand_manipulation_test.agents.rsl_rl_contact_ppo_cfg:ContactPPORunnerCfg",
}
PUSH_ENTRY_POINTS = {
    "entry_point": "hand_manipulation_test.push_env:HandManipulationPushEnv",
    "env_cfg_entry_point": "hand_manipulation_test.config.ur5e.push_env_cfg:UR5eInspirePushEnvCfg",
    "rsl_rl_cfg_entry_point": "hand_manipulation_test.agents.rsl_rl_push_ppo_cfg:PushPPORunnerCfg",
}
USD_DEPENDENCY = re.compile(rb"(?i)(?:[A-Za-z][A-Za-z0-9+.-]*://)?[A-Za-z0-9_./\\:+-]+\.(?:usd|usda|usdc|usdz)")


def _tree(path: Path) -> ast.Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    compile(tree, str(path), "exec")
    return tree


def _class(path: Path, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(_tree(path)) if isinstance(node, ast.ClassDef) and node.name == name)


def _assignments(node) -> dict[str, ast.expr]:
    result = {}
    for statement in node.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = statement.targets[0]
            if isinstance(name, ast.Name):
                result[name.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            result[statement.target.id] = statement.value
    return result


def _calls(node, constructor: str) -> dict[str, ast.Call]:
    return {
        name: value for name, value in _assignments(node).items()
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == constructor
    }


class PublicTaskContractTest(unittest.TestCase):
    def test_registration_resolves_to_local_classes_without_importing_kit(self):
        tree = _tree(PACKAGE_ROOT / "__init__.py")
        bindings = {name: ast.literal_eval(value) for name, value in _assignments(tree).items()}
        self.assertEqual(bindings["TASK_ID"], TASK_ID)
        self.assertEqual(bindings["CONTACT_TASK_ID"], CONTACT_TASK_ID)
        self.assertEqual(bindings["PUSH_TASK_ID"], PUSH_TASK_ID)
        registrations = [
            node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "register"
        ]
        for task_id, entry_points in ((TASK_ID, ENTRY_POINTS), (CONTACT_TASK_ID, CONTACT_ENTRY_POINTS), (PUSH_TASK_ID, PUSH_ENTRY_POINTS)):
            matching = []
            for call in registrations:
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                id_value = keywords["id"]
                registered_id = bindings[id_value.id] if isinstance(id_value, ast.Name) else ast.literal_eval(id_value)
                if registered_id == task_id:
                    matching.append(keywords)
            self.assertEqual(len(matching), 1, task_id)
            registration = matching[0]
            self.assertEqual(ast.literal_eval(registration["entry_point"]), entry_points["entry_point"])
            self.assertTrue(ast.literal_eval(registration["disable_env_checker"]))
            kwargs = ast.literal_eval(registration["kwargs"])
            for name in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point"):
                self.assertEqual(kwargs[name], entry_points[name])
            for value in entry_points.values():
                module_name, class_name = value.split(":")
                path = PROJECT_ROOT.joinpath(*module_name.split(".")).with_suffix(".py")
                self.assertTrue(path.is_file(), str(path))
                self.assertIsInstance(_class(path, class_name), ast.ClassDef)

    def test_environment_uses_standard_manager_step_and_reset(self):
        env_class = _class(PACKAGE_ROOT / "env.py", "HandManipulationTestEnv")
        self.assertEqual([ast.unparse(base) for base in env_class.bases], ["ManagerBasedRLEnv"])
        custom_methods = [node.name for node in env_class.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.assertEqual(custom_methods, [], "Task logic belongs in manager terms")
        cfg_class = _class(PACKAGE_ROOT / "config" / "ur5e" / "reach_env_cfg.py", "UR5eInspireReachEnvCfg")
        self.assertEqual([ast.unparse(base) for base in cfg_class.bases], ["HandManipulationTestEnvCfg"])
        contact_env = _class(PACKAGE_ROOT / "contact_env.py", "HandManipulationContactEnv")
        self.assertEqual([ast.unparse(base) for base in contact_env.bases], ["HandManipulationTestEnv"])
        self.assertFalse(any(isinstance(node, ast.FunctionDef) for node in contact_env.body))
        contact_cfg = _class(PACKAGE_ROOT / "config" / "ur5e" / "contact_env_cfg.py", "UR5eInspireContactEnvCfg")
        self.assertEqual([ast.unparse(base) for base in contact_cfg.bases], ["UR5eInspireReachEnvCfg"])
        push_env = _class(PACKAGE_ROOT / "push_env.py", "HandManipulationPushEnv")
        self.assertEqual([ast.unparse(base) for base in push_env.bases], ["HandManipulationContactEnv"])
        self.assertFalse(any(isinstance(node, ast.FunctionDef) for node in push_env.body))

    def test_observation_terms_are_ordered_for_the_55d_policy_contract(self):
        constants = _assignments(_tree(PACKAGE_ROOT / "constants.py"))
        layout = ast.literal_eval(constants["OBSERVATION_LAYOUT"])
        expected = (
            ("arm_joint_position", 6), ("arm_joint_velocity", 6),
            ("eef_relative_position", 3), ("eef_relative_orientation", 3),
            ("hand_state", 2), ("surface_header_and_tactile", 18),
            ("wrist_wrench_c", 6), ("initial_target_relative_position", 3),
            ("last_action", 8),
        )
        self.assertEqual(layout, expected)
        self.assertEqual(sum(width for _, width in layout), 55)
        self.assertEqual(ast.literal_eval(constants["ACTION_DIM"]), 8)
        terms = _calls(_class(PACKAGE_ROOT / "env_cfg.py", "PolicyCfg"), "ObservationTermCfg")
        self.assertEqual(tuple(terms), tuple(name for name, _ in expected))
        for name in ("eef_relative_position", "initial_target_relative_position"):
            self.assertFalse({keyword.arg for keyword in terms[name].keywords} & {"scale", "clip"})

    def test_only_the_two_authorized_rewards_and_timeout_are_declared(self):
        path = PACKAGE_ROOT / "env_cfg.py"
        rewards = _calls(_class(path, "RewardsCfg"), "RewardTermCfg")
        self.assertEqual(set(rewards), {"eef_distance", "action_rate"})
        distance_keywords = {keyword.arg: keyword.value for keyword in rewards["eef_distance"].keywords}
        self.assertEqual(ast.literal_eval(distance_keywords["weight"]), 4.0)
        rate_keywords = {keyword.arg: keyword.value for keyword in rewards["action_rate"].keywords}
        self.assertEqual(abs(ast.literal_eval(rate_keywords["weight"])), 0.06)
        terminations = _calls(_class(path, "TerminationsCfg"), "TerminationTermCfg")
        self.assertEqual(set(terminations), {"time_out"})
        timeout = {keyword.arg: keyword.value for keyword in terminations["time_out"].keywords}
        self.assertTrue(ast.literal_eval(timeout["time_out"]))

    def test_runtime_scripts_use_new_task_and_reaching_configs(self):
        for name in ("train.py", "play.py", "smoke_env.py"):
            path = PROJECT_ROOT / "scripts" / name
            tree = _tree(path)
            self.assertEqual(ast.literal_eval(_assignments(tree)["TASK_ID"]), TASK_ID)
            self.assertEqual(ast.literal_eval(_assignments(tree)["CONTACT_TASK_ID"]), CONTACT_TASK_ID)
            self.assertEqual(ast.literal_eval(_assignments(tree)["PUSH_TASK_ID"]), PUSH_TASK_ID)
            source = path.read_text(encoding="utf-8")
            config_keys = {
                ast.literal_eval(node.args[1]) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "load_cfg_from_registry" and len(node.args) == 2
            }
            self.assertIn("env_cfg_entry_point", config_keys)
            if name in ("train.py", "play.py"):
                self.assertIn("rsl_rl_cfg_entry_point", config_keys)
            self.assertIn('"--num_envs"', source)
        play = (PROJECT_ROOT / "scripts" / "play.py").read_text(encoding="utf-8")
        self.assertIn("commands.target_position.debug_vis", play)
        self.assertIn("scene.ee_frame.debug_vis", play)
        self.assertIn('"--disable-markers"', play)


class StandaloneInstallContractTest(unittest.TestCase):
    def test_no_runtime_imports_of_existing_or_example_packages(self):
        forbidden = {"hand_manipulation_rl", "sweeping_policy", "sweep_rl", "inspire_tactile", "axia80_feasibility"}
        violations = []
        for base in (PACKAGE_ROOT, PROJECT_ROOT / "scripts"):
            for path in base.rglob("*.py"):
                for node in ast.walk(_tree(path)):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        continue
                    for module in modules:
                        if set(module.split(".")) & forbidden:
                            violations.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: {module}")
        self.assertEqual(violations, [])

    def test_robot_usd_composition_is_bundled_and_relative(self):
        asset_root = PACKAGE_ROOT / "assets" / "data"
        layers = sorted(asset_root.rglob("*.usd"))
        self.assertEqual(len(list((asset_root / "ur5e_inspire_usd").rglob("*.usd"))), 5)
        for layer in layers:
            data = layer.read_bytes()
            self.assertGreater(len(data), 0)
            for prefix in (b"/home/", b"/Users/", b"omniverse://", b"http://", b"https://"):
                self.assertNotIn(prefix, data, str(layer))
            for reference in {match.decode() for match in USD_DEPENDENCY.findall(data)}:
                self.assertNotIn("://", reference)
                self.assertFalse(PurePosixPath(reference).is_absolute())
                self.assertFalse(PureWindowsPath(reference).is_absolute())
                resolved = (layer.parent / reference).resolve()
                self.assertTrue(resolved.is_relative_to(asset_root.resolve()), reference)
                self.assertTrue(resolved.is_file(), reference)
        setup = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("assets/data/ur5e_inspire_usd/*", setup)
        self.assertIn("assets/data/ur5e_inspire_usd/configuration/*", setup)


if __name__ == "__main__":
    unittest.main()
