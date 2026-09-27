"""Pure-Python static contracts for the standalone training package.

This module deliberately does not import Torch, Isaac Lab, Gymnasium, or USD
Python bindings.  It checks the source/asset boundary that must remain valid
even on a machine where Isaac Sim cannot be launched.

Binary ``usdc`` files cannot be interpreted semantically with the standard
library.  Their string tables are nevertheless stored as printable byte
sequences, so the USD check below conservatively extracts composition asset
paths ending in a USD extension.  It also rejects common absolute URI/path
prefixes.  The simulator smoke test remains responsible for resolving the
layers through the USD API.
"""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "hand_manipulation_rl"
ASSET_DATA_ROOT = PACKAGE_ROOT / "assets" / "data"

TASK_ID = "Isaac-Blind-Sweep-Inspire-v0"
ENV_ENTRY_POINT = "hand_manipulation_rl.env:BlindSweepEnv"
ENV_CFG_ENTRY_POINT = "hand_manipulation_rl.env_cfg:BlindSweepEnvCfg"
RUNNER_CFG_ENTRY_POINT = (
    "hand_manipulation_rl.agents.rsl_rl_ppo_cfg_02:BlindSweepReferencePPORunnerCfg"
)

FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "sweep_rl",
        "sweeping_policy",
        "inspire_tactile",
        "inspire_tactile_ros",
        "axia80_feasibility",
    }
)

USD_DEPENDENCY_PATTERN = re.compile(
    rb"(?i)(?:[A-Za-z][A-Za-z0-9+.-]*://)?[A-Za-z0-9_./\\:+-]+\.(?:usd|usda|usdc|usdz)"
)
ABSOLUTE_USD_PREFIXES = (
    b"/home/",
    b"/Users/",
    b"/opt/",
    b"/usr/",
    b"file://",
    b"http://",
    b"https://",
    b"omniverse://",
)


def _syntax_tree(path: Path) -> ast.Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # ast.parse alone accepts a few invalid constructs (for example repeated
    # keyword arguments). Compiling the AST performs the remaining syntax-only
    # checks without importing or executing the module.
    compile(tree, str(path), "exec")
    return tree


def _literal_bindings(tree: ast.Module) -> dict[str, object]:
    """Resolve top-level literal assignments, including aliases to literals."""

    values: dict[str, object] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        try:
            values[target.id] = ast.literal_eval(value)
        except (ValueError, TypeError):
            if isinstance(value, ast.Name) and value.id in values:
                values[target.id] = values[value.id]
    return values


def _resolve_static_value(node: ast.AST, bindings: dict[str, object]) -> object:
    if isinstance(node, ast.Name) and node.id in bindings:
        return bindings[node.id]
    if isinstance(node, ast.Dict):
        return {
            _resolve_static_value(key, bindings): _resolve_static_value(value, bindings)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, (ast.Tuple, ast.List)):
        values = [_resolve_static_value(item, bindings) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    return ast.literal_eval(node)


def _registration_calls(tree: ast.Module) -> list[dict[str, object]]:
    bindings = _literal_bindings(tree)
    registrations: list[dict[str, object]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function_name = ""
        if isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            function_name = node.func.id
        if function_name != "register":
            continue
        keywords: dict[str, object] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                keywords[keyword.arg] = _resolve_static_value(keyword.value, bindings)
            except (ValueError, TypeError, KeyError):
                keywords[keyword.arg] = None
        registrations.append(keywords)
    return registrations


def _entry_point_target(entry_point: str) -> tuple[Path, str]:
    module_name, separator, attribute = entry_point.partition(":")
    if not separator or not module_name.startswith("hand_manipulation_rl.") or not attribute:
        raise AssertionError(f"Entry point is not a local module: {entry_point!r}")
    relative_module = module_name.split(".")[1:]
    module_path = PACKAGE_ROOT.joinpath(*relative_module).with_suffix(".py")
    return module_path, attribute


def _defined_top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in _syntax_tree(path).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _class_property_integer(path: Path, class_name: str, property_name: str) -> int:
    for node in _syntax_tree(path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) or child.name != property_name:
                continue
            for statement in child.body:
                if isinstance(statement, ast.Return):
                    value = ast.literal_eval(statement.value)
                    if isinstance(value, int):
                        return value
        raise AssertionError(f"{class_name}.{property_name} has no literal integer return")
    raise AssertionError(f"Class {class_name} was not found in {path}")


def _class_call_assignments(path: Path, class_name: str) -> dict[str, str]:
    """Return ``field = Constructor(...)`` pairs declared directly in a class."""

    for node in _syntax_tree(path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        assignments: dict[str, str] = {}
        for child in node.body:
            if not isinstance(child, ast.Assign) or len(child.targets) != 1:
                continue
            target = child.targets[0]
            if not isinstance(target, ast.Name) or not isinstance(child.value, ast.Call):
                continue
            function = child.value.func
            if isinstance(function, ast.Name):
                assignments[target.id] = function.id
            elif isinstance(function, ast.Attribute):
                assignments[target.id] = function.attr
        return assignments
    raise AssertionError(f"Class {class_name} was not found in {path}")


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    """Return a class method without importing its dependency-heavy module."""

    for node in _syntax_tree(path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == method_name:
                return child
        raise AssertionError(f"Method {class_name}.{method_name} was not found in {path}")
    raise AssertionError(f"Class {class_name} was not found in {path}")


def _method_assignments(method: ast.FunctionDef) -> dict[str, str]:
    """Return normalized assignment expressions from a parsed method."""

    assignments: dict[str, str] = {}
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            assignments[ast.unparse(node.targets[0])] = ast.unparse(node.value)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            assignments[ast.unparse(node.target)] = ast.unparse(node.value)
    return assignments


def _imported_module_names(path: Path) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(_syntax_tree(path)):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            # Cover the two usual dynamic import spellings when their target is
            # a literal.  Calls with computed names are intentionally left to
            # runtime smoke tests.
            is_dunder_import = isinstance(node.func, ast.Name) and node.func.id == "__import__"
            is_import_module = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
            )
            if (is_dunder_import or is_import_module) and node.args:
                try:
                    module_name = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    continue
                if isinstance(module_name, str):
                    imports.append((node.lineno, module_name))
    return imports


def _is_forbidden_module(module_name: str) -> bool:
    components = tuple(part.replace("-", "_") for part in module_name.split(".") if part)
    return any(component in FORBIDDEN_IMPORT_ROOTS for component in components)


def _usd_dependencies(path: Path) -> set[str]:
    return {
        match.decode("utf-8", errors="strict")
        for match in USD_DEPENDENCY_PATTERN.findall(path.read_bytes())
    }


class StandaloneImportContractTest(unittest.TestCase):
    def test_project_does_not_import_reference_implementations(self):
        violations: list[str] = []
        for path in sorted(PROJECT_ROOT.rglob("*.py")):
            for line, module_name in _imported_module_names(path):
                if _is_forbidden_module(module_name):
                    violations.append(f"{path.relative_to(PROJECT_ROOT)}:{line}: {module_name}")
        self.assertEqual(
            violations,
            [],
            "Standalone package imports a reference implementation:\n" + "\n".join(violations),
        )

    def test_train_and_play_do_not_dispatch_repository_scripts(self):
        for name in ("train.py", "play.py"):
            path = PROJECT_ROOT / "scripts" / name
            source = path.read_text(encoding="utf-8")
            imports = {module for _, module in _imported_module_names(path)}
            self.assertNotIn("runpy", imports, f"{name} dispatches an external script")
            self.assertNotIn("REPOSITORY_ROOT", source, f"{name} depends on repository layout")
            self.assertNotIn(
                '"IsaacLab" / "scripts"',
                source,
                f"{name} dispatches an IsaacLab repository script",
            )


class DimensionContractTest(unittest.TestCase):
    def test_observation_layout_is_exactly_57_and_action_is_6_plus_2(self):
        state_path = PACKAGE_ROOT / "mdp" / "episode_state.py"
        bindings = _literal_bindings(_syntax_tree(state_path))
        self.assertEqual(bindings.get("ACTION_DIM"), 8)
        layout = bindings.get("OBSERVATION_LAYOUT")
        self.assertIsInstance(layout, tuple)
        self.assertEqual(sum(width for _, width in layout), 57)
        self.assertIn(("command_direction_y_s", 1), layout)
        self.assertEqual(layout[-1], ("last_action", 8))

        actions_path = PACKAGE_ROOT / "mdp" / "actions.py"
        arm_dim = _class_property_integer(actions_path, "CurrentFrameOscAction", "action_dim")
        hand_dim = _class_property_integer(actions_path, "InspireHandSynergyAction", "action_dim")
        self.assertEqual((arm_dim, hand_dim), (6, 2))
        self.assertEqual(arm_dim + hand_dim, bindings["ACTION_DIM"])

        configured_actions = _class_call_assignments(PACKAGE_ROOT / "env_cfg.py", "ActionsCfg")
        self.assertEqual(
            configured_actions,
            {
                "arm_action": "CurrentFrameOscActionCfg",
                "hand_action": "InspireHandSynergyActionCfg",
            },
        )


class RuntimeAdapterContractTest(unittest.TestCase):
    def test_sweep_policy_reference_ppo_02_is_the_new_default(self):
        path = PACKAGE_ROOT / "agents" / "rsl_rl_ppo_cfg_02.py"
        tree = _syntax_tree(path)
        cfg_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BlindSweepReferencePPORunnerCfg"
        )
        literals: dict[str, object] = {}
        calls: dict[str, ast.Call] = {}
        for node in cfg_class.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(node.value, ast.Call):
                calls[target.id] = node.value
            else:
                literals[target.id] = ast.literal_eval(node.value)
        self.assertEqual(literals["num_steps_per_env"], 36)
        self.assertEqual(literals["max_iterations"], 10_000)
        self.assertEqual(literals["save_interval"], 50)
        self.assertIsNone(literals["clip_actions"])
        self.assertEqual(literals["experiment_name"], "UR5e_shelf_sweep_random")
        self.assertEqual(literals["run_name"], "")

        actor_keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in calls["actor"].keywords
            if keyword.arg in {"hidden_dims", "activation", "obs_normalization"}
        }
        critic_keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in calls["critic"].keywords
            if keyword.arg in {"hidden_dims", "activation", "obs_normalization"}
        }
        self.assertEqual(actor_keywords["hidden_dims"], [256, 128, 64])
        self.assertEqual(actor_keywords["activation"], "elu")
        self.assertFalse(actor_keywords["obs_normalization"])
        self.assertEqual(critic_keywords["hidden_dims"], [256, 128, 64])
        self.assertEqual(critic_keywords["activation"], "elu")
        self.assertFalse(critic_keywords["obs_normalization"])

        actor_distribution = next(
            keyword.value
            for keyword in calls["actor"].keywords
            if keyword.arg == "distribution_cfg"
        )
        self.assertIsInstance(actor_distribution, ast.Call)
        self.assertEqual(actor_distribution.func.attr, "GaussianDistributionCfg")
        self.assertEqual(
            {
                keyword.arg: ast.literal_eval(keyword.value)
                for keyword in actor_distribution.keywords
            },
            {"init_std": 1.0},
        )

        algorithm = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in calls["algorithm"].keywords
        }
        self.assertEqual(
            algorithm,
            {
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.005,
                "num_learning_epochs": 8,
                "num_mini_batches": 4,
                "learning_rate": 1.0e-3,
                "schedule": "adaptive",
                "gamma": 0.98,
                "lam": 0.95,
                "desired_kl": 0.02,
                "max_grad_norm": 1.0,
            },
        )

    def test_reference_robot_and_wide_shallow_shelf_geometry(self):
        cfg_bindings = _literal_bindings(_syntax_tree(PACKAGE_ROOT / "env_cfg.py"))
        self.assertEqual(cfg_bindings["BOARD_SIZE_M"], (0.36, 1.0, 0.04))
        self.assertEqual(cfg_bindings["BOARD_CENTER_S_M"], (-0.70, 0.0, 1.03))
        self.assertGreater(cfg_bindings["BOARD_SIZE_M"][1], cfg_bindings["BOARD_SIZE_M"][0])
        self.assertEqual(
            cfg_bindings["BOARD_CENTER_S_M"][2] + 0.5 * cfg_bindings["BOARD_SIZE_M"][2],
            1.05,
        )

        env_tree = _syntax_tree(PACKAGE_ROOT / "env_cfg.py")
        cube_initial = next(
            node.value
            for node in env_tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CUBE_INITIAL_CENTER_S_M"
        )
        self.assertIsInstance(cube_initial, ast.Tuple)
        self.assertEqual(ast.literal_eval(cube_initial.elts[0]), -0.675)
        self.assertEqual(ast.literal_eval(cube_initial.elts[1]), 0.0)

        task_class = next(
            node
            for node in env_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BlindSweepTaskCfg"
        )
        task_literals: dict[str, object] = {}
        for child in task_class.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                try:
                    task_literals[child.target.id] = ast.literal_eval(child.value)
                except (ValueError, TypeError):
                    pass
        self.assertEqual(task_literals["object_xy_range_low"], (-0.77, -0.22))
        self.assertEqual(task_literals["object_xy_range_high"], (-0.58, 0.22))
        self.assertEqual(task_literals["command_distance_range"], (0.18, 0.18))

        robot_tree = _syntax_tree(PACKAGE_ROOT / "assets" / "robot.py")
        make_cfg = next(
            node
            for node in robot_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_robot_cfg"
        )
        assignments = _method_assignments(make_cfg)
        self.assertEqual(
            ast.literal_eval(assignments["initial_arm"]),
            {
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.2,
                "elbow_joint": 2.2,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.57,
                "wrist_3_joint": 0.785,
            },
        )
        make_source = ast.unparse(make_cfg)
        self.assertIn("pos=(0.0, 0.0, 0.79505)", make_source)
        self.assertIn("rot=(0.0, 0.0, 0.0, 1.0)", make_source)
        self.assertNotIn("armature=0.01", make_source)

    def test_reset_collision_gate_uses_live_fk_and_conservative_mesh_bounds(self):
        bounds_method = _class_method(
            PACKAGE_ROOT / "env.py",
            "BlindSweepEnv",
            "_initialize_reset_collision_bounds",
        )
        bounds_source = ast.unparse(bounds_method)
        self.assertIn("UsdPhysics.CollisionAPI", bounds_source)
        self.assertIn("UsdGeom.Boundable", bounds_source)
        self.assertIn("ComputeExtentFromPlugins", bounds_source)
        self.assertIn("Usd.TraverseInstanceProxies", bounds_source)

        collision_method = _class_method(
            PACKAGE_ROOT / "env.py",
            "BlindSweepEnv",
            "reset_collision_free_mask",
        )
        collision_source = ast.unparse(collision_method)
        self.assertIn("robot.data.body_pos_w[env_ids]", collision_source)
        self.assertIn("robot.data.body_quat_w[env_ids]", collision_source)
        self.assertIn("target.data.root_pos_w[env_ids]", collision_source)
        self.assertIn("target.data.root_quat_w[env_ids]", collision_source)
        self.assertIn("board.data.root_pos_w[env_ids]", collision_source)
        self.assertIn("board.data.root_quat_w[env_ids]", collision_source)
        self.assertEqual(collision_source.count("oriented_box_overlap"), 2)
        self.assertNotIn("overlap_shape", collision_source)

        solve_method = _class_method(
            PACKAGE_ROOT / "mdp" / "events.py",
            "ConditionalPoseIKReset",
            "_solve_spec",
        )
        solve_source = ast.unparse(solve_method)
        self.assertIn("env.reset_collision_free_mask", solve_source)
        self.assertIn("clearance_ok & collision_free", solve_source)
        self.assertIn("q[:, self.hand_joint_ids] = sampled_hand_targets", solve_source)
        self.assertIn("final_q[:, self.hand_joint_ids] = sampled_hand_targets", solve_source)

    def test_reset_sensor_buffer_is_valid_without_a_hidden_rollout_step(self):
        finalize = _class_method(PACKAGE_ROOT / "env.py", "BlindSweepEnv", "_finalize_reset")
        assignments = _method_assignments(finalize)
        self.assertEqual(assignments.get("self.sensor_valid[env_ids]"), "True")
        self.assertEqual(assignments.get("self.sensor_data_fresh[env_ids]"), "False")
        self.assertEqual(assignments.get("self.invalid_reset_latched[env_ids]"), "False")
        finalize_source = ast.unparse(finalize)
        self.assertIn("self.reset_collision_free[env_ids].all()", finalize_source)
        self.assertIn("self.reset_collision_free_mask(env_ids)", finalize_source)
        self.assertNotIn("calibr", finalize_source.lower())

        observation_method = next(
            node
            for node in _syntax_tree(PACKAGE_ROOT / "mdp" / "observations.py").body
            if isinstance(node, ast.FunctionDef) and node.name == "policy_observation"
        )
        observation_source = ast.unparse(observation_method)
        self.assertIn("env.sensor_valid & env.sensor_data_fresh", observation_source)

    def test_every_policy_action_is_applied_for_the_full_decimation_window(self):
        step = _class_method(PACKAGE_ROOT / "env.py", "BlindSweepEnv", "step")
        step_source = ast.unparse(step)
        self.assertEqual(step_source.count("self.action_manager.process_action(bounded_action)"), 1)
        self.assertNotIn("process_actions_for_envs", step_source)
        self.assertNotIn("reset_contact_validation_pending", step_source)
        self.assertLess(
            step_source.index("self.action_manager.process_action(bounded_action)"),
            step_source.index("for _ in range(self.cfg.decimation)"),
        )
        self.assertLess(
            step_source.index("if self.invalid_reset_latched.any()"),
            step_source.index("self.reward_buf = self.reward_manager.compute"),
        )

    def test_physx_jacobians_are_shifted_from_link_com_to_c(self):
        action_method = _class_method(
            PACKAGE_ROOT / "mdp" / "actions.py",
            "CurrentFrameOscAction",
            "_compute_c_jacobian",
        )
        action_source = ast.unparse(action_method)
        self.assertIn("body_com_pos_b", action_source)
        self.assertIn("r_comc_b", action_source)

        ik_method = _class_method(
            PACKAGE_ROOT / "mdp" / "events.py",
            "ConditionalPoseIKReset",
            "_solve_spec",
        )
        ik_source = ast.unparse(ik_method)
        self.assertIn("body_com_pos_b", ik_source)
        self.assertIn("r_comc_w", ik_source)

    def test_osc_uses_full_inertial_dynamics_decoupling(self):
        tree = _syntax_tree(PACKAGE_ROOT / "env_cfg.py")
        arm_calls = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef) or node.name != "ActionsCfg":
                continue
            for child in node.body:
                if (
                    isinstance(child, ast.Assign)
                    and len(child.targets) == 1
                    and isinstance(child.targets[0], ast.Name)
                    and child.targets[0].id == "arm_action"
                    and isinstance(child.value, ast.Call)
                ):
                    arm_calls.append(child.value)
        self.assertEqual(len(arm_calls), 1)
        keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in arm_calls[0].keywords
            if keyword.arg in {
                "motion_stiffness",
                "motion_damping_ratio",
                "gravity_compensation",
                "inertial_dynamics_decoupling",
                "partial_inertial_dynamics_decoupling",
            }
        }
        self.assertEqual(
            keywords,
            {
                "motion_stiffness": (100.0, 100.0, 100.0, 100.0, 100.0, 100.0),
                "motion_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                "gravity_compensation": False,
                "inertial_dynamics_decoupling": True,
                "partial_inertial_dynamics_decoupling": False,
            },
        )

    def test_ft_reader_performs_no_gravity_or_tare_cancellation(self):
        sensors_tree = _syntax_tree(PACKAGE_ROOT / "sensors.py")
        reader = next(
            node
            for node in sensors_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FixedJointWrenchReader"
        )
        reader_source = ast.unparse(reader)
        for forbidden in (
            "gravity_compensation",
            "calibrate_bias",
            "set_bias_f",
            "bias_candidate_f",
        ):
            self.assertNotIn(forbidden, reader_source)
        read_method = next(
            node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == "read"
        )
        assignments = _method_assignments(read_method)
        self.assertEqual(assignments.get("measured_f"), "raw_f")
        self.assertIn("transform_wrench_f_to_c", ast.unparse(read_method))

    def test_task_geometry_overrides_are_copied_into_scene_configs(self):
        cfg_path = PACKAGE_ROOT / "env_cfg.py"
        post_init = _class_method(cfg_path, "BlindSweepEnvCfg", "__post_init__")
        assignments = _method_assignments(post_init)

        self.assertEqual(
            assignments.get("self.scene.board.spawn.size"),
            "self.task.board_size",
        )
        self.assertEqual(
            assignments.get("self.scene.board.init_state.pos"),
            "self.task.board_center",
        )
        self.assertEqual(
            assignments.get("self.scene.target_object.spawn.size"),
            "(self.task.cube_size,) * 3",
        )
        self.assertEqual(
            assignments.get("self.scene.target_object.init_state.pos"),
            "cube_initial_center",
        )
        self.assertEqual(
            assignments.get("self.scene.target_object.init_state.rot"),
            "self.task.initial_cube_quat_w",
        )
        self.assertFalse(
            any(target.endswith(".prim_path") for target in assignments),
            "Geometry synchronization must not rewrite ContactSensor paths",
        )

    def test_start_box_and_sampled_goal_use_the_same_usable_board(self):
        cfg_validate = _class_method(PACKAGE_ROOT / "env_cfg.py", "BlindSweepTaskCfg", "validate")
        validate_source = ast.unparse(cfg_validate)
        for required in (
            "self.object_xy_range_low",
            "self.object_xy_range_high",
            "usable_lower",
            "usable_upper",
        ):
            self.assertIn(required, validate_source)

        sampler = _class_method(PACKAGE_ROOT / "env.py", "BlindSweepEnv", "sample_episode_specs")
        assignments = _method_assignments(sampler)
        goal_expression = assignments.get("goal_xy", "")
        self.assertIn("xy + distance.unsqueeze(-1) * direction_xy", goal_expression)
        self.assertIn("xy >= board_min", assignments.get("start_valid", ""))
        self.assertIn("xy <= board_max", assignments.get("start_valid", ""))
        self.assertIn("goal_xy >= board_min", assignments.get("goal_valid", ""))
        self.assertIn("goal_xy <= board_max", assignments.get("goal_valid", ""))
        self.assertEqual(assignments.get("valid"), "start_valid & goal_valid")

    def test_timeout_is_truncation_and_is_masked_by_terminal_outcomes(self):
        cfg_tree = _syntax_tree(PACKAGE_ROOT / "env_cfg.py")
        timeout_terms = [
            node
            for node in ast.walk(cfg_tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "time_out" for target in node.targets)
            and isinstance(node.value, ast.Call)
        ]
        self.assertEqual(len(timeout_terms), 1)
        timeout_keywords = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in timeout_terms[0].value.keywords
            if keyword.arg == "time_out"
        }
        self.assertEqual(timeout_keywords, {"time_out": True})

        step = _class_method(PACKAGE_ROOT / "env.py", "BlindSweepEnv", "step")
        assignments = _method_assignments(step)
        self.assertEqual(
            assignments.get("self.reset_time_outs"),
            "self.termination_manager.time_outs & ~self.reset_terminated",
        )
        self.assertEqual(
            assignments.get("self.reset_buf"),
            "self.reset_terminated | self.reset_time_outs",
        )


class RegistrationContractTest(unittest.TestCase):
    def test_registration_uses_local_string_entry_points(self):
        init_path = PACKAGE_ROOT / "__init__.py"
        tree = _syntax_tree(init_path)
        bindings = _literal_bindings(tree)
        self.assertEqual(bindings.get("TASK_ID"), TASK_ID)

        registrations = [call for call in _registration_calls(tree) if call.get("id") == TASK_ID]
        self.assertEqual(len(registrations), 1, f"Expected one static registration for {TASK_ID}")
        registration = registrations[0]
        self.assertEqual(registration.get("entry_point"), ENV_ENTRY_POINT)
        kwargs = registration.get("kwargs")
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(kwargs.get("env_cfg_entry_point"), ENV_CFG_ENTRY_POINT)
        self.assertEqual(kwargs.get("rsl_rl_cfg_entry_point"), RUNNER_CFG_ENTRY_POINT)

        for entry_point in (ENV_ENTRY_POINT, ENV_CFG_ENTRY_POINT, RUNNER_CFG_ENTRY_POINT):
            module_path, attribute = _entry_point_target(entry_point)
            self.assertTrue(module_path.is_file(), f"Missing entry-point module: {module_path}")
            self.assertIn(attribute, _defined_top_level_names(module_path))


class LocalAssetContractTest(unittest.TestCase):
    def test_required_package_local_usd_layers_exist(self):
        asset_root = ASSET_DATA_ROOT / "ur5e_inspire_usd"
        required = (
            asset_root / "ur5e_inspire.usd",
            asset_root / "configuration" / "ur5e_inspire_base.usd",
            asset_root / "configuration" / "ur5e_inspire_physics.usd",
            asset_root / "configuration" / "ur5e_inspire_robot.usd",
            asset_root / "configuration" / "ur5e_inspire_sensor.usd",
        )
        for path in required:
            self.assertTrue(path.is_file(), f"Missing package-local USD layer: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty package-local USD layer: {path}")

        setup_source = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("assets/data/ur5e_inspire_usd/*", setup_source)
        self.assertIn("assets/data/ur5e_inspire_usd/configuration/*", setup_source)

    def test_usd_composition_dependencies_are_relative_and_package_local(self):
        usd_layers = sorted(ASSET_DATA_ROOT.rglob("*.usd"))
        self.assertGreater(len(usd_layers), 0)
        problems: list[str] = []
        for layer in usd_layers:
            raw = layer.read_bytes()
            lowered = raw.lower()
            for prefix in ABSOLUTE_USD_PREFIXES:
                if prefix.lower() in lowered:
                    problems.append(f"{layer.relative_to(PROJECT_ROOT)} embeds {prefix.decode()!r}")

            for dependency in sorted(_usd_dependencies(layer)):
                is_uri = "://" in dependency
                is_absolute = PurePosixPath(dependency).is_absolute() or PureWindowsPath(dependency).is_absolute()
                if is_uri or is_absolute:
                    problems.append(f"{layer.relative_to(PROJECT_ROOT)} -> absolute {dependency!r}")
                    continue
                dependency_path = (layer.parent / dependency).resolve()
                try:
                    dependency_path.relative_to(ASSET_DATA_ROOT.resolve())
                except ValueError:
                    problems.append(f"{layer.relative_to(PROJECT_ROOT)} -> escapes package {dependency!r}")
                    continue
                if not dependency_path.is_file():
                    problems.append(f"{layer.relative_to(PROJECT_ROOT)} -> missing {dependency!r}")

        self.assertEqual(problems, [], "Invalid USD composition dependency:\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
