"""Read live definitions for CPU tests that exercise one configuration variant.

The catalog contains names and import bindings only. Every executable body is
read from the current package, so a historical test unit can span the new
manager/category files without loading unrelated sibling configuration classes.
It does not provide import compatibility to the production package.
"""

from __future__ import annotations

import ast
import copy
from fnmatch import fnmatchcase
import importlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace


MODULE_ROOT = Path(__file__).resolve().parents[1] / "inspire_sweep_baseline"
CATALOG = json.loads(Path(__file__).with_name("_source_units.json").read_text(encoding="utf-8"))
PREFIX = "inspire_sweep_baseline."


def _unit(path):
    try:
        relative = Path(path).resolve().relative_to(MODULE_ROOT.resolve())
    except ValueError:
        return None
    name = ".".join(relative.with_suffix("").parts)
    return name if name in CATALOG["definitions"] else None


def _names(node):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return [node.name]
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return [child.id for target in targets for child in ast.walk(target)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)]


def _symbol(unit, name):
    destination = CATALOG["symbols"][f"{PREFIX}{unit}:{name}"]
    module, symbol = destination.removeprefix(PREFIX).split(":")
    return module, symbol


def _canonical(ref):
    if ref[0] == "symbol":
        if f"{PREFIX}{ref[1]}:{ref[2]}" not in CATALOG["symbols"]:
            return tuple(ref)
        return ("symbol", *_symbol(ref[1], ref[2]))
    return tuple(ref)


def source_tree(path):
    unit = _unit(path)
    if unit is None:
        return ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    body, seen, trees = [], set(), {}
    for name in CATALOG["definitions"][unit]:
        destination, symbol = _symbol(unit, name)
        if destination not in trees:
            actual_path = MODULE_ROOT.joinpath(*destination.split(".")).with_suffix(".py")
            trees[destination] = ast.parse(actual_path.read_text(encoding="utf-8-sig"))
        node = next(node for node in trees[destination].body if symbol in _names(node))
        key = destination, node.lineno
        if key in seen:
            continue
        seen.add(key)
        node = copy.deepcopy(node)
        node._source_unit = unit
        node._source_destination = destination
        body.append(node)
    imports, imported = [], set()
    for node in body:
        destination = node._source_destination
        used = {child.id for child in ast.walk(node) if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)}
        for statement in trees[destination].body:
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            if not any((alias.asname or alias.name.split(".")[0]) in used for alias in statement.names):
                continue
            key = ast.dump(statement)
            if key not in imported:
                imports.append(copy.deepcopy(statement))
                imported.add(key)
    return ast.Module(body=imports + body, type_ignores=[])


def source_text(path, *args, **kwargs):
    if _unit(path) is None:
        kwargs.setdefault("encoding", "utf-8")
        return Path(path).read_text(*args, **kwargs)
    return ast.unparse(source_tree(path))


def _fixture_values(unit, namespace):
    values = {}
    for alias, ref in CATALOG["bindings"].get(unit, {}).items():
        if alias not in namespace:
            continue
        value = namespace[alias]
        if ref[0] != "module":
            values[_canonical(ref)] = value
            continue
        values[tuple(ref)] = value
        for name, binding in CATALOG["bindings"].get(ref[1], {}).items():
            if hasattr(value, name) and binding[0] != "module":
                values[_canonical(binding)] = getattr(value, name)
    return values


def execute_definitions(path, namespace, selected_classes=None):
    tree = source_tree(path)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    if selected_classes is not None:
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in selected_classes]
    unit = _unit(path)
    values = _fixture_values(unit, namespace) if unit else {}
    for node in tree.body:
        destination = getattr(node, "_source_destination", None)
        if destination:
            # Bind the generated import names to the same deliberate CPU doubles
            # supplied by the test. The production AST itself remains unchanged.
            for alias, ref in CATALOG["imports"].get(destination, {}).items():
                if tuple(ref) in values:
                    namespace[alias] = values[tuple(ref)]
                elif len(ref) == 3 and ref[2] in namespace:
                    namespace[alias] = namespace[ref[2]]
                elif len(ref) == 3:
                    for original_alias, original_ref in CATALOG["bindings"].get(unit, {}).items():
                        candidate = namespace.get(original_alias)
                        if original_ref[0] == "module" and hasattr(candidate, ref[2]):
                            namespace[alias] = getattr(candidate, ref[2])
                            break
        body = [node]
        if selected_classes is not None:
            body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
        if unit:
            for name in _names(node):
                if name in CATALOG["definitions"][unit]:
                    values[("symbol", *_symbol(unit, name))] = namespace[name]
    return SimpleNamespace(**namespace)


def current_path(unit):
    """Return the sole current module for an unsplit test unit."""
    modules = {_symbol(unit, name)[0] for name in CATALOG["definitions"][unit]}
    if len(modules) != 1:
        raise ValueError(f"Test unit spans multiple current files: {unit}")
    return MODULE_ROOT.joinpath(*modules.pop().split(".")).with_suffix(".py")


def unit_paths(directory="", pattern="*.py"):
    """Select logical source units explicitly, without scanning removed folders."""
    prefix = directory.replace("/", ".").strip(".")
    return [MODULE_ROOT.joinpath(*unit.split(".")).with_suffix(".py")
            for unit in sorted(CATALOG["definitions"])
            if unit.rpartition(".")[0] == prefix
            and fnmatchcase(unit.rsplit(".", 1)[-1] + ".py", pattern)]


def import_current(module, package="_baseline_cpu_pkg"):
    return importlib.import_module(f"{package}.{module}")


def import_unit(unit, package="_baseline_cpu_pkg"):
    """Expose one test's logical set of terms from their actual current modules."""
    modules = {}
    for name in CATALOG["definitions"][unit]:
        destination, _ = _symbol(unit, name)
        modules[destination] = import_current(destination, package)
    if len(modules) == 1:
        return next(iter(modules.values()))

    class TermSet(ModuleType):
        def __setattr__(self, name, value):
            for destination, module in modules.items():
                if name in vars(module):
                    setattr(module, name, value)
            super().__setattr__(name, value)

    result = TermSet(f"_test_terms_{unit}")
    for module in modules.values():
        result.__dict__.update({name: value for name, value in vars(module).items()
                                if not name.startswith("__")})
    for name in CATALOG["definitions"][unit]:
        destination, symbol = _symbol(unit, name)
        result.__dict__[name] = getattr(modules[destination], symbol)
    return result
