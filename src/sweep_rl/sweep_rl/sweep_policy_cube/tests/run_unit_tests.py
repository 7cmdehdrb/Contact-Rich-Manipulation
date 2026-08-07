"""Run simulator-independent tests without requiring pytest or Isaac Sim."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    test_root = Path(__file__).parent
    total = 0
    for filename in (
        "test_reset_math.py",
        "test_reward_math.py",
        "test_static_contract.py",
    ):
        module = _load_module(test_root / filename)
        tests = [
            getattr(module, name)
            for name in dir(module)
            if name.startswith("test_")
        ]
        for test in tests:
            test()
        total += len(tests)
        print(f"{filename}: {len(tests)} passed")
    print(f"sweep_policy_cube simulator-independent tests: {total} passed")


if __name__ == "__main__":
    main()
