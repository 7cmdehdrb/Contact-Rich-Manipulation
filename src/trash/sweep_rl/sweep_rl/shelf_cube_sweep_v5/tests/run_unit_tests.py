"""Run standalone v5 static contracts without importing Isaac Sim."""

from __future__ import annotations

import inspect

import test_static_contract


def main() -> None:
    tests = [
        function
        for name, function in inspect.getmembers(
            test_static_contract, inspect.isfunction
        )
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"standalone shelf_cube_sweep_v5 tests: {len(tests)} passed")


if __name__ == "__main__":
    main()
