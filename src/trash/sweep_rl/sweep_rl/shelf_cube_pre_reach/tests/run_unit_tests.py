#!/usr/bin/env python3
"""Run Cube pre-reach static tests without Isaac Sim."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> None:
    path = Path(__file__).with_name("test_static_contract.py")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("test_")
    ]
    for test in tests:
        test()
    print(f"shelf_cube_pre_reach simulator-independent tests: {len(tests)} passed")


if __name__ == "__main__":
    main()
