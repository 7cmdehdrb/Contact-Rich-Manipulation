"""Check backend selection at the launchers' pre-SimulationApp boundary."""
import argparse
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def cli(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "cli_args.py"
    spec = importlib.util.spec_from_file_location("_renderer_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.metadata, "version", lambda _: "5.1.0.0")
    return module


def test_windows_51_preserves_other_options_and_is_idempotent(cli):
    args = argparse.Namespace(kit_args="--/app/window/width=800")
    cli.apply_renderer_compatibility(args)
    cli.apply_renderer_compatibility(args)
    assert args.kit_args == "--/app/window/width=800 --/app/vulkan=false"


@pytest.mark.parametrize("backend", ["--vulkan", "--d3d12", "--/app/vulkan=true", "--/app/vulkan=false", "--/app/d3d12=true"])
def test_explicit_backend_is_preserved(cli, backend):
    args = argparse.Namespace(kit_args=backend)
    cli.apply_renderer_compatibility(args)
    assert args.kit_args == backend


@pytest.mark.parametrize("platform,version", [("linux", "5.1.0.0"), ("win32", "5.0.0"), ("win32", "6.0.0")])
def test_other_platforms_and_versions_are_unchanged(cli, monkeypatch, platform, version):
    monkeypatch.setattr(cli.sys, "platform", platform)
    monkeypatch.setattr(cli.metadata, "version", lambda _: version)
    args = argparse.Namespace(kit_args="")
    cli.apply_renderer_compatibility(args)
    assert args.kit_args == ""
