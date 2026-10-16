"""Probe all 17 tactile channels at their center and four inset corners."""

import argparse
from datetime import datetime
import math
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE_ROOT), str(PACKAGE_ROOT.parent / "inspire_sweep_baseline")]


def arguments():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=85, help="Parallel independent hand/probe pairs.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--channels", nargs="+", type=int, default=None, help="Optional zero-based channel subset (0..16).")
    parser.add_argument("--pad-layout", choices=("original", "exposed"), default="original", help="Test-only pad exposure; source Sweep assets are read-only.")
    parser.add_argument("--pad-clearance", type=float, default=0.001, help="Clearance of exposed pad lower face above parent hull [m].")
    parser.add_argument("--corner-inset", type=float, default=0.2)
    parser.add_argument("--external-forces-every-iteration", action=argparse.BooleanOptionalAction, default=True, help="Apply the force-controlled probe load throughout TGS solver iterations.")
    parser.add_argument("--solver", choices=("TGS", "PGS"), default="TGS")
    parser.add_argument("--physics-hz", type=float, default=100.0)
    parser.add_argument("--probe-radius", type=float, default=0.00075)
    parser.add_argument("--probe-length", type=float, default=0.03)
    parser.add_argument("--clearance", type=float, default=0.015)
    parser.add_argument("--indentation", type=float, default=0.003)
    parser.add_argument("--stiffness", type=float, default=100.0, help="Probe Cartesian position servo [N/m].")
    parser.add_argument("--max-force", type=float, default=1.0, help="Cap on commanded probe force [N].")
    parser.add_argument("--threshold", type=float, default=0.05, help="Tactile detection and reference load threshold [N].")
    parser.add_argument("--baseline-seconds", type=float, default=0.15)
    parser.add_argument("--approach-seconds", type=float, default=0.5)
    parser.add_argument("--hold-seconds", type=float, default=0.3)
    parser.add_argument("--retract-seconds", type=float, default=0.5)
    parser.add_argument("--recovery-seconds", type=float, default=0.2)
    parser.add_argument("--warmup-seconds", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inspire_tactile_validation") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--strict", action="store_true", help="Exit 2 if any trial has no valid detected contact or fails recovery.")
    parser.add_argument("--real-time", action="store_true", help="Pace GUI display at the physics rate.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.repeats < 1:
        parser.error("num-envs and repeats must be positive")
    if not 0 < args.corner_inset < 1:
        parser.error("corner-inset must be between 0 and 1")
    for name in ("pad_clearance", "physics_hz", "probe_radius", "probe_length", "clearance", "indentation", "stiffness", "max_force", "threshold", "baseline_seconds", "approach_seconds", "hold_seconds", "retract_seconds", "recovery_seconds", "warmup_seconds"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.channels is not None and (len(set(args.channels)) != len(args.channels) or any(i < 0 or i > 16 for i in args.channels)):
        parser.error("channels must be distinct integers between 0 and 16")
    return args


if __name__ == "__main__":
    args = arguments()
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(args)
    exit_code = 1
    try:
        from inspire_tactile_validation.experiment import run
        success = run(args, launcher.app)
        exit_code = 2 if args.strict and not success else 0
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Kit shuts down the process itself; set its return code before close.
        import omni.kit.app
        from isaaclab.sim import SimulationContext
        context = SimulationContext.instance()
        if context is not None:
            context._disable_app_control_on_stop_handle = True
        omni.kit.app.get_app().post_quit(exit_code)
        launcher.app.close(wait_for_replicator=False)
    raise SystemExit(exit_code)
