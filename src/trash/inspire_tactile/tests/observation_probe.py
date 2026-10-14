"""GPU regression fixture: identical open/Close commands, raw vs projected readout.

Run twice using env_isaaclab Python (one simulation per process):
  python src/inspire_tactile/tests/observation_probe.py --mode raw --output /tmp/raw.jsonl
  python src/inspire_tactile/tests/observation_probe.py --mode projected --output /tmp/projected.jsonl
Both logs include every physics step's joint angles and cylinder state.
"""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inspire_tactile.model import build_model, repo_root
from inspire_tactile.manual_control import JointControls
import manual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "projected"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=8)
    parser.add_argument("--without_object", action="store_true")
    opts = parser.parse_args()
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True, "device": "cuda:0"})
    code = 0
    try:
        urdf, model = build_model(repo_root(), repo_root() / "src/inspire_tactile/generated")
        controls = JointControls(model)
        original_step = controls.step
        elapsed = 0.0

        def close_at_one_second(dt):
            nonlocal elapsed
            elapsed += dt
            if elapsed >= 1.0:
                controls.preset("close")
            return original_step(dt)

        controls.step = close_at_one_second
        args = SimpleNamespace(headless=True, device="cuda:0", duration=opts.duration,
                               object_offset=(.008, .050, .165), object_lowering=.020,
                               without_object=opts.without_object, fix_object=False, force_conversion=False,
                               threshold=.01, ros2=False, ros_port=9873, ros_hz=30, print_hz=240,
                               log=opts.output, tactile_mode=opts.mode, region_tolerance=.002)
        manual.run(args, launcher.app, urdf, model, controls)
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        import omni.kit.app
        omni.kit.app.get_app().post_quit(code)
        sys.stdout.flush()
        sys.stderr.flush()
        launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
