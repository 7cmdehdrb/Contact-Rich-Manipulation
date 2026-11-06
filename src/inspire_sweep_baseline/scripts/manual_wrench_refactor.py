"""Launch Refactor v0 for Cartesian EEF / 2D hand control without a policy.

Run from the workspace root with the Isaac Lab Python environment:
    python src/inspire_sweep_baseline/scripts/manual_wrench_refactor.py
"""

import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def arguments():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--linear-speed', type=float, default=.04, help='Cartesian target increment limit [m/s].')
    parser.add_argument('--angular-speed', type=float, default=.4, help='Rotation increment limit [rad/s].')
    parser.add_argument('--shelf-usd', type=str, help='Optional local copy or URI of the original shelf USD.')
    parser.add_argument('--steps', type=int, default=0, help='Exit after N loop ticks; 0 keeps the GUI open.')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.linear_speed, args.angular_speed)):
        parser.error('Movement speeds must be positive and finite')
    if args.steps < 0 or (args.headless and args.steps == 0):
        parser.error('--steps must be nonnegative; headless requires --steps > 0')
    return args


def main():
    args = arguments()
    from isaaclab.app import AppLauncher
    from cli_args import apply_renderer_compatibility

    apply_renderer_compatibility(args)
    launcher = AppLauncher(args)
    env = panel = None
    exit_code = 0
    try:
        import torch
        from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorEnvCfg
        from inspire_sweep_baseline.runtime.wrench_refactor import WrenchRefactorEnv
        from inspire_sweep_baseline.mdp.action_functions.wrench_refactor import WrenchRefactorOSCAction
        from inspire_sweep_baseline.runtime.wrench_refactor_manual import (
            WrenchRefactorManualSession, WrenchRefactorManualPanel,
        )

        cfg = WrenchRefactorEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = args.seed
        # Refactor v3's corrected offset-EEF Jacobian and link velocity;
        # preserve the v0 scene, physical properties and two-channel hand.
        cfg.actions.arm_action.class_type = WrenchRefactorOSCAction
        cfg.viewer.eye, cfg.viewer.lookat = (.7, 1.0, 1.9), (-.55, 0., 1.1)
        if args.shelf_usd:
            cfg.scene.shelf.spawn.usd_path = args.shelf_usd
        env = WrenchRefactorEnv(cfg)
        env.reset(seed=args.seed)
        session = WrenchRefactorManualSession(env, args.linear_speed, args.angular_speed)
        if not args.headless:
            panel = WrenchRefactorManualPanel(session)
        print('MANUAL READY: world XYZ/RPY, 2D hand, independent shelf/robot resets; no checkpoint.', flush=True)
        ticks = 0
        with torch.inference_mode():
            while launcher.app.is_running() and (args.steps == 0 or ticks < args.steps):
                started = time.perf_counter()
                if env.sim.is_stopped() or (panel is not None and not panel.window.visible):
                    break
                if env.sim.is_playing():
                    if panel:
                        panel.update()
                    if not session.paused:
                        session.step()
                env.sim.render()
                ticks += 1
                remaining = env.step_dt - (time.perf_counter() - started)
                if remaining > 0 and not args.headless:
                    time.sleep(remaining)
        print(f'MANUAL DONE: physics_steps={session.steps}, ticks={ticks}', flush=True)
    except KeyboardInterrupt:
        print('MANUAL STOPPED', flush=True)
    except BaseException:
        exit_code = 1
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if panel:
            panel.close()
        if env:
            env.close()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(exit_code)
        launcher.app.close()


if __name__ == '__main__':
    main()
