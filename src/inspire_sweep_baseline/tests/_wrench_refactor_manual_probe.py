"""Real PhysX checks for Cartesian manual control and isolated reset boundaries."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, default=Path('outputs/wrench_refactor_manual_validation'))
parser.add_argument('--shelf-usd')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
env = panel = None
exit_code = 0
try:
    import torch
    from inspire_sweep_baseline.envs.wrench_refactor_envs import WrenchRefactorEnvCfg
    from inspire_sweep_baseline.runtime.wrench_refactor import WrenchRefactorEnv
    from inspire_sweep_baseline.mdp.action_functions.wrench_refactor import WrenchRefactorOSCAction
    from inspire_sweep_baseline.runtime.wrench_refactor_manual import (
        ShelfSnapshot, WrenchRefactorManualSession, WrenchRefactorManualPanel,
    )

    cfg = WrenchRefactorEnvCfg()
    before = cfg.to_dict()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = 1, args.device, 42
    cfg.actions.arm_action.class_type = WrenchRefactorOSCAction
    cfg.viewer.eye, cfg.viewer.lookat = (.7, 1.0, 1.9), (-.55, 0., 1.1)
    if args.shelf_usd:
        cfg.scene.shelf.spawn.usd_path = args.shelf_usd
    env = WrenchRefactorEnv(cfg)
    env.reset(seed=42)
    session = WrenchRefactorManualSession(env)
    if not args.headless:
        panel = WrenchRefactorManualPanel(session)
        panel.update()
        for _ in range(5):
            env.sim.render()
        import omni.kit.renderer_capture
        args.output.mkdir(parents=True, exist_ok=True)
        omni.kit.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(
            str((args.output / 'panel.png').resolve()))
        for _ in range(5):
            env.sim.render()
    mass = env.scene['object_collection'].root_physx_view.get_masses().clone()
    materials = [m.properties.clone() for m in env._clutter_physics.materials]

    def robot_state():
        r = env.scene['robot'].data
        return r.root_link_state_w.clone(), r.joint_pos.clone(), r.joint_vel.clone()

    def equal_robot(state):
        for current, saved in zip(robot_state(), state):
            torch.testing.assert_close(current, saved, atol=1e-6, rtol=0.)

    def equal_shelf(saved):
        current = ShelfSnapshot.capture(env)
        for a, b in [(current.shelf, saved.shelf), (current.objects, saved.objects),
                     (current.goal_position, saved.goal_position), (current.target_initial_state, saved.target_initial_state)]:
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0.)
        for name in saved.metadata:
            torch.testing.assert_close(current.metadata[name], saved.metadata[name])

    # Isolated shelf randomization must leave even nonzero robot velocities alone.
    robot = env.scene['robot']
    robot.write_joint_velocity_to_sim(torch.full_like(robot.data.joint_vel, .01))
    state = robot_state()
    old_layout = session.snapshot.objects.clone()
    session.reset_shelf()
    equal_robot(state)
    assert not torch.equal(old_layout, session.snapshot.objects)
    assert session.paused
    # Include nonzero shelf and object velocities in a checkpoint.
    shelf = env.scene['shelf']
    shelf.write_root_link_velocity_to_sim(torch.full_like(shelf.data.root_link_vel_w, .001))
    objects = env.scene['object_collection']
    changed = objects.data.object_link_state_w.clone()
    changed[..., 7:] = .002
    objects.write_object_link_state_to_sim(changed)
    session.save_shelf()
    saved = session.snapshot
    for _ in range(3):
        session.reset_shelf()
        # Keep the explicit checkpoint, as randomization normally creates a new one.
        session.snapshot = saved
        state = robot_state()
        session.restore_shelf()
        equal_robot(state)
        equal_shelf(saved)
        expected_flags = (~env._active_object_mask).T.reshape(-1, 1).cpu().to(torch.uint8)
        assert torch.equal(env._wrench_gravity_flags, expected_flags)
    session.reset_shelf()
    saved = ShelfSnapshot.capture(env)
    session.reset_robot()
    equal_shelf(saved)
    torch.testing.assert_close(robot.data.joint_pos, robot.data.default_joint_pos)
    assert torch.count_nonzero(robot.data.joint_vel) == 0
    session.reset_all()
    torch.testing.assert_close(robot.data.joint_pos, robot.data.default_joint_pos)
    equal_shelf(session.snapshot)
    assert torch.equal(env.scene['object_collection'].root_physx_view.get_masses(), mass)
    assert all(torch.equal(m.properties, saved) for m, saved in zip(env._clutter_physics.materials, materials))

    with torch.inference_mode():
        start, _ = session.current_pose()
        values = session.target_values()
        values[2] += .025
        session.set_target(values[:3], values[3:])
        session.hand[:] = torch.tensor([[.6, .3]], device=env.device)
        if panel:
            panel.hand_models[0].set_value(.6)
            panel.hand_models[1].set_value(.3)
        snapshot = session.snapshot
        for tick in range(550):  # Beyond the production 10s timeout, without autoreset.
            session.step()
            if panel and tick % 10 == 0:
                panel.update()
                env.sim.render()
        end, _ = session.current_pose()
        displacement = (end - start)[0].tolist()
        target_error = float((end - session.target_position).norm())
        print(f'CARTESIAN_TRACKING displacement={displacement}, error={target_error}', flush=True)
        assert displacement[2] > .01 and target_error < .015, (displacement, target_error)
        assert session.snapshot is snapshot
        assert env.episode_length_buf.item() == 0  # RL lifecycle is not evaluated.
        hand = env.action_manager.get_term('gripper_action')
        torch.testing.assert_close(hand.processed_actions, torch.tensor([[.6, .3]], device=env.device))
        assert torch.isfinite(robot.data.joint_pos).all()
        axis_errors = []
        for axis in (0, 1):
            values = session.target_values()
            values[axis] += .01
            session.set_target(values[:3], values[3:])
            for tick in range(150):
                session.step()
                if panel and tick % 10 == 0:
                    panel.update()
                    env.sim.render()
            position, _ = session.current_pose()
            error = float((position - session.target_position).norm())
            assert error < .003, (axis, error)
            axis_errors.append(error)
        from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul
        values = session.target_values()
        values[5] += 3.
        session.set_target(values[:3], values[3:])
        for tick in range(250):
            session.step()
            if panel and tick % 10 == 0:
                panel.update()
                env.sim.render()
        _, rotation = session.current_pose()
        rotation_error = float(axis_angle_from_quat(quat_mul(session.target_rotation, quat_conjugate(rotation))).norm())
        print(f'CARTESIAN_XY_ROTATION errors_m={axis_errors}, rotation_error_rad={rotation_error}', flush=True)
        assert rotation_error < math.radians(1.), rotation_error
        state = robot_state()
        session.restore_shelf()
        equal_robot(state)
        equal_shelf(snapshot)
        # GUI button dispatch uses the same public operations, outside render callbacks.
        if panel:
            old_x = float(session.target_position[0, 0])
            panel._jog(0, 1)
            panel.update()
            assert abs(float(session.target_position[0, 0]) - old_x - .005) < 1e-6
            panel.pose_models[2].set_value(session.target_values()[2] + .005)
            panel._apply()
            panel.update()
            panel.jobs.append('reset_shelf')
            panel.jobs.append('restore_shelf')
            panel.jobs.append('reset_robot')
            panel.jobs.append('reset_all')
            panel.update()
            env.sim.render()
    report = dict(isolated_shelf_reset=True, isolated_robot_reset=True, snapshot_state_and_velocity=True,
                  repeated_restore=True, gravity_mask=True, fixed_materials=True, full_reset=True,
                  no_autoreset_11_seconds=True, cartesian_displacement_m=displacement, hand_2d=True,
                  cartesian_target_error_m=target_error,
                  xy_target_errors_m=axis_errors, rotation_target_error_rad=rotation_error,
                  gui_panel=panel is not None)
    # A fresh production config remains byte-for-byte equivalent as a Python dict.
    assert WrenchRefactorEnvCfg().to_dict() == before
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / ('gui.json' if panel else 'headless.json')).write_text(json.dumps(report, indent=2))
    print('MANUAL_PROBE_OK ' + json.dumps(report), flush=True)
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
    app.close()
