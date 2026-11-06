"""Interactive Cartesian control and shelf checkpoints for Wrench Refactor.

The production environment/configuration is unchanged. This session advances
only its action manager and physics; RL rewards, terminations and automatic
resets are deliberately not evaluated during human interaction.
"""

from collections import deque
from dataclasses import dataclass
import math

import torch


SHELF_FIELDS = (
    'target_id', 'target_width', 'sweep_dir', '_active_object_mask',
    '_initial_obstacle_observation', '_initial_object_positions_w',
    '_revision_spawn_positions', '_final_actual_population', '_revision_layout_complete',
)


@dataclass
class ShelfSnapshot:
    shelf: torch.Tensor
    objects: torch.Tensor
    metadata: dict[str, torch.Tensor]
    goal_position: torch.Tensor
    target_initial_state: torch.Tensor

    @classmethod
    def capture(cls, env):
        goal = env.command_manager.get_term('target_goal_pos')
        shelf = env.scene['shelf'].data
        objects = env.scene['object_collection'].data
        # Use explicit link poses + COM velocities with the matching legacy
        # state setters. This Isaac Lab version's link-velocity setter also
        # overwrites its cached link-state velocity with COM velocity.
        return cls(
            torch.cat((shelf.root_link_pose_w, shelf.root_com_vel_w), dim=-1),
            torch.cat((objects.object_link_pose_w, objects.object_com_vel_w), dim=-1),
            {name: getattr(env, name).clone() for name in SHELF_FIELDS},
            goal.pos_command_w.clone(), goal.target_init_state_w.clone(),
        )

    def restore(self, env):
        from ..mdp.event_functions.wrench_refactor import apply_fixed_pool_gravity_mask

        env.scene['shelf'].write_root_state_to_sim(self.shelf.clone())
        env.scene['object_collection'].write_object_state_to_sim(self.objects.clone())
        for name, value in self.metadata.items():
            getattr(env, name).copy_(value)
        goal = env.command_manager.get_term('target_goal_pos')
        goal.pos_command_w.copy_(self.goal_position)
        goal.target_init_state_w.copy_(self.target_initial_state)
        apply_fixed_pool_gravity_mask(env, torch.arange(env.num_envs, device=env.device))


def cartesian_action(current_position, current_rotation, target_position, target_rotation,
                     frame_rotation, position_scale, orientation_scale, max_translation, max_rotation):
    """World pose target -> bounded RPY input in the supplied command frame.

    The target remains absolute across steps, so releasing a jog button holds
    that target instead of repeatedly commanding zero relative displacement.
    """
    from isaaclab.utils.math import (
        axis_angle_from_quat, euler_xyz_from_quat, quat_apply_inverse,
        quat_conjugate, quat_from_angle_axis, quat_mul,
    )

    def bounded(vector, limit):
        return vector * (limit / vector.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp_max(1.)

    translation = bounded(quat_apply_inverse(frame_rotation, target_position - current_position), max_translation)
    world_rotation = axis_angle_from_quat(quat_mul(target_rotation, quat_conjugate(current_rotation)))
    rotation = bounded(quat_apply_inverse(frame_rotation, world_rotation), max_rotation)
    angle = rotation.norm(dim=-1)
    delta = quat_from_angle_axis(angle, rotation / angle[:, None].clamp_min(1e-9))
    rpy = torch.stack(euler_xyz_from_quat(delta), dim=-1)
    # Isaac Lab reports Euler angles in [0, 2*pi); action needs signed deltas.
    rpy = torch.remainder(rpy + math.pi, 2 * math.pi) - math.pi
    return torch.cat((translation / position_scale, rpy / orientation_scale), dim=-1)


def advance_reference(position, rotation, target_position, target_rotation, distance, angle):
    """Slew the desired pose independently of tracking error, allowing contact force."""
    from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul

    delta = target_position - position
    next_position = position + delta * (distance / delta.norm(dim=-1, keepdim=True).clamp_min(1e-9)).clamp_max(1.)
    rotation_delta = axis_angle_from_quat(quat_mul(target_rotation, quat_conjugate(rotation)))
    magnitude = rotation_delta.norm(dim=-1)
    limited = magnitude.clamp_max(angle)
    delta_quat = quat_from_angle_axis(limited, rotation_delta / magnitude[:, None].clamp_min(1e-9))
    return next_position, quat_mul(delta_quat, rotation)


class WrenchRefactorManualSession:
    """One production Refactor v0 environment, independently resettable parts."""

    def __init__(self, env, linear_speed=.04, angular_speed=.4):
        if env.num_envs != 1:
            raise ValueError('Manual control requires exactly one environment')
        if not all(math.isfinite(v) and v > 0 for v in (linear_speed, angular_speed)):
            raise ValueError('Movement speeds must be positive and finite')
        self.env = env
        self.linear_speed, self.angular_speed = linear_speed, angular_speed
        self.hand = torch.ones((1, 2), device=env.device)
        self.paused = False
        self.steps = 0
        self.snapshot_number = 0
        self.status = 'Ready'
        self.hold_actual()
        self.save_shelf()

    def current_pose(self):
        frame = self.env.scene['ee_frame'].data
        return frame.target_pos_w[:, 0].clone(), frame.target_quat_w[:, 0].clone()

    def hold_actual(self):
        self.target_position, self.target_rotation = self.current_pose()
        self.reference_position = self.target_position.clone()
        self.reference_rotation = self.target_rotation.clone()
        self.status = 'EEF target set to actual pose'

    def set_target(self, xyz, rpy_degrees):
        from isaaclab.utils.math import quat_from_euler_xyz

        if len(xyz) != 3 or len(rpy_degrees) != 3 or not all(
            math.isfinite(value) for value in (*xyz, *rpy_degrees)
        ):
            raise ValueError('XYZ and RPY must contain six finite numbers')
        self.target_position[:] = torch.tensor(xyz, device=self.env.device)
        angles = torch.tensor(rpy_degrees, device=self.env.device).deg2rad().reshape(1, 3)
        self.target_rotation[:] = quat_from_euler_xyz(*angles.unbind(-1))
        self.status = 'Cartesian target applied'

    def target_values(self):
        from isaaclab.utils.math import euler_xyz_from_quat

        angles = torch.stack(euler_xyz_from_quat(self.target_rotation), -1)
        angles = torch.remainder(angles + math.pi, 2 * math.pi) - math.pi
        return [*self.target_position[0].tolist(), *angles[0].rad2deg().tolist()]

    def _refresh(self):
        # Forward kinematics only: reset buttons must never integrate the
        # untouched robot or shelf as a side effect of refreshing the GUI.
        for sensor in self.env.scene.sensors.values():
            sensor.reset()
        self.env.sensor_data_fresh[:] = False
        self.env.sim.forward()
        self.env.scene.update(0.)

    def save_shelf(self):
        self.snapshot = ShelfSnapshot.capture(self.env)
        self.snapshot_number += 1
        self.status = f'Shelf snapshot {self.snapshot_number} saved'

    def restore_shelf(self):
        self.snapshot.restore(self.env)
        self.env.scene['shelf'].reset()
        self.env.scene['object_collection'].reset()
        self._refresh()
        self.paused = True
        self.status = f'Shelf snapshot {self.snapshot_number} restored; physics paused'

    def reset_shelf(self):
        from ..mdp.event_functions.wrench_refactor import reset_scene

        env = self.env
        shelf = env.scene['shelf']
        state = shelf.data.default_root_state.clone()
        state[:, :3] += env.scene.env_origins
        shelf.write_root_state_to_sim(state)
        shelf.reset()
        env.scene['object_collection'].reset()
        reset_scene(env, torch.arange(1, device=env.device), **env.cfg.events.object_spawn.params)
        env._initial_object_positions_w.copy_(env.scene['object_collection'].data.object_pos_w)
        env.command_manager.reset(torch.arange(env.num_envs, device=env.device))
        self._refresh()
        self.save_shelf()
        self.paused = True
        self.status = 'New shelf layout + snapshot; robot unchanged; physics paused'

    def reset_robot(self):
        robot = self.env.scene['robot']
        state = robot.data.default_root_state.clone()
        state[:, :3] += self.env.scene.env_origins
        robot.write_root_state_to_sim(state)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.joint_vel))
        robot.reset()
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel))
        robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_vel))
        self.env.action_manager.reset()
        self.hand[:] = 1.
        self._refresh()
        self.hold_actual()
        self.paused = True
        self.status = 'Robot pose reset; shelf unchanged; physics paused'

    def reset_all(self):
        self.env.reset()
        self.hand[:] = 1.
        self._refresh()
        self.hold_actual()
        self.save_shelf()
        self.paused = True
        self.status = 'Full reset + new shelf snapshot; physics paused'

    def step(self):
        env = self.env
        position, rotation = self.current_pose()
        arm = env.cfg.actions.arm_action
        self.reference_position, self.reference_rotation = advance_reference(
            self.reference_position, self.reference_rotation, self.target_position, self.target_rotation,
            self.linear_speed * env.step_dt, self.angular_speed * env.step_dt,
        )
        command = cartesian_action(
            position, rotation, self.reference_position, self.reference_rotation,
            rotation, arm.position_scale, arm.orientation_scale,
            arm.position_scale, arm.orientation_scale,
        )
        action = torch.cat((command, self.hand.clamp(0., 1.)), dim=-1)
        if not torch.isfinite(action).all():
            raise RuntimeError('Nonfinite manual action')
        env.sensor_data_fresh[:] = True
        env.action_manager.process_action(action)
        for _ in range(env.cfg.decimation):
            env._sim_step_counter += 1
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        self.steps += 1
        if not torch.isfinite(env.scene['robot'].data.joint_pos).all():
            raise RuntimeError('Nonfinite robot joint state')


class WrenchRefactorManualPanel:
    """Native Kit panel. Callbacks queue physics mutations for the outer loop."""

    def __init__(self, session):
        import omni.ui as ui

        self.session = session
        self.jobs = deque()
        self.window = ui.Window(
            'Wrench Refactor - Manual Control', width=540, height=720,
            position_x=max(20, ui.Workspace.get_main_window_width() - 570), position_y=60,
        )
        with self.window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    ui.Label('EEF Cartesian target | WORLD axes | XYZ meters / RPY degrees', height=24)
                    self.actual = ui.Label('', height=42, word_wrap=True)
                    names = ('X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw')
                    values = session.target_values()
                    self.pose_models = [None] * 6
                    for row in range(3):
                        with ui.HStack(height=26):
                            for index in (row, row + 3):
                                ui.Label(names[index], width=55)
                                model = ui.SimpleFloatModel(values[index])
                                ui.FloatField(model=model, width=195)
                                self.pose_models[index] = model
                    with ui.HStack(height=30):
                        ui.Button('Apply XYZ / RPY', clicked_fn=self._apply)
                        ui.Button('Hold actual EEF', clicked_fn=lambda: self.jobs.append('hold_actual'))
                    with ui.HStack(height=26):
                        ui.Label('Jog: meters / degrees', width=210)
                        self.jog_distance = ui.SimpleFloatModel(.005)
                        self.jog_angle = ui.SimpleFloatModel(2.)
                        ui.FloatField(model=self.jog_distance)
                        ui.FloatField(model=self.jog_angle)
                    for row in range(3):
                        with ui.HStack(height=25):
                            for axis in (row, row + 3):
                                ui.Label(names[axis], width=55)
                                ui.Button('-', width=90, clicked_fn=lambda a=axis: self._jog(a, -1))
                                ui.Button('+', width=90, clicked_fn=lambda a=axis: self._jog(a, 1))
                    ui.Separator(height=8)
                    ui.Label('Hand 2D | 0 = closed, 1 = open', height=24)
                    self.hand_models = []
                    for name in ('Finger flexion', 'Thumb opposition'):
                        with ui.HStack(height=28):
                            ui.Label(name, width=150)
                            model = ui.SimpleFloatModel(1.)
                            ui.FloatSlider(model=model, min=0., max=1.)
                            ui.FloatField(model=model, width=65)
                            self.hand_models.append(model)
                    ui.Separator(height=8)
                    with ui.HStack(height=32):
                        ui.Button('Reset ALL (new layout)', clicked_fn=lambda: self.jobs.append('reset_all'))
                        ui.Button('Reset SHELF (new layout)', clicked_fn=lambda: self.jobs.append('reset_shelf'))
                    with ui.HStack(height=32):
                        ui.Button('Restore shelf snapshot', clicked_fn=lambda: self.jobs.append('restore_shelf'))
                        ui.Button('Save shelf snapshot', clicked_fn=lambda: self.jobs.append('save_shelf'))
                    with ui.HStack(height=32):
                        ui.Button('Reset ROBOT pose', clicked_fn=lambda: self.jobs.append('reset_robot'))
                        self.pause_button = ui.Button('Pause physics', clicked_fn=lambda: self.jobs.append('pause'))
                    ui.Label('Reset / restore pauses physics. Resume when ready.\n'
                             'A new layout replaces the saved shelf snapshot.', height=38, word_wrap=True)
                    self.status = ui.Label(session.status, height=42, word_wrap=True)

    def _apply(self):
        self.jobs.append(('target', [model.as_float for model in self.pose_models]))

    def _jog(self, axis, direction):
        distance = self.jog_distance.as_float if axis < 3 else self.jog_angle.as_float
        self.jobs.append(('jog', axis, direction * distance))

    def _sync(self):
        for model, value in zip(self.pose_models, self.session.target_values()):
            model.set_value(value)
        for model, value in zip(self.hand_models, self.session.hand[0].tolist()):
            model.set_value(value)

    def update(self):
        session = self.session
        # Hand edits are live; copy them before jobs so reset_robot/all can
        # replace stale GUI values with the production fully-open preset.
        hand = [model.as_float for model in self.hand_models]
        if all(math.isfinite(value) for value in hand):
            session.hand[:] = torch.tensor(hand, device=session.env.device).clamp(0., 1.)
        while self.jobs:
            job = self.jobs.popleft()
            try:
                if isinstance(job, tuple):
                    values = job[1] if job[0] == 'target' else session.target_values()
                    if job[0] == 'jog':
                        values[job[1]] += job[2]
                    session.set_target(values[:3], values[3:])
                elif job == 'pause':
                    session.paused = not session.paused
                    session.status = 'Physics paused' if session.paused else 'Physics running'
                else:
                    getattr(session, job)()
                self._sync()
            except ValueError as error:
                session.status = str(error)
        position, _ = session.current_pose()
        error = float((position - session.target_position).norm())
        self.actual.text = ('Actual XYZ: ' + ', '.join(f'{value:.4f}' for value in position[0].tolist())
                            + f' m\nPosition error: {error * 1000:.1f} mm')
        self.pause_button.text = 'Resume physics' if session.paused else 'Pause physics'
        self.status.text = f'{session.status} | snapshot {session.snapshot_number}'

    def close(self):
        self.window.destroy()
