"""Episode-initial Actor position and live object state for the Critic."""

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse, quat_unique, subtract_frame_transforms


class InitialObjectPositionInRoot(ManagerTermBase):
    """Latch the initial object position in robot-root coordinates per reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._position_b = torch.zeros((env.num_envs, 3), device=env.device)
        self._pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        # ObservationManager resets before CommandManager. Capture lazily after
        # the goal command has snapshotted the newly spawned object's link pose.
        self._pending[slice(None) if env_ids is None else env_ids] = True

    def __call__(self, env, command_name="target_goal_pos"):
        ids = self._pending.nonzero(as_tuple=False).flatten()
        if ids.numel():
            robot = env.scene["robot"]
            initial = env.command_manager.get_term(command_name).target_init_state_w
            position, _ = subtract_frame_transforms(
                robot.data.root_pos_w[ids], robot.data.root_quat_w[ids], initial[ids, :3],
            )
            self._position_b[ids] = position
            self._pending[ids] = False
        # ObservationManager may apply in-place modifiers; never expose cache.
        return self._position_b.clone()


def live_object_state_in_root(env):
    """13D [position, quaternion, linear velocity, angular velocity] Scene GT.

    Position/velocity refer to the link origin (the cylinder bottom), matching
    the initial position's origin. Robot root is fixed; world velocities are
    rotated into its axes. This term has no actor noise or initial-position latch.
    """
    robot = env.scene["robot"]
    objects = env.scene["object_collection"]
    rows = torch.arange(env.num_envs, device=env.device)
    state = objects.data.object_link_state_w[rows, env.target_id[:, 0].long()]
    root_q = robot.data.root_quat_w
    position, quaternion = subtract_frame_transforms(
        robot.data.root_pos_w, root_q, state[:, :3], state[:, 3:7],
    )
    linear = quat_apply_inverse(root_q, state[:, 7:10])
    angular = quat_apply_inverse(root_q, state[:, 10:13])
    return torch.cat((position, quat_unique(quaternion), linear, angular), dim=-1)
