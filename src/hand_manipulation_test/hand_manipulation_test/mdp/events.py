"""Direct deterministic reset without an object-relative IK placement."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def reset_robot_to_initial_state(env, env_ids: Sequence[int] | torch.Tensor | None) -> None:
    robot = env.scene["robot"]
    index = torch.arange(env.num_envs, device=env.device) if env_ids is None else torch.as_tensor(
        env_ids, device=env.device, dtype=torch.long
    )
    root_state = robot.data.default_root_state[index].clone()
    root_state[:, :3] += env.scene.env_origins[index]
    root_state[:, 7:] = 0.0
    joint_position = robot.data.default_joint_pos[index].clone()
    joint_velocity = torch.zeros_like(joint_position)
    robot.write_root_pose_to_sim(root_state[:, :7], env_ids=index)
    robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=index)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=index)
    robot.set_joint_position_target(joint_position, env_ids=index)
    robot.set_joint_velocity_target(joint_velocity, env_ids=index)
    robot.set_joint_effort_target(torch.zeros_like(joint_position), env_ids=index)
    # Joint writes invalidate ArticulationData.body_link_pose_w. The command's
    # subsequent lazy pose read refreshes FK without advancing physics time.


__all__ = ["reset_robot_to_initial_state"]
