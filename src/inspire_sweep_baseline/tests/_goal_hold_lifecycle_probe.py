"""GPU lifecycle check with robot/object pinned; not a policy evaluation.

Exercise an actual environment through the default ten-second timeout.
Pin simulated states before physics substeps to isolate reward/reset semantics
from policy skill and the separately documented OSC dynamics instability.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(
    "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2",
    "Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2",
), default="Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg

import inspire_sweep_baseline  # noqa: F401

tasks = (args.task,)
try:
    with torch.inference_mode():
        for task in tasks:
            cfg = parse_env_cfg(task, device=args.device or "cuda:0", num_envs=2)
            cfg.seed = 42
            env = gym.make(task, cfg=cfg).unwrapped
            original_apply = None
            try:
                env.reset()
                assert "success" not in env.termination_manager.active_terms
                robot = env.scene["robot"]
                objects = env.scene["object_collection"]
                position = env.scene["shelf"].data.root_pos_w + torch.tensor(
                    [0., 0., 1.05], device=env.device,
                )
                fixed_state = torch.zeros((env.num_envs, 1, 13), device=env.device)
                fixed_state[:, 0, :3] = position
                fixed_state[:, 0, 3] = 1.
                command = env.command_manager.get_term("target_goal_pos")
                original_apply = env.action_manager.apply_action

                def pinned_apply():
                    robot.write_joint_state_to_sim(
                        robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_vel),
                    )
                    objects.write_object_link_state_to_sim(fixed_state)
                    command.pos_command_w[:] = position
                    original_apply()

                env.action_manager.apply_action = pinned_apply
                bonus_index = env.reward_manager.active_terms.index("success")
                hold_index = env.reward_manager.active_terms.index("goal_hold")
                paid = torch.zeros(env.num_envs, device=env.device)
                held = torch.zeros_like(paid)
                actions = torch.zeros((env.num_envs, 8), device=env.device)
                actions[:, 6:] = 1.
                steps = env.max_episode_length
                for step in range(1, steps + 1):
                    _, _, terminated, timeout, _ = env.step(actions)
                    paid += env.reward_manager._step_reward[:, bonus_index] * env.step_dt
                    held += env.reward_manager._step_reward[:, hold_index] * env.step_dt
                    assert not terminated.any(), (task, step, "unexpected failure/success reset")
                    assert bool(timeout.all()) == (step == steps), (task, step, "early/missing timeout")
                    if step == 25:
                        assert torch.allclose(paid, torch.full_like(paid, 100.)), (task, paid)
                    if step % 100 == 0:
                        print(f"GOAL_HOLD_PROGRESS {task} {step}/{steps}", flush=True)
                assert torch.allclose(paid, torch.full_like(paid, 100.)), (task, paid)
                assert torch.allclose(held, torch.full_like(held, 195.2), atol=0.02), (task, held)
                # Auto-reset must rearm the episode latch and clear the timer.
                env.step(actions)
                assert env.reward_manager._step_reward[:, bonus_index].count_nonzero() == 0
                assert torch.allclose(
                    env.reward_manager._step_reward[:, hold_index] * env.step_dt,
                    torch.full_like(held, 0.016), atol=1e-5,
                )
                print("GOAL_HOLD_LIFECYCLE " + json.dumps({
                    "task": task, "timeout_step": steps, "episode_seconds": cfg.episode_length_s,
                    "success_bonus": paid.tolist(), "hold_reward": held.tolist(),
                    "passed": True, "pinned_states": True,
                }), flush=True)
            finally:
                if original_apply is not None:
                    env.action_manager.apply_action = original_apply
                env.close()
finally:
    app.app.close()
