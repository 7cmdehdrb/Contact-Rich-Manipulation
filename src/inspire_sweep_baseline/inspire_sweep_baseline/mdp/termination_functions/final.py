"""Final v1 time-limit accounting without early goal termination."""

from isaaclab.envs.mdp import time_out
from .sweep import shelf_collision
from .goal_terminations import shelf_displacement, _object_position
from ..common.contact_sweep_clutter_fusion_holding import goal_radius


def shelf_termination(env, threshold=0.1, check_fingers=False,
                      wrist_clearance=0.03, displacement_threshold=0.05):
    # TEMP per-step Test-v2 diagnostics: comment out this block to disable.
    if type(env.cfg).__name__ == "RevisionV2TestEnvCfg":
        shelf = env.scene["shelf"].data
        height = shelf.root_pos_w[:, 2] + 1.05
        wrist = env.scene["wrist_frame"].data.target_pos_w[:, 0, 2] - height
        velocity = shelf.root_vel_w.norm(dim=-1)
        initial = shelf.default_root_state[:, :3] + env.scene.env_origins
        displacement = (shelf.root_pos_w - initial).norm(dim=-1)
        for env_id in range(env.num_envs):
            speed = float(velocity[env_id])
            clearance = float(wrist[env_id])
            distance = float(displacement[env_id])
            finger_failed = False
            if check_fingers:
                fingers = env.scene["finger_frame"].data.target_pos_w[env_id, :2, 2] - height[env_id]
                finger_failed = bool((fingers < 0.01).any())
                print(f"  env={env_id} finger_clearance={fingers.tolist()}m < 0.01m "
                      f"-> {finger_failed}", flush=True)
            print(
                f"[Revision-Test-v2 shelf_failure] env={env_id} "
                f"step={int(env.episode_length_buf[env_id])} "
                f"root_vel_6d_norm={speed:.6f} > {threshold:.6f} -> {speed > threshold}; "
                f"wrist_clearance={clearance:.6f}m < {wrist_clearance:.6f}m "
                f"-> {clearance < wrist_clearance}; "
                f"displacement={distance:.6f}m > {displacement_threshold:.6f}m "
                f"-> {distance > displacement_threshold}; check_fingers={check_fingers}; "
                f"terminated={speed > threshold or clearance < wrist_clearance or distance > displacement_threshold or finger_failed}",
                flush=True,
            )
    # END TEMP per-step Test-v2 diagnostics.
    return (shelf_collision(env, threshold, check_fingers, wrist_clearance)
            | shelf_displacement(env, displacement_threshold))


def timeout_goal_reached(env, goal_hold_term_name="goal_hold"):
    """Only the live XY radius is tested, with no speed or hold-time gate."""
    goal = env.command_manager.get_command("target_goal_pos")
    return (_object_position(env)[:, :2] - goal[:, :2]).norm(dim=-1) <= goal_radius(env, goal_hold_term_name)


def success_at_timeout(env, goal_hold_term_name="goal_hold"):
    return time_out(env) & timeout_goal_reached(env, goal_hold_term_name)


def unsuccessful_timeout(env, goal_hold_term_name="goal_hold"):
    return time_out(env) & ~timeout_goal_reached(env, goal_hold_term_name)


def goal_accuracy_success_at_timeout(env, goal_hold_term_name="goal_hold"):
    """Collect metrics and use v1's inclusive timeout partition."""
    env.record_goal_accuracy_metrics()
    return success_at_timeout(env, goal_hold_term_name)


def goal_accuracy_unsuccessful_timeout(env, goal_hold_term_name="goal_hold"):
    return unsuccessful_timeout(env, goal_hold_term_name)
