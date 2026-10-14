"""Pushing rewards from the shared, memoized physical episode transition."""

from __future__ import annotations


def _state(env):
    return env.command_manager.get_term("target_position").state()


def push_approach_reward(env):
    return _state(env).approach_delta / env.step_dt


def push_progress_reward(env):
    # An impulse in the bounded distance potential. Cancel the manager's dt
    # integration so sensitivity and episode budget do not depend on policy Hz.
    return _state(env).progress_delta / env.step_dt


def push_backslide_reward(env):
    return _state(env).backslide_delta / env.step_dt


def push_first_contact_reward(env):
    return _state(env).first_contact.float() / env.step_dt


def push_contact_reward(env):
    # A deliberately tiny rate, rather than an impulse, rewards maintaining
    # real, aligned, grounded side contact. Its manager weight is 0.005.
    return _state(env).maintaining_contact


def push_alignment_reward(env):
    return _state(env).alignment_penalty


def push_success_reward(env):
    return _state(env).success.float() / env.step_dt


def push_failure_reward(env):
    return -_state(env).failure.float() / env.step_dt


__all__ = ["push_approach_reward", "push_progress_reward", "push_backslide_reward", "push_first_contact_reward",
           "push_contact_reward", "push_alignment_reward", "push_success_reward", "push_failure_reward"]
