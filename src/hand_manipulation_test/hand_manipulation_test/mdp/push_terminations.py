"""Pushing outcome precedence: physical failure, settled success, timeout."""

from __future__ import annotations


def _state(env):
    return env.command_manager.get_term("target_position").state()


def push_failure(env):
    return _state(env).failure


def push_success(env):
    state = _state(env)
    return state.success & ~state.failure


def push_time_out(env):
    state = _state(env)
    return (env.episode_length_buf >= env.max_episode_length) & ~state.failure & ~state.success


__all__ = ["push_failure", "push_success", "push_time_out"]
