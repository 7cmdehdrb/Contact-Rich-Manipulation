"""Check numerical boundaries without changing healthy PPO updates or rewards.

Large finite observations can overflow losses and poison std during the next
update. Stop at the first invalid boundary and preserve evidence, rather than
silently replacing data, dropping batches, or restarting the simulator.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO


class NumericalTrainingError(RuntimeError):
    """An unsafe rollout or update was rejected; inspect numerical_failure.pt."""


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if hasattr(value, "items"):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(item) for item in value]
    return value


class NumericalPPO(PPO):
    """Checkpoint-compatible safety boundary around the installed PPO algorithm.

    Limits reject samples; they never clip physical observations or rewards.
    They are deliberately broad numerical ceilings, not sensor calibrations.
    """

    def __init__(self, *args, max_abs_observation=1e6, max_abs_value=1e6,
                 max_abs_return=1e6, max_value_loss=1e6, min_std=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        for name, value in {
            "max_abs_observation": max_abs_observation, "max_abs_value": max_abs_value,
            "max_abs_return": max_abs_return, "max_value_loss": max_value_loss, "min_std": min_std,
        }.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            setattr(self, name, value)
        self.diagnostic_dir = None
        self.iteration = None
        self._observations = None
        self._snapshot = None
        self.iteration_reader = None
        self.std_projections = 0
        # MLP output checks run before Normal.update/sample, including PPO's
        # minibatch actor reevaluation, which bypasses act().
        self.optimizer.register_step_pre_hook(self._before_step)
        self.optimizer.register_step_post_hook(self._after_step)

    @contextmanager
    def _model_checks(self):
        # Temporary hooks must not leak into deepcopy/JIT/ONNX policy exports.
        actor = self.actor.mlp.register_forward_hook(self._actor_output)
        critic = self.critic.register_forward_hook(self._critic_output)
        try:
            yield
        finally:
            actor.remove()
            critic.remove()

    def _check(self, name, value, limit=None):
        valid = torch.isfinite(value)
        if limit is not None:
            valid = valid & (value.abs() <= limit)
        if not bool(valid.all()):
            indices = (~valid).nonzero()[:16].detach().cpu().tolist()
            self._fail(f"{name}: invalid or above ceiling {limit}; indices={indices}")

    def _fail(self, reason):
        if self.diagnostic_dir is not None:
            directory = Path(self.diagnostic_dir)
            directory.mkdir(parents=True, exist_ok=True)
            payload = {
                "reason": reason, "iteration": self.iteration,
                "observations": _cpu(dict(self._observations.items())) if self._observations is not None else None,
                "state_before_update": _cpu(self._snapshot), "current_state": _cpu(self.save()),
                "rollout": {name: _cpu(getattr(self.storage, name)) for name in
                            ("observations", "actions", "rewards", "dones", "values", "returns", "advantages")},
                "gradients": {f"{kind}/{name}": _cpu(p.grad) for kind, model in
                              (("actor", self.actor), ("critic", self.critic))
                              for name, p in model.named_parameters() if p.grad is not None},
            }
            # Save on failure only; no recurring checkpoint or log cleanup.
            torch.save(payload, directory / "numerical_failure.pt")
            (directory / "numerical_failure.json").write_text(
                json.dumps({"reason": reason, "iteration": self.iteration}, indent=2) + "\n", encoding="utf-8")
        raise NumericalTrainingError(reason + "; unsafe training data/update rejected; no automatic restart")

    def _check_observations(self, obs):
        self._observations = obs
        for name, value in obs.items():
            # Dummy obstacle coordinates are intentionally ignored by the real
            # encoder. Apply the same mask for validation, leaving obs intact.
            if name == "obstacles" and value.shape[-1] % 4 == 0:
                rows = value.unflatten(-1, (value.shape[-1] // 4, 4))
                self._check("observation/obstacles/present", rows[..., 0], self.max_abs_observation)
                value = torch.where(rows[..., :1] > .5, rows, torch.zeros_like(rows))
            self._check(f"observation/{name}", value, self.max_abs_observation)

    def _check_std(self):
        distribution = self.actor.distribution
        if hasattr(distribution, "std_param"):
            self._check("actor/std", distribution.std_param)
            if bool((distribution.std_param <= 0).any()):
                self._fail("actor/std: non-positive checkpoint or policy parameter")
        elif hasattr(distribution, "log_std_param"):
            self._check("actor/log_std", distribution.log_std_param)
            self._check("actor/std", distribution.log_std_param.exp())

    def _actor_output(self, module, inputs, output):
        self._check("actor/mean", output, self.max_abs_observation)
        self._check_std()

    def _critic_output(self, module, inputs, output):
        self._check("critic/value", output, self.max_abs_value)

    def act(self, obs):
        if self.iteration_reader is not None:
            self.iteration = self.iteration_reader()
        self._check_observations(obs)
        with self._model_checks():
            return super().act(obs)

    def process_env_step(self, obs, rewards, dones, extras):
        self._check_observations(obs)
        self._check("environment/rewards", rewards, self.max_abs_return)
        super().process_env_step(obs, rewards, dones, extras)
        # PPO clears transition after copying it into storage.
        self._check("bootstrap/stored_rewards", self.storage.rewards[self.storage.step - 1], self.max_abs_return)

    def compute_returns(self, obs):
        self._check_observations(obs)
        with self._model_checks():
            super().compute_returns(obs)
        self._check("rollout/returns", self.storage.returns, self.max_abs_return)
        self._check("rollout/advantages", self.storage.advantages, self.max_abs_return)

    def _before_step(self, optimizer, args, kwargs):
        self._check_named([
            (f"gradient/{kind}/{name}", parameter.grad)
            for kind, model in (("actor", self.actor), ("critic", self.critic))
            for name, parameter in model.named_parameters() if parameter.grad is not None
        ])

    def _check_named(self, tensors):
        # One device synchronization per boundary rather than per parameter.
        if tensors and not bool(torch.stack([torch.isfinite(value).all() for _, value in tensors]).all()):
            for name, value in tensors:
                self._check(name, value)

    def _check_state(self):
        tensors = [(f"parameter/{kind}/{name}", parameter)
                   for kind, model in (("actor", self.actor), ("critic", self.critic))
                   for name, parameter in model.named_parameters()]
        tensors.extend((f"optimizer/{index}/{key}", value)
                       for index, state in enumerate(self.optimizer.state.values())
                       for key, value in state.items() if torch.is_tensor(value))
        # Adam's scalar step counters may be on CPU while moments are on CUDA.
        for device in {value.device for _, value in tensors}:
            self._check_named([(name, value) for name, value in tensors if value.device == device])

    def _after_step(self, optimizer, args, kwargs):
        self._check_state()
        # Project scalar noise onto the positive domain while preserving the
        # parameter name/shape and all existing scalar checkpoints/Adam state.
        distribution = self.actor.distribution
        if hasattr(distribution, "std_param"):
            with torch.no_grad():
                low = distribution.std_param < self.min_std
                if bool(low.any()):
                    self.std_projections += int(low.sum())
                    distribution.std_param.clamp_(min=self.min_std)

    def update(self):
        self._snapshot = copy.deepcopy(self.save())
        try:
            with self._model_checks():
                losses = super().update()
            for name, value in losses.items():
                if not math.isfinite(value) or (name == "value" and value > self.max_value_loss):
                    self._fail(f"loss/{name}: {value}")
            return losses
        except NumericalTrainingError:
            # Do not resume/retry. Restore only model/optimizer memory so a
            # caller cannot accidentally save the poisoned update as healthy.
            super().load(self._snapshot, None, True)
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            raise
        finally:
            self._snapshot = None

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self._check_state()
        self._check_std()
        return result
