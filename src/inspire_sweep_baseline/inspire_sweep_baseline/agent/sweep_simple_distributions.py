"""Bounded Finger policy with transformed-density PPO likelihoods and exports."""
import math

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import GaussianDistribution

from ..assets.hand_kinematics import FINGER_ACTION_LOWER, FINGER_ACTION_UPPER


class _FingerBoundedOutput(nn.Module):
    def __init__(self):
        super().__init__()
        lower, upper = torch.tensor(FINGER_ACTION_LOWER), torch.tensor(FINGER_ACTION_UPPER)
        self.register_buffer('center', (upper + lower) / 2)
        self.register_buffer('half_range', (upper - lower) / 2)

    def forward(self, latent):
        return self.center + self.half_range * latent.tanh()


class FingerBoundedGaussian(GaussianDistribution):
    """Affine tanh Gaussian; tanh belongs to the policy, NOT Last Action.

    Stored params are the underlying Normal parameters. KL is invariant under
    the common bijection; entropy uses a reparameterized one-sample estimate of
    its Jacobian correction. Log probability inverts the bounded raw command.
    """
    def __init__(self, output_dim, init_std=1.0):
        if output_dim != 12:
            raise ValueError('Finger policy requires 12 outputs')
        super().__init__(output_dim, init_std=init_std, std_type='log')
        self.transform = _FingerBoundedOutput()

    def update(self, mlp_output):
        # Positive, finite exploration scale even after large optimizer updates.
        std = self.log_std_param.clamp(-5., 2.).exp().expand_as(mlp_output)
        self._distribution = Normal(mlp_output, std)

    def sample(self):
        return self.transform(self._distribution.sample())

    def deterministic_output(self, mlp_output):
        return self.transform(mlp_output)

    def as_deterministic_output_module(self):
        import copy
        return copy.deepcopy(self.transform)

    @property
    def mean(self):
        # Deterministic policy center, not the exact transformed expectation.
        return self.transform(self._distribution.mean)

    @property
    def params(self):
        return (self._distribution.mean, self._distribution.stddev)

    def _log_jacobian(self, latent):
        return self.transform.half_range.log() + 2 * (
            math.log(2.) - latent - torch.nn.functional.softplus(-2 * latent))

    @property
    def entropy(self):
        latent = self._distribution.rsample()
        return (self._distribution.entropy() + self._log_jacobian(latent)).sum(-1)

    def log_prob(self, outputs):
        normalized = (outputs - self.transform.center) / self.transform.half_range
        # Floating-point tanh/affine arithmetic can round extreme samples to an endpoint.
        normalized = normalized.clamp(-1 + 1e-6, 1 - 1e-6)
        latent = torch.atanh(normalized)
        return (self._distribution.log_prob(latent) - self._log_jacobian(latent)).sum(-1)
