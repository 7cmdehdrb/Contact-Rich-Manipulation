"""Ordinary Gaussian policy; execution clipping belongs to the environment."""
import torch
from torch import nn
from rsl_rl.modules import GaussianDistribution

from ..assets.hand_kinematics import FINGER_ACTION_LOWER, FINGER_ACTION_UPPER


class FingerGaussian(GaussianDistribution):
    """Untransformed raw samples and Normal likelihoods, including PPO storage.

    Only initialization differs from the library Gaussian. No action squashing,
    inverse transform, observation normalization, or density correction is used.
    Deterministic/export outputs are also raw commands: deployment must apply
    the same per-axis execution bounds and scales as the environment.
    """

    action_names = ('translation_x', 'translation_y', 'translation_z',
                    'rotation_x', 'rotation_y', 'rotation_z',
                    'thumb', 'index', 'middle', 'ring', 'little', 'thumb_opposition')

    def __init__(self, output_dim, init_std=None, report_clip_fraction=True):
        if output_dim != 12:
            raise ValueError('Finger policy requires 12 outputs')
        lower, upper = torch.tensor(FINGER_ACTION_LOWER), torch.tensor(FINGER_ACTION_UPPER)
        if init_std is None:
            init_std = torch.cat((torch.full((3,), .2), torch.full((3,), .5),
                                  (upper[6:] - lower[6:]) / 6))
        std = torch.as_tensor(init_std, dtype=torch.float32).expand(output_dim)
        if not bool((torch.isfinite(std) & (std > 0)).all()):
            raise ValueError('Initial standard deviations must be finite and positive')
        super().__init__(output_dim, init_std=std, std_type='log')
        self.report_clip_fraction = report_clip_fraction
        # Constants move with the module but are not checkpoint parameters.
        self.register_buffer('_action_lower', lower, persistent=False)
        self.register_buffer('_action_upper', upper, persistent=False)

    def init_mlp_weights(self, mlp: nn.Module):
        # Small initial state dependence around zero arm / midpoint hand means.
        last = [layer for layer in mlp.modules() if isinstance(layer, nn.Linear)][-1]
        with torch.no_grad():
            last.weight.mul_(.01)
            last.bias.copy_((self._action_lower + self._action_upper) / 2)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # The old latent mean/std have a different meaning. Even non-strict
        # loading must not silently reinterpret a transformed-policy checkpoint.
        if any(key.startswith(prefix + 'transform.') for key in state_dict):
            error_msgs.append('Cannot load a tanh policy checkpoint into FingerGaussian; '
                              'start a new training run (action semantics changed).')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def action_diagnostics(self, raw_actions):
        """Per-policy-step, environment-mean telemetry in original raw units."""
        mean, std = self.mean.detach().mean(0), self.std.detach().mean(0)
        metrics = [('mean', mean), ('std', std)]
        if self.report_clip_fraction:
            clipped = ((raw_actions < self._action_lower) | (raw_actions > self._action_upper)).float().mean(0)
            metrics.insert(0, ('clip_fraction', clipped))
        return {f'PolicyAction/{kind}/{name}': values[i].clone()
                for kind, values in metrics
                for i, name in enumerate(self.action_names)}
