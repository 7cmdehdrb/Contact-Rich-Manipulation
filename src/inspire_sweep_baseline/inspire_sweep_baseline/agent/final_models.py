"""Final v1/v2's 57D policy input, with the unchanged masked obstacle encoder."""

from rsl_rl.models import MLPModel
from .contact_sweep_clutter_fusion_models import ContactSweepClutterFusionModel


class FinalV1Model(ContactSweepClutterFusionModel):
    def _get_obs_dim(self, obs, obs_groups, obs_set):
        if obs_set not in ('actor', 'critic'):
            raise ValueError('Final observation set must be actor or critic')
        expected = ['policy', self.obstacle_group]
        if obs_set == 'critic':
            expected.insert(1, 'privileged')
        if obs_groups[obs_set] != expected:
            raise ValueError(f'{obs_set} observation groups must be {expected} in order')
        if obs['policy'].ndim != 2 or obs['policy'].shape[-1] != 57:
            raise ValueError('Final v1/v2 require 57D policy observations')
        obstacles = obs[self.obstacle_group]
        if obstacles.ndim != 2 or obstacles.shape[-1] != 4 * self.obstacle_observation_slots:
            raise ValueError(f'Expected {4 * self.obstacle_observation_slots}D obstacle observations')
        if obs_set == 'critic' and (obs['privileged'].ndim != 2 or obs['privileged'].shape[-1] != 3):
            raise ValueError('Privileged target position must have shape (num_envs, 3)')
        direct = [name for name in expected if name != self.obstacle_group]
        return MLPModel._get_obs_dim(self, obs, {obs_set: direct}, obs_set)
