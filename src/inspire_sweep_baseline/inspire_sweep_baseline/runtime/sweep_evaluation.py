"""Read-only policy-step probe installed only by collect_sweep_results.py.

Capture after the existing reward computation and before ManagerBasedRLEnv's
auto-reset. Never call a reward term or observation manager a second time.
"""
from pathlib import Path

import numpy as np
import torch


class SweepEvaluationProbe:
    def __init__(self, env, output, episodes_per_env, chunk_steps=128):
        self.env = env
        self.output = Path(output)
        self.quota = episodes_per_env
        self.chunk_steps = chunk_steps
        self.completed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.rows = torch.arange(env.num_envs, device=env.device)
        self.initial = torch.zeros((env.num_envs, 3), device=env.device)
        self.shelf_pos = torch.zeros_like(self.initial)
        self.shelf_quat = torch.zeros((env.num_envs, 4), device=env.device)
        self.pending = []
        self.chunks = 0
        self.samples = 0
        self.last_step = None
        self.contact_cfg = self._contact_config()
        self.original_compute = env.reward_manager.compute

        def compute(*args, **kwargs):
            result = self.original_compute(*args, **kwargs)
            self.capture()
            return result

        self.compute_hook = compute
        env.reward_manager.compute = compute

    def _contact_config(self):
        manager = self.env.reward_manager
        # These are explicit adapters for existing semantics, not reward>0 tests.
        windows = {'GoalContactMaintenance', 'TargetContactMaintenance', 'TargetContactDecayMaintenance'}
        for name in manager.active_terms:
            cfg = manager.get_term_cfg(name)
            kind = type(cfg.func).__name__
            module = type(cfg.func).__module__
            if not module.startswith('inspire_sweep_baseline.'):
                continue
            if kind in windows:
                self.contact_term = cfg.func
                self.contact_kind = 'target_resultant_2_of_3_height_gated'
                return cfg
            if kind == 'ContactConditionedSweep':
                self.contact_term = None
                self.contact_kind = 'target_pair_magnitudes_instant_height_gated'
                return cfg
        raise ValueError('No supported Contact adapter. Expected Simple/Finger/Prof target contact terms; '
                         'add an explicit adapter for this task rather than inferring contact from reward.')

    @property
    def finished(self):
        return bool((self.completed >= self.quota).all())

    def prepare_step(self):
        """Snapshot new episodes after reset, before any physics advances."""
        env = self.env
        new = env.episode_length_buf == 0
        data = env.scene['object_collection'].data
        ids = env.target_id[:, 0].long()
        self.initial[new] = data.object_pos_w[self.rows, ids][new]
        shelf = env.scene['shelf'].data
        self.shelf_pos[new] = shelf.root_pos_w[new]
        self.shelf_quat[new] = shelf.root_quat_w[new]

    def capture(self):
        from isaaclab.utils.math import quat_apply_inverse
        from ..mdp.common.contact_sweep_clutter_fusion import SHELF_SURFACE_HEIGHT
        from ..mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name

        env = self.env
        if self.last_step == env.common_step_counter:
            raise RuntimeError('Evaluation probe sampled twice in one policy step')
        self.last_step = env.common_step_counter
        objects = env.scene['object_collection']
        ids = env.target_id[:, 0].long()
        matrices = torch.stack([env.scene[object_hand_sensor_name(name)].data.force_matrix_w
                                for name in objects.object_names], dim=1)
        matrix = matrices[self.rows, ids]
        resultant = -matrix.sum(dim=(1, 2))
        pair_force = matrix.norm(dim=-1).sum(dim=(1, 2))
        resultant_force = resultant.norm(dim=-1)
        fresh = env.sensor_data_fresh
        eef_z = env.scene['ee_frame'].data.target_pos_w[:, 0, 2]
        surface_z = env._revision_initial_shelf_pos[:, 2] + SHELF_SURFACE_HEIGHT
        height_limit = self.contact_cfg.params.get('eef_height_limit', .15)
        threshold = self.contact_cfg.params.get('threshold_n', .01)
        height_valid = (torch.isfinite(eef_z) & torch.isfinite(surface_z)
                        & (eef_z < surface_z + height_limit))
        force = resultant_force if self.contact_term is not None else pair_force
        instant = fresh & torch.isfinite(force) & (force >= threshold) & height_valid
        if self.contact_term is not None:
            if self.contact_term._cached_step != env.common_step_counter:
                raise RuntimeError('Contact term did not update this step (possibly zero reward weight)')
            accepted = (self.contact_term._history.sum(-1) >= 2) & height_valid
            credit = self.contact_term._cached_reward
        else:
            accepted = instant
            credit = torch.full_like(force, float('nan'))

        def local(position):
            return quat_apply_inverse(self.shelf_quat, position - self.shelf_pos)

        manager = env.termination_manager
        sample = {
            'env_id': self.rows, 'episode_id': self.completed.clone(),
            'episode_step': env.episode_length_buf.clone(),
            'global_step': torch.full_like(self.rows, env.common_step_counter),
            'initial_shelf': local(self.initial),
            'goal_shelf': local(env.command_manager.get_command('target_goal_pos')[:, :3]),
            'object_shelf': local(objects.data.object_pos_w[self.rows, ids]),
            'velocity_w': objects.data.object_lin_vel_w[self.rows, ids],
            'target_id': ids, 'shelf_pos_w': self.shelf_pos, 'shelf_quat_w': self.shelf_quat,
            'force_resultant_n': resultant_force, 'force_pair_sum_n': pair_force,
            'sensor_fresh': fresh, 'height_valid': height_valid,
            'contact_instant': instant, 'contact_accepted': accepted,
            'contact_reward_credit': credit,
            'terminated': manager.terminated, 'time_out': manager.time_outs,
            # Distinguish time-limit expiry from other terms marked time_out.
            'horizon_reached': env.episode_length_buf >= env.max_episode_length,
        }
        for name in manager.active_terms:
            sample['term__' + name] = manager.get_term(name)
        active = self.completed < self.quota
        # Boolean indexing copies tensors before reset mutates any live buffers.
        cpu = {key: value[active].detach().cpu().numpy().copy() for key, value in sample.items()}
        self.pending.append(cpu)
        self.samples += len(cpu['env_id'])
        self.completed += (manager.dones & active).long()
        if len(self.pending) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if self.pending:
            path = self.output / f'steps_{self.chunks:06d}.npz'
            # A crash cannot leave a partially written file matching steps_*.npz.
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **{key: np.concatenate([s[key] for s in self.pending])
                                               for key in self.pending[0]})
            temporary.replace(path)
            self.pending.clear()
            self.chunks += 1

    def close(self):
        self.env.reward_manager.compute = self.original_compute
        self.flush()
