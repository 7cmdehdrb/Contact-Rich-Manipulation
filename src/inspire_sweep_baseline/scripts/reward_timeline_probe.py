"""Read-only reward instrumentation; preserve callable order, caches and reset."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class _CallableProbe:
    def __init__(self, original, owner, index, name):
        self.original, self.owner, self.index, self.name = original, owner, index, name

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, env, **params):
        if self.name == 'wrench_change':
            self.owner.wrench_previous = self.original._previous_wrench.clone()
            self.owner.wrench_previous_valid = self.original._previous_valid.clone()
        value = self.original(env, **params)
        self.owner.raw[:, self.index] = value
        self.owner.evaluated[:, self.index] = True
        if self.name == 'wrench_change':
            self.owner.wrench_current = self.original._previous_wrench.clone()
        return value


class RewardTimelineProbe:
    def __init__(self, env, rows, output, report):
        self.env, self.rows, self.output = env, rows, Path(output)
        manager = env.reward_manager
        self.names = list(manager.active_terms)
        self.raw = torch.zeros((env.num_envs,len(self.names)),device=env.device)
        self.evaluated = torch.zeros_like(self.raw,dtype=torch.bool)
        self.weights = torch.tensor([manager.get_term_cfg(name).weight for name in self.names],device=env.device)
        metadata=[]
        for i,name in enumerate(self.names):
            cfg=manager.get_term_cfg(name)
            original=cfg.func
            metadata.append(dict(name=name,weight=cfg.weight,params=cfg.params,
                callable=f'{getattr(original,"__module__",type(original).__module__)}:{getattr(original,"__qualname__",type(original).__qualname__)}'))
            cfg.func=_CallableProbe(original,self,i,name)
        (self.output/'reward_terms.json').write_text(json.dumps(metadata,indent=2,default=str),encoding='utf-8')
        report.update(reward_terms=self.names,reward_weights=self.weights.cpu().tolist(),
            reward_raw_definition='Original callable output once; may be a rate or distance/step_dt. Zero weight: raw zero with evaluated=False.',
            reward_contribution_definition='Original raw * live weight * policy dt, as applied by RewardManager; total is before reset.',
            initial_cohort='episode=0 only, one completed episode per initial environment; later auto-resets retained but excluded from paired comparison')
        if hasattr(env._stabilize_state, 'reach_delta'):
            report['v1_diagnostics'] = {
                'reach_geometry': 'world-space meters; reference angle is radians clockwise from shelf -X at reset',
                'sweep_angle_xy': 'radians, zero when sweep_gate is false',
                'components': 'reach_positive/negative/net, sweep_positive/direction_negative/retreat are weighted per-step contributions',
                'joint_failure_event': 'actual termination-manager boolean, captured before auto-reset',
            }
        original_compute=manager.compute

        def compute(dt):
            self.raw.zero_();self.evaluated.zero_()
            value=original_compute(dt)
            live_weights=self.raw.new_tensor([manager.get_term_cfg(n).weight for n in self.names])
            contribution=self.raw*live_weights*dt
            records=dict(reward_raw=self.raw,reward_weighted_rate=manager._step_reward,
                reward_contribution=contribution,reward_total=value,reward_evaluated=self.evaluated,
                reward_weights=live_weights[None].expand(env.num_envs,-1),
                wrench_previous=self.wrench_previous,wrench_current=self.wrench_current,
                wrench_previous_valid=self.wrench_previous_valid)
            state = env._stabilize_state
            if hasattr(state, 'reach_delta'):
                for name in ('reach_target_center', 'reach_radius', 'reach_region', 'reach_reference_angle',
                             'reach_reference_point', 'reach_distance', 'reach_excess', 'reach_previous_excess',
                             'reach_delta', 'sweep_angle_xy', 'sweep_direction_coefficient', 'sweep_gate',
                             'sweep_side_multiplier', 'sweep_positive', 'sweep_direction_negative'):
                    records[name] = getattr(state, name)
                reach = state.reach_delta * live_weights[self.names.index('reaching')]
                sweep_weight = live_weights[self.names.index('sweep_progress')]
                records.update(reach_positive=reach.clamp_min(0.), reach_negative=reach.clamp_max(0.),
                               reach_net=reach, sweep_positive=state.sweep_positive * sweep_weight,
                               sweep_direction_negative=state.sweep_direction_negative * sweep_weight,
                               sweep_retreat=state.retreat * live_weights[self.names.index('sweep_retreat')],
                               joint_failure_event=env.termination_manager.get_term('arm_joint_velocity'))
            for name,tensor in records.items():
                self.rows.setdefault(name,[]).append(tensor.detach().cpu().numpy().copy())
            return value
        manager.compute=compute
        self.hand_ids=manager.get_term_cfg('hand_joint_velocity').params['asset_cfg'].joint_ids
        self.arm_ids=manager.get_term_cfg('arm_joint_velocity').params['asset_cfg'].joint_ids
        self.eef_body=env.scene['robot'].body_names.index('axia80_link')

    def state_inputs(self):
        from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion_contacts import hand_obstacle_contact_force_w
        env=self.env; robot=env.scene['robot'].data; objects=env.scene['object_collection'].data
        shelf=env.scene['shelf'].data; eef=env.scene['ee_frame'].data.target_pos_w[:,0]
        offset=eef-robot.body_link_pos_w[:,self.eef_body]
        eef_velocity=robot.body_link_lin_vel_w[:,self.eef_body]+torch.cross(robot.body_link_ang_vel_w[:,self.eef_body],offset,dim=-1)
        surface=shelf.root_pos_w.clone();surface[:,2]+=1.06
        return dict(eef_velocity_w=eef_velocity,previous_action=env.action_manager.prev_action,
            arm_joint_velocities=robot.joint_vel[:,self.arm_ids],hand_joint_velocities=robot.joint_vel[:,self.hand_ids],
            obstacle_force_w=hand_obstacle_contact_force_w(env),
            obstacle_displacement_squared=(objects.object_pos_w-env._initial_object_positions_w).square().sum(-1),
            shelf_near_eef=(eef-surface).norm(dim=-1)<.2,
            shelf_moved_proxy=((shelf.root_pos_w-env._revision_initial_shelf_pos).norm(dim=-1)+shelf.root_vel_w.norm(dim=-1))>.005,
            finger_height=env.scene['finger_frame'].data.target_pos_w[:,:2,2]-surface[:,2,None],
            shelf_penalty_wrist_height=env.scene['wrist_frame'].data.target_pos_w[:,0,2]-surface[:,2],
            sensor_fresh=env.sensor_data_fresh)

    def save_initial(self, observations):
        from isaaclab.utils.io import dump_yaml
        env=self.env
        bank=dict(robot_joint_pos=env.scene['robot'].data.joint_pos,
            robot_joint_vel=env.scene['robot'].data.joint_vel,
            object_state=env.scene['object_collection'].data.object_state_w,
            object_bounds=env._revision_bounds_t,active_objects=env._active_object_mask,
            shelf_state=env.scene['shelf'].data.root_state_w,
            goal=env.command_manager.get_command('target_goal_pos'),sweep_direction=env.sweep_dir,
            initial_obstacle_observation=env._initial_obstacle_observation,
            surface_target=env._surface_target_w,surface_obstacles=env._surface_obstacles_w,
            surface_valid=env._surface_valid)
        bank.update(eef_pos_w=env.scene['ee_frame'].data.target_pos_w[:, 0],
                    eef_quat_w=env.scene['ee_frame'].data.target_quat_w[:, 0],
                    shelf_reference_pos_w=env._revision_initial_shelf_pos,
                    shelf_reference_quat_w=env._revision_initial_shelf_quat)
        for key,value in observations.items():
            if isinstance(value,torch.Tensor):bank['observation_'+key]=value
        bank={k:v.detach().cpu().numpy().copy() for k,v in bank.items()}
        np.savez_compressed(self.output/'initial_scene.npz',**bank)
        hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in bank.items()}
        (self.output/'initial_scene_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
        torch.save(dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all()),self.output/'torch_rng_state.pt')
        dump_yaml(str(self.output/'environment.yaml'),env.cfg)
        (self.output/'initial_scene_notes.json').write_text(json.dumps(dict(
            object_names=env.scene['object_collection'].object_names,robot_joint_names=env.scene['robot'].joint_names,
            observations='Policy observations retained with original corruption; no future ground-truth inputs added.',
            replay='Initial physical and observation snapshots support auditing. Exact resume also needs task internal state, RNGs and PhysX contact solver state; this is not a full simulator checkpoint.'),indent=2),encoding='utf-8')
