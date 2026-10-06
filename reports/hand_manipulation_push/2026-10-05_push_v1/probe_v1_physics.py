"""Real OSC side-contact feasibility probe; never teleports robot or Cube during rollout."""
import argparse
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/hand_manipulation_test"))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--left-height", type=float, default=.075)
parser.add_argument("--right-height", type=float, default=.03)
parser.add_argument("--left-backoff", type=float, default=.008)
parser.add_argument("--right-backoff", type=float, default=.008)
parser.add_argument("--cube-x", type=float, default=-.70)
parser.add_argument("--hand-open", type=float, default=None)
parser.add_argument("--palm-shift", type=float, default=0.)
parser.add_argument("--left-roll-deg", type=float, default=30.)
parser.add_argument("--table-width", type=float, default=.36)
parser.add_argument("--pad-exposure", type=float, default=0.)
parser.add_argument("--anchored-rotation", action="store_true")
parser.add_argument("--hand-stiffness", type=float, default=None)
parser.add_argument("--thumb-up", action="store_true")
parser.add_argument("--steps", type=int, default=350)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("v1_physics.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, fast_shutdown=True)
app = launcher.app

def main():
    import gymnasium as gym
    import torch
    import hand_manipulation_test
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, compute_pose_error, quat_from_matrix, quat_mul
    from hand_manipulation_test.push_math import push_palm_rotation
    from hand_manipulation_test.geometry import control_point_pose_w
    task = "Isaac-Hand-Manipulation-Push-v1"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    cfg.commands.target_position.debug_vis = False
    cfg.scene.ee_frame.debug_vis = False
    cfg.task.left_finger_down_rad = 0.
    cfg.task.table_size = (args.table_width,1.,.04)
    cfg.scene.table.spawn.size = cfg.task.table_size
    if args.hand_stiffness is not None:
        drive=cfg.scene.robot.actuators["hand"]
        drive.stiffness=args.hand_stiffness; drive.damping=.5; drive.effort_limit_sim=1.5
    if args.pad_exposure:
        from pxr import Gf, UsdGeom
        from isaaclab.sim.utils import clone, get_current_stage
        from hand_manipulation_test.assets.contact_robot import spawn_contact_robot
        from hand_manipulation_test.assets.robot import PALM_SENSOR_BODY_NAMES
        @clone
        def exposed_spawn(path,spawner,translation=None,orientation=None,**kwargs):
            root=spawn_contact_robot.__wrapped__(path,spawner,translation=translation,orientation=orientation,**kwargs)
            stage=get_current_stage(); root_path=str(root.GetPath()); cache=UsdGeom.XformCache()
            normal=cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path+"/inspire_base_link")).TransformDir(Gf.Vec3d(0,1,0))
            for name in PALM_SENSOR_BODY_NAMES:
                if name=="inspire_palm_force_sensor": continue
                body=stage.GetPrimAtPath(root_path+"/"+name); collision=stage.GetPrimAtPath(root_path+"/"+name+"/collisions")
                body_transform=cache.GetLocalToWorldTransform(body)
                sign=1 if Gf.Dot(body_transform.TransformDir(Gf.Vec3d(0,0,1)),normal)>0 else -1
                displacement=body_transform.TransformDir(Gf.Vec3d(0,0,sign*args.pad_exposure))
                world=cache.GetLocalToWorldTransform(collision); world.SetTranslateOnly(world.ExtractTranslation()+displacement)
                local=world*cache.GetLocalToWorldTransform(collision.GetParent()).GetInverse()
                xf=UsdGeom.Xformable(collision); xf.ClearXformOpOrder(); xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local)
            return root
        cfg.scene.robot.spawn.func=exposed_spawn
    if args.cube_x != -.70:
        x_offset = args.cube_x-cfg.task.table_center[0]
        cfg.task.object_xy_offset_low = (x_offset,-.06)
        cfg.task.object_xy_offset_high = (x_offset,.06)
    env = gym.make(task, cfg=cfg).unwrapped
    report = {"left_height_m": args.left_height, "right_height_m": args.right_height, "steps": []}
    try:
        safe = env.event_manager.get_term_cfg("safe_hand").func
        command = env.command_manager.get_term("target_position")
        arm = env.action_manager.get_term("arm_action")
        hand = env.action_manager.get_term("hand_action")
        original_sample = safe._sample_pose
        def shifted_sample(index):
            original_sample(index)
            safe.desired_palm_position_w[index,0] += args.palm_shift
            direction = command.direction_w[index]
            rotation = push_palm_rotation(direction)
            angle = direction.new_full((len(index),),math.radians(args.left_roll_deg))
            angle = torch.where(direction[:,1]<0,angle,torch.zeros_like(angle))
            up = torch.zeros_like(direction); up[:,2]=1
            finger = angle.cos()[:,None]*rotation[:,:,2]-angle.sin()[:,None]*up
            rotation = torch.stack((torch.linalg.cross(direction,finger,dim=-1),direction,finger),dim=-1)
            if args.thumb_up:
                rotation=torch.stack((up,direction,torch.linalg.cross(up,direction,dim=-1)),dim=-1)
            safe.desired_c_pos_w[index] = safe.desired_palm_position_w[index]+(rotation@(safe.c_offset_h-safe.palm_reference_h).unsqueeze(-1)).squeeze(-1)
            safe.desired_c_quat_w[index] = quat_mul(quat_from_matrix(rotation),safe.c_quat_h.expand(len(index),-1))
        safe._sample_pose = shifted_sample
        ids = torch.arange(4, device=env.device)
        position = torch.tensor([[args.cube_x,-.06,1.08],[args.cube_x,.06,1.08]]*2, device=env.device)
        angles = torch.tensor([0.,0.,math.pi,math.pi],device=env.device)
        command.set_reset_specs(ids, position, angles, .2)
        env.reset(seed=42)
        desired_q = safe.desired_c_quat_w.clone()
        if args.anchored_rotation:
            from isaaclab.utils.math import quat_conjugate
            anchored_q=quat_mul(quat_conjugate(env.scene["robot"].data.root_link_quat_w),desired_q)
            original_process=arm.process_actions_for_envs
            def anchored_process(actions,index):
                original_process(actions,index)
                arm._desired_c_pose_b[index,3:]=anchored_q[index]
                arm._osc.set_command(arm._desired_c_pose_b)
            arm.process_actions_for_envs=anchored_process
        height = torch.tensor([args.right_height]*2+[args.left_height]*2,device=env.device)
        start = command.target_pos_w.clone()
        failures = torch.zeros(4,device=env.device,dtype=torch.long)
        peaks = torch.zeros(4,device=env.device)
        max_forward = torch.zeros(4,device=env.device)
        contacts = torch.zeros(4,device=env.device,dtype=torch.long)
        valid_contacts = contacts.clone()
        for step in range(args.steps):
            direction = command.direction_w
            palm, _ = safe.palm_reference_pose_w()
            current_c, current_q = control_point_pose_w(env)
            cube_pos = env.scene["target_object"].data.root_pos_w
            backoff = .14 if step < 80 else direction.new_tensor([args.right_backoff]*2+[args.left_backoff]*2)[:,None]
            target_palm = cube_pos - backoff * direction
            target_palm[:,0] += args.palm_shift
            target_palm[:,2] += height
            delta = target_palm-palm-quat_apply(current_q,arm.position_target_error_c_m)
            action = torch.zeros((4,8),device=env.device)
            action[:,:3] = quat_apply_inverse(current_q,delta)/action.new_tensor(cfg.actions.arm_action.translation_scale)
            _, rotation_error = compute_pose_error(current_c,current_q,current_c,desired_q,rot_error_type="axis_angle")
            action[:,3:6] = quat_apply_inverse(current_q,rotation_error)/action.new_tensor(cfg.actions.arm_action.rotation_scale)
            action[:,6:] = hand.synergy_to_action(hand.actual_synergy)
            if args.hand_open is not None:
                action[:,6:] = hand.synergy_to_action(action.new_full((4,2),args.hand_open))
            _, reward, terminated, truncated, extra = env.step(action.clamp(-1,1))
            state = command.state()
            raw_force = env.scene["cube_palm_contacts"].data.force_matrix_w_history
            peak = raw_force.norm(dim=-1).flatten(1).amax(-1)
            peaks = torch.maximum(peaks,peak)
            forward = ((env.scene["target_object"].data.root_pos_w-start)*direction).sum(-1)
            max_forward = torch.maximum(max_forward,forward)
            contacts += state.palm_cube_contact.long()
            valid_contacts += state.valid_push_contact.long()
            failures += terminated.long()
            if step%10==0 or bool(terminated.any()):
                report["steps"].append({"step":step,"force_n":peak.tolist(),"forward_m":forward.tolist(),
                    "palm_local_w":(palm-env.scene.env_origins).tolist(),"contact":state.palm_cube_contact.tolist(),
                    "valid_contact":state.valid_push_contact.tolist(),"grounded":state.grounded.tolist(),
                    "normal_alignment_cos":state.palm_alignment_cos.tolist(),"force_alignment_cos":state.force_alignment_cos.tolist(),
                    "cube_local_w":(env.scene["target_object"].data.root_pos_w-env.scene.env_origins).tolist(),
                    "terminated":terminated.tolist(),"target_error_m":arm.position_target_error_c_m.norm(dim=-1).tolist()})
        report["summary"]={"failure_count":failures.tolist(),"force_peak_n":peaks.tolist(),
                           "max_forward_m":max_forward.tolist(),"raw_contact_steps":contacts.tolist(),
                           "valid_contact_steps":valid_contacts.tolist()}
        args.output.write_text(json.dumps(report,indent=2))
        print("V1_PHYSICS",json.dumps(report["summary"]),flush=True)
    finally:
        env.close()

status=0
try:
    main()
except Exception:
    traceback.print_exc()
    status=1
finally:
    import omni.kit.app
    sys.stdout.flush()
    omni.kit.app.get_app().post_quit(status)
    app.close()
raise SystemExit(status)
