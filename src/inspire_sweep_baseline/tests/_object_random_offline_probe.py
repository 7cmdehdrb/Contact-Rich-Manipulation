"""GPU regression: finish one PPO iteration, then disconnect this process only.

Linux/libseccomp test harness, not part of environment initialization. No host
firewall, network interface, DNS, or other process is changed.
"""
from pathlib import Path
import ctypes
import errno
import json
import os
import socket
import stat
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OUTPUT = Path('outputs/object_random_validation/offline_training')
OUTPUT.mkdir(parents=True, exist_ok=True)


def disconnect_process():
    """TSYNC a kernel filter to native workers, then shut down existing IP sockets."""
    lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
    class ArgCompare(ctypes.Structure):
        _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_uint),
                    ('datum_a', ctypes.c_uint64), ('datum_b', ctypes.c_uint64)]
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_attr_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                          ctypes.c_uint, ctypes.POINTER(ArgCompare)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(0x7fff0000)  # SCMP_ACT_ALLOW
    if not ctx:
        raise RuntimeError('seccomp_init failed')
    try:
        # SCMP_FLTATR_CTL_TSYNC: also affect threads created before this call.
        assert lib.seccomp_attr_set(ctx, 4, 1) == 0
        denied = 0x00050000 | errno.ENETDOWN
        for domain in (socket.AF_INET, socket.AF_INET6):
            compare = ArgCompare(0, 4, domain, 0)  # SCMP_CMP_EQ, socket(domain,...)
            assert lib.seccomp_rule_add_array(ctx, denied,
                lib.seccomp_syscall_resolve_name(b'socket'), 1, ctypes.byref(compare)) == 0
        # Block reconnects on pre-existing unconnected sockets as well.
        assert lib.seccomp_rule_add_array(ctx, denied,
            lib.seccomp_syscall_resolve_name(b'connect'), 0, None) == 0
        result = lib.seccomp_load(ctx)
        if result != 0:
            raise RuntimeError(f'seccomp_load/TSYNC failed: {result}')
    finally:
        lib.seccomp_release(ctx)
    shut_down = 0
    for fd in list(Path('/proc/self/fd').iterdir()):
        try:
            if not stat.S_ISSOCK(os.fstat(int(fd.name)).st_mode):
                continue
            # Duplicate the fd; never close a native worker's original descriptor.
            with socket.socket(fileno=os.dup(int(fd.name))) as sock:
                if sock.getsockopt(socket.SOL_SOCKET, socket.SO_DOMAIN) in (socket.AF_INET, socket.AF_INET6):
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                        shut_down += 1
                    except OSError:
                        pass
        except OSError:
            pass
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError as exc:
            assert exc.errno == errno.ENETDOWN, exc
        else:
            sock.close()
            raise AssertionError('Kernel network block did not apply')
    return shut_down


from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
try:
    import importlib.metadata
    import torch
    import gymnasium as gym
    import imageio.v3 as iio
    import inspire_sweep_baseline
    import isaaclab.sim as sim
    from isaaclab.sensors import CameraCfg
    from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from inspire_sweep_baseline.agent.curriculum_runner import CurriculumOnPolicyRunner

    task = 'Isaac-Inspire-Contact-Sweep-Clutter-Fusion-Holding-OSC-ObjectRandom-v0'
    cfg = parse_env_cfg(task, device='cuda:0', num_envs=4)
    cfg.seed = 42
    cfg.log_dir = str(OUTPUT)
    cfg.scene.offline_camera = CameraCfg(
        prim_path='{ENV_REGEX_NS}/OfflineCamera', width=640, height=360, data_types=['rgb'],
        spawn=sim.PinholeCameraCfg(focal_length=18., horizontal_aperture=24.))
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    agent = handle_deprecated_rsl_rl_cfg(agent, importlib.metadata.version('rsl-rl-lib'))
    assert agent.logger == 'tensorboard', agent.logger
    agent.device = cfg.sim.device
    agent.save_interval = 2  # Exercise local saves repeatedly; do not touch training logs.
    env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
    raw = env.unwrapped
    runner = CurriculumOnPolicyRunner(env, agent.to_dict(), log_dir=str(OUTPUT/'tensorboard'), device=raw.device)
    camera = raw.scene['offline_camera']
    origins = raw.scene.env_origins
    camera.set_world_poses_from_view(
        origins+torch.tensor([.05,-.95,1.65],device=raw.device),
        origins+torch.tensor([-.72,0.,1.10],device=raw.device))
    # Cut off after the first actual PPO update, inside one uninterrupted learn().
    original_log = runner.logger.log
    reset_counts = [0]*raw.num_envs
    original_reset = raw._reset_idx
    report = dict(task=task, num_envs=4, logger=agent.logger, offline=False)
    def counted_reset(ids):
        if report['offline']:
            for index in ids.cpu().tolist(): reset_counts[index] += 1
        return original_reset(ids)
    raw._reset_idx = counted_reset
    def log_and_disconnect(*args, **kwargs):
        original_log(*args, **kwargs)
        if not report['offline']:
            # Prove this process could reach the real asset service before cutoff.
            online = socket.create_connection(('omniverse-content-production.s3-us-west-2.amazonaws.com',443),timeout=10)
            report['closed_ip_sockets'] = disconnect_process()
            try:
                online.sendall(b'offline-probe')
            except OSError:
                report['existing_connection_blocked'] = True
            else:
                raise AssertionError('Existing TCP socket still writable')
            online.close()
            report['offline'] = True
            report['cutoff_after_iteration'] = runner.current_learning_iteration
            with open(OUTPUT/'status.json','w') as f: json.dump(report,f,indent=2)
            print('OFFLINE CUTOFF: kernel blocks IPv4/IPv6 sockets and reconnects on all threads',flush=True)
    runner.logger.log = log_and_disconnect
    runner.learn(num_learning_iterations=13, init_at_random_ep_len=False)
    # Explicit full reset and render after offline PPO, including assets moved into view.
    with torch.inference_mode():
        env.reset()
        for _ in range(8): raw.sim.render()
        camera.update(raw.step_dt,force_recompute=True)
    iio.imwrite(OUTPUT/'offline_preview.png',camera.data.output['rgb'][0,:,:,:3].cpu().numpy())
    checkpoint = OUTPUT/'tensorboard/model_12.pt'
    assert checkpoint.exists()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['iter'] == 12
    assert all(count > 1 for count in reset_counts), reset_counts
    assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
    assert all(torch.isfinite(p).all() for p in runner.alg.critic.parameters())
    assert raw._active_object_mask.sum().item() == 60
    report.update(status='passed', final_iteration=runner.current_learning_iteration,
                  offline_iterations=12, offline_control_steps=12*agent.num_steps_per_env,
                  offline_time_s=12*agent.num_steps_per_env*raw.step_dt,
                  offline_resets_per_env=reset_counts, checkpoint=str(checkpoint), preview='offline_preview.png')
    with open(OUTPUT/'result.json','w') as f: json.dump(report,f,indent=2)
    print('OFFLINE PPO/RESET/RENDER/CHECKPOINT PASSED',json.dumps(report),flush=True)
    env.close()
except BaseException:
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
finally:
    app.close(skip_cleanup=True)
