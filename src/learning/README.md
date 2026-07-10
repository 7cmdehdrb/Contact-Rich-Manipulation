# Newton UR5e Reaching

Newton Physics 기반 UR5e point-reaching 학습 패키지.

Current status:

- 기본 backend는 `newton`이다.
- UR5e xacro는 실행 시 `src/learning/generated/ur5e.urdf`로 자동 변환된다.
- `NewtonReachJointEnv`: policy action is 6D joint-space delta.
- `NewtonReachCartesianEnv`: policy action is 3D EEF displacement, converted internally with differential IK.
- Observation is 21D: joint position, joint velocity, EEF position, target position, target minus EEF.
- Reward follows the IsaacLab reach shape: coarse position tracking penalty plus `1 - tanh(distance / fine_tracking_std)` fine tracking reward, with small action-rate and joint-velocity penalties.
- UR5e description is expected at `src/Universal_Robots_ROS2_Description`.

Use the `newton` conda environment.

Smoke check:

```bash
conda run --no-capture-output -n newton python -m src.learning.scripts.train_joint --num-envs 1 --smoke-steps 200
conda run --no-capture-output -n newton python -m src.learning.scripts.train_cartesian --num-envs 1 --smoke-steps 200
```

Start PPO training:

```bash
conda run --no-capture-output -n newton python -m src.learning.scripts.train_joint --num-envs 128 --iterations 1000
conda run --no-capture-output -n newton python -m src.learning.scripts.train_cartesian --num-envs 128 --iterations 1000
```

If CUDA is available in the `newton` environment:

```bash
conda run --no-capture-output -n newton python -m src.learning.scripts.train_joint --device cuda:0 --rl-device cuda:0 --num-envs 128 --iterations 1000
```

Viewer controls:

- Default is visual Newton GL viewer: `viewer: gl`, `headless: false`.
- Use `--viewer null` only for headless CI/smoke tests.
- Use `--headless` to keep the GL backend headless.
- Rendering is throttled by `--render-fps 30`; use `--render-every N` to render less often.
- PPO metrics are reported by rsl_rl at iteration boundaries.

```bash
conda run --no-capture-output -n newton python -m src.learning.scripts.train_joint --num-envs 32 --iterations 100
```

```bash
conda run --no-capture-output -n newton python -m src.learning.scripts.train_cartesian --device cuda:0 --rl-device cuda:0 --num-envs 128 --iterations 1000
```

The first run may spend time compiling Warp/Newton kernels. Generated URDF assets and training logs are not source files.
