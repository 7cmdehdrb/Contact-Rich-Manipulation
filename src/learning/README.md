# Newton UR5e Reaching

Newton Physics 기반 UR5e point-reaching 학습 패키지.

Current status:

- 기본 backend는 `newton`이다.
- UR5e xacro는 실행 시 `src/learning/generated/ur5e.urdf`로 자동 변환된다.
- `NewtonReachJointEnv`: policy action is 6D joint-space delta.
- `NewtonReachCartesianEnv`: policy action is 3D EEF displacement, converted internally with differential IK.
- Observation is 21D: joint position, joint velocity, EEF position, target position, target minus EEF.
- UR5e description is expected at `src/Universal_Robots_ROS2_Description`.

Use the `newton` conda environment.

Smoke check:

```bash
conda run -n newton python -m src.learning.scripts.train_joint --num-envs 4 --smoke-steps 2
conda run -n newton python -m src.learning.scripts.train_cartesian --num-envs 4 --smoke-steps 2
```

Start PPO training:

```bash
conda run -n newton python -m src.learning.scripts.train_joint --num-envs 128 --iterations 1000
conda run -n newton python -m src.learning.scripts.train_cartesian --num-envs 128 --iterations 1000
```

If CUDA is available in the `newton` environment:

```bash
conda run -n newton python -m src.learning.scripts.train_joint --device cuda:0 --rl-device cuda:0 --num-envs 128 --iterations 1000
```

The first run may spend time compiling Warp/Newton kernels. Generated URDF assets and training logs are not source files.


```bash
conda run -n newton python -m src.learning.scripts.train_joint --device cuda:0 --rl-device cuda:0 --num-envs 128 --iterations 1000
```

```bash
conda run -n newton python -m src.learning.scripts.train_cartesian --device cuda:0 --rl-device cuda:0 --num-envs 128 --iterations 1000
```
