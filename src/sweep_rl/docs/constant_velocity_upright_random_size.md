# ConstantVelocity Gripper Exclusion

환경 ID:

```text
Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0
```

> `UprightRandomSize`는 기존 스크립트와 checkpoint 디렉터리 호환성을 위한 이름이다.
> 현재 환경에는 upright 자세 보상이나 물체 크기 랜덤화가 없다.

구현 클래스 `UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg`는
`UR5eOscSweepConstantVelocityEnvCfg`를 상속한다. 부모의 12-D Action, 55-D
Observation, 4-D 속도 command, reward와 안전 종료를 유지하면서 외측 pad 접근과
gripper 내부 삽입 방지를 추가한다.

## 부모 환경과의 차이

### 외측 pad 접근

`push_pose_error`만 교체된다.

| 항목 | 값 |
|---|---:|
| Weight | `-1.0` |
| 거리 scale | `0.10 m` |
| pad 정면 stand-off | EEF-local `X = ±0.065 m` |
| 좌·우 pad 중심 | EEF-local `Y = ±0.055 m` |
| 목표 높이 | EEF-local `Z = 0` |
| raw penalty 최대값 | `3.0` |

현재 물체가 EEF-local 좌·우 어느 pad에 가까운지에 따라 pad 중심을 선택한다. 밀기
방향을 EEF frame으로 변환해 사용할 pad 면의 `±X` 부호를 정하므로 특정 world-frame
orientation을 강제하지 않는다.

### Gripper 내부 삽입 실패

`object_inside_gripper` termination을 추가한다. 물체 중심이 EEF-local exclusion box에
들어가면 실패한다.

```text
XYZ half extents = (0.040, 0.040, 0.058) m
```

box는 EEF와 함께 회전한다. 이 종료는 `failure_termination`의 남은 horizon 비용에도
포함되므로 정책이 고의 종료로 running cost를 회피할 수 없다.

## 유지되는 계약

| 구분 | 값 |
|---|---|
| Action | 12-D variable-stiffness OSC |
| Observation | 55-D |
| Command | `[direction_x, direction_y, distance_m, target_speed_mps]` |
| 물체 | 고정 `0.06 m` cube, `0.35 kg` |
| Scene | 부모와 동일, `replicate_physics=True` |
| Episode | 8초 |
| PPO experiment | `ur5e_osc_sweep_constant_velocity_upright_random_size` |

세부 Action/Observation/Reward는
[ConstantVelocity 계약](constant_velocity_action_reward_observation.md)을 참고한다.

## 학습과 플레이

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size.py \
  --num_envs 2048 --device cuda:0 --headless
```

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

과거 upright/size-randomization 의미로 학습한 checkpoint와 reward 의미가 다르므로 새
run으로 학습한다.

문서 내용은 2026-07-19 현재 코드 기준이다.
