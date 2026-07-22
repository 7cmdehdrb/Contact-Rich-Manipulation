# ConstantVelocity UprightRandomSize HomeReturn

환경 ID:

```text
Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0
```

이 환경은
`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0`를 상속한다.
부모 환경의 일정 속도 sweep, 목표점 정지, 외측 pad 접근, gripper 내부 삽입 실패를
유지하고, 물체를 놓은 뒤 UR5e가 Home joint pose로 복귀하는 두 번째 단계를 추가한다.
환경 ID의 `UprightRandomSize`는 호환성용 이름이며, 현재 upright 자세 보상이나 물체
크기 랜덤화는 없다.

`example/Sweep-Policy`의 목표 도달 후 homing reward를 참고했지만, Home에 도달하지
않고도 성공 종료되던 부분은 가져오지 않았다. 새 환경은 Home, 저속, 비접촉, 물체
목표점 유지 조건을 모두 충족해야 성공한다.

## Phase

정책 Observation에 `task_phase` 1차원을 추가한다.

| 값 | Phase | 동작 |
|---:|---|---|
| 0 | `SWEEP` | 부모 Gripper Exclusion reward로 물체를 목표점까지 이동 |
| 1 | `HOME` | 물체와 분리한 뒤 기본 arm joint pose로 복귀 |

Phase 전환은 다음 조건을 0.30초 연속 만족할 때 발생하며 한번 전환되면 episode가
끝날 때까지 HOME으로 유지된다.

```text
endpoint_error < 0.020 m
object_speed < 0.020 m/s
```

HOME phase에서는 기존 `target_contact`, `stopped_at_goal`, `success`를 포함한 sweep
reward를 모두 끈다. 따라서 목표점에서 계속 접촉하며 기존 보상을 누적하는 정책은
이득을 얻지 못한다.

## Action과 Observation

Action은 부모 환경과 동일한 12-D variable-stiffness OSC다.

```text
[normalized_stiffness(6), relative_pose(6)]
```

Observation은 부모의 55-D에 phase 1-D를 추가한 56-D다.

| Observation | 차원 |
|---|---:|
| 부모 Gripper Exclusion Observation | 55 |
| `task_phase` | 1 |
| **합계** | **56** |

Home pose는 reset에서 약간 perturb된 시작 자세가 아니라 asset에 정의된 canonical
`robot.data.default_joint_pos`의 6개 arm joint를 뜻한다.

## HOME phase Reward

Isaac Lab은 각 항을 `raw × weight × step_dt`로 합산한다.

| Reward term | Weight | 내용 |
|---|---:|---|
| `home_joint_pose` | `+15.0` | default joint와의 오차 Gaussian, `joint_std=0.35 rad` |
| `home_joint_error` | `-3.0` | 평균 절대 joint 오차 / `0.75 rad` |
| `home_clearance` | `+3.0` | EEF-물체 거리가 `0.22 m`까지 증가할수록 smooth 보상 |
| `post_goal_contact` | `-12.0` | HOME phase에서 robot 어느 부분이든 target과 접촉 |
| `goal_hold_error` | `-10.0` | 복귀 중 물체의 normalized endpoint 오차 |
| `post_goal_object_speed` | `-3.0` | 복귀 중 물체를 다시 움직이는 속도 |
| `post_goal_object_displacement` | `-8.0` | HOME 진입 순간의 물체 위치에서 벗어난 거리 |
| `home_time` | `-0.5` | HOME phase 매초 지연 비용 |
| `home_success` | `+50.0` | 최종 성공 조건을 만족하는 동안의 sparse 보상 |

`post_goal_contact`는 기존 두 contact pad만 검사하지 않는다. TargetCube에 추가한
`target_robot_contact` 센서가 `/Robot/.*` 전체를 filter하므로 gripper 본체, pad,
가상 EEF link를 포함한 로봇 전체와 target의 접촉을 감지한다. 판정 threshold는
`0.25 N`이다.

안전 실패 reward, object acceleration, F/T torque, Action rate, joint velocity,
commanded effort, torque saturation 항은 두 phase 모두 계속 적용된다.

## 성공과 종료

다음 조건을 0.25초 연속 만족해야 성공 종료된다.

```text
task_phase == HOME
모든 arm joint의 |current - default| < 0.12 rad
모든 arm joint의 |velocity| < 0.15 rad/s
endpoint_error < 0.025 m
object_speed < 0.025 m/s
robot-target filtered contact 없음
HOME 진입 위치 대비 물체 변위 < 0.010 m
```

HOME 진입 순간의 물체 위치는 command term에 별도로 저장된다. 복귀 중 이 위치에서
`0.015 m` 이상 벗어나거나 물체 속도가 `0.10 m/s`를 넘으면
`post_goal_object_moved` 실패로 즉시 종료한다. 따라서 목표 허용 반경 안에서 물체를
다시 밀고도 성공하는 우회 동작을 허용하지 않는다.

부모 환경의 안전 종료 조건은 그대로 유지한다. 두 단계 수행 시간을 위해 episode
timeout은 `8 s`에서 `12 s`로 늘렸다.

## 학습

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size_home.py \
  --num_envs 2048 --device cuda:0 --headless
```

기존 Gripper Exclusion 정책으로부터 이어서 학습하려면 observation이 55-D에서
56-D로 바뀌므로 actor의 첫 layer shape가 달라진다는 점에 주의해야 한다. 기존
checkpoint를 그대로 resume할 수 없으며, phase 입력을 수용하도록 checkpoint를
변환하지 않는 한 새 정책으로 학습해야 한다.

## 플레이

기존 `HomeReturn-v0` 환경과 Isaac Lab 표준 `play.py`는 cube 환경으로 유지한다. Can
재생은 다음 별도 환경 ID와 전용 script를 사용한다.

```text
Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-Can-v0
src/sweep_rl/scripts/play_constant_velocity_home_can.py
```

Can variant의 pushing target은 다음 USD다.

```text
omniverse://192.168.0.13/Library/Shelf/Objects/Can_6/Can_6.usd
```

Can USD의 rigid-body root에는 `RigidBodyAPI`만 있고 `MassAPI`가 없으므로
`UsdFileCfg.mass_props`를 사용하지 않는다. 대신 `prestartup`의 `set_target_mass` event가
실제 `{ENV_REGEX_NS}/TargetCube` root에 `MassAPI`를 먼저 생성하고 질량을 설정한다. 기본
질량은 `0.35 kg`이다. Can의 local Z 원점은 바닥면이므로 초기 root Z는 table top인
`0.775 m`다. Target–robot contact filter는 `/Robot/.*` wildcard 한 개가 아니라 18개
rigid-body 경로를 명시적으로 사용하여 PhysX filter-count 불일치를 방지한다.

Cube는 rigid root가 중심이지만 Can_6는 바닥에 있다. 전용 variant는 Can 높이
`0.1191307 m`의 절반인 `0.0595654 m`를 `initial_target_pose`와
`current_target_pose` 관측의 Z에 더한다. 따라서 policy에는 Can 중심이 전달되고 전체
관측 shape는 기존 checkpoint와 같은 56-D로 유지된다.

질량을 `1.25 kg`으로 재생하려면 다음처럼 전용 script를 실행한다. Task, checkpoint와
`num_envs=1`은 script의 기본값이다.

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_constant_velocity_home_can.py \
  --object_mass 1.25 \
  --device cuda:0
```

필요할 때 `--target_z_offset`으로 관측 보정값을 바꿀 수 있다. Can 환경에서는
`env.scene.target_object.spawn.mass_props.mass=...`를 사용하지 않는다.

TensorBoard에서는 `Metrics/desired_motion/home_phase`가 0에서 1로 전환되는지,
`Metrics/desired_motion/parked_displacement`가 `0.015 m` 아래로 유지되는지,
`Episode_Reward/post_goal_contact`가 감소하는지, `Episode_Termination/success`가
증가하는지를 함께 확인한다.

문서 내용은 2026-07-19 현재 코드 기준이다.
