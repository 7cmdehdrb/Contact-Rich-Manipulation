# 등록된 UR5e OSC Sweep 환경

이 문서는 `sweep_rl/osc_sweep/__init__.py`에 등록된 Gym 환경의 상속 관계,
학습·플레이 방법, Action, Observation, Reward 차이를 현재 코드 기준으로 정리한다.
문서 내용은 2026-07-19 현재 `sweep_rl/osc_sweep` 코드 기준이다.

## 1. 사전 준비와 공통 실행 규칙

먼저 Sweep RL 패키지를 Isaac Lab Python에 설치한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl
```

설치하지 않고 실행한다면 명령 앞에 다음 환경 변수를 붙인다.

```bash
PYTHONPATH="$PWD/src/sweep_rl:$PYTHONPATH"
```

체크포인트는 환경별 PPO 설정의 `experiment_name`에 따라 기본적으로 다음 위치에
저장된다.

```text
logs/rsl_rl/<experiment_name>/<run>/model_<iteration>.pt
```

### 플레이 스크립트 선택

- 기본, Play, WideRandomization, TactileLocalization은 5-D force command를 사용한다.
  목표점과 목표/측정 접촉력을 표시하는 `src/sweep_rl/scripts/play_sweep.py`를 사용할
  수 있다.
- ConstantVelocity 계열은 4-D speed command를 사용한다. `play_sweep.py`는 5-D force
  command를 전제로 작성되어 있으므로 이 계열에는 Isaac Lab 표준
  `rsl_rl/play.py`를 사용한다.

## 2. 상속 구조

```text
UR5eOscSweepEnvCfg
├── UR5eOscSweepEnvCfg_PLAY
├── UR5eOscSweepWideRandomizationEnvCfg
│   └── UR5eOscSweepTactileLocalizationEnvCfg
└── UR5eOscSweepConstantVelocityEnvCfg
    └── UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg
        └── UR5eOscSweepConstantVelocityUprightRandomSizeHomeEnvCfg
```

| 환경 ID | 기반 환경 | Observation | Command | experiment_name |
|---|---|---:|---:|---|
| `Isaac-Sweep-Object-UR5e-OSC-v0` | 기본 | 62-D | 5-D force | `ur5e_osc_sweep` |
| `Isaac-Sweep-Object-UR5e-OSC-Play-v0` | 기본 | 62-D | 5-D force | `ur5e_osc_sweep` |
| `Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0` | 기본 | 62-D | 5-D force | `ur5e_osc_sweep_wide_randomization` |
| `Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0` | Wide | 56-D | 5-D force | `ur5e_osc_sweep_tactile_localization` |
| `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0` | 기본 | 55-D | 4-D speed | `ur5e_osc_sweep_constant_velocity` |
| `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0` | ConstantVelocity + external-pad approach + gripper-interior termination | 55-D | 4-D speed | `ur5e_osc_sweep_constant_velocity_upright_random_size` |
| `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0` | Gripper Exclusion + HomeReturn | 56-D | 4-D speed | `ur5e_osc_sweep_constant_velocity_upright_random_size_home` |

## 3. 기본 환경: `Isaac-Sweep-Object-UR5e-OSC-v0`

구현 클래스는 `UR5eOscSweepEnvCfg`다. 이후 force-command 환경의 기준이 된다.

### 실행

학습:

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --num_envs 2048 \
  --device cuda:0 \
  --headless
```

플레이에는 동일 환경을 사용해 매 episode의 무작위 방향을 유지하거나, 다음 절의
결정론적 `Play-v0`를 사용할 수 있다.

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_sweep.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 1 \
  --device cuda:0
```

### Command

```text
[direction_x, direction_y, distance_m, desired_force_N, force_tolerance_N]
```

- 방향: `[-pi, pi]`
- 이동 거리: `0.10~0.22 m`
- 목표 접촉력: `8~25 N`
- 접촉력 tolerance: `3~6 N`

### Action: 12-D

```text
[normalized_stiffness(6), relative_pose(6)]
```

| 구간 | 의미 | 변환 |
|---|---|---|
| `0:6` | task-space stiffness | `[-1, 1]`에서 `[20, 300]`으로 변환 |
| `6:9` | EEF 상대 `dx, dy, dz` | 축별 최대 `0.025 m/step` |
| `9:12` | EEF 상대 RPY | 축별 최대 `0.12 rad/step`, axis-angle로 변환 |

OSC torque는 UR5e effort limit의 90%에서 clamp된다. 정책 주기는 30 Hz이고 physics는
120 Hz다. 그리퍼는 정책 Action에 포함되지 않는다.

### Observation: 62-D

| 순서 | 항목 | 차원 | 좌표계/내용 | Noise |
|---:|---|---:|---|---|
| 1 | `joint_pos` | 6 | 팔 관절 위치 | uniform `±0.002` |
| 2 | `joint_vel` | 6 | 팔 관절 속도 | uniform `±0.01` |
| 3 | `joint_effort` | 6 | 팔 applied effort | 없음 |
| 4 | `eef_pose` | 6 | robot base 기준 `xyz + RPY` | 없음 |
| 5 | `ft_sensor` | 6 | sensor frame의 `force + torque` | 없음 |
| 6 | `contact_point` | 3 | robot base 기준 target 접촉점, 미접촉 시 0 | 없음 |
| 7 | `initial_target_pose` | 6 | reset 시 target `xyz + RPY` | 없음 |
| 8 | `current_target_pose` | 6 | 현재 target `xyz + RPY` | 없음 |
| 9 | `desired_motion` | 5 | 위 5-D command | 없음 |
| 10 | `last_action` | 12 | 직전 Action | 없음 |
| | **합계** | **62** | | |

### Reward

Isaac Lab Reward Manager는 각 항을 매 step `raw_value × weight × step_dt`로 합산한다.

| Reward term | Weight | 내용 |
|---|---:|---|
| `reaching` | `+1.5` | 초기 물체 뒤 `0.065 m` pre-contact pose 접근 |
| `target_contact` | `+0.1` | target-specific 접촉 binary 보상 |
| `side_direction` | `+1.5` | gripper 넓은 면과 명령 방향 정렬 |
| `side_center_contact` | `+2.5` | 넓은 contact pad 중앙 접촉 품질 |
| `force_tracking` | `+4.0` | 중앙 접촉 품질로 gating한 목표 접촉력 추종 |
| `velocity_progress` | `+4.0` | 명령 방향 물체 속도 |
| `normalized_progress` | `+1.0` | 종방향 이동 거리 / 명령 거리 |
| `direction_alignment` | `+1.5` | 실제 XY 변위와 명령 방향 cosine 정렬 |
| `endpoint_tracking` | `+6.0` | 목표점 오차의 Gaussian, `std=0.035 m` |
| `success` | `+10.0` | endpoint `<0.025 m`, normalized lateral `<0.12` |
| `lateral_error` | `-2.0` | 명령 방향에 수직인 normalized 변위 |
| `overshoot` | `-4.0` | 명령 거리 이후의 normalized overshoot |
| `off_center_contact` | `-1.5` | pad 모서리·좁은 면·내측 면 접촉 |
| `ft_torque` | `-0.02` | F/T torque norm 중 `1.5 Nm` 초과분 |
| `action_rate` | `-0.02` | 연속 Action 변화량 제곱 |
| `joint_velocity` | `-0.002` | 관절 속도 제곱 |
| `commanded_effort` | `-0.03` | effort limit로 정규화한 OSC torque 제곱 |
| `torque_saturation` | `-0.5` | Action clipping/비정상 값 또는 torque saturation |

공통 종료 조건은 8초 timeout, 목표 도달, 물체 낙하/과도한 기울기, F/T
`100 N` 또는 `15 Nm` 초과, 팔 관절 속도 `6.5 rad/s` 초과다.

## 4. 플레이 환경: `Isaac-Sweep-Object-UR5e-OSC-Play-v0`

`UR5eOscSweepEnvCfg_PLAY`은 기본 `UR5eOscSweepEnvCfg`를 상속한다.

### 기본 환경에서 바뀐 부분

- scene environment 수: `2048 → 16` 기본값
- Observation corruption/noise: 비활성화
- 명령 방향: `pi/2`로 고정
- Action, Observation 항목과 차원, Reward, termination, PPO 구조: 기본 환경과 동일

### 실행

이 환경은 기본 환경의 checkpoint를 재생하기 위한 설정이다. 권장 학습 방법은
`Isaac-Sweep-Object-UR5e-OSC-v0`에서 학습하는 것이다. 기술적으로 직접 학습하려면
다음 명령을 사용할 수 있지만, 16개 환경·고정 방향·noise 비활성 설정이고 기본
환경과 같은 `experiment_name`을 사용하므로 일반 학습에는 권장하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-Play-v0 \
  --num_envs 16 \
  --device cuda:0 \
  --headless
```

플레이:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_sweep.py \
  --task Isaac-Sweep-Object-UR5e-OSC-Play-v0 \
  --checkpoint /absolute/path/to/base_model.pt \
  --num_envs 1 \
  --device cuda:0
```

`--checkpoint`를 생략하면 `logs/rsl_rl/ur5e_osc_sweep`에서 `--load_run`과 일치하는
최신 checkpoint를 검색한다.

## 5. Wide Randomization: `Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0`

`UR5eOscSweepWideRandomizationEnvCfg`는 기본 `UR5eOscSweepEnvCfg`를 상속한다.

### 기본 환경에서 바뀐 부분

- 목표 접촉력 범위: `8~25 N → 8~50 N`
- target cube mass: episode마다 uniform `0.3~3.0 kg`
- sampled mass에 맞춰 inertia 재계산
- 모든 arm/gripper actuator의 PD stiffness와 damping을 0으로 설정
- Observation 62-D와 Reward weight는 기본 환경과 동일
- PPO hyperparameter는 동일하고 experiment directory만 분리

### 실행

학습:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_wide_randomization.py \
  --num_envs 2048 \
  --device cuda:0 \
  --headless
```

플레이:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_sweep.py \
  --task Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0 \
  --checkpoint /absolute/path/to/wide_model.pt \
  --num_envs 1 \
  --device cuda:0
```

플레이에서도 mass와 command randomization은 유지된다.

## 6. Tactile Localization: `Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0`

`UR5eOscSweepTactileLocalizationEnvCfg`는
`UR5eOscSweepWideRandomizationEnvCfg`를 상속한다. 따라서 Wide의 `8~50 N` force,
`0.3~3.0 kg` mass randomization과 passive actuator 설정을 모두 포함한다.

### Wide 환경에서 바뀐 부분

Observation에서 `current_target_pose` 6-D만 제거한다.

| Observation | 차원 |
|---|---:|
| Wide Observation | 62 |
| 제거: `current_target_pose` | -6 |
| **Tactile Observation** | **56** |

나머지 `joint state`, EEF pose, F/T wrench, 접촉점, 초기 target pose, command,
last Action은 그대로 관측한다. Reward와 termination은 simulator의 실제 target pose를
privileged state로 계속 사용할 수 있다.

Reward 변경:

| Reward term | Wide | Tactile | 변경 내용 |
|---|---:|---:|---|
| `normalized_progress` | `+1.0` | `+3.0` | 목표 방향 이동 강화 |
| `endpoint_tracking` | `+6.0` | `+12.0` | endpoint 배치 강화 |
| `endpoint_tracking.coarse_std` | 없음 | `0.12 m` | 먼 거리에서도 gradient 제공 |
| `endpoint_tracking.coarse_weight` | `0` | `0.35` | wide Gaussian 35% 혼합 |
| `success` | `+10.0` | `+40.0` | 목표 도달 우선순위 강화 |

추가 actuator 변경:

- UR5e arm simulated effort limit을 기존 값의 `1.5배`로 확대
- OSC `effort_limit_scale`: `0.9 → 1.0`
- F/T safety termination은 그대로 유지

### 실행

학습:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_tactile_localization.py \
  --num_envs 2048 \
  --device cuda:0 \
  --headless
```

플레이:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_sweep.py \
  --task Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0 \
  --checkpoint /absolute/path/to/tactile_model.pt \
  --num_envs 1 \
  --device cuda:0
```

## 7. Constant Velocity: `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0`

`UR5eOscSweepConstantVelocityEnvCfg`는 기본 `UR5eOscSweepEnvCfg`를 직접 상속하지만,
Command, Action term, Observation, Reward, termination을 교체한다. scene과 reset 위치
randomization은 기본 환경을 상속한다.

### 실행

학습:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity.py \
  --num_envs 2048 \
  --device cuda:0 \
  --headless
```

플레이:

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0 \
  --checkpoint /absolute/path/to/constant_velocity_model.pt \
  --num_envs 1 \
  --device cuda:0
```

### Command 변경: 4-D

```text
[direction_x, direction_y, distance_m, target_speed_mps]
```

- 방향: `[-pi, pi]`
- 이동 거리: `0.10~0.22 m`
- 목표 순항 속도: `0.08 m/s` 고정
- 목표 force와 force tolerance 제거
- 시작 `0.025 m`에서 가속하고 마지막 `0.04 m`에서 감속하는 속도 profile 사용

### Action 변경

12-D stiffness + relative pose 구조와 scale은 기본 환경과 같다. 차이는
`OpenGripperSweepOperationalSpaceAction`이 모든 physics step에 gripper open position
`0.0`을 다시 적용한다는 점이다. gripper PD gain은 stiffness `2000`, damping `100`이다.

### Observation 변경: 55-D

| 순서 | 항목 | 차원 | 기본 환경 대비 |
|---:|---|---:|---|
| 1 | `joint_pos` | 6 | 유지 |
| 2 | `joint_vel` | 6 | 유지 |
| 3 | `joint_effort` | 6 | 유지 |
| 4 | `eef_pose` | 6 | 유지 |
| 5 | `initial_target_pose` | 6 | 유지 |
| 6 | `current_target_pose` | 6 | 유지 |
| 7 | `object_linear_velocity` | 3 | 추가, noise `±0.005` |
| 8 | `desired_motion` | 4 | force command 대신 speed command |
| 9 | `last_action` | 12 | 유지 |
| | **합계** | **55** | `ft_sensor`와 `contact_point`는 제거 |

### Reward 교체

| Reward term | Weight | 내용 |
|---|---:|---|
| `push_pose_error` | `-0.35` | 현재 물체를 따라가는 pre-contact pose 거리 오차 |
| `side_direction_error` | `-0.25` | 물체 근처 gripper 면 방향 오차 |
| `target_contact` | `+0.50` | 작은 target 접촉 bridge |
| `side_center_contact` | `+0.75` | pad 중앙의 양질 접촉 |
| `contact_forward_progress` | `+3.0` | 접촉 중 명령 방향으로 발생한 실제 전진 |
| `velocity_tracking` | `+10.0` | 가속-순항-감속 profile 추종, 정지 시 0 |
| `endpoint_error` | `-5.0` | 목표점 거리 / 명령 거리, 최대 2 |
| `stopped_at_goal` | `+20.0` | 위치 오차와 속도가 작은 상태의 Gaussian 보상 |
| `success` | `+40.0` | endpoint `<0.020 m`, lateral `<0.10`, speed `<0.020 m/s` |
| `failure_termination` | `-8.0` | 안전 실패 시 남은 episode horizon 비용 |
| `lateral_error` | `-3.0` | normalized lateral displacement |
| `overshoot` | `-8.0` | 목표 거리 초과 |
| `stall` | `-6.0` | 시작 0.40초 후 목표 속도의 50% 미만인 속도 부족 |
| `object_acceleration` | `-0.15` | 물체 선가속도 제곱 |
| `ft_torque` | `-0.02` | F/T torque `1.5 Nm` 초과분 |
| `action_rate` | `-0.02` | Action 변화량 제곱 |
| `joint_velocity` | `-0.002` | 관절 속도 제곱 |
| `commanded_effort` | `-0.03` | normalized OSC torque 제곱 |
| `torque_saturation` | `-0.5` | Action/torque saturation indicator |

안전 실패 비용은 실제 합산 결과 기준으로 다음과 같다.

```text
-8.0 × max(남은 episode 시간, 1.0 s)
```

성공 termination은 endpoint/lateral/speed 조건을 `0.30 s` 연속 유지해야 한다. PPO의
초기 Gaussian 표준편차는 기본 환경의 `0.8` 대신 `0.5`다.

## 8. Gripper Exclusion: `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0`

`UR5eOscSweepConstantVelocityUprightRandomSizeEnvCfg`는
`UR5eOscSweepConstantVelocityEnvCfg`를 상속한다.

환경 ID와 Python 클래스 이름은 기존 실행 스크립트 및 checkpoint 경로 호환성을 위해
유지한다. 현재 동작에는 upright 자세 보상이나 물체 크기 randomization이 없다.

### ConstantVelocity에서 바뀐 부분

Action, 55-D Observation, 4-D speed command와 scene/reset은
`ConstantVelocity-v0`와 동일하다. Reward는 기존 `push_pose_error` 하나만 다음처럼
교체한다.

| Reward term | Weight | 변경 내용 |
|---|---:|---|
| `push_pose_error` | `-1.0` | EEF-local에서 물체가 pad 정면 `X=+/-0.065 m`, 좌우 pad 중심 `Y=+/-0.055 m`, `Z=0`에 오도록 하는 단순 거리 오차. 가까운 pad를 대칭적으로 선택 |

이는 특정 world orientation을 요구하지 않으면서 dense 접근 목표가 gripper gap
중앙을 가리키던 문제를 제거한다. raw value는 선택된 목표 상대 위치와 현재 물체
상대 위치의 Euclidean distance를 `0.10 m`로 나눈 뒤 최대 `3.0`으로 제한한 값이다.
허용된 양쪽 pad 중심에서는 0이며 gap 중앙의 pre-contact 위치에서는 약 0.55다.

기존 `side_direction_error`가 밀기 방향에 맞는 pad 면을 유도하고, 새
`push_pose_error`는 그 면의 좌우 pad 중 하나를 물체에 정렬한다. 두 항 모두 특정
world-frame gripper orientation을 지정하지 않는다. 추가 termination은 다음과 같다.

| Termination term | 조건 |
|---|---|
| `object_inside_gripper` | 물체 중심이 EEF-local gripper 내부 exclusion box (`XYZ half extents = 0.040, 0.040, 0.058 m`)에 진입하면 실패 종료 |

exclusion box는 EEF와 함께 회전하므로 world-frame orientation을 강제하지 않는다.
종료를 고의로 이용하지 못하도록 이 항은 기존 `failure_termination`과 동일한 남은
episode 시간 비용을 받는다.

그 밖의 설정은 `ConstantVelocity-v0`에서 그대로 상속한다.

- Action: 동일한 12-D variable-stiffness OSC
- Observation: 동일한 55-D policy observation
- Command: 동일한 4-D direction/distance/speed command
- 물체: 고정 `0.06 m` 정육면체, 질량 `0.35 kg`
- scene: `replicate_physics=True`
- upright orientation reward와 물체 크기 randomization: 사용하지 않음
- PPO hyperparameter: `ConstantVelocity-v0`와 동일하며 `experiment_name`만 기존 호환
  이름 `ur5e_osc_sweep_constant_velocity_upright_random_size` 사용

### 실행

학습:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size.py \
  --num_envs 2048 \
  --device cuda:0 \
  --headless
```

플레이:

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

reward 의미가 이전 UprightRandomSize 구현과 달라졌으므로 과거 checkpoint를 resume하지
않고 새 run으로 학습한다.

## 9. Home Return: `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0`

`UR5eOscSweepConstantVelocityUprightRandomSizeHomeEnvCfg`는 위 Gripper Exclusion
환경을 상속한다. 클래스와 환경 ID의 `UprightRandomSize`는 호환성을 위한 과거 이름일
뿐이며, 이 환경에도 upright reward나 size randomization은 없다. 기존 sweep 이후
`task_phase=1`로 전환해 UR5e의 6개 arm joint를 default Home pose로 복귀시킨다.

변경점:

- Observation: 부모 55-D + `task_phase` 1-D = 56-D
- episode: `8 s → 12 s`
- scene: target 전체와 robot 전체의 접촉을 감지하는 sensor 추가,
  `replicate_physics=False`
- SWEEP phase: 부모 환경의 external-pad `push_pose_error`와
  `object_inside_gripper` termination 사용
- HOME phase: sweep/contact/endpoint/success shaping을 비활성화하고 아래 Home reward 사용
- `object_inside_gripper`와 기존 safety termination은 두 phase 모두 계속 활성화
- 성공: Home joint 오차 `<0.12 rad`, joint speed `<0.15 rad/s`, target endpoint/speed
  유지, 전체 robot-target 비접촉, HOME 진입 위치 대비 변위 `<0.010 m`를 0.25초 유지
- 실패: HOME 진입 위치에서 target이 `0.015 m` 넘게 벗어나거나 속도가
  `0.10 m/s`를 넘으면 `post_goal_object_moved`로 즉시 종료

부모 환경에서 추가로 바뀌는 Reward는 다음과 같다.

| Reward term | Weight | 활성 phase | 내용 |
|---|---:|---:|---|
| `home_joint_pose` | `+15.0` | HOME | default Home joint pose의 Gaussian 보상 |
| `home_joint_error` | `-3.0` | HOME | normalized Home joint 오차 |
| `home_clearance` | `+3.0` | HOME | EEF와 물체 사이 `0.22 m` 안전거리 확보 |
| `post_goal_contact` | `-12.0` | HOME | 전체 robot-target 접촉 |
| `goal_hold_error` | `-10.0` | HOME | 배치한 물체의 endpoint 이탈 |
| `post_goal_object_speed` | `-3.0` | HOME | 배치 후 물체 속도 |
| `post_goal_object_displacement` | `-8.0` | HOME | HOME 진입 시 저장한 물체 위치에서의 변위 |
| `home_time` | `-0.5` | HOME | Home 복귀 지연 시간 |
| `home_success` | `+50.0` | HOME | Home pose, 정지 물체, 비접촉 조건 동시 만족 |

`failure_termination`, `object_acceleration`, `ft_torque`, `action_rate`,
`joint_velocity`, `commanded_effort`, `torque_saturation`은 공통 안전·정규화 항으로 두
phase 모두 유지된다.

TensorBoard command metric에는 `home_phase`와 `parked_displacement`가 추가된다.
`parked_displacement`는 SWEEP phase에서 0이며 HOME phase에서 저장 위치 대비 현재
물체 중심의 거리 `[m]`를 기록한다.

학습:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size_home.py \
  --num_envs 2048 --device cuda:0 --headless
```

플레이:

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0 \
  --checkpoint /absolute/path/to/home_return_model.pt \
  --num_envs 1 --device cuda:0
```

## 10. 빠른 선택 가이드

| 목적 | 권장 환경 |
|---|---|
| force command 기반 기본 동작 학습 | `Isaac-Sweep-Object-UR5e-OSC-v0` |
| 기본 checkpoint를 고정 방향/noise 없이 확인 | `Isaac-Sweep-Object-UR5e-OSC-Play-v0` |
| 넓은 force·mass 범위에서 강건성 학습 | `Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0` |
| 현재 target pose 없이 촉각·로봇 상태로 위치 추론 | `Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0` |
| 접촉력 목표 없이 일정 속도와 endpoint 정지 학습 | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0` |
| 일정 속도 + 외측 pad 접근 + gripper 내부 삽입 금지 | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0` |
| 외측 pad sweep과 물체 배치 후 비접촉 Home joint 복귀까지 학습 | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0` |
