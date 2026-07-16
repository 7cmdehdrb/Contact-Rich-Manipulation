# UR5e Variable-Stiffness OSC Sweep 학습

## 1. 구현 개요

새 태스크 ID는 다음과 같다.

- 학습: `Isaac-Sweep-Object-UR5e-OSC-v0`
- 소규모 확인: `Isaac-Sweep-Object-UR5e-OSC-Play-v0`

환경은 Isaac Lab Manager-based API와 RSL-RL PPO를 사용한다. 각 vectorized
environment에는 다음 요소가 존재한다.

```text
UR5e tool
  -> fixed joint
  -> VirtualFTSensor
  -> fixed joint
  -> Robotiq 2F-85 (open)
  -> SweepToolCenter (virtual EEF link)
  -> Left/RightSweepContactPad (target-filtered contact sensors)

OpenTable + TargetCube
```

`VirtualFTSensor`의 incoming joint wrench를 사용하므로 F/T 관측은 처음부터
GPU tensor이며 순서는 `[Fx, Fy, Fz, Tx, Ty, Tz]`이다. 좌표계는 센서 body
좌표계이다.

Robotiq는 별도 정책 action 없이 open joint target을 유지한다.
`SweepToolCenter`는 충돌이 없는 작은 virtual rigid link이며, 실제 열린
그리퍼의 중앙점과 OSC 제어점을 동시에 정의한다. 좌우 contact pad는 그리퍼
접촉면보다 조금 돌출된 얇은 충돌 body로, TargetCube와의 filtered contact
point와 force만 수집한다.

## 2. 에셋 경로

기본값은 `ur5e_2f85_ft_test.py`에서 검증한 Nucleus URL이다. 다른 서버나
로컬 USD를 쓰려면 실행 전에 다음 환경 변수를 설정한다.

```bash
export SWEEP_UR5E_USD_PATH="omniverse://SERVER/path/to/ur5e.usd"
export SWEEP_ROBOTIQ_USD_PATH="omniverse://SERVER/path/to/Robotiq_2F_85_edit.usd"
```

mount rotation, virtual EEF offset, contact pad offset/size는
`Ur5eRobotiqFtSpawnerCfg`에서 조정할 수 있다.

## 3. Observation

Policy observation은 아래 순서의 62차원 vector이다.

| 순서 | 항목 | 차원 | 좌표계 / 단위 |
|---:|---|---:|---|
| 1 | Arm joint position | 6 | rad |
| 2 | Arm joint velocity | 6 | rad/s |
| 3 | Arm applied effort | 6 | N·m |
| 4 | Virtual EEF pose | 6 | robot base, xyz + RPY |
| 5 | Virtual F/T wrench | 6 | sensor frame, N + N·m |
| 6 | Target contact point | 3 | robot base, m |
| 7 | Initial target pose | 6 | robot base, xyz + RPY |
| 8 | Current target pose | 6 | robot base, xyz + RPY |
| 9 | Desired motion | 5 | 아래 정의 |
| 10 | Last action | 12 | stiffness + relative pose |

Contact가 없으면 contact point는 `(0, 0, 0)`이다. 두 pad가 동시에
접촉하면 target-filtered normal force 크기로 contact point를 가중 평균한다.

### Desired Motion(5)

원 요청의 `XY direction(2) + distance(1) + force magnitude(1)`는 합계가
4차원이므로, reward에서 필요한 force band를 명시적으로 표현하기 위해
다섯 번째 값을 추가했다.

```text
[direction_x, direction_y, distance_m, desired_force_N, force_tolerance_N]
```

`direction_xy`는 항상 정규화된다. 기본 sampling 범위는 다음과 같다.

- direction angle: `[-pi, pi]`
- distance: `[0.10, 0.22] m`
- desired force: `[8, 25] N`
- force tolerance: `[3, 6] N`

## 4. Action과 OSC

정책 action은 요청 순서 그대로 12차원이다.

```text
[normalized_stiffness(6), relative_pose(6)]
```

- stiffness 입력 `[-1, 1]`은 `[20, 300]`의 task-space diagonal stiffness로
  선형 변환된다.
- relative pose는 `[dx, dy, dz, droll, dpitch, dyaw]`이다.
- position 최대 변화는 action step당 `0.025 m`이다.
- rotation 최대 변화는 action step당 `0.12 rad`이다.
- RPY increment는 OSC 내부 입력용 axis-angle로 변환된다.
- OSC 출력 torque는 UR5e joint effort limit의 90%에서 clamp된다.
- arm actuator의 stiffness/damping은 0이며 OSC torque가 arm을 직접 구동한다.
- gripper actuator는 별도 position target으로 열린 상태를 유지한다.

정책 주기는 30 Hz이고 physics는 120 Hz이다.

## 5. Reward

총 reward는 다음 목표를 우선하도록 구성했다.

1. cube 뒤쪽의 pre-contact pose 접근
2. TargetCube와 접촉
3. 요청 force magnitude와 tolerance 준수
4. 요청 방향으로의 object velocity와 누적 progress
5. 실제 displacement 방향 정렬
6. 요청 길이의 endpoint 도달
7. lateral motion과 overshoot 억제
8. F/T torque, joint velocity, OSC effort, action 변화, torque saturation 억제

주요 항목은 다음과 같다.

```text
force_tracking =
    exp(-((measured_target_contact_force - desired_force) / tolerance)^2)
    * is_target_contact

normalized_progress =
    dot(current_position - initial_position, desired_direction) / desired_distance

endpoint_tracking =
    exp(-(distance(current_position, desired_goal) / 0.035)^2)
```

기존 sweep reward와 달리 object width/type/ID 같은 물체별 정보는 사용하지
않는다. 대상은 하나의 cube로 고정하며, 요청된 initial/current pose와
desired motion만 사용한다. F/T torque 초과량과 OSC joint torque도 penalty에
포함한다.

## 6. Termination

- episode timeout: 8 s
- endpoint와 lateral tolerance를 만족한 성공
- cube가 table 아래로 떨어지거나 과도하게 기울어짐
- F/T sensor force `100 N` 또는 torque `15 N·m` 초과
- arm joint speed `6.5 rad/s` 초과

## 7. 설치 및 학습

Sweep RL 패키지를 Isaac Lab Python 환경에 editable install한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl
```

이 저장소의 `train.py`는 설치된 `sweep_rl`을 자동 import하여 Gym
태스크를 등록한다.

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --headless \
  --device cuda:0
```

설치하지 않고 실행할 때는 `PYTHONPATH`를 사용할 수 있다.

```bash
PYTHONPATH="$PWD/src/sweep_rl:$PYTHONPATH" \
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --headless \
  --device cuda:0
```

4개 환경, 1 iteration smoke test:

```bash
PYTHONPATH="$PWD/src/sweep_rl:$PYTHONPATH" \
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --headless \
  --device cuda:0
```

기본 학습은 2048 environments, PPO rollout 32 steps, 최대 12000
iterations이다. GPU memory에 맞추어 `--num_envs`를 먼저 조정하는 것을
권장한다.

학습 checkpoint 재생:

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-Play-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --device cuda:0
```

## 8. 주요 파일

- `sweep_rl/osc_sweep/assets.py`: UR5e/F/T/Robotiq assembly
- `sweep_rl/osc_sweep/env_cfg.py`: scene와 Manager term 구성
- `sweep_rl/osc_sweep/mdp/actions.py`: 12-D variable-stiffness OSC
- `sweep_rl/osc_sweep/mdp/commands.py`: desired motion sampling
- `sweep_rl/osc_sweep/mdp/observations.py`: 62-D observation
- `sweep_rl/osc_sweep/mdp/rewards.py`: sweep/force/torque reward
- `sweep_rl/osc_sweep/mdp/terminations.py`: success와 safety termination
- `sweep_rl/osc_sweep/agents/rsl_rl_ppo_cfg.py`: RSL-RL PPO
