# Independent UR5e OSC Sweep + HomeReturn 환경

## 1. 환경 개요

이 환경은 UR5e와 항상 열린 Robotiq 2F-85 gripper를 이용하여 정육면체 물체를
지정된 방향, 거리, 속도로 민 뒤, 물체를 다시 건드리지 않고 로봇을 Home joint
pose로 복귀시키는 2단계 강화학습 환경이다.

Gym 환경 ID는 다음과 같다.

```text
Isaac-Sweep-Object-UR5e-OSC-Independent-v0
```

하나의 episode는 다음 상태 순서로 진행된다.

```text
SWEEP
  1. 물체의 지정된 면으로 접근
  2. gripper 외측 pad로 접촉
  3. 접촉을 유지하며 목표 motion을 수행
  4. 목표 위치에서 물체를 정지
       ↓
HOME
  5. 물체로부터 이탈
  6. 같은 OSC action으로 UR5e를 Home joint pose까지 복귀
       ↓
SUCCESS
```

SWEEP와 HOME 모두 동일한 12-D OSC policy가 제어한다. HOME phase에서 별도의
joint-position controller나 scripted trajectory로 전환하지 않는다. Phase가 바뀌면
observation의 `task_phase`와 학습 신호만 바뀌며, policy가 joint state를 보고 OSC
relative pose를 연속적으로 출력하여 Home으로 복귀해야 한다.

기존 `osc_sweep` 환경 Config나 MDP 구현을 상속하지 않는 독립 환경이다. 다만 동일한
UR5e/F/T/Robotiq USD assembly를 사용하기 위해 robot asset factory만 공유한다.

## 2. 시뮬레이션과 Episode

| 항목 | 설정 |
|---|---:|
| Physics 주기 | 120 Hz |
| Policy 주기 | 30 Hz |
| Decimation | 4 physics step / action |
| 최대 episode 시간 | 20 s |
| 병렬 환경 기본값 | 2,048 |
| 환경 간격 | 2.0 m |
| Physics replication | 비활성화 |

Physics replication은 물체 크기를 병렬 환경마다 다르게 적용하고 물체 전체를 이용한
robot-contact sensor를 구성하기 위해 비활성화한다.

물체가 목표점에 도착했다고 즉시 HOME으로 전환하지 않는다. 아래 조건을 0.30초 동안
연속으로 만족해야 SWEEP에서 HOME으로 전환한다.

```text
object endpoint error < 0.025 m
object linear speed  < 0.020 m/s
```

이 dwell 조건은 목표점을 통과하는 순간적인 상태를 성공으로 오인하지 않고, 물체가
실제로 목표 위치에서 정지했는지 확인하기 위한 것이다.

## 3. Observation

### 3.1 Actor와 Critic 입력

Actor와 Critic은 모두 동일한 `policy` observation group만 사용한다. 별도의
asymmetric 또는 privileged critic observation은 없다.

```text
Actor observation  : 56-D policy observation
Critic observation : 56-D policy observation
```

현재 물체 pose와 velocity는 Actor뿐 아니라 Critic 입력에도 포함되지 않는다. 이 값은
Push reward, phase 전환, 성공 판정, 안전 termination을 계산할 때만 simulator 내부의
privileged state로 사용한다. 따라서 학습 policy는 noisy proprioception, F/T, 접촉점,
초기 물체 pose와 action history를 이용해 현재 물체 상태를 간접적으로 추정해야 한다.

### 3.2 Observation 구성

Observation은 아래 순서대로 concatenate된다.

| 순서 | Observation | 차원 | 기준 좌표계 및 의미 | Noise |
|---:|---|---:|---|---|
| 1 | Joint Position | 6 | UR5e arm joint position | uniform ±0.002 rad |
| 2 | Joint Velocity | 6 | UR5e arm joint velocity | uniform ±0.01 rad/s |
| 3 | Joint Effort | 6 | UR5e arm의 simulated joint effort | uniform ±0.5 Nm |
| 4 | End-Effector Pose | 6 | robot base 기준 `xyz + RPY` | 없음 |
| 5 | F/T Sensor | 6 | virtual wrist F/T body frame의 `force + torque` | force ±0.5 N, torque ±0.02 Nm |
| 6 | Contact Point | 3 | robot base 기준 force-weighted target 접촉점 | uniform ±0.002 m |
| 7 | Initial Target Pose | 6 | reset 직후 robot base 기준 `xyz + RPY` | position ±0.003 m, RPY ±0.02 rad |
| 8 | Desired Motion | 4 | `direction_x, direction_y, distance, push_velocity` | 없음 |
| 9 | Task Phase | 1 | SWEEP=`0`, HOME=`1` | 없음 |
| 10 | Last Action | 12 | 직전 normalized OSC policy action | 없음 |
| | **합계** | **56** | | |

`Desired Motion`의 방향은 각도가 아니라 단위 XY 방향 벡터로 제공된다.

```text
[direction_x, direction_y, distance_m, target_speed_mps]
```

`Contact Point`는 좌·우 gripper 외측 pad와 target 물체 사이의 접촉 force 크기로
가중 평균한 점이다. 유효 접촉이 없을 때는 `(0, 0, 0)`을 반환한다. 접촉점 noise는
유효 접촉이 있을 때만 적용되며, 미접촉 상태의 zero sentinel은 noise로 훼손하지 않는다.

물체 크기, 질량, 현재 pose, 현재 velocity, OSC calibration randomization 값은
observation으로 제공하지 않는다.

## 4. Action

### 4.1 12-D Action 정의

Policy action은 다음 순서의 12-D normalized vector이다.

```text
[stiffness_x, stiffness_y, stiffness_z,
 stiffness_roll, stiffness_pitch, stiffness_yaw,
 dx, dy, dz, droll, dpitch, dyaw]
```

모든 입력은 policy 및 runner에서 `[-1, 1]`로 제한된다.

| 구간 | 차원 | 변환 및 의미 |
|---|---:|---|
| Normalized stiffness | 6 | 각 축을 `[20, 300]` 범위의 OSC diagonal stiffness로 선형 변환 |
| Temporal target position | 3 | action당 최대 ±0.025 m의 EEF 상대 이동 |
| Temporal target orientation | 3 | action당 최대 ±0.12 rad의 상대 RPY 회전 |

상대 RPY 명령은 quaternion을 거쳐 axis-angle로 변환된 뒤 Isaac Lab OSC에 전달된다.
Temporal target은 고정된 절대 pose가 아니라 매 policy step의 현재 EEF pose를 기준으로
누적되지 않는 상대 target을 생성한다.

Stiffness 6-D는 full matrix의 독립적인 모든 원소를 출력하는 것이 아니라 task-space
6축에 대응하는 diagonal stiffness matrix를 구성한다. Damping은 critical damping
형태로 stiffness에서 계산되며, episode별 작은 calibration randomization을 적용한다.

OSC가 계산한 joint torque는 UR5e simulated effort limit의 기본 90%를 기준으로
clamp한다. NaN/Inf action은 0으로 치환하며 범위 밖 action과 torque clipping 상태는
내부 saturation 상태로 기록한다. 현재 reward에는 별도 action-rate 또는 torque
penalty를 추가하지 않았다.

### 4.2 Gripper 제어

Gripper는 policy action에 포함되지 않는다. 모든 reset과 매 physics step에 open joint
target `0.0`을 다시 적용한다. Gripper actuator는 이 target을 유지하기 위해 stiffness
`2000`, damping `100`의 PD gain을 사용한다.

따라서 policy는 gripper를 닫아 물체를 잡을 수 없으며, 열린 gripper의 외측 contact
pad로 물체를 밀어야 한다.

### 4.3 HomeReturn의 OSC 제어

HOME phase에서도 action 구조와 controller는 바뀌지 않는다. Policy는 다음 정보로
HomeReturn action을 생성한다.

- `task_phase=1`
- 현재 joint position과 velocity
- 현재 EEF pose
- F/T 및 contact point
- 직전 OSC action

Home target은 robot asset에 정의된 6개 arm joint의 default pose이다. Reward와 성공
판정은 이 joint pose를 기준으로 계산하지만, 실제 robot command는 끝까지 OSC
stiffness와 relative EEF pose만 사용한다.

## 5. Critic과 학습 신호

### 5.1 Critic 구성

RSL-RL actor와 critic은 모두 `[512, 256, 128]` ELU MLP를 사용하고 observation
normalization을 활성화한다. Critic은 simulator의 현재 물체 pose나 velocity를 직접
입력받지 않는다. 다만 reward와 termination 계산은 정확한 simulator state를 사용하기
때문에, 이 privileged 정보는 value target을 만드는 학습 신호에 간접적으로 반영된다.

Isaac Lab Reward Manager는 각 policy step에서 아래 값을 누적한다.

```text
episode reward contribution = raw reward × term weight × step_dt
```

등록된 reward term은 사용자가 승인한 다음 네 개뿐이다.

```text
reaching
contact
push
home_return
```

방향 오차, lateral error, overshoot, Home contact 등은 별도의 다섯 번째 reward term으로
등록하지 않고 해당 목적에 속하는 `push` 또는 `home_return` 내부 penalty로 구성한다.

### 5.2 Reaching Reward

| 항목 | 값 |
|---|---:|
| Manager weight | +1.0 |
| 활성 phase | SWEEP, target contact 전 |
| 거리 scale | 0.12 m |
| 물체 표면 clearance | 0.008 m |
| table-side pad 높이 offset | 0.055 m |

목표 접근점은 현재 target 중심에서 push 반대 방향으로 이동한 점이다. 수평 stand-off는
고정값이 아니라 `물체 반지름 + 0.008 m`로 계산하므로, random size 물체에서도 목표가
표면을 기준으로 유지된다. EEF 높이는 열린 gripper의 gap이 아니라 table-side 외측 pad가
물체 옆면에 도달하도록 0.055 m 올린다.

정규화 거리 `e = distance / 0.12`에 대해 raw reward는 다음 형태이다.

```text
exp(-e²) - 0.20 × clamp(e, max=3)
```

따라서 접근점 근처에서는 양의 reward를 받고 멀리 떨어져 있으면 penalty를 받는다.
Target pad contact가 생기면 Reaching reward는 0이 되고 Contact/Push 목적이 중심이 된다.

### 5.3 Contact Reward

| 항목 | 값 |
|---|---:|
| Manager weight | +1.5 |
| 활성 phase | SWEEP |
| 유효 접촉 force threshold | 0.25 N |

좌·우 외측 pad 중 하나가 target 물체와 유효하게 접촉하면 raw reward `1`, 그렇지 않으면
`0`을 반환한다. Table이나 다른 robot link와의 접촉은 이 reward로 인정하지 않는다.

지속 접촉은 매 step의 binary reward와 별도의 contact-loss termination을 함께 사용해
학습한다. 최초 접촉 전 접근 시간은 contact-loss로 종료하지 않는다.

### 5.4 Push Reward와 Penalty

| 항목 | 값 |
|---|---:|
| Manager weight | +2.0 |
| 활성 phase | SWEEP |
| Velocity tracking std | 0.035 m/s |
| 가속 구간 | 최초 0.030 m |
| 감속 구간 | 마지막 0.050 m |
| Endpoint std | 0.035 m |
| 정지 속도 std | 0.025 m/s |

Push reward는 정확한 현재 물체 pose와 velocity를 사용하는 privileged 학습 신호이다.
이 값들은 observation에 포함되지 않는다.

목표 속도는 episode 전체에서 갑자기 켜지고 꺼지는 상수값이 아니다.

1. 시작 0.030 m 동안 목표 속도의 25%에서 sampled cruise speed까지 smooth하게 가속한다.
2. 중간 구간에서는 sampled target speed를 유지한다.
3. endpoint 전 0.050 m 동안 목표 속도를 0까지 smooth하게 감속한다.

Push raw reward는 다음 요소를 하나의 term으로 결합한다.

| 내부 요소 | 계수 | 의미 |
|---|---:|---|
| Contact velocity tracking | +2.5 | target velocity vector가 목표 profile과 일치하는 정도. Pad contact 중에만 지급 |
| Directional progress | +0.75 | 목표 방향 object 속도를 commanded speed로 정규화 |
| Stopped at endpoint | +4.0 | endpoint 오차와 object speed가 동시에 작은 Gaussian reward |
| Lateral error | -2.0 | 목표 진행선에서 벗어난 거리를 command distance로 정규화 |
| Overshoot | -3.0 | 목표 거리보다 더 밀어낸 거리를 command distance로 정규화 |

방향 reward는 역방향 이동에 음수를 줄 수 있도록 `[-1, 1]`로 제한한다. Endpoint
reward는 위치만 맞고 물체가 빠르게 움직이는 통과 상태에는 큰 값을 주지 않는다.

### 5.5 HomeReturn Reward와 Penalty

| 항목 | 값 |
|---|---:|
| Manager weight | +2.0 |
| 활성 phase | HOME |
| Home joint Gaussian std | 0.35 rad |
| Joint error normalization | 0.75 rad |
| Contact threshold | 0.25 N |
| Parked displacement scale | 0.010 m |

HomeReturn raw reward는 다음 요소로 구성된다.

| 내부 요소 | 계수 | 의미 |
|---|---:|---|
| Home pose tracking | +3.0 | 6개 joint의 mean squared wrapped error에 대한 Gaussian reward |
| Mean joint error | -1.0 | 평균 절대 joint error를 0.75 rad로 정규화한 penalty |
| Robot-target contact | -4.0 | HOME 중 target과 robot 어느 부분이든 접촉하면 penalty |
| Object displacement | -3.0 | HOME 진입 시 저장한 parked pose에서 이동한 거리 penalty |

접촉과 물체 이동은 reward penalty만 적용되는 것이 아니라 아래 termination 조건으로도
강제된다. HOME reward에는 별도 scripted path나 EEF waypoint가 없으므로 policy는
joint/EEF 상태와 충돌 결과를 이용해 안전한 복귀 경로를 학습해야 한다.

### 5.6 의도적으로 제외한 Penalty

Reward 항을 추가하지 않는다는 제약에 따라 다음 항은 별도 reward로 등록하지 않았다.

- Action-rate penalty
- Joint-velocity penalty
- Commanded-effort penalty
- Torque-saturation penalty
- Object-acceleration penalty
- Episode 조기 실패 penalty

수치적으로 위험한 joint speed, F/T overload, gripper 내부 진입 등은 reward가 아니라
명시적 termination으로 처리한다.

## 6. Termination

### 6.1 성공 Termination

HOME phase에서 아래 조건을 0.25초 동안 연속으로 만족하면 episode가 성공 종료된다.

| 조건 | Threshold |
|---|---:|
| 모든 arm joint의 Home position error | < 0.12 rad |
| 모든 arm joint speed | < 0.15 rad/s |
| target endpoint error | < 0.030 m |
| target linear speed | < 0.025 m/s |
| HOME 진입 pose 대비 target displacement | < 0.010 m |
| robot-target contact | 없음, threshold 0.25 N |

SWEEP 목표 달성은 최종 성공 termination이 아니다. 목표점에서 정지하면 HOME phase로
전환되고, HomeReturn까지 완료해야 episode 성공으로 기록된다.

### 6.2 Contact Loss Termination

Target과 최초 유효 pad contact가 발생한 뒤, SWEEP phase에서 접촉이 연속 0.75초 이상
사라지면 실패 종료한다.

- 최초 접촉 전 Reaching 구간에는 적용하지 않는다.
- 0.75초 안에 재접촉하면 timer를 0으로 reset한다.
- HOME phase에는 적용하지 않는다.
- 목표점 정지 dwell은 0.30초이므로 정상적인 phase 전환이 contact-loss보다 먼저
  완료될 수 있다.

### 6.3 Gripper Exclusion Termination

Target 중심을 현재 EEF frame으로 변환한 뒤 아래 local exclusion box 안에 들어오면
즉시 실패한다.

```text
EEF-local center half extents:
X = 0.040 m
Y = 0.040 m
Z = 0.058 m
```

이 box는 EEF orientation과 함께 회전한다. 따라서 특정 world-frame gripper 자세를
강제하지 않으면서, 물체를 열린 finger 사이로 넣어 미는 동작만 금지한다.

### 6.4 HOME Contact와 Object Disturbance Termination

SWEEP 종료 순간에는 pad가 물체와 접촉 중일 수 있으므로 HOME 진입 즉시 접촉 실패로
처리하면 정상적인 이탈이 불가능하다. 다음 규칙으로 초기 이탈과 재접촉을 구분한다.

1. HOME 진입 후 최대 0.30초 동안 기존 접촉에서 이탈할 수 있다.
2. 한번 contact-free 상태가 된 뒤 target과 다시 접촉하면 즉시 실패한다.
3. 0.30초가 지나도록 접촉을 유지해도 실패한다.

접촉 여부와 별개로 HOME 진입 시 저장한 물체 상태에서 아래 중 하나가 발생하면
`object_disturbed_home` 실패로 종료한다.

```text
parked pose 대비 displacement > 0.015 m
또는 object linear speed       > 0.10 m/s
```

### 6.5 공통 Safety Termination

| Termination | 실패 조건 |
|---|---|
| Timeout | episode 시간이 20 s에 도달 |
| Invalid target pose | target center 높이 < 0.76 m 또는 roll/pitch 절댓값 > 0.80 rad |
| Excessive F/T | force norm > 100 N 또는 torque norm > 15 Nm |
| Arm speed | arm joint 중 하나라도 절대 속도 > 6.5 rad/s |

Timeout은 Isaac Lab의 time-out termination으로 기록되고 나머지는 실패 termination으로
기록된다. 실패 termination에 연결된 별도 reward penalty는 없다.

## 7. Randomization

### 7.1 Target 물체

| 항목 | 범위 | Sampling 시점 |
|---|---:|---|
| 정육면체 한 변 길이 | 0.04–0.08 m | simulator startup, 병렬 환경별 |
| 질량 | 0.25–2.0 kg | 매 episode reset |
| 시작 X offset | -0.05–0.05 m | 매 episode reset |
| 시작 Y offset | -0.14–0.14 m | 매 episode reset |
| 시작 yaw | -π–π rad | 매 episode reset |

기본 물체 중심은 robot base 기준 대략 `(0.50, 0.0)`이다. Z 위치는 고정 offset으로
randomize하지 않고 실제 sampled cube size의 절반을 table top 높이 `0.775 m`에 더해
항상 물체 바닥면이 table 위에 놓이도록 계산한다. Reset 시 linear/angular velocity는
0으로 초기화한다.

Isaac Lab의 rigid-body scale 변경은 physics startup 전에만 안전하게 적용할 수 있다.
따라서 크기는 병렬 환경마다 서로 다르지만 동일 simulator 실행 안에서 해당 환경의
episode가 reset되어도 유지된다. 질량과 시작 pose는 매 episode 다시 sampling한다.
질량 변경 시 inertia도 새 질량에 맞춰 다시 계산한다.

### 7.2 Desired Motion

| 항목 | 범위 |
|---|---:|
| 방향 | XY 전체 방향, angle -π–π |
| 명목 거리 | 0.12–0.35 m |
| 목표 cruise speed | 0.04–0.12 m/s |

방향과 거리를 단순히 독립 sampling하면 긴 command가 table 또는 robot 작업영역 밖에
목표점을 만들 수 있다. 이 환경은 현재 물체 위치, sampled 물체 반지름, boundary
margin을 이용하여 해당 방향으로 허용되는 최대 거리를 먼저 계산한다.

사용하는 object-center 안전영역은 다음과 같다.

```text
X workspace:  0.18–0.82 m
Y workspace: -0.36–0.36 m
추가 boundary margin: 0.015 m + cube half-size
```

Sampling 절차는 다음과 같다.

1. 전체 평면에서 방향을 sampling한다.
2. 그 방향에서 안전영역 경계까지 가능한 최대 거리를 계산한다.
3. 최소 거리 0.12 m를 확보할 수 없는 방향은 다시 sampling한다.
4. 가능한 상한과 0.35 m 중 작은 값까지 실제 거리를 sampling한다.
5. 반복 sampling이 실패하면 workspace 중심 방향을 안전한 fallback으로 사용한다.

따라서 observation에 보이는 distance는 항상 실제 sampled 방향과 초기 위치에서 수행
가능한 값이다. 모든 episode가 반드시 0.35 m를 미는 것은 아니지만, 기존 0.22 m보다
긴 command가 가능한 위치와 방향에서는 최대 0.35 m까지 학습한다.

### 7.3 OSC Parameter Randomization

OSC randomization은 policy action의 의미가 크게 바뀌어 수렴을 방해하지 않도록 작은
범위만 사용한다. 각 값은 매 episode action reset에서 환경별로 sampling하며 policy에
노출하지 않는다.

| 항목 | Multiplicative 범위 | 의미 |
|---|---:|---|
| Stiffness calibration | 0.95–1.05 | policy가 명령한 6축 stiffness에 적용 |
| Damping calibration | 0.95–1.05 | stiffness 기반 damping gain에 적용 |
| Effort-limit calibration | 0.97–1.03 | 기본 90% torque clamp에 적용 |

Stiffness calibration 적용 후에도 최종 stiffness는 `[20, 300]` 범위로 clamp된다.

### 7.4 Robot Reset Randomization

매 episode 시작 시 UR5e arm은 asset의 default Home joint pose 주위에서 각 joint를
uniform ±0.04 rad 범위로 perturb한다. 초기 joint velocity는 0이다. 이 초기 자세는
최종 Home target 자체를 바꾸지 않으며, HomeReturn 성공은 perturb되지 않은 canonical
default joint pose를 기준으로 평가한다.

### 7.5 Randomize하지 않는 항목

현재 버전은 다음 항목을 고정한다.

- Table 크기와 pose
- Table 및 target friction/restitution
- Observation noise 범위
- Physics timestep과 policy decimation
- Gripper open target과 gripper PD gain
- F/T 및 safety threshold
- Home joint target

추가 domain randomization은 reward 항 추가와 별개이지만, 학습 난이도와 task 의미를
바꿀 수 있으므로 범위를 확장할 때는 별도로 검토해야 한다.

## 8. 학습 및 재생

Windows PowerShell 학습 예시는 다음과 같다.

```powershell
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\train_independent_sweep.py `
  --num_envs 2048 --device cuda:0 --headless
```

기본 experiment 이름은 다음과 같다.

```text
ur5e_osc_sweep_independent
```

재생에는 Isaac Lab 기본 RSL-RL player를 사용한다.

```powershell
.\IsaacLab\isaaclab.bat -p `
  IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py `
  --task Isaac-Sweep-Object-UR5e-OSC-Independent-v0 `
  --checkpoint C:\absolute\path\model.pt `
  --num_envs 1 --device cuda:0
```

이 환경은 observation 차원, phase state, action 의미가 기존 checkpoint와 다르므로
기존 sweep 계열 checkpoint를 그대로 resume하지 않고 새 run으로 학습하는 것을
전제로 한다.
