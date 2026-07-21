# Isaac-Sweep-Object-UR5e-OSC-Independent-v0

이 문서는 `osc_sweep_independent` 패키지의 현재 구현을 기준으로 작성한 환경 명세다.
특히 policy가 실제로 받는 관측, 출력하는 액션, Reward Manager에 등록된 보상과
Domain Randomization의 적용 시점 및 범위를 설명한다.

환경 등록 ID는 다음과 같다.

```text
Isaac-Sweep-Object-UR5e-OSC-Independent-v0
```

주요 구현 파일은 다음과 같다.

| 구성 | 구현 파일 |
|---|---|
| Scene, observation/action/reward/event 설정 | `env_cfg.py` |
| 12-D variable-stiffness OSC action | `mdp/actions.py` |
| sweep command, phase 전환, 목표점 marker | `mdp/commands.py` |
| 관측 계산과 observation noise | `mdp/observations.py` |
| 네 개의 reward 계산 | `mdp/rewards.py` |
| 물체 크기 및 reset pose randomization | `mdp/events.py` |
| 성공·실패 termination | `mdp/terminations.py` |
| PPO actor/critic 설정 | `agents/rsl_rl_ppo_cfg.py` |

## 1. Task와 Scene

UR5e와 항상 열린 Robotiq 2F-85 gripper로 선반 위 정육면체를 지정 방향과 거리만큼
밀고, 목표점에서 정지시킨 뒤 물체를 다시 건드리지 않으면서 Home joint pose로
복귀하는 작업이다.

```text
REACH(phase=0)
  물체의 push 반대편 옆면으로 접근
       ↓ target-pad filtered contact > 0.25 N
SWEEP(phase=1)
  접촉 유지 → 목표 속도로 밀기 → 목표점에서 정지
       ↓ endpoint error/speed dwell 0.30 s
HOME(phase=2)
  물체에서 이탈 → OSC만으로 Home joint pose 복귀
       ↓ 안정 조건 dwell 0.25 s
SUCCESS
```

세 phase 모두 동일한 actor와 동일한 12-D OSC action을 사용한다. HOME에서
scripted trajectory나 별도의 joint-position controller로 전환하지 않는다.

### 1.1 Scene 구성

| 항목 | 설정 |
|---|---|
| Robot | UR5e + virtual F/T body + Robotiq 2F-85 + 좌·우 sweep contact pad |
| Shelf USD | `omniverse://192.168.0.13/Library/Shelf/Arena/Collected_speedrack_shape/speedrack_shape.usd` |
| Shelf pose | position `(-0.7, 0.0, 0.0)`, quaternion `(1, 0, 0, 0)` |
| Target | 기본 0.06 m 정육면체, 기본 질량 0.35 kg |
| Target 기본 pose | position `(-0.60, 0.0, 1.05)` |
| Physics | 120 Hz |
| Policy | 30 Hz, decimation 4 |
| Episode 제한 | 20 s |
| 기본 병렬 환경 수 | 2,048 |
| Physics replication | 비활성화 |

Robot 초기 arm joint pose는 다음과 같다.

```text
[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
= [0.0, -2.2, 2.2, 0.0, 1.57, 0.785] rad
```

F/T 센서는 `VirtualFTSensor`, OSC 기준 EEF는 `SweepToolCenter`다. Robot의 각 rigid
body에 개별 filtered ContactSensor를 두어 filter 0은 `TargetCube`, filter 1은
`Shelf/rack`, filter 2 이후는 18개 robot rigid body를 측정한다. 이 구조는 하나의
wildcard sensor가 여러 robot body를 동시에
매칭하면서 발생하던 PhysX filter-count 오류를 피한다.

Robot articulation의 self-collision physics도 활성화한다. Joint로 직접 연결되거나
중립 gripper pose에서 구조적으로 맞닿는 5개 쌍만 종료 판정에서 제외하고, 그 밖의
비인접 UR 링크·gripper 링크 및 UR–gripper 접촉은 `0.1 N` 초과 시 실패 종료한다.

### 1.2 Sweep command와 phase 전환

Episode마다 다음 command가 생성된다.

```text
c = [direction_x, direction_y, distance, target_speed]
```

| 성분 | 범위 |
|---|---:|
| XY 방향 각도 | `[-π, π]` |
| 거리 | `[0.12, 0.35]` m |
| cruise speed | `[0.04, 0.12]` m/s |

방향은 단위 벡터다. 물체 크기와 workspace 경계를 고려해 최소 0.12 m를 안전하게 밀 수
있는 방향만 채택하며, 실제 distance 상한은 해당 방향의 경계까지 거리와 0.35 m 중
작은 값이다. 최대 32회 방향 sampling이 실패하면 workspace 중심 방향을 사용한다.

```text
Robot-root object-center workspace X: [ 0.50, 0.90] m
Robot-root object-center workspace Y: [-0.50, 0.50] m
Boundary margin                      : 0.015 m + sampled cube half-size
```

이 범위는 composed shelf USD의 중간 상판 world bounds X `[-0.90,-0.50]`,
Y `[-0.50,0.50]`를 실측한 값이다. 로봇 root가 Z축으로 180도 회전해 있으므로 command
frame에서는 X가 `[0.50,0.90]`이 된다. REACH에서 좌·우 pad 중 하나가 TargetCube에
`0.25 N`을 초과하는 filtered contact를 만들면 SWEEP으로 전환한다. SWEEP 중 다음
조건을 0.30초 연속 만족하면 HOME으로 전환한다.

```text
||object_position - goal_position|| < 0.025 m
||object_linear_velocity||          < 0.020 m/s
```

HOME 진입 순간의 물체 위치는 `parked_object_pos_w`에 저장되어 HomeReturn reward와
성공·실패 판정에 사용된다.

## 2. Observation

### 2.1 Policy observation 계약

Policy observation은 순서가 고정된 56-D vector다. Actor와 Critic 모두 같은 `policy`
group을 사용하며 별도의 privileged critic observation은 없다.

| 순서 | 항목 | 차원 | 값과 좌표계 | Additive noise |
|---:|---|---:|---|---|
| 1 | `joint_pos` | 6 | UR5e arm absolute joint position, rad | 각 축 uniform `[-0.002, 0.002]` rad |
| 2 | `joint_vel` | 6 | UR5e arm joint velocity, rad/s | 각 축 uniform `[-0.01, 0.01]` rad/s |
| 3 | `joint_effort` | 6 | arm에 실제 적용된 torque, Nm | 각 축 uniform `[-0.5, 0.5]` Nm |
| 4 | `eef_pose` | 6 | robot root 기준 `x,y,z,roll,pitch,yaw` | 없음 |
| 5 | `ft_sensor` | 6 | virtual F/T body frame의 `Fx,Fy,Fz,Tx,Ty,Tz` | force ±0.5 N, torque ±0.02 Nm |
| 6 | `contact_point` | 3 | robot root 기준 target 접촉점 | 유효 측정에만 각 축 ±0.002 m |
| 7 | `initial_target_pose` | 6 | reset 시 robot root 기준 target `xyz+RPY` | xyz ±0.003 m, RPY ±0.02 rad |
| 8 | `desired_motion` | 4 | `direction_x,direction_y,distance,speed` | 없음 |
| 9 | `task_phase` | 1 | REACH=`0.0`, SWEEP=`1.0`, HOME=`2.0` | 없음 |
| 10 | `last_action` | 12 | 직전 normalized policy action | 없음 |
| | **합계** | **56** | | |

Observation group에서 `enable_corruption=True`이므로 위 noise는 학습과 실행 시 실제
관측 vector에 적용된다. 환경은 단위별 수동 scaling을 적용하지 않지만 RSL-RL actor와
critic 모두 running observation normalization을 활성화한다.

### 2.2 좌표계와 센서 처리

`eef_pose`와 `initial_target_pose`는 world pose를 robot root frame으로 변환한 뒤
quaternion을 roll/pitch/yaw로 바꾼 값이다.

F/T 관측은 다음 부호 규칙을 사용한다.

```text
ft_sensor = -body_incoming_joint_wrench_b[VirtualFTSensor]
```

`contact_point`는 좌·우 sweep pad가 TargetCube에 가한 filtered force의 크기로 접촉점을
가중 평균한다. 누적 force weight가 0.25 N 이하이면 유효 접촉이 아닌 것으로 판단해
정확히 `(0,0,0)`을 반환한다. `MaskedUniformNoiseCfg`는 이 zero sentinel에는 noise를
추가하지 않으므로 policy가 미접촉과 작은 좌표값을 구분할 수 있다.

### 2.3 Policy에 제공하지 않는 privileged state

다음 값은 56-D 관측에 포함되지 않는다.

- 현재 물체 pose와 linear/angular velocity
- 현재 물체 크기와 질량
- target/shelf friction sample
- 목표점 world position
- stiffness, damping, effort calibration sample
- shelf contact force 전체 값

현재 물체 상태는 Push reward, phase 전환, success와 safety termination에만 사용된다.
따라서 actor는 initial pose, desired motion, proprioception, F/T, contact point와 action
history를 이용해 물체의 현재 상태를 간접 추정해야 한다.

## 3. Action

### 3.1 12-D policy action

Policy가 출력하는 action 순서는 다음과 같다.

```text
a = [k_x, k_y, k_z, k_roll, k_pitch, k_yaw,
     dx,  dy,  dz,  droll,  dpitch,  dyaw]
```

RSL-RL runner와 환경 양쪽에서 action을 `[-1,1]`로 제한한다. NaN 또는 Inf 성분은
0으로 치환된다.

첫 6개 stiffness action `a_k`는 축별로 다음과 같이 변환된다.

```text
K_raw = 20 + 0.5 × (a_k + 1) × (300 - 20)
K     = clamp(K_raw × stiffness_calibration, 20, 300)
```

따라서 policy가 직접 명령하는 값은 task-space 6축 diagonal stiffness이며 범위는
`[20,300]`이다. Full 6×6 stiffness matrix의 비대각 성분을 출력하는 구조는 아니다.

뒤의 6개 pose action은 한 policy step 동안의 상대 EEF target으로 변환된다.

```text
Δposition = [dx,dy,dz] × 0.025 m
ΔRPY      = [droll,dpitch,dyaw] × 0.12 rad
```

상대 RPY는 quaternion으로 변환한 뒤 axis-angle 표현으로 바뀌어 Isaac Lab OSC에
전달된다. 최종 OSC 입력 내부 순서는 `relative_pose(6) + stiffness(6)`으로 재배열된다.

### 3.2 OSC 설정과 torque 제한

| 항목 | 설정 |
|---|---|
| Target type | relative pose |
| Controlled task axes | translation 3축 + rotation 3축 모두 활성 |
| Inertial dynamics decoupling | 활성 |
| Gravity compensation | 활성 |
| Null-space control | 비활성 |
| Nominal motion stiffness | translation 120, rotation 35 |
| Damping ratio | 6축 모두 1.0 |
| Torque clamp | simulated joint effort limit의 `0.9 × effort_calibration` |

Damping/stiffness 표의 nominal 값은 controller configuration의 초기값이며,
`variable_kp` mode에서는 매 policy step마다 위 12-D action에서 계산한 stiffness가 실제
명령값으로 들어간다. 예를 들어 calibration 전 normalized stiffness action이 0이면
6축 모두 160이 된다.

Damping gain은 stiffness에서 계산된 값에 episode별 damping calibration을 곱한다. OSC
출력 torque에 NaN/Inf가 있으면 해당 환경의 torque를 0으로 만들며, joint effort limit을
넘는 torque는 clamp한다. 범위 밖 action이나 torque clamp는 내부
`torque_saturated` flag에 기록되지만 별도의 reward penalty는 없다.

### 3.3 Gripper와 HOME action

Gripper joint는 action에 포함되지 않는다. Reset 및 모든 physics step에서 open target
`0.0`을 다시 적용하므로 policy는 물체를 잡지 못하고 외측 pad로 밀어야 한다.

HOME에서도 action 의미는 바뀌지 않는다. Policy는 `task_phase=2`, joint/EEF 상태와
센서 관측을 보고 상대 EEF pose와 stiffness를 연속 출력하여 canonical Home joint
pose로 복귀한다.

## 4. Reward

Reward Manager에 등록된 term은 정확히 네 개다.

| Term | Manager weight | 활성 phase |
|---|---:|---|
| `reaching` | 1.0 | REACH |
| `contact` | 1.5 | SWEEP |
| `push` | 2.0 | SWEEP |
| `home_return` | 2.0 | HOME |

Isaac Lab은 policy step마다 다음 값을 episode reward에 누적한다.

```text
weighted contribution = raw_term × manager_weight × step_dt
step_dt = 1 / 30 s
```

아래 식에서 `I(condition)`은 조건이 참일 때 1, 아니면 0이다.

### 4.1 Reaching

현재 물체 중심 `p_obj`, command 방향 world vector `d`, sampled cube size `s`에 대해
접촉 전 목표점을 계산한다.

```text
stand_off   = s/2 + 0.008
p_pre       = p_obj - stand_off × d
p_pre.z    += 0.055
e           = clamp(||p_eef - p_pre|| / 0.12, max=3)
r_reaching  = (exp(-e²) - 0.20e)
              × I(REACH) × I(no target-pad contact)
```

물체 크기에 따라 stand-off가 바뀌므로 크기가 달라도 물체 표면 기준 접근 위치가
유지된다. Target contact가 발생하면 reaching은 0이 되고 contact/push reward가 학습을
주도한다.

### 4.2 Contact

```text
r_contact = I(target-pad filtered contact force > 0.25 N) × I(SWEEP)
```

좌·우 외측 pad 중 하나라도 TargetCube와 유효 접촉하면 raw reward 1을 준다. Robot의
다른 link나 shelf와의 접촉은 이 reward로 인정하지 않는다.

### 4.3 Push

초기 물체 위치로부터 robot-root frame 변위 `Δp`, command 단위 방향 `d`, command 거리
`L`, cruise speed `v_c`를 사용한다.

```text
progress = dot(Δp_xy, d)
lateral  = ||Δp_xy - progress × d||
remaining = L - progress
```

`smoothstep(x)=x²(3-2x)`이고 입력은 `[0,1]`로 clamp한다.

```text
accel = 0.25 + 0.75 × smoothstep(clamp(progress / 0.030, 0, 1))
stop  = smoothstep(clamp(remaining / 0.050, 0, 1))
v_des = v_c × accel × stop
v_des_world = direction_world × v_des
```

즉 시작 시 cruise speed의 25%에서 출발해 최초 0.030 m 동안 가속하고, endpoint 전
0.050 m 동안 0까지 감속한다.

```text
velocity_tracking = exp(-( ||v_obj-v_des_world|| / 0.035 )²)
direction_progress = clamp(dot(v_obj,d_world)/v_c, -1, 1)
normalized_lateral = lateral / L
overshoot = relu(progress-L) / L
stopped_at_goal = exp(-(endpoint_error/0.035)²
                      -(||v_obj||/0.025)²)

r_push = [2.5 × velocity_tracking × I(target-pad contact)
          +0.75 × direction_progress
          +4.0 × stopped_at_goal
          -2.0 × normalized_lateral
          -3.0 × overshoot] × I(SWEEP)
```

Push는 현재 물체 pose와 velocity를 사용하는 privileged reward다. 위치만 목표에 맞고
빠르게 통과하는 경우에는 `stopped_at_goal`을 크게 받지 못한다.

### 4.4 HomeReturn

각 arm joint의 wrapped Home 오차를 `q_err`, HOME 진입 시 저장한 물체 위치에서 현재
변위를 `Δp_parked`라 한다.

```text
pose_tracking = exp(-mean(q_err²) / 0.35²)
joint_error   = mean(abs(q_err)) / 0.75
disturbance   = clamp(||Δp_parked|| / 0.010, max=4)

r_home = [3.0 × pose_tracking
          -1.0 × joint_error
          -4.0 × I(any robot-target contact > 0.25 N)
          -3.0 × disturbance] × I(HOME)
```

이 reward는 Home pose 접근뿐 아니라 물체 재접촉 및 parked object 이동도 같은 term
안에서 벌점으로 처리한다.

### 4.5 의도적으로 없는 reward

다음 항목은 별도 RewardTerm으로 등록하지 않는다.

- action-rate, joint-velocity, torque 또는 torque-saturation penalty
- object acceleration penalty
- 조기 실패 penalty
- shelf collision penalty

위험 상태는 reward shaping 대신 F/T, joint speed, shelf collision, contact loss 등의
termination으로 처리한다.

## 5. Domain Randomization

### 5.1 적용 시점 요약

| 시점 | Randomization | Episode reset 시 재추출 여부 |
|---|---|---|
| `prestartup` | TargetCube 크기 | 아니요. simulator 실행 동안 env별 고정 |
| `startup` | TargetCube와 shelf의 physics material | 아니요. simulator 실행 동안 env별 고정 |
| `reset` | arm 초기 joint offset, target 질량, target 시작 pose/yaw | 예 |
| action term `reset` | OSC stiffness/damping/effort calibration | 예 |
| command `reset` | sweep 방향, 거리, cruise speed | 예 |
| observation 계산 | sensor/proprioception additive noise | 매 관측마다 적용 |

`prestartup`과 `startup` randomization은 병렬 환경들이 한 simulator 실행에서 서로 다른
조건을 갖도록 한다. 단일 환경으로 실행하면 크기와 마찰은 그 실행에서 한 sample만
사용하므로, 충분한 다양성을 얻으려면 병렬 환경 또는 여러 simulator 실행이 필요하다.

### 5.2 Target geometry, mass와 초기 pose

| 항목 | Distribution/범위 |
|---|---:|
| Cube 한 변 길이 | uniform `[0.04,0.08]` m |
| 질량 | uniform `[0.25,2.0]` kg |
| 시작 X offset | uniform `[-0.05,0.05]` m |
| 시작 Y offset | uniform `[-0.14,0.14]` m |
| 시작 yaw | uniform `[-π,π]` rad |
| 시작 linear/angular velocity | 0으로 reset |

크기는 기본 0.06 m cube에 uniform scale을 적용하며 env별 sampled size를 별도 buffer에
저장한다. Reset 시 물체 중심 Z는 다음 식으로 계산한다.

```text
object_center_z = shelf_surface_height(1.05 m) + sampled_size/2
```

따라서 모든 크기의 물체 바닥이 선반 표면에 놓인다. 질량 변경 시 inertia도 다시
계산한다.

### 5.3 Object–Shelf friction

TargetCube와 shelf material을 서로 독립적으로 randomize한다.

| Asset | Static friction | Dynamic friction | Restitution | Buckets |
|---|---:|---:|---:|---:|
| TargetCube | `[0.40,1.10]` | `[0.25,0.90]` | 0.0 | 64 |
| Shelf | `[0.40,1.10]` | `[0.25,0.90]` | 0.0 | 64 |

`make_consistent=True`이므로 sampled dynamic friction이 static friction보다 커지지 않게
보정한다. 실제 object–shelf 접촉에 사용되는 계수는 PhysX가 양쪽 material을 결합한
결과다. 양쪽을 모두 randomize하여 실제 선반 표면의 재질, 마모와 오염 차이에 대한
정책 강건성을 높인다.

### 5.4 Robot reset과 OSC calibration

매 episode에서 6개 arm joint를 canonical Home pose 기준 uniform `[-0.04,0.04]` rad로
perturb하고 joint velocity는 0으로 reset한다. Home reward와 success target은 perturb된
pose가 아니라 canonical Home pose다.

| OSC 항목 | Multiplicative 범위 |
|---|---:|
| Stiffness calibration | `[0.95,1.05]` |
| Damping calibration | `[0.95,1.05]` |
| Effort-limit calibration | `[0.97,1.03]` |

이 calibration sample은 observation에 노출되지 않는다. Stiffness는 calibration 적용
후에도 `[20,300]`으로 clamp된다.

### 5.5 Observation noise

Observation noise 역시 sim-to-real을 위한 sensor randomization이다. 구체적인 범위는
2.1절 표와 같으며, 독립 uniform additive noise를 매 observation 계산에 적용한다.
미접촉 contact-point zero sentinel만 예외적으로 noise를 적용하지 않는다.

### 5.6 Randomize하지 않는 항목

- Shelf USD와 shelf pose
- Robot base pose와 canonical Home joint pose
- Physics timestep, policy decimation과 episode 시간
- Gripper open target 및 gripper actuator gain
- F/T/contact/safety threshold
- Observation noise의 분포 범위 자체

## 6. 목표 위치 시각화

Debug visualization은 GUI의 단일 환경 실행(`--num_envs 1`, `--headless` 미사용)에서만
활성화된다. 각각 정확히 1 prototype/1 instance만 갖는 두 개의 독립 PointInstancer
visual Sphere로 구성되며, Fabric 초기화 전 `prestartup` event에서 생성한다. Headless
학습 또는 병렬 환경에서는 마커를 생성하지 않아 학습 비용과 Fabric prototype 경고를
방지한다. 마커 유무는 command, observation, reward에 영향을 주지 않는다.

| Marker | 의미 | 경로 |
|---|---|---|
| 반투명 파란 구 | episode 시작 시 물체 중심 | `/Visuals/Command/sweep_target_positions/initial` |
| 선명한 자홍색 구 | sampled sweep 목표 중심 | `/Visuals/Command/sweep_target_positions/goal` |

두 구는 물체나 상판에 가리지 않도록 실제 중심보다 Z축으로 `0.10 m` 높여 표시한다.
XY 위치는 실제 시작점/목표점과 동일하다. 시작점과 목표점을 하나의 다중-prototype
PointInstancer에 섞지 않아 Fabric prototype 수 불일치를 방지한다. 각 visualizer에는
항상 위치 한 개만 전달한다. 목표 구의 반지름은 `0.050 m`이며 gripper pad와 혼동하지
않도록 자홍색을 사용한다.

GUI에서 확인하려면 다음과 같이 `--headless` 없이 한 환경을 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/train_independent_sweep.py \
  --num_envs 1 --device cuda:0
```

## 7. 관련 문서

Termination 전체 조건, PPO 설정 및 학습·재생 명령은
[independent_osc_sweep.md](../../docs/independent_osc_sweep.md)에 정리되어 있다. 수치가
변경되면 이 문서와 해당 운영 문서를 함께 갱신해야 한다.

동일한 scene, observation, action, randomization과 termination을 상속하면서 네 개의
composite reward를 phase별 세부 term으로 분해한 환경은
[independent_osc_sweep_detailed.md](../../docs/independent_osc_sweep_detailed.md)에 정리되어
있다. Gym ID는 `Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0`이다.
