# Shelf Reach → Cube Pre-Reach → Cube Sweep 개발 기록

이 문서는 Sweep-Policy의 UR5e + Robotiq 선반 장면을 출발점으로 하여 다음 환경을
순차적으로 구축하고 조정한 과정을 정리한다.

- `shelf_reach`
  - `Isaac-Reach-Shelf-UR5e-Gripper-v0`
- `shelf_cube_pre_reach`
  - `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0`
  - `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1`
- `shelf_cube_sweep`
  - `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0`
  - `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1`
  - `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v2`

문서의 “현재 설정”은 이 저장소의 최종 코드를 기준으로 한다. 개발 중 시험했다가
철회한 설정은 별도로 표시한다.

## 1. 환경 계보

```text
example/Sweep-Policy의 선반·UR5e/Robotiq 자산
  │
  └─ Isaac-Reach-Shelf-UR5e-Gripper-v0
       가상 TCP pose Reach
       │
       ├─ Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0
       │    물리 Cube + Cube-relative moving Reach + 선반 접촉 패널티
       │    │
       │    └─ Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1
       │         Reach reward와 동작 비용 재조정
       │
       ├─ Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0
       │    CubePreReach-v0 + 고정 Sweep goal + pushing_target
       │
       └─ Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1
            CubePreReach-v1 + 속도/자세/전도 안정화
            │
            └─ Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v2
                 저속 Sweep + 고정 Shelf + 강화된 충돌·속도 제약 + PPO 안정화
```

Sweep v0는 PreReach v0에서, Sweep v1은 PreReach v1에서 갈라진다. Sweep v2는
Sweep v1을 직접 상속한다. 따라서 버전 번호가 같더라도 모든 환경이 일렬로
상속되는 구조는 아니다.

## 2. 공통 기반: Sweep-Policy 장면에서 Manager-based Reach로

### 2.1 출발점

`example/Sweep-Policy`에서 사용하던 결합형 UR5e + Robotiq USD와 선반 USD를
재사용했다. 원본 Sweep-Policy의 `pushing_target`에도 별도의 명시적 phase state나
Reach-complete gate는 없었다. 다만 end-effector와 wrist가 물체 뒤의 offset에 충분히
가까운지 확인하는 `zeta_m` 조건이 reward 내부 gate로 존재했다.

새 환경 계열에서는 한 번에 Sweep 전체를 옮기기 전에 다음 세 단계를 분리했다.

1. 가상 pose를 추적하는 기본 Reach를 검증한다.
2. 실제 Cube 뒤의 moving pose까지 접근하는 Pre-Reach를 검증한다.
3. Pre-Reach를 유지하면서 Cube를 고정 goal까지 미는 Sweep reward를 추가한다.

이 분리는 “팔이 목표 pose를 추적하지 못하는 문제”와 “접촉 후 Cube를 제대로 미는
문제”를 서로 다른 환경에서 진단하기 위한 것이었다.

## 3. `Isaac-Reach-Shelf-UR5e-Gripper-v0`

소스: `src/sweep_rl/sweep_rl/shelf_reach`

### 3.1 추가된 내용

물리 target object 없이 UR5e/Robotiq TCP가 robot-base frame의 가상 7D pose command를
추종하는 최소 환경을 만들었다.

- TCP: `robotiq_base_link` local `+X 0.13 m`
- action: UR5e arm 6축 relative joint-position action
- action scale: `0.5`, default joint pose 기준 offset
- gripper: action과 observation에서 제외하고 열린 자세 `0.0` 유지
- observation: `25-D`
  - arm joint position 6
  - arm joint velocity 6
  - pose command 7
  - previous action 6
- command: 4초마다 재샘플링되는 가상 pose
- orientation: `roll=pi/2`, `pitch=0`, `yaw=0`
- simulation/control: `60 Hz / 30 Hz`
- episode: `12 s = 360 control steps`
- termination: timeout만 사용

고정 `roll=pi/2`는 원본 기본 wrist 자세보다 TCP를 더 수직으로 세워, TCP local Y가
robot-base `+Z`를 향하도록 정한 목표 orientation이다.

### 3.2 기본 reward와 PPO

```text
position coarse = -0.2 * d_pos
position fine   = +0.1 * (1 - tanh(d_pos / 0.1))
orientation     = -0.1 * d_rot
action_rate     = -0.0001 * sum(delta_action^2)
joint_vel       = -0.0001 * sum(joint_velocity^2)
```

4500 common steps 이후 curriculum이 action-rate와 joint-velocity weight를 각각
`-0.005`, `-0.001`로 바꾼다.

초기 PPO는 다음 계열의 모든 v0/v1 환경이 기본적으로 상속했다.

- actor/critic: `[64, 64]`, `elu`
- Gaussian initial std: `1.0`, direct scalar std
- learning rate: `1e-3`, adaptive schedule
- entropy coefficient: `0.01`
- learning epochs/update: `8`
- rollout: 환경당 `24 step`
- maximum learning iterations: `10000`

이 단계의 목적은 물체 접촉이나 선반 안전 문제 없이 robot, TCP frame, pose reward와
RSL-RL 학습 경로를 먼저 검증하는 것이었다.

## 4. `CubePreReach-v0`: 물리 Cube와 moving Reach 도입

소스: `src/sweep_rl/sweep_rl/shelf_cube_pre_reach`

### 4.1 Shelf Reach에 추가된 내용

`Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0`는 기본 Reach에 실제 Cube를
추가하고, 고정된 임의 pose 대신 현재 Cube 뒤의 pose를 매 control step 계산한다.

```text
Cube size     = 0.08 x 0.08 x 0.20 m
Cube mass     = 1.5 kg
Cube center   = (-0.70, -0.10, 1.15) m
reach_offset  = (0, -cube_width * 1.1, +0.03) m
              = (0, -0.088, +0.03) m
```

Y offset을 `width * 1.1`로 둔 이유는 Cube 뒤쪽에 한 폭보다 약간 큰 접근 여유를
확보하기 위해서다. Z offset `+0.03 m`는 그리퍼가 Cube 하부를 밀 수 있게 하면서도
선반 바닥과 직접 충돌하지 않도록 확보한 높이다.

### 4.2 observation 확장

기존 25-D에 Cube 정보를 추가하여 29-D가 되었다.

| 항목 | 차원 | 비고 |
|---|---:|---|
| arm joint position | 6 | 기존 유지 |
| arm joint velocity | 6 | 기존 유지 |
| Cube-relative pose command | 7 | 매 step 움직임 |
| Cube center position | 3 | robot-base frame |
| Cube width | 1 | 환경 상수 |
| previous action | 6 | 기존 유지 |

Cube orientation, up-axis, linear/angular velocity는 정책 observation에 넣지 않았다.
이후 tilt나 속도를 reward 계산에 사용하더라도 simulator-side privileged 정보로만
사용한다는 원칙을 유지했다.

### 4.3 선반 충돌 패널티 추가

Cube-relative 목표를 쫓으면 로봇이나 그리퍼가 선반 지지판을 치는 행동이 나타날 수
있으므로 shelf-side contact sensor와 `shelf_collision`을 추가했다.

- robot/gripper의 필터된 contact만 검사
- Cube와 선반의 정상 접촉은 제외
- 접촉 force `> 1 N`
- shelf-local 중앙 지지면 `z=1.05 +/- 0.02 m`
- XY 범위 안의 접촉만 인정
- reward weight `-5.0`

정책에 contact observation을 주는 대신 reward 계산에서만 센서 정보를 사용했다.
PreReach-v0의 다른 reward, action, curriculum과 PPO는 Shelf Reach v0를 그대로
상속한다.

## 5. `CubePreReach-v1`: Reach reward 재조정

### 5.1 발생한 문제

기본 coarse/fine 조합은 Cube 바로 뒤의 moving offset에 정밀하게 도달시키는 목적에
비해 reward scale과 근거리 gradient가 충분히 직접적이지 않았다. 동시에 자세와
빠른 action/joint motion 비용의 상대 비중을 다시 맞출 필요가 있었다.

### 5.2 조정 내용

두 position reward를 하나의 exponential reward로 교체했다.

```text
R_position = 3.0 * exp(-10 * d_pos)
```

그 밖의 변경은 다음과 같다.

- fine-grained tanh position reward 제거
- orientation error weight: `-0.1 -> -0.7`
- action-rate weight: `-0.0001 -> -0.03`
- joint-velocity weight: `-0.0001 -> -0.03`
- Cube geometry, observation 29-D, relative action, shelf penalty는 v0와 동일

주의할 점은 PreReach-v1 자체는 부모 curriculum을 제거하지 않았다는 것이다.
curriculum 제거는 이후 Sweep-v1에서 이루어진다.

## 6. `CubeSweep-v0`: Reach 이후 implicit Sweep 추가

소스: `src/sweep_rl/sweep_rl/shelf_cube_sweep`

### 6.1 추가된 내용

PreReach-v0에 다음 기능을 추가했다.

- episode 초기 Cube 위치 기준 고정 Sweep goal
  - `goal = initial_cube_position + (0, +0.18, 0) m`
- wrist 위치를 별도로 계산하는 frame transformer
- moving push point와 EE/wrist 거리를 이용한 `zeta_m`
- Cube의 goal 거리와 Y 속도를 이용한 `pushing_target`
- Sweep reward weight `+6.0`

현재 reward의 핵심은 다음과 같다.

```text
push_point = current_cube_position + (0, -cube_width*1.1, +0.03)
zeta_m = (EE-to-push-point distance < 0.08)
         and (wrist Y error < 0.08)

if 0.05 < abs(cube_vy) < 0.10:
    velocity_reward = +0.5
elif abs(cube_vy) >= 0.10:
    velocity_reward = -0.5
else:
    velocity_reward = 0

if goal_distance < 0.03:
    raw_push = 2.0 * exp(-5 * goal_distance)
else:
    raw_push = zeta_m * (1 - goal_distance/0.18 + velocity_reward)
```

raw push reward의 이론상 success 최대는 `2.0`이고 v0 weight를 포함하면 `12.0`이다.

### 6.2 “Reach에는 도달하지만 Sweep하지 않음” 문제

초기 학습 `2026-08-07_23-18-41`에서 EE가 Reach 위치에는 충분히 도달했지만
Sweep 행동으로 이어지지 않는 현상이 확인되었다.

원본 Sweep-Policy와 현재 구조 모두 별도의 phase gate는 없었다. 실제 원인은
`pushing_target` 내부의 공간 gate가 정책 입장에서 충분히 쉽게 열리지 않거나,
Reach 목표와 push-point 조건이 미세하게 불일치할 수 있다는 점이었다.

이에 다음을 정리했다.

- Reach Y offset을 `cube_width * 1.1`로 통일
- reward의 push point도 같은 offset과 `+0.03 m` 높이를 사용
- `zeta_m` gate distance를 `0.08 m`로 두어 Reach 완료 부근에서 쉽게 활성화
- 고정 phase 전환 로직은 추가하지 않고 reward gate 기반 implicit 전환 유지

즉 정책은 별도의 `REACH -> SWEEP` 상태 변수를 받지 않는다. moving Reach pose에
가까워지면 `zeta_m`이 열리고 Sweep reward가 행동을 이어서 유도한다.

### 6.3 선반을 치는 문제

Reach/Sweep reward만 높이면 목표에 빨리 가기 위해 선반 바닥을 가로지르는 행동이
나타났다. 이 문제를 억제하기 위해 PreReach에서 도입한 filtered shelf collision
패널티를 유지하고, 이후 버전에서 weight와 검사 면을 더 강화했다.

## 7. `CubeSweep-v1`: 속도, 전도, wrist 자세 개선

### 7.1 학습에서 확인된 세 가지 문제

`2026-08-08_02-18-53` 학습과 재생에서 다음 문제가 관찰되었다.

1. 로봇 joint 움직임이 지나치게 빠름
2. Cube를 평행 이동시키기보다 넘어뜨리는 행동이 많음
3. 마지막 gripper/wrist가 수직이 아니라 약 45도로 남음

원인을 다음과 같이 정리했다.

- relative action scale과 큰 actuator limit만으로는 step별 target 점프를 직접
  제한하지 못함
- action/joint 비용 curriculum이 후반에 비용을 약화시키는 방향으로 작동 가능
- push reward는 Cube 이동만 보며, Cube tilt 자체는 차단하지 않음
- Cube footprint와 질량 분포가 높은 접촉에 대해 쉽게 전도될 수 있음
- command orientation이 수직이어도 기본 `wrist_3_joint=0.785 rad`와 wrist reset
  scaling 때문에 초기·학습 자세가 45도 방향으로 남을 수 있음

### 7.2 full-range absolute action과 rate limit

joint의 전체 가동 범위를 잃지 않으면서 한 control step의 target 변화만 제한하는
action term을 추가했다.

```text
requested_target = map(clamp(action, -1, 1), full_soft_joint_limits)
target_delta = clamp(requested_target - previous_target,
                     -max_delta, +max_delta)
applied_target = previous_target + target_delta
```

- shoulder/elbow: `0.05 rad/step`, 30 Hz 기준 최대 target slew `1.5 rad/s`
- wrist: `0.07 rad/step`, 30 Hz 기준 최대 target slew `2.1 rad/s`
- actuator hard limit은 기존 `3.14/6.28 rad/s` 유지
- reset 시 limiter 기준을 실제 reset joint position으로 다시 초기화

이 방식은 “step별 변화율”만 제한한다. 충분한 step이 주어지면 joint soft limit 전체에
도달할 수 있으므로 joint 가동 범위를 줄이지 않는다.

또한 Sweep-v1에서는 curriculum을 제거하고 `action_rate=-0.03`,
`joint_vel=-0.03`을 episode 및 학습 전체에 고정했다.

### 7.3 Cube 전도 억제

Cube 높이와 nominal contact point는 바꾸지 않았다. `+0.03 m` 접촉 높이는 선반과
그리퍼 사이의 안전 여유 때문에 유지했다. 대신 다음 물성 및 reward 조정을 사용했다.

- Cube footprint: `0.08 x 0.08 -> 0.11 x 0.11 m`
- COM: geometric center보다 `0.05 m` 아래
- diagonal inertia: `(0.008, 0.008, 0.003) kg m^2`
- Cube tilt가 10도에 도달하면 `pushing_target`이 0이 되는 upright-quality gate

Cube quaternion/up-axis는 이 reward 계산에서만 읽는다. Cube angular velocity를
포함한 추가 동적 정보는 policy observation에 넣지 않았다. tilt가 10도를 넘더라도
episode termination을 발생시키지는 않는다.

### 7.4 수직 wrist 고정

- 기본 `wrist_3_joint`: `pi/2`
- reset random scaling 대상에서 wrist 3축 제외
- orientation error weight: `-0.7 -> -1.5`

orientation weight는 위치·Sweep reward보다 압도적으로 커지지 않도록 `-1.5`에
제한했다. 목표는 orientation만 맞추고 접근하지 않는 local optimum을 만드는 것이
아니라, 접촉 중 수직 자세를 유지하는 것이다.

### 7.5 Push reward scale 해석

Sweep-v1의 push weight는 `12.0`이고 raw success 최대는 `2.0`이므로 이론상 최대
reward rate는 `24.0`이다. 따라서 `2026-08-08_14-19-08` 로그에서
`pushing_target`이 약 20까지 올라간 것은 weight가 중복 적용된 오류가 아니라,
정책이 상당 시간 높은 raw push reward를 받은 결과로 해석할 수 있다.

## 8. `CubeSweep-v2`: 저속·안전 Sweep으로 재구성

### 8.1 v2의 목표

v1은 Sweep 자체는 학습했지만, 더 느린 joint motion과 더 느린 Cube 이동, 고정된
선반, 강한 충돌 회피가 필요했다. v2는 v1의 수직 wrist, 낮은 COM, upright gate를
유지하면서 속도와 환경 안정성을 강화했다.

### 8.2 Cube footprint 롤백과 Reach offset 변화

- Cube footprint: `0.11 x 0.11 -> 0.08 x 0.08 m`
- 낮은 COM과 inertia는 유지
- Cube 높이, 질량, nominal contact Z `+0.03 m` 유지

Reach Y offset은 독립 상수가 아니라 `-cube_width * 1.1`이다. 따라서 v1의
`-0.121 m`에서 v2의 `-0.088 m`로 작아진 것은 별도의 Reach tuning이 아니라 Cube
폭을 원래 크기로 롤백한 결과다.

### 8.3 joint 속도 추가 감소

| 항목 | v1 | v2 |
|---|---:|---:|
| shoulder/elbow target delta | `0.05 rad/step` | `0.03 rad/step` |
| wrist target delta | `0.07 rad/step` | `0.04 rad/step` |
| shoulder/elbow target slew | `1.5 rad/s` | `0.9 rad/s` |
| wrist target slew | `2.1 rad/s` | `1.2 rad/s` |
| shoulder/elbow actuator limit | `3.14 rad/s` | `1.5 rad/s` |
| wrist actuator limit | `6.28 rad/s` | `2.0 rad/s` |

v2에서는 per-step target rate limit과 actuator hard limit을 동시에 사용한다.

### 8.4 느린 Push reward의 시행착오

처음에는 v0/v1의 `pushing_target` velocity shaping을 끄고 signed `+Y` 속도에 대한
별도 `slow_push_speed` reward를 두는 방안을 시도했다. 이후 reward 구조가 분산되는
것을 피하기 위해 이 변경을 롤백하고, 기존 `pushing_target` 내부 shaping을 유지한
채 속도 구간만 낮췄다.

```text
v0/v1 desired band: 0.05 < abs(cube_vy) < 0.10 m/s
v2 desired band:    0.03 < abs(cube_vy) < 0.06 m/s
```

현재 v2에는 별도의 `slow_push_speed` reward term이 없다.

### 8.5 episode step과 learning iteration 오해 수정

v2 구축 중 `max step 2000` 요청을 episode horizon으로 해석해 episode 길이를 크게
늘린 시도가 있었다. 이후 의도는 학습 iteration에 관한 것이었음이 확인되었고,
최종적으로 다음과 같이 정리했다.

- episode maximum control step: `360`
- episode duration: `12 s`
- PPO maximum learning iterations: 기존 `10000` 유지

즉 episode를 2000 control step으로 늘리지 않으며, learning iteration도 별도로
줄이지 않는다.

### 8.6 Shelf 전도 termination 도입과 철회

개발 중 Shelf up-axis를 검사하여 선반이 넘어지면 termination하는 조건을 추가했다.
그러나 이 조건을 넣은 뒤 episode가 너무 쉽게 끊기며 학습이 사실상 진행되지 않는
문제가 발생했다.

최종 해결은 termination threshold를 완화하는 대신 문제의 자유도 자체를 제거하는
것이었다.

- Shelf를 collision 가능한 kinematic rigid body로 고정
- Shelf gravity 비활성화
- Shelf tipping termination 제거
- 현재 모든 환경의 termination은 timeout만 유지

Cube와 robot은 여전히 Shelf와 접촉하지만 Shelf는 힘을 받아 이동하거나 넘어지지
않는다.

### 8.7 Shelf collision 강화

Shelf를 kinematic으로 고정하면 “Shelf를 밀어 움직이는 비용”이 물리적으로 사라진다.
로봇이 Shelf를 강하게 치는 행동을 계속 억제하기 위해 collision penalty를 강화했다.

- weight: `-5.0 -> -10.0`
- 중앙 Cube 지지면 하나에서 Shelf의 세 수평면으로 검사 확대
- 검사 높이: `0.70`, `1.05`, `1.50 m`
- 한 step에 여러 면을 접촉해도 OR로 합쳐 penalty는 한 번만 적용

### 8.8 `cube_vy` metric과 episode speed latch 도입

저속 Push가 실제로 지켜지는지 별도로 확인할 수 있도록 signed world-frame
`cube_vy`를 command metric으로 추가했다.

```text
TensorBoard key: Metrics/ee_pose/cube_vy
```

이 metric은 policy/critic observation이 아니다. 또한 v2는 다음 상태형 reward latch를
사용한다.

```text
if abs(cube_vy) > 0.06 m/s at any step:
    speed_limit_exceeded = True

if speed_limit_exceeded:
    pushing_target = 0 for the rest of the episode
```

초과한 바로 그 step부터 reward가 0이 되며, 환경별 reset에서 latch가 해제된다.
따라서 한 번 과속한 뒤 goal에 도달하여 success reward를 회수하는 전략을 차단한다.

### 8.9 episode speed latch 제거 실험

이후 학습에서 로봇이 Cube와 접촉하지 않는 국소 최적점으로 수렴하는 현상이
관찰되었다. 순간적인 과속 한 번이 episode의 남은 `pushing_target` 보상을 모두
차단하여, 정책이 접촉 자체를 회피하도록 유도했는지 검증하기 위해 v2의 상태형
speed latch를 제거했다.

- `Metrics/ee_pose/cube_vy` 기록은 유지
- `0.03 < abs(cube_vy) < 0.06 m/s`의 `+0.5` 속도 항 유지
- `abs(cube_vy) >= 0.06 m/s`인 현재 step의 `-0.5` 속도 항 유지
- 과속 이력을 다음 step으로 전달하지 않음
- 속도가 다시 낮아지면 이후 step에서 `pushing_target` 보상을 다시 받을 수 있음

### 8.10 v2 self-collision 제약

정책이 arm link 또는 gripper를 로봇 본체와 충돌시킨 상태로 회전하는 전략을 막기
위해 v2용 PhysX self-collision과 링크별 contact sensing을 구현했다.

- 직접 연결된 arm link와 gripper 내부 linkage pair는 검사에서 제외
- 나머지 비인접 arm-arm 및 gripper-arm pair의 최대 normal force 계산
- `2 N`부터 `20 N`까지 force 비례 연속 패널티
- `20 N` 초과 시 즉시 failure termination
- `5 N` 초과가 2 control step 지속되어도 failure termination
- `time_out=False`
- 조기 충돌 실제 terminal cost 약 `-40`, episode 말기 약 `-10`
- `self_collision_force_max`와 self-collision termination rate 기록
- 기존 `0.75-1.25` shoulder/elbow reset이 실제 `upper_arm-wrist_2` 충돌을 생성하여
  v2 reset 범위를 `0.95-1.05`로 축소

이후 std 오류 당시 상태를 재현하는 비교 실험을 위해 등록된 v2에서 이 기능의
호출을 모두 해제했다. 구현 파일과 opt-in scene/reward/termination config는
보존했지만, 현재 v2는 self-contact sensor를 생성하지 않고 관련 패널티·termination·
metric도 활성화하지 않는다. reset 범위도 당시 값인 `0.75-1.25`로 돌아갔다.

## 9. v2 PPO 발산과 안정화

### 9.1 중단 현상

`logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v2/2026-08-08_20-55-17`은 약
1850 iteration까지 의도한 방향으로 학습되었지만 그 뒤 PPO가 수치적으로 발산했다.

확인된 흐름은 다음과 같았다.

- checkpoint 1850 부근까지 push reward 약 `20.21 / 24`
- EE position error 약 `2.29 cm`
- shelf collision은 낮은 수준
- iteration 1855부터 value loss가 급격히 증가
- 이후 `inf`, `nan`으로 전파
- direct scalar Gaussian std가 음수가 되어
  `normal expects all elements of std >= 0.0` 오류로 중단

즉 환경이 전혀 학습되지 않은 것이 아니라, 유효한 정책을 학습한 뒤 optimizer/value
scale이 무너지면서 actor distribution까지 오염된 문제였다.

### 9.2 v2 전용 안정화 설정

v0/v1 PPO는 그대로 두고 v2 runner만 다음과 같이 바꿨다.

| PPO 항목 | 기존 | v2 안정화 |
|---|---:|---:|
| Gaussian initial std | `1.0` | `0.5` |
| std parameterization | direct scalar | `log` |
| learning rate | `1e-3` | `3e-4` |
| schedule | adaptive | fixed |
| entropy coefficient | `0.01` | `0.001` |
| learning epochs/update | `8` | `5` |
| runner action clip | 없음 | `1.0` |

`max_grad_norm=1.0`, mini-batch 수 `4`, rollout `24`, hidden layer `[64, 64]`,
maximum learning iterations `10000`은 유지한다. log-std는 내부 parameter가 어떤
실수 값이더라도 실제 표준편차를 지수 변환해 양수로 만들므로, direct std가 음수가
되는 실패를 구조적으로 방지한다.

distribution parameterization이 달라졌기 때문에 안정화 전 v2 checkpoint를 그대로
resume하기보다 새 run으로 학습하는 것이 안전하다.

### 9.3 self-collision 적용 run과 재현용 롤백

`logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v2/2026-08-09_01-26-09`에서는
self-collision 제약과 안정화 PPO를 함께 사용했다. 857 iteration 시점 기준으로
position error는 약 `0.085 m`, `pushing_target`은 약 `2.25`, mean std는 약
`0.084`였으며, self-collision을 피하면서 다른 local optimum으로 수렴했다.

두 변경이 동시에 들어간 상태에서는 원인 분리가 어려우므로, 비교 기준을
`2026-08-08_20-55-17` 당시의 환경과 PPO로 복원했다.

- v2 PPO 전용 override 제거: v1 runner의 scalar std, `1e-3` adaptive learning rate,
  entropy `0.01`, epoch `8`, action clip 없음을 다시 상속
- `cube_vy` 및 self-collision command metric 호출 해제
- self-collision scene sensor, reward, termination 연결 해제
- shoulder/elbow reset 범위 `0.75-1.25` 복원
- 별도 dense Reach 보상은 추가하지 않음
- self-collision 구현 코드와 opt-in config는 삭제하지 않고 보존

이 롤백은 direct scalar std가 다시 음수가 될 수 있는 위험까지 포함한 의도적인 재현
설정이다.

### 9.4 v3 fixed learning-rate 분리 실험

두 v2 run에서 공통적으로 adaptive schedule이 공용 actor/critic learning rate를
`1e-2`까지 올린 뒤 value loss가 폭발하는 현상을 확인했다. 환경 요인을 고정한 채
optimizer schedule만 검증하기 위해 Sweep v3를 추가했다.

- 환경 MDP와 episode 설정은 v2와 동일
- PPO learning rate: `1e-3 -> 3e-4`
- PPO schedule: `adaptive -> fixed`
- scalar std, entropy `0.01`, epoch `8`, mini-batch `4`, action clip 없음은 유지

따라서 v3는 adaptive LR 상승 경로가 발산의 주요 원인이었는지 분리해서 확인하는
환경으로 시작했다. 이후 Cube 하단 COM 회전이 `root_lin_vel_w`의 COM 속도로
계산되어 제자리 회전에도 Push 속도 보상이 발생하는 local optimum을 확인했다.
이에 v3의 `pushing_target`만 Cube root-link actor 원점의 XY 평면 속력을 사용하도록
개선했다. direct scalar std와 observation normalization 미사용은 그대로이므로 모든
NaN 가능성을 구조적으로 제거한 설정은 아니다.

## 10. 현재 환경 비교

### 10.1 Scene, observation, action

| 환경 | 물리 Cube | policy obs | action | Shelf |
|---|---|---:|---|---|
| Shelf Reach v0 | 없음 | 25-D | relative, scale 0.5 | dynamic rigid body |
| PreReach v0 | 0.08 m 폭 | 29-D | relative, scale 0.5 | dynamic + 중앙면 sensor |
| PreReach v1 | 0.08 m 폭 | 29-D | relative, scale 0.5 | PreReach v0와 동일 |
| Sweep v0 | 0.08 m 폭 | 29-D | relative, scale 0.5 | dynamic, collision -5 |
| Sweep v1 | 0.11 m 폭, low COM | 29-D | full-range + 0.05/0.07 rate limit | dynamic, collision -5 |
| Sweep v2 | 0.08 m 폭, low COM | 29-D | full-range + 0.03/0.04 rate limit | kinematic, 3면 collision -10 |
| Sweep v3 | Sweep v2와 동일 | 29-D | Sweep v2와 동일 | Sweep v2와 동일 |

### 10.2 주요 reward 차이

| 환경 | Position | Orientation | Motion cost | Push |
|---|---|---:|---:|---|
| Shelf Reach v0 | coarse `-0.2*d` + fine tanh | `-0.1` | 초기 `-0.0001/-0.0001` + curriculum | 없음 |
| PreReach v0 | Shelf Reach와 동일 | `-0.1` | Shelf Reach와 동일 | 없음 |
| PreReach v1 | `+3*exp(-10*d)` | `-0.7` | `-0.03/-0.03`, 부모 curriculum 존재 | 없음 |
| Sweep v0 | PreReach v0와 동일 | `-0.1` | 부모 curriculum 존재 | weight `6`, 0.05-0.10 m/s |
| Sweep v1 | `+3*exp(-10*d)` | `-1.5` | `-0.03/-0.03`, curriculum 제거 | weight `12`, upright gate |
| Sweep v2 | Sweep v1과 동일 | `-1.5` | Sweep v1과 동일 | weight `12`, 0.03-0.06 m/s, speed latch 없음 |
| Sweep v3 | Sweep v2와 동일 | `-1.5` | Sweep v2와 동일 | v2 식 + root-link XY 속력 기준 |

모든 현재 환경의 episode horizon은 360 control steps이고 termination은 timeout만
사용한다. Cube tilt, Cube goal 도달, Shelf/robot 충돌은 조기 termination이 아니다.

## 11. 개발 과정에서 얻은 설계 원칙

1. **Reach와 Sweep을 분리해 진단한다.**  
   TCP 추적 실패와 접촉 정책 실패를 같은 reward 곡선만 보고 구분하기 어렵다.

2. **명시적 phase state가 없어도 gate는 존재할 수 있다.**  
   이 환경의 전환은 상태 머신이 아니라 `zeta_m` reward gate로 구현되어 있다.

3. **policy observation과 privileged 계산을 구분한다.**  
   Cube pose·속도·tilt를 reward에 쓴다고 해서 반드시 정책 입력에 추가할 필요는 없다.

4. **가동 범위와 속도 제한은 별개다.**  
   full joint range를 유지하면서 per-step target delta와 actuator velocity limit으로
   속도만 낮출 수 있다.

5. **termination은 학습 신호를 쉽게 제거한다.**  
   Shelf 전도처럼 환경 모델링으로 제거할 수 있는 실패는 잦은 termination보다
   kinematic 고정과 contact penalty가 더 안정적이었다.

6. **물성 변경과 접촉 목표를 함께 검토한다.**  
   Cube를 무조건 크게 하거나 contact point를 내리면 다른 collision 문제가 생긴다.
   v1/v2에서는 contact Z를 유지하고 COM/inertia/footprint를 조정했다.

7. **reward 최대치와 로그 단위를 구분한다.**  
   raw reward, weight 적용 reward rate, episode 집계 metric을 혼동하면 정상적인 값도
   중복 보상처럼 보일 수 있다.

8. **환경 학습 성공과 PPO 수치 안정성은 별개다.**  
   v2는 좋은 정책 지표를 보인 뒤에도 value/std 발산으로 중단되었다. 장기 학습에는
   reward 설계뿐 아니라 distribution parameterization과 optimizer 설정도 중요하다.

## 12. 관련 소스와 로그

| 내용 | 경로 |
|---|---|
| 기본 Shelf Reach | `src/sweep_rl/sweep_rl/shelf_reach` |
| Cube Pre-Reach | `src/sweep_rl/sweep_rl/shelf_cube_pre_reach` |
| Cube Sweep | `src/sweep_rl/sweep_rl/shelf_cube_sweep` |
| 원본 참고 환경 | `example/Sweep-Policy` |
| 최초 Reach 후 Sweep 정체 분석 | `logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v1/2026-08-07_23-18-41` |
| 고속·전도·45도 wrist 분석 | `logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v1/2026-08-08_02-18-53` |
| v1 push reward scale 확인 | `logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v1/2026-08-08_14-19-08` |
| v2 PPO 발산 분석 | `logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v2/2026-08-08_20-55-17` |
| v2 self-collision 적용 후 local optimum 분석 | `logs/rsl_rl/sweep_shelf_ur5e_gripper_cube_v2/2026-08-09_01-26-09` |
