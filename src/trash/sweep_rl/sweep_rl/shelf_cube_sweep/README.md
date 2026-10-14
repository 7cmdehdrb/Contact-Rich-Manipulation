# UR5e Gripper Shelf Cube Sweep

UR5e + Robotiq gripper가 선반 위 Cube 뒤쪽으로 접근한 다음, Cube를 world
`+Y` 방향의 목표 지점까지 미는 Isaac Lab 환경이다. 모든 환경은
`shelf_cube_pre_reach`의 Cube-relative Reach, 선반 충돌 패널티와 자세 추적을
상속한다.

## 환경 ID

| 버전 | 학습 환경 | 재생 환경 | 로그 디렉터리 |
|---|---|---|---|
| v0 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0` | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0` | `sweep_shelf_ur5e_gripper_cube` |
| v1 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v1` | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v1` | `sweep_shelf_ur5e_gripper_cube_v1` |
| v2 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v2` | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v2` | `sweep_shelf_ur5e_gripper_cube_v2` |
| v3 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v3` | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v3` | `sweep_shelf_ur5e_gripper_cube_v3` |
| v4 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v4` | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v4` | `sweep_shelf_ur5e_gripper_cube_v4` |

## 공통 목표와 관측

Reach point는 매 control step 현재 Cube 중심을 기준으로 계산된다.

```text
reach_offset = (0.0, -cube_width * 1.1, +0.03) m
sweep_goal   = episode 초기 Cube 위치 + (0.0, +0.18, 0.0) m
zeta_gate    = 0.08 m
```

v0-v3 정책 관측은 29차원이다. Cube의 동적 상태 중 정책에 들어가는 정보는 robot-base
frame에서의 Position뿐이다. `cube_width`는 고정된 환경 상수로 제공된다. Cube의
quaternion/up-axis와 angular velocity는 정책 관측에 포함하지 않으며, reward 또는
termination 같은 simulator-side privileged 계산에서만 사용할 수 있다.

## v0

기본 Cube 크기는 `0.08 x 0.08 x 0.20 m`, 질량은 `1.5 kg`이다. Sweep reward
weight는 `6.0`이며 기존 속도 shaping을 사용한다.

```text
if 0.05 < abs(cube_vy) < 0.10:
    velocity_reward = +0.5
elif abs(cube_vy) >= 0.10:
    velocity_reward = -0.5
else:
    velocity_reward = 0.0
```

## v1

v1은 다음 안정화 설정을 추가한다.

- Cube footprint: `0.11 x 0.11 m`
- Cube COM: geometric center보다 `0.05 m` 아래
- Cube diagonal inertia: `(0.008, 0.008, 0.003) kg m^2`
- full-range absolute joint action
- per-control-step joint target rate limit
  - shoulder/elbow: `0.05 rad/step` (`1.5 rad/s` at 30 Hz)
  - wrist: `0.07 rad/step` (`2.1 rad/s` at 30 Hz)
- 기본 `wrist_3_joint = pi/2`, reset 시 wrist scaling 제외
- orientation error weight: `-1.5`
- `action_rate`/`joint_vel` reward-weight curriculum 제거
- Sweep reward weight: `12.0`
- Cube tilt가 `10 deg`에 도달하면 Sweep reward가 0이 되는 upright-quality gate

joint action `[-1, 1]`은 각 joint의 전체 soft position limit에 매핑된다. rate
limit은 한 step의 target 변화량만 제한하므로 joint의 전체 가동 범위는 줄지 않는다.

## v2

v2는 v1의 수직 wrist, 낮은 COM, upright gate와 선반 충돌 패널티를 유지하면서
더 느린 Sweep을 학습하도록 구성한다.

### Episode와 물체

| 설정 | 값 |
|---|---:|
| 최대 episode step | `360` |
| control frequency | `30 Hz` |
| 최대 episode 시간 | `12.0 s` |
| Cube 크기 | `0.08 x 0.08 x 0.20 m` |
| Cube 질량 | `1.5 kg` |
| Cube COM offset | `(0.0, 0.0, -0.05) m` |
| Cube diagonal inertia | `(0.008, 0.008, 0.003) kg m^2` |
| Shelf rigid body | kinematic, gravity disabled |

Cube footprint만 원래 크기로 복구하며 높이, 질량, nominal contact point
`+0.03 m`와 낮은 COM은 유지한다.

v2의 Shelf는 충돌 가능한 kinematic body로 고정한다. 따라서 로봇 및 Cube와의
접촉은 유지되지만 힘을 받아 이동하거나 넘어지지 않는다.

### Joint 속도

| Joint group | Target delta limit | 30 Hz target slew | Actuator hard limit |
|---|---:|---:|---:|
| shoulder/elbow | `0.03 rad/step` | `0.9 rad/s` | `1.5 rad/s` |
| wrist | `0.04 rad/step` | `1.2 rad/s` | `2.0 rad/s` |

### 느린 Push 속도 shaping

v2도 v0/v1과 동일하게 `pushing_target` 내부의 `abs(cube_vy)` velocity shaping을
사용한다. 다만 더 느린 Push를 유도하도록 보상/패널티 경계를
`0.05-0.10 m/s`에서 `0.03-0.06 m/s`로 낮춘다. 별도 `slow_push_speed` reward는
사용하지 않는다.

```text
if 0.03 < abs(cube_vy) < 0.06:
    velocity_reward = +0.5
elif abs(cube_vy) >= 0.06:
    velocity_reward = -0.5
else:
    velocity_reward = 0.0
```

이 속도 항은 success 영역 밖에서 EE/wrist Sweep gate가 활성화될 때
`pushing_target`의 거리 shaping에 더해진다. 기존 방식과 마찬가지로 signed 방향이
아닌 속력 `abs(cube_vy)`를 사용하므로 `-Y` 이동도 동일하게 평가한다.

v3는 같은 `0.03-0.06 m/s` 구간을 유지하되 속도 기준을 Cube COM의 `abs(v_y)`에서
Cube root-link actor 원점의 평면 속력 `norm(v_xy)`로 변경한다. 따라서 하단 COM
오프셋과 각속도의 `omega x r` 성분은 v3의 Push 속도 보상에 포함되지 않는다.

v2는 더 이상 episode speed latch를 사용하지 않는다. 따라서 한 step에서
`abs(cube_vy) >= 0.06 m/s`가 되어 `-0.5` 속도 항을 받더라도 다음 step의 속도가
낮아지면 정상적으로 `pushing_target` reward를 다시 받을 수 있다.

## 전체 MDP 명세

아래 내용은 상속된 설정을 포함한 실제 최종 환경 구성을 기준으로 한다. v0/v1/v2
설명이 앞 절과 중복되더라도 관측, 액션, 보상과 termination을 한곳에서 모두 확인할
수 있도록 정리했다.

### 전체 정책 관측

v0-v3의 `policy` observation group은 아래 6개 term을 순서대로 concatenate한
29차원 벡터다.

| Observation term | 차원 | 값 | 학습 시 noise |
|---|---:|---|---:|
| `joint_pos` | 6 | 6개 arm joint의 default pose 기준 상대 위치 | uniform `[-0.01, 0.01]` |
| `joint_vel` | 6 | 6개 arm joint의 default velocity 기준 상대 속도 | uniform `[-0.01, 0.01]` |
| `pose_command` | 7 | robot-base frame의 moving Reach position 3개와 목표 quaternion 4개 | 없음 |
| `target_object_position` | 3 | robot-base frame에서의 Cube 중심 Position | 없음 |
| `cube_width` | 1 | 고정 Cube width: v0/v2 `0.08`, v1 `0.11` | 없음 |
| `actions` | 6 | 직전 policy raw arm action | 없음 |

```text
6 + 6 + 7 + 3 + 1 + 6 = 29
```

v4는 여기에 현재 EE position 3개와 quaternion 4개를 추가하고, `actions`가 binary
gripper를 포함한 7차원이 되므로 총 37차원이다.

- 학습 환경에서는 observation corruption이 활성화되어 `joint_pos`와 `joint_vel`에
  위 noise가 적용된다.
- `Play` 환경에서는 `enable_corruption=False`이므로 모든 observation noise가
  비활성화된다.
- `actions`는 actuator에 최종 적용된 rate-limited joint target이 아니라 policy가
  직전에 출력한 raw action이다.
- Cube quaternion, up-axis, linear/angular velocity는 policy observation에 없다.
- `cube_vy`와 self-collision metric을 계산하는 구현은 남아 있지만, 현재 등록된
  v2 환경에서는 기록 기능을 호출하지 않는다. 어느 값도 policy/critic observation에
  추가되지 않는다.
- RSL-RL runner는 `actor: [policy]`, `critic: [policy]`를 사용한다. 따라서 critic만
  받는 별도 privileged observation group도 현재는 없다.
- reward는 policy observation과 별개로 Cube root Position, quaternion, EE/wrist frame과
  shelf contact sensor를 simulator state에서 직접 읽는다. Push 속도는 v0-v2에서 COM
  `+Y` 선속도, v3에서 root-link actor 원점의 XY 평면 속력을 사용한다.
- Cube angular velocity는 policy observation에 없으며, 현재 활성 reward와
  termination에서도 사용하지 않는다.

### 전체 액션

v0-v3의 action dimension은 6이며 다음 arm joint 순서를 사용한다.

```text
shoulder_pan_joint
shoulder_lift_joint
elbow_joint
wrist_1_joint
wrist_2_joint
wrist_3_joint
```

Gripper joint는 policy action에 포함되지 않는다. reset 시 finger/knuckle joint의
default position `0.0`을 사용하므로 학습 가능한 open/close action은 없다.

v4는 아래 6개 arm action에 원본 Sweep-Policy의 binary gripper action 1개를 더한
7차원 action을 사용한다.

#### v0 액션

v0는 `JointPositionActionCfg`를 사용한다.

```text
joint_target = default_joint_position + 0.5 * raw_action
```

- `use_default_offset=True`
- 별도의 per-step target rate-limit 없음
- task와 RSL-RL runner에 명시적인 raw action clip 없음
- actuator velocity hard limit:
  - shoulder/elbow: `3.14 rad/s`
  - wrist: `6.28 rad/s`
- 기본 `wrist_3_joint`: `0.785 rad`
- reset 시 6개 arm joint 모두 default pose의 `0.75-1.25` 배로 scaling

#### v1/v2/v3 full-range absolute 액션

v1, v2와 v3는 raw action을 각 joint의 전체 finite soft position limit에 매핑한 뒤,
이전 applied target을 기준으로 한 control-step target rate-limit을 적용한다.

```text
requested_target = unscale(clamp(raw_action, -1, 1), soft_lower, soft_upper)
target_delta = clamp(requested_target - previous_target, -max_delta, +max_delta)
applied_target = previous_target + target_delta
```

| 버전 | shoulder/elbow max delta | wrist max delta | shoulder/elbow actuator limit | wrist actuator limit |
|---|---:|---:|---:|---:|
| v1 | `0.05 rad/step` | `0.07 rad/step` | `3.14 rad/s` | `6.28 rad/s` |
| v2 | `0.03 rad/step` | `0.04 rad/step` | `1.5 rad/s` | `2.0 rad/s` |

- control frequency는 `30 Hz`이므로 v1 target slew는 각각 `1.5`, `2.1 rad/s`이고
  v2 target slew는 각각 `0.9`, `1.2 rad/s`다.
- rate-limit은 한 step의 변화량만 제한하며 최종 reachable joint range를 제한하지
  않는다.
- reset 시 rate limiter의 이전 target은 reset 후 현재 joint position으로 초기화된다.
- v1/v2 기본 `wrist_3_joint`는 `pi/2`다.
- v1/v2 reset scaling은 shoulder 2개와 elbow에만 `0.75-1.25`로 적용되며, wrist
  3개는 vertical default pose를 그대로 사용한다.
- v1/v2 모두 별도의 runner-level raw action clip을 사용하지 않는다.

### 전체 보상

다음 기호를 사용한다.

```text
d_pos      = TCP와 moving Reach command 사이의 L2 position distance
d_rot      = TCP와 command quaternion 사이의 shortest-path rotation error [rad]
delta_a    = current raw action - previous raw action
q_dot      = 6개 arm joint velocity
I_shelf    = 선반 바닥 충돌 여부 {0, 1}
d_goal     = Cube 중심과 고정 Sweep goal 사이의 L2 distance
v_y        = Cube world-frame +Y linear velocity
zeta       = EE/wrist Sweep gate {0, 1}
q_upright  = Cube upright quality [0, 1]
```

#### 버전별 활성 reward term

표의 식에는 최종 weight가 이미 반영되어 있다.

| Reward term | v0 | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| `end_effector_position_tracking` | `-0.2 * d_pos` | `+3.0 * exp(-10*d_pos)` | `+3.0 * exp(-10*d_pos)` | v2와 동일 | v3와 동일 |
| `end_effector_position_tracking_fine_grained` | `+0.1 * (1-tanh(d_pos/0.1))` | 비활성 | 비활성 | 비활성 | 비활성 |
| `end_effector_orientation_tracking` | `-0.1 * d_rot` | `-1.5 * d_rot` | `-1.5 * d_rot` | v2와 동일 | v3와 동일 |
| `action_rate` | `-0.0001 * sum(delta_a^2)` | `-0.03 * sum(delta_a^2)` | `-0.03 * sum(delta_a^2)` | v2와 동일 | v3와 동일 |
| `joint_vel` | `-0.0001 * sum(q_dot^2)` | `-0.03 * sum(q_dot^2)` | `-0.03 * sum(q_dot^2)` | v2와 동일 | v3와 동일 |
| `shelf_collision` | `-5.0 * I_shelf` | `-5.0 * I_shelf` | `-10.0 * I_shelf` | v2와 동일 | v3와 동일 |
| `pushing_target` | `+6.0 * R_push_raw` | `+12.0 * q_upright * R_push_raw` | `+12.0 * q_upright * R_push_raw_v2` | v2 식 + root-link XY 속력 | `+12.0 * R_push_v4`, COM Y 속도와 upright gate 없음 |
| `self_collision_force` | 비활성 | 비활성 | 비활성 | 비활성 | 비활성 |
| `self_collision_terminal` | 비활성 | 비활성 | 비활성 | 비활성 | 비활성 |

v0에만 reward-weight curriculum이 남아 있다. `common_step_counter > 4500`이 되면
step-wise하게 다음 weight로 교체된다.

```text
action_rate: -0.0001 -> -0.005
joint_vel:   -0.0001 -> -0.001
```

v1/v2/v3/v4는 `curriculum=None`이며 `action_rate=-0.03`, `joint_vel=-0.03`을 학습
처음부터 끝까지 유지한다.

#### Shelf collision 패널티

`shelf_collision`은 Cube와 선반의 정상 접촉이 아니라 robot/gripper와 선반의
수평 바닥판 접촉만 검사한다. v0/v1은 Cube가 놓인 중앙면만 검사하고, v2는 Shelf의
세 수평면을 모두 검사한다. 다음 조건을 만족하는 filtered contact가 하나라도 있으면
`I_shelf=1`이다.

```text
contact force > 1.0 N
shelf-local x in [-0.20, +0.20] m
shelf-local y in [-0.50, +0.50] m
v0/v1: abs(contact_z - 1.05) <= 0.02 m
v2: any(abs(contact_z - z_surface) <= 0.02 m
        for z_surface in [0.70, 1.05, 1.50])
```

패널티는 v0/v1에서 해당 control step마다 `-5.0`, v2에서 `-10.0`이다. v2에서는
Shelf가 더 이상 힘에 밀리지 않으므로 상부 Block을 치는 행동의 비용을 두 배로
높인다. 같은 control step에 여러 수평면과 접촉하더라도 OR 조건으로 합쳐 raw
penalty는 `1`이며 `-10.0`을 한 번만 적용한다.

#### v2 Self-collision 구현 상태

아래 self-collision 센서·패널티·termination 구현은 소스에 보존되어 있지만, 현재
등록된 v2의 scene/reward/termination 설정에는 연결하지 않는다. 따라서 학습 중
PhysX self-collision, 링크별 contact sensor, 관련 reward와 termination은 모두
비활성이다.

다시 연결할 경우에는 직접 연결된 arm link 쌍, `wrist_3_link-robotiq_base_link`와
그리퍼 내부 linkage 쌍을 제외하고 비인접 pair의 최대 normal contact force
`F_self`를 사용하도록 구현되어 있다.

- `F_self <= 2 N`: 패널티 없음
- `2 N < F_self < 20 N`: 접촉력에 비례하는 연속 패널티
- `F_self > 20 N`: 즉시 termination
- `F_self > 5 N`이 2 control step 연속: termination

terminal penalty는 RewardManager의 `step_dt` scaling을 역보정한다. 따라서 설정값이
단순 weight가 아니라 실제 episode return에 반영되며, episode 초반 충돌은 약 `-40`,
마지막 충돌은 약 `-10`이다.

#### `pushing_target` 공통 gate와 reward

현재 Cube를 기준으로 moving push point를 계산한다.

```text
push_point = cube_position + (0, -cube_width*1.1, +0.03)
contact_distance = norm(push_point - ee_position)
wrist_y_error = abs(push_point_y - wrist_y)
zeta = (contact_distance < 0.08) and (wrist_y_error < 0.08)
```

v0-v2의 velocity shaping은 signed velocity가 아니라 `abs(v_y)`를 사용한다.
v0/v1의 경계는 다음과 같다.

```text
if 0.05 < abs(v_y) < 0.10:
    R_velocity = +0.5
elif abs(v_y) >= 0.10:
    R_velocity = -0.5
else:
    R_velocity = 0.0
```

v0/v1 raw push reward:

```text
if d_goal < 0.03:
    R_push_raw = 2.0 * exp(-5.0*d_goal)
else:
    R_push_raw = zeta * ((1.0 - d_goal/0.18) + R_velocity)
```

v2도 동일한 식을 사용하되 속도 경계만 낮춘다.

```text
if 0.03 < abs(v_y) < 0.06:
    R_velocity_v2 = +0.5
elif abs(v_y) >= 0.06:
    R_velocity_v2 = -0.5
else:
    R_velocity_v2 = 0.0

if d_goal < 0.03:
    R_push_raw_v2 = 2.0 * exp(-5.0*d_goal)
else:
    R_push_raw_v2 = zeta * ((1.0 - d_goal/0.18) + R_velocity_v2)
```

v3는 v2의 식과 속도 경계를 그대로 사용하되 다음 actor 원점 평면 속력을 식의
`speed`로 사용한다.

```text
speed_v3 = norm(cube.root_link_lin_vel_w[0:2])
```

이는 COM 속도가 아니므로 COM 오프셋에 의한 `omega x r` 회전 성분을 제거한다.

v0는 tilt gate가 없다. v1/v2/v3는 다음 upright quality를 전체 push reward에 곱한다.

```text
up_z = world frame에서 Cube local +Z axis의 Z component
min_up_z = cos(10 deg)
q_upright = clamp((up_z - min_up_z)/(1-min_up_z), 0, 1)
if tilt >= 10 deg:
    q_upright = 0
```

따라서 v1/v2/v3는 success 영역 안에 있더라도 Cube가 10도 이상 기울어져 있으면
`pushing_target` reward가 0이다.

### 전체 termination

현재 v0/v1/v2/v3/v4의 활성 termination은 `time_out` 하나뿐이다.

| 버전 | Control frequency | 최대 step | 최대 시간 | 처리 |
|---|---:|---:|---:|---|
| v0 | `30 Hz` | `360` | `12.0 s` | time-limit truncation |
| v1 | `30 Hz` | `360` | `12.0 s` | time-limit truncation |
| v2 | `30 Hz` | `360` | `12.0 s` | time-limit truncation |
| v3 | `30 Hz` | `360` | `12.0 s` | time-limit truncation |

현재 다음 조건은 termination이 아니다.

- Cube가 Sweep goal의 `0.03 m` success 영역에 들어감
- Cube tilt가 `10 deg` 이상이 됨
- Cube가 넘어짐
- robot/gripper가 선반 바닥과 충돌함
- Cube 또는 robot 속도가 커짐

위 목록은 여전히 termination이 아니다. success는 reward만 바꾸며 episode를
조기 종료하지 않는다. Cube tilt는 v1/v2의 Sweep 관련 reward를 0으로 만들지만
episode를 종료하지 않고, 선반 상부 Block 충돌도 버전별 패널티만 부여한다.

### v2 PPO 설정

현재 v2는 std 음수 문제가 발생했던 `2026-08-08_20-55-17` run의 설정을 재현하기
위해 v1 runner 설정을 그대로 상속한다.

| 설정 | v2 값 |
|---|---:|
| actor Gaussian initial std | `1.0` |
| std parameterization | `scalar` |
| learning rate | `1.0e-3` |
| learning-rate schedule | `adaptive` |
| entropy coefficient | `0.01` |
| learning epochs/update | `8` |
| mini-batches/update | `4` |
| max gradient norm | `1.0` |
| runner action clip | 없음 |

actor/critic hidden layer는 기존과 같은 `[64, 64]`, activation은 `elu`이고,
`max_iterations=10000`, rollout은 환경당 `24 step`으로 유지한다. 이 설정은 direct
scalar std가 음수가 되어 `normal expects all elements of std >= 0.0`로 중단될 수
있다는 점까지 의도적으로 이전 상태로 복원한 것이다.

### v3 PPO 설정

v3의 scene, action, observation, termination과 episode 길이는 v2와 동일하다.
`pushing_target`의 속도 측정 기준과 PPO의 다음 두 항목을 변경한다.

- v2: Cube COM의 `abs(v_y)`
- v3: Cube root-link actor 원점의 평면 속력 `norm(v_xy)`

v3는 하단으로 5 cm 이동된 COM의 회전 운동이 Push 선속도로 계산되지 않게 한다.
속도 보상/패널티 구간 `0.03-0.06 m/s`, 거리 shaping, Sweep gate 및 upright
quality는 v2와 동일하다.

| 설정 | v2 | v3 |
|---|---:|---:|
| learning rate | `1.0e-3` | `3.0e-4` |
| schedule | `adaptive` | `fixed` |

초기 scalar std `1.0`, entropy `0.01`, epoch `8`, mini-batch `4`, action clip 없음 등
나머지 설정은 모두 v2를 그대로 상속한다. fixed schedule은 actor KL이 작을 때 공용
optimizer learning rate가 `1e-2`까지 증가하는 경로를 제거한다. 따라서 관찰된
value/std 발산 가능성을 낮추지만 scalar std의 양수 제약이나 NaN 방지는 추가하지
않으므로 수치 오류를 구조적으로 완전히 차단하지는 않는다.

### v4 설정

v4는 v3의 scene, episode 길이, actuator 속도 제한과 PPO learning-rate 정책을
상속하고, Sweep-Policy 원본과의 차이 중 행동 유도에 직접 관여하는 항목을 다음과
같이 변경한다.

- 정책 관측에 robot-base frame의 현재 EE pose `(position, quaternion)` 7차원을
  추가한다.
- 속도 shaping은 Cube COM의 `abs(v_y)`만 사용한다. 따라서 X 방향 속도는 보상에
  들어가지 않는다.
- 속도 보상 구간은 `0.05 < abs(v_y) < 0.10 m/s`, high-speed penalty는
  `abs(v_y) >= 0.10 m/s`에서 시작한다.
- Cube-relative push point의 Y offset을 `-cube_width`, Sweep gate를 `0.04 m`로
  변경한다.
- `cube_upright_quality` 구현은 보존하지만 v4의 `pushing_target`에서는 호출하지
  않는다.
- action은 원본처럼 arm 6축 default-relative joint-position action
  (`scale=0.5`)과 binary gripper 1축으로 구성한다.
- actor hidden layer만 `[256, 128, 64]`로 변경한다. critic은 상속된 `[64, 64]`를
  유지한다.

현재 EE pose 7차원과 7차원 last action을 포함하므로 v4의 policy observation은
총 37차원이고 action은 7차원이다. v3 checkpoint는 관측/action 차원이 달라 v4에서
직접 resume할 수 없다.

## Windows 11 실행

저장소 루트에서 Isaac Lab Python 환경을 활성화한 다음 PowerShell에서 실행한다.

v3 학습:

```powershell
.\IsaacLab\isaaclab.bat -p .\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v3 --num_envs 2048 --headless
```

학습 결과는 기본적으로 다음 위치에 저장된다.

```text
logs\rsl_rl\sweep_shelf_ur5e_gripper_cube_v3\<run>\
```

v3 재생:

```powershell
.\IsaacLab\isaaclab.bat -p .\IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v3 --checkpoint logs\rsl_rl\sweep_shelf_ur5e_gripper_cube_v3\<run>\model_<step>.pt --num_envs 1
```

v3의 새 PPO 설정을 비교하려면 기존 v2 checkpoint를 resume하지 말고 처음부터
학습하는 것을 권장한다.

v4 학습:

```powershell
.\IsaacLab\isaaclab.bat -p .\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v4 --num_envs 2048 --headless
```

학습 결과는 기본적으로 다음 위치에 저장된다.

```text
logs\rsl_rl\sweep_shelf_ur5e_gripper_cube_v4\<run>\
```

v4 재생:

```powershell
.\IsaacLab\isaaclab.bat -p .\IsaacLab\scripts\reinforcement_learning\rsl_rl\play.py --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v4 --checkpoint logs\rsl_rl\sweep_shelf_ur5e_gripper_cube_v4\<run>\model_<step>.pt --num_envs 1
```

## v5

v5는 기존 상속 구조에서 분리되어 인접한 `shelf_cube_sweep_v5` 패키지로 이동했다.
이 폴더는 v0-v4만 소유한다. v5 명세와 실행 방법은
[`../shelf_cube_sweep_v5/README.md`](../shelf_cube_sweep_v5/README.md)를 참고한다.

## 테스트

v2 simulator smoke test:

```powershell
.\IsaacLab\isaaclab.bat -p .\src\sweep_rl\sweep_rl\shelf_cube_sweep\tests\smoke_env_v2.py --num_envs 1 --headless
```
