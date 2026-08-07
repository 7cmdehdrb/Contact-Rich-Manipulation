# Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0

`example/Sweep-Policy`의 Fixed `+Y` 환경을 현재 `sweep_rl` 패키지에서 독립적으로
실행할 수 있게 이식한 단일 Cube 환경이다. 기존 `sweep_basic`과 OSC 환경은 변경하지
않는다.

환경 변경 시 유지해야 하는 필수 설계 원칙은 [`PRINCIPLES.md`](./PRINCIPLES.md)에
기록한다.

## Task

- Robot: UR5e + Robotiq 2F-85
- Robot asset: 원본 Fixed Sweep-Policy의 일체형 `Collected_UR5e_v4/UR5e_v4.usd`
- Arm action: 6축 default-relative joint position command, scale `0.5`. PPO action
  clipping은 적용하지 않음
- Gripper: action에서 제외하고 open position `0`으로 고정. Implicit actuator의
  stiffness `2000`, damping `1000`으로 열린 자세를 유지
- 전체 action 차원: `6`
- Episode: `10 s`, simulation `100 Hz`, control `50 Hz`
- Target: 진단용 `0.06 × 0.06 × 0.12 m`, `0.5 kg` Cuboid 하나
- Goal: reset된 Cube 중심에서 world `+Y`로 `0.18 m`

Cube는 원본 환경의 6개 XY 위치 중 하나에서 시작한다.

```text
x ∈ {-0.70, -0.55}
y ∈ {-0.20, 0.00, 0.20}
z = 1.11
```

Z 높이를 기존 `0.06 m`의 `2배`인 `0.12 m`로 높였으며, 바닥면은 계속 선반
표면 `z=1.05 m`에 놓인다. 이에 따라 Cube/goal 중심 Z는 `1.11 m`, reaching
offset의 world Z는 `1.14 m`가 된다. Y 폭과 `target_obj_width`는 `0.06 m`로
유지한다.

Manipulator와의 도달 여유를 늘리기 위해 Shelf와 모든 spawn slot을 world `+X`로
`0.05 m` 이동했다. Shelf 중심 X는 `-0.65 m`이며, Shelf와 target 사이의 상대적인
X/Y 배치는 기존과 동일하다.

각 환경에 독립적으로 XY `±0.02 m` jitter와 random yaw를 적용한다. Cube 이외의
병·컵·캔·머그 등의 task object는 생성하지 않는다. Ground, Shelf, Robot, light는
환경 구성 요소이므로 유지한다.

## Observation

Policy observation은 `32-D` 벡터다. Reaching point 자체를 추가로 제공하지 않으며,
Cube 위치와 폭의 observation noise만 제거했다.

| 순서 | 항목 | 차원 | 설명 |
|---:|---|---:|---|
| 1 | `joint_pos` | 6 | UR5e arm 6축의 default-relative position |
| 2 | `joint_vel` | 6 | UR5e arm의 default-relative velocity |
| 3 | `actions` | 6 | 이전 arm action |
| 4 | `target_obs_state` | 3 | robot base frame의 정확한 Cube position |
| 5 | `target_obj_width` | 1 | 정확한 Cube Y 폭 `0.06 m` |
| 6 | `ee_pose` | 7 | robot base frame의 원본 Sweep-Policy EE position과 quaternion |
| 7 | `goal_pos` | 3 | robot base frame의 고정 goal position |

## Reward와 termination

접근·pushing·관측에는 Fixed Sweep-Policy와 동일한 `robotiq_base_link` 기준 EE frame을
사용한다. 원본과 동일하게 EE는 local-X forward `0.130 m`, finger clearance proxy는
local-X `0.130 m`와 local-Y `±0.070 m`, wrist proxy는 local-X `-0.140 m`이다. 이
frame들은 관측과 reward/termination 계산용이며 별도 collision/contact pad를 생성하지
않는다. 실제 충돌 geometry도 런타임에 조립한 별도 Robotiq 자산이 아니라 원본 일체형
USD의 collision hull을 그대로 사용한다.

접근 목표는 sweep 방향의 반대편 offset으로 계산한다.

```text
offset_x = target_x
offset_y = target_y - target_width * sign(sweep_dir_y)
offset_z = target_z + 0.03
```

`sweep_dir_y`는 goal Y와 현재 target Y의 차이에서 얻는다. 현재 설정은 이 offset에
정확히 도달할 수 있는지를 분리해서 확인하는 Reaching-only 테스트 벤치다. Isaac Lab
Reach 환경과 동일한 coarse L2 error와 fine tanh kernel을 사용한다.

```text
distance = ||offset_pos - ee_pos||₂
R_coarse_raw = distance
R_fine_raw = 1 - tanh(distance / 0.1)
```

Orientation은 quaternion 전체 자세를 강제하지 않고 EE의 y축과 shelf의 z축 사이
최단 각도 오차를 사용한다. 두 축의 dot product를 `align`이라 하면 raw 값은
`acos(clamp(align, -1, 1))`이고, `-0.1` weight로 패널티를 부과한다. 따라서 두 축이
같은 방향이면 최대값 `0`, 직교하면 `-0.1 * pi/2`, 반대 방향이면 `-0.1 * pi`이다.
회전행렬의 열을 사용해 각 로컬 축의 world 방향을 계산하며, 축 주위 회전은
제한하지 않는다. Pushing, homing, shelf-collision reward와
object/drop, push-fast, shelf-collision, hand-velocity termination은 삭제하지 않고
`env_cfg.py`에서 주석 처리했다. 활성 termination은 `time_out`뿐이다.

Action과 관측 차원이 각각 `6-D`, `32-D`로 바뀌었으므로 기존 checkpoint는 로드할
수 없으며 새 run을 시작해야 한다. PPO는 초기 action std `0.25`, entropy coefficient
`0.001`, discount factor `0.99`를 사용한다. PPO 출력에는 별도의 action clipping을
적용하지 않으며, 급격한 실제 관절 운동은 `joint_vel` reward로 억제한다.

| Reward | Weight |
|---|---:|
| `end_effector_position_tracking` | `-0.2` |
| `end_effector_position_tracking_fine_grained` | `+0.1` |
| `orientation` | `-0.1` |
| `action_rate` | `-0.001` |
| `joint_vel` | `-0.0001` |

이 구성에서 이전 `Episode_Reward/reaching`의 이론적 최대 `5.0`은 더 이상 비교
지표가 아니다. 정확도는 `Episode_Reward/end_effector_position_tracking`이 `0`에
접근하는지와 `Episode_Reward/end_effector_position_tracking_fine_grained`가 이론적
최대 `0.1`에 접근하는지를 함께 확인한다.

## 실행

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl

./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0 \
  --num_envs 4096 \
  --headless
```

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0 \
  --checkpoint logs/rsl_rl/ur5e_sweep_policy_cube/2026-08-07_12-56-24/model_450.pt \
  --num_envs 4 
```

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/sweep_rl/sweep_policy_cube/tests/run_unit_tests.py

./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/sweep_rl/sweep_policy_cube/tests/smoke_env.py \
  --headless --num_envs 8
```

Robot 기본 경로는 원본 환경에 사용된 로컬 일체형 USD다. 다른 위치에 복사했다면
`SWEEP_POLICY_ROBOT_USD_PATH`로 재정의한다. Shelf는 `SWEEP_SHELF_USD_PATH`로
재정의할 수 있다. Goal marker와 frame visualization은
`SWEEP_POLICY_CUBE_DEBUG_VIS=1`로 활성화할 수 있다.
