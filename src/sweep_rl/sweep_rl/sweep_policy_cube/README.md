# Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0

`example/Sweep-Policy`에서 이식한 UR5e·Shelf·Cube scene에 Isaac Lab의 표준 Reach
학습 구성을 적용한 환경이다. 기존 `sweep_basic`과 OSC 환경은 변경하지 않는다.

환경 변경 시 유지해야 하는 필수 설계 원칙은 [`PRINCIPLES.md`](./PRINCIPLES.md)에
기록한다.

## Task

- Robot: UR5e + Robotiq 2F-85
- Robot asset: 원본 Fixed Sweep-Policy의 일체형 `Collected_UR5e_v4/UR5e_v4.usd`
- Arm action: 공식 UR10 Reach와 같은 6축 default-relative joint position command,
  scale `0.5`. PPO/action term의 별도 clipping은 사용하지 않음
- Gripper: action에서 제외하고 open position `0`으로 고정. Implicit actuator의
  stiffness `2000`, damping `1000`으로 열린 자세를 유지
- 전체 action 차원: `6`
- Episode: `10 s`, simulation `100 Hz`, control `50 Hz`
- Target: 진단용 `0.06 × 0.06 × 0.15 m`, `0.5 kg` Cuboid 하나
- Goal: reset된 Cube 중심에서 world `+Y`로 `0.18 m`

Cube는 world `+Y`로 미는 동안 shelf의 `+Y` 가장자리에 치우치지 않도록 구성한
6개 XY 위치 중 하나에서 시작한다.

```text
x ∈ {-0.75, -0.60}
y ∈ {-0.20, -0.10, 0.00}
z = 1.125
```

Cube 높이는 `0.15 m`이며, 바닥면은 계속 선반 표면 `z=1.05 m`에 놓인다. 이에
따라 Cube/goal 중심 Z는 `1.125 m`다. Reaching z offset은 기존 `+0.03 m`를
변경하지 않으므로 reaching 목표의 world Z는 `1.155 m`가 된다. Y 폭과
`target_obj_width`는 `0.06 m`로 유지한다.

Shelf 중심 X는 원본 배치인 `-0.70 m`다. 이전의 manipulator 방향 `+X 0.05 m`
이동을 되돌렸으며, spawn slot도 함께 `-X 0.05 m` 옮겨 Shelf와 target 사이의
상대적인 깊이는 유지한다.

각 환경에 독립적으로 XY `±0.02 m` jitter와 random yaw를 적용한다. Cube 이외의
병·컵·캔·머그 등의 task object는 생성하지 않는다. Ground, Shelf, Robot, light는
환경 구성 요소이므로 유지한다.

Episode reset은 먼저 모든 joint를 default로 되돌려 gripper open target을 복원한 뒤,
UR5e arm 6축에만 공식 UR10 Reach의 default-position scale randomization
`[0.75, 1.25]`를 적용한다. Gripper joint는 randomization 대상이 아니다.

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

Observation과 offset 목표는 현재 환경의 `32-D` 구성을 유지하지만, action·reward·PPO는
Isaac Lab UR10 Reach 기준을 따른다. PPO는 24-step rollout, 최대 1000 iteration,
`[64, 64]` actor/critic, 초기 action std `1.0`, entropy coefficient `0.01`, discount
factor `0.99`, desired KL `0.01`을 사용한다. 별도 action clipping은 없고 실제 관절
위치는 로봇 USD의 물리 joint limit을 적용받는다. 네트워크 구조가 변경됐으므로 기존
checkpoint 대신 새 run을 시작해야 한다.

| Reward | Weight |
|---|---:|
| `end_effector_position_tracking` | `-0.2` |
| `end_effector_position_tracking_fine_grained` | `+0.1` |
| `end_effector_orientation_tracking` | `-0.1` |
| `action_rate` | `-0.0001` |
| `joint_vel` | `-0.0001` |

공식 Reach curriculum과 같이 4500 environment step 동안 `action_rate` weight는
`-0.005`, `joint_vel` weight는 `-0.001`까지 선형 변경한다.

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
  --checkpoint logs/rsl_rl/ur5e_sweep_policy_cube/<new-run>/model_1000.pt \
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
