# Sweep Basic

`Isaac-Sweep-Object-UR5e-Basic-v0`는 열린 Robotiq 2F-85가 장착된 UR5e로
선반 위 near-cubic box를 world `+Y` 방향으로 `0.40 m` 미는 Manager-based RL
환경이다. 목표 속도는 모든 episode에서 `0.10 m/s`다. 정책 observation에는
센서값이 없지만 실제 접촉 reward와 안전 termination에는 contact sensor를 사용한다.

Scene의 ground, shelf, robot, virtual EEF frame, 단일 target과 light 구성은
유지한다. 정책은 6축 arm만 제어하며 gripper는 action term이 매 physics step
open target `0.0`으로 고정한다. 기존 checkpoint는 새 31-D observation 및
reward 계약과 호환되지 않는다.

## Single-Stage Task

환경에는 REACH/SWEEP/HOME phase, phase one-hot, high waypoint 또는 descent
latch가 없다. 모든 reward는 episode 전체에서 동시에 활성화된다.

- Gripper base의 local `(0, 0, 0.16)` offset을 EEF contact center로 사용한다.
- 최종 push point는 target 중심에서 world `-Y`로 `0.040 m`, shelf-up으로
  `0.055 m` 떨어진 위치다. reaching과 pushing reward는 항상 이 점을 직접 추종한다.
- Goal은 reset target 위치에서 world `+Y`로 `0.40 m` 떨어진 점이다.
- 별도 contact-pad rigid body, pad pose 또는 pad contact observation은 사용하지 않는다.
- 실제 gripper rigid body와 target 사이 filtered contact force가 `>0.25 N`인 상태를
  `0.10 s` 연속 유지하면 해당 episode가 target에 진입한 것으로 내부 기록한다.
- Episode 절반까지 진입하지 못하면 이후 처음 진입할 때까지 강한 패널티를 받는다.
- Goal 거리 `<0.03 m`, target 3D speed `<0.02 m/s`가 되면 성공 종료한다.

## Observation과 Action

Actor는 noise 없는 31-D `policy` observation을 사용한다. 목표 위치는 바뀌지 않지만
정책이 endpoint를 명시적으로 알 수 있도록 robot-base frame의 goal position을 넣는다.

| 순서 | Term | 차원 | 값 |
|---:|---|---:|---|
| 1 | `joint_pos` | 6 | UR5e arm absolute joint position |
| 2 | `joint_vel` | 6 | UR5e arm joint velocity |
| 3 | `last_action` | 6 | 직전 arm raw action |
| 4 | `target_object_state` | 3 | robot-base frame target position |
| 5 | `goal_position` | 3 | robot-base frame의 고정 goal position |
| 6 | `eef_pose` | 7 | robot-base frame virtual EEF XYZ + WXYZ quaternion |

`target_linear_velocity`는 actor observation에서 제외한다. Critic은 위 31-D 정보 뒤에
robot-base frame target linear velocity 3-D를 privileged observation으로 추가한
34-D `critic` group을 사용한다.

Action `a∈[-1,1]^6`은 매 control step 다음 current-relative joint target으로
변환되고, 그 target을 physics decimation 동안 고정한다.

```text
q_target = clamp(q_current + 0.05 * a, q_soft_lower, q_soft_upper)
```

따라서 arm workspace는 더 이상 `q_default ± 0.5 rad`로 제한되지 않는다.

Gripper joint는 observation/action 차원에 포함되지 않는다.

## Scene과 Reset

- Target 크기: `0.06 x 0.06 x 0.07 m`, nominal mass `0.50 kg`.
- Target spawn: world `x∈[-0.72,-0.68]`, `y∈[-0.12,-0.08]`, upright,
  zero velocity.
- Goal y 범위: `[0.28,0.32]`. 시작점은 shelf 중앙 가까이로 제한하고,
  폭 `[-0.50,0.50]`의 상판 끝에서 충분한 여유를 가진다.
- Target yaw, arm reset offset, mass/friction domain randomization은 사용하지 않는다.
- Shelf USD와 pose는 유지하되 kinematic, gravity-disabled body로 고정한다.

## Reward

Isaac Lab reward manager가 아래 weighted term을 합하고 control-step
`dt=0.02 s`를 적용한다.

| Term | Weight | 의미 |
|---|---:|---|
| `action_rate_l2` | `-0.01` | 연속 action 차분 제곱합 |
| `joint_vel_l2` | `-0.01` | arm velocity 제곱합 |
| `shelf_collision` | `-2.0` | 상판 위 finger/wrist clearance 침범 |
| `reward_for_hand_reaching` | `+2.0` | EEF 중앙의 최종 push point 추종 |
| `align_ee_target` | `+2.0` | TCP y-axis와 shelf z-axis의 signed-square 정렬 |
| `pushing_target` | `+6.0` | endpoint 거리, push pose gate, target world-y 속도의 piecewise reward |
| `target_contact` | `+1.0` | 실제 gripper rigid body와 target의 접촉 |
| `contact_forward_progress` | `+4.0` | 실제 접촉 중 world `+Y` 진행 속도 |
| `velocity_tracking` | `+8.0` | 정지 reward가 없는 `0.10 m/s` 속도 tracking |
| `endpoint_error` | `-2.0` | 남은 goal 거리 |
| `stopped_at_goal` | `+15.0` | endpoint 위치와 정지의 결합 tracking |
| `sweep_success` | `+30.0` | endpoint 및 정지 sparse bonus |
| `lateral_error` | `-2.0` | 명령 방향 밖 target 변위 |
| `overshoot` | `-4.0` | `0.40 m` 이후 초과 진행 |
| `stall` | `-2.0` | 2초 grace 이후 transit 정지 |
| `midpoint_no_entry` | `-10.0` | Episode 50% 이후 실제 접촉 진입 전까지 지속 패널티 |
| `shelf_collision_failure` | `-8.0` | shelf 충돌 시 남은 episode 시간에 비례한 비용 |

모든 term은 phase mask 없이 동시에 계산된다. `midpoint_no_entry`는 `dt=0.02 s`에서
step당 `-0.2`를 적용하므로 15초 episode의 후반 전체가 미진입 상태라면 return에
약 `-75`가 누적된다. 늦게라도 0.10초 연속 실제 접촉을 달성하면 즉시 중단되며,
접촉 이력은 환경 reset 시 초기화된다. 이 내부 이력은 policy에 관측되지 않는다.

`align_ee_target`은 TCP y-axis와 shelf z-axis의 dot product `align`에 대해
`sign(align) * align^2`을 반환한다. Reward manager weight `2.0`을 적용하므로 반대 방향
정렬은 음수, 올바른 수직 정렬은 양수가 된다. 별도 `keep_ee_upright` term은 없다.

`pushing_target`은 EEF contact center가 움직이는 push point의 `0.04 m` 이내이고
wrist world-y가 push point world-y의 `0.04 m` 이내일 때만 `zeta_m=1`로 둔다.
Target world-y 속도의 절댓값이 `(0.05, 0.10) m/s`이면 `+0.5`, `0.10 m/s`
이상이면 `-0.5`, 그 외에는 `0`이다. Raw reward는 다음과 같다.

```text
if distance < 0.03:
    R_push_raw = 2.0 * exp(-5.0 * distance)
else:
    R_push_raw = zeta_m * ((1.0 - distance / 0.18) + obj_vel_rew)

R_push = 6.0 * R_push_raw
```

정책 observation 31-D에는 contact 또는 target velocity 정보가 없다.

`shelf_collision`은 shelf 이동을 측정하지 않는다. Shelf frame의 상판 bounds
`X=[-0.20,0.20]`, `Y=[-0.50,0.50]` 안에서 finger `0.02 m`, wrist `0.08 m`
clearance보다 낮은 정도만 penalize한다.

## Termination

| Term | 조건 |
|---|---|
| `time_out` | episode `15.0 s` 초과 |
| `success` | goal 거리 `<0.03 m`이고 target 3D speed `<0.02 m/s` |
| `object_drop` | reset target root z 대비 절대 변화 `>0.04 m` |
| `push_fast` | 3 control step 연속 overspeed. 한계는 `0.30 → 0.20 → 0.15 m/s` |
| `shelf_collision` | UR5e 또는 Gripper와 Shelf의 filtered contact force `>0.1 N` |

`push_fast` 한계는 control step `100,000`과 `250,000`에서 순차적으로 낮아진다.
최종 정책의 안전 한계는 원래 계약과 같은 `0.15 m/s`다.
`shelf_collision`은 Shelf 위에 놓인 TargetCube의 정상 접촉은 무시하고 robot rigid
body와의 실제 PhysX contact만 감지하여 즉시 episode를 종료한다.

## Physics-frame Debug Visualization

USD selection gizmo 대신 실제 PhysX/Fabric tensor 위치를 확인하려면 play 실행 전에
`SWEEP_BASIC_DEBUG_VIS=1`을 설정한다. 다음 marker가 표시된다.

| 색 | 위치 |
|---|---|
| 파랑 | target physics root |
| 노랑 | pre-contact point |
| 빨강 | physical push point |
| 초록 | target goal |
| 자홍 | 현재 EEF contact center |

## 학습과 검증

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl

SWEEP_BASIC_DEBUG_VIS=1 ./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-Basic-v0 \
  --num_envs 2048 --device cuda:0 --headless
```

### Play

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-Basic-v0 \
  --checkpoint logs/rsl_rl/ur5e_sweep_basic/2026-08-06_18-15-01/model_300.pt \
  --num_envs 1 --device cuda:0
```

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/sweep_rl/sweep_basic/tests/run_unit_tests.py

./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/sweep_rl/sweep_basic/tests/smoke_env.py --headless
```

Asset 위치는 `SWEEP_UR5E_USD_PATH`, `SWEEP_ROBOTIQ_USD_PATH`,
`SWEEP_SHELF_USD_PATH` 환경 변수로 바꿀 수 있다.

현재-relative action과 actor 31-D/critic 34-D observation 및 reward 계약 변경으로
기존 checkpoint는 호환되지
않으므로 새 run으로 학습해야 한다.
