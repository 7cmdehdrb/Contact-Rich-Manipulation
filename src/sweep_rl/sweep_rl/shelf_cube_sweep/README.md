# UR5e Gripper Shelf Cube Sweep

`shelf_cube_pre_reach`의 충돌 안전 Reach 환경을 상속하여 Cube를 world `+Y` 방향으로
`0.18 m` 미는 별도 task다. 기존 PreReach task에는 Sweep sensor, goal 및 reward를
추가하지 않는다.

| 용도 | Gym ID |
|---|---|
| 학습 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0` |
| 재생 | `Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0` |

## 상속 구조

```text
shelf_reach
  -> shelf_cube_pre_reach
       - moving Cube-relative Reach point
       - 기존 coarse/fine position error
       - orientation 유지
       - shelf floor collision penalty
  -> shelf_cube_sweep
       - wrist frame
       - fixed object goal
       - pushing_target reward
```

Reach point는 매 step 현재 Cube 중심에서 다음 offset으로 계산된다.

```text
(0.0, -0.06 * 1.2, +0.03) = (0.0, -0.072, +0.03) m
```

Object goal은 episode 초기 Cube 위치에서 `(0.0, +0.18, 0.0) m` 떨어진 곳에
고정된다. 명시적인 phase state는 사용하지 않고 `zeta_m`으로 Sweep shaping을 연다.

```text
distance = ||goal_pos_w - target_pos_w||_2
zeta_m = 1 if ee_offset_error < 0.04 and wrist_y_error < 0.04 else 0

if 0.05 < |target_y_velocity| < 0.10:
    obj_vel_rew = +0.5
elif |target_y_velocity| >= 0.10:
    obj_vel_rew = -0.5
else:
    obj_vel_rew = 0.0

if distance < 0.03:
    R_push_raw = 2.0 * exp(-5.0 * distance)
else:
    R_push_raw = zeta_m * ((1.0 - distance / 0.18) + obj_vel_rew)

R_push = 6.0 * R_push_raw
```

## 실행

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0 \
  --num_envs 4096 --headless
```

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0 \
  --checkpoint logs/rsl_rl/sweep_shelf_ur5e_gripper_cube/<run>/model_<step>.pt \
  --num_envs 1
```
