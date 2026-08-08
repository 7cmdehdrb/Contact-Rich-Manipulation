# UR5e Gripper Shelf Cube Pre-Reach

`Isaac-Reach-Shelf-UR5e-Gripper-v0`를 상속하여, 선반 위의 Cube를 `+Y`로 밀기
직전 자세까지 도달하는 환경이다.

## 환경

| 용도 | Gym ID |
|---|---|
| 학습 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0` |
| 재생 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0` |
| v1 학습 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1` |
| v1 재생 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v1` |

Scene, 6D arm action, 열린 Gripper, TCP 정의, reward, curriculum과 episode timing은
부모 Shelf Reach 환경을 그대로 사용한다.

## Cube와 목표

- 크기: `0.08 × 0.08 × 0.20 m`
- 질량: `1.5 kg`
- 초기 중심: world `(-0.70, -0.10, 1.15) m`
- 선반 표면: world `z=1.05 m`

초기 Y 위치 `-0.10 m`는 `+Y` 선반 가장자리까지 약 `0.60 m`의 sweep 공간을
남긴다. Pre-reach 위치도 `-Y` 가장자리에서 충분히 떨어져 있어 Gripper가 선반
외곽으로 깊게 접근하지 않는다.

가상 목표 position은 매 step 현재 Cube 중심에서 다음 world offset으로 계산된다.

```text
goal = cube_position + (0, -0.08 * 1.1, +0.03)
     = cube_position + (0, -0.088, +0.03) m
```

목표 orientation은 부모 환경과 동일한 `roll=pi/2, pitch=0, yaw=0`이다. 따라서
TCP local X는 전방, local Y는 천장을 향한다.

## v1 positional reward

v0는 부모 Shelf Reach의 coarse/fine position reward를 유지한다. v1은 두 position
reward를 다음 단일 항목으로 교체한다.

```text
distance = ||current_ee_pos_w - moving_cube_offset_pos_w||_2
R_position_v1 = 3.0 * exp(-10.0 * distance)
```

Moving offset, orientation reward, action/joint penalty와 shelf-floor collision penalty는
v0와 동일하다.

## Shelf floor collision penalty

선반 USD의 Cube 지지판(`/Shelf/rack/Cube_02`, local top `z=1.05 m`)에 한정해
접촉 패널티를 계산하며, 이 정보는 policy observation에 추가하지 않는다. 선반의
`rack` rigid body에 Contact sensor를 활성화하고 UR5e 7개 link와 Robotiq 9개 link를
각각 필터로 등록한다. 접촉점은 shelf frame으로 변환한 뒤 지지판의 XY 범위와
상면 높이 `1.05 +/- 0.02 m` 안에 있는 경우만 충돌로 판정한다.

Cube와 선반의 접촉 및 선반 구성요소 사이의 접촉은 필터 목록에 포함되지 않으므로
패널티가 발생하지 않는다.

```text
robot_floor_contact = filtered_force > 1.0 N and contact_point_on_floor
raw_shelf_collision = 1.0 if any(robot_floor_contact) else 0.0
weight = -5.0
```

충돌 weight는 v1 Reach 최대 reward rate `+3.0`보다 크게 설정하여, 검출된 선반
접촉을 통해 얻을 수 있는 Reach 이득보다 안전 비용이 우선하도록 한다.

## Observation

기존 25D Reach observation에 offset 적용 전 Cube 중심과 폭을 추가한 `29-D`다.

| 순서 | 항목 | 차원 |
|---:|---|---:|
| 1 | arm relative joint position | 6 |
| 2 | arm relative joint velocity | 6 |
| 3 | Cube-relative pre-reach pose command | 7 |
| 4 | 원본 Cube center position, robot-base frame | 3 |
| 5 | Cube width | 1 |
| 6 | previous arm action | 6 |

## 실행

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl

./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0 \
  --num_envs 4096 --headless
```

v1 학습은 task ID만 변경한다.

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v1 \
  --num_envs 4096 --headless
```

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0 \
  --checkpoint logs/rsl_rl/reach_shelf_ur5e_gripper_cube_pre_reach/2026-08-07_15-58-15/model_400.pt --num_envs 1
```
