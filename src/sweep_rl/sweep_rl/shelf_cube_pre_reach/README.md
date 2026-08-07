# UR5e Gripper Shelf Cube Pre-Reach

`Isaac-Reach-Shelf-UR5e-Gripper-v0`를 상속하여, 선반 위의 Cube를 `+Y`로 밀기
직전 자세까지 도달하는 환경이다.

## 환경

| 용도 | Gym ID |
|---|---|
| 학습 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0` |
| 재생 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0` |

Scene, 6D arm action, 열린 Gripper, TCP 정의, reward, curriculum과 episode timing은
부모 Shelf Reach 환경을 그대로 사용한다.

## Cube와 목표

- 크기: `0.06 × 0.06 × 0.15 m`
- 질량: `0.5 kg`
- 초기 중심: world `(-0.70, -0.10, 1.125) m`
- 선반 표면: world `z=1.05 m`

초기 Y 위치 `-0.10 m`는 `+Y` 선반 가장자리까지 약 `0.60 m`의 sweep 공간을
남긴다. Pre-reach 위치도 `-Y` 가장자리에서 충분히 떨어져 있어 Gripper가 선반
외곽으로 깊게 접근하지 않는다.

가상 목표 position은 매 step 현재 Cube 중심에서 다음 world offset으로 계산된다.

```text
goal = cube_position + (0, -0.06 * 1.2, +0.03)
     = cube_position + (0, -0.072, +0.03) m
```

목표 orientation은 부모 환경과 동일한 `roll=pi/2, pitch=0, yaw=0`이다. 따라서
TCP local X는 전방, local Y는 천장을 향한다.

## Shelf collision penalty

Shelf 상태는 reward 계산에만 사용하며 policy observation에는 추가하지 않는다.
현재 Shelf 위치와 초기 위치의 거리, 그리고 6-D root velocity norm의 합으로 충돌을
판정한다.

```text
motion = ||shelf_position - initial_shelf_position||₂ + ||shelf_root_velocity||₂
raw_shelf_collision = 1.0 if motion > 0.005 else 0.0
weight = -0.02
```

`-0.02`는 부모 Reach 환경의 최대 양의 position-tracking reward `0.1`의 20%에
해당한다. 따라서 충돌 step의 최종 reward 기여는 `-0.02`다.

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

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1
```
