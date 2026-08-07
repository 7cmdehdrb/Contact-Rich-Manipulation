# UR5e Gripper Shelf Reach

`example/Sweep-Policy`의 shelf scene에서 UR5e + Robotiq TCP가 가상 7D pose를
추종하는 Manager-based RL 환경이다. 물리 target object와
`RigidObjectCollectionCfg`는 생성하지 않는다.

## 환경

| 용도 | Gym ID |
|---|---|
| 학습 | `Isaac-Reach-Shelf-UR5e-Gripper-v0` |
| 재생 | `Isaac-Reach-Shelf-UR5e-Gripper-Play-v0` |

- TCP: `robotiq_base_link` local `+X 0.13 m`
- Action: UR5e arm 6축 relative joint-position command, scale `0.5`
- Gripper: action과 observation에서 제외하고 open joint position `0`으로 고정
- Observation: arm position `6` + arm velocity `6` + pose command `7` + last action `6` = `25-D`
- Episode: `12 s`, simulation `60 Hz`, control `30 Hz`

가상 pose command는 robot-base frame에서 4초마다 다음 범위로 재샘플링된다.

```text
x     = [ 0.55,  0.75] m
y     = [-0.20,  0.20] m
z     = [ 0.30,  0.50] m
roll  = pi / 2
pitch = 0
yaw   = 0
```

원본 Sweep-Policy 초기 TCP의 약 `pi/4` roll에서 45도를 더 회전한 고정
orientation이다. TCP local X축은 robot-base `+X` 전방을 유지하고 local Y축은
robot-base `+Z` 천장 방향에 정확히 정렬된다. 오른손 좌표계에 따라 local Z축은
robot-base `-Y`를 향한다. 목표가 갱신되어도 임의 yaw는 적용하지 않는다.

학습 환경은 marker를 만들지 않는다. Play 환경만 목표 pose와 현재 TCP를 나타내는
비물리 frame marker를 표시한다.

## 자산 경로

기본 robot USD는 원본 Sweep-Policy의 결합형 자산을 사용한다.

```text
./asset/Shelf_USD/Robots/UR5e/Collected_UR5e_v4/.collect.mapping.json
```

다른 머신에서는 다음 환경변수로 경로를 바꿀 수 있다.

```bash
export SWEEP_POLICY_ROBOT_USD_PATH=/absolute/path/to/UR5e_v4.usd
export SWEEP_SHELF_USD_PATH=omniverse://server/path/to/speedrack_shape.usd
```

## 설치와 실행

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl

./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Reach-Shelf-UR5e-Gripper-v0 \
  --num_envs 4096 --headless
```

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Reach-Shelf-UR5e-Gripper-Play-v0 \
  --checkpoint /absolute/path/to/model.pt \
  --num_envs 1
```

## 검증

Simulator-independent contract test:

```bash
python src/sweep_rl/sweep_rl/shelf_reach/tests/run_unit_tests.py
```

Isaac Lab smoke test:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/sweep_rl/shelf_reach/tests/smoke_env.py \
  --num_envs 4 --headless
```
