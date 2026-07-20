# ConstantVelocity Sweep 학습

기본 환경 ID는 `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0`다.
`UR5eOscSweepEnvCfg`의 scene/reset을 상속하고 Command, Action term, Observation,
Reward, termination을 일정 속도 밀기 목적에 맞게 교체한다.

## 학습 계약 요약

| 구분 | 값 |
|---|---|
| Command | `[direction_x, direction_y, distance_m, target_speed_mps]` |
| 방향 | `[-π, π]` |
| 거리 | `0.10–0.22 m` |
| 순항 속도 | `0.08 m/s` |
| Action | 12-D variable-stiffness OSC |
| Observation | 55-D |
| 정책/physics 주기 | 30 Hz / 120 Hz |
| Episode | 8초 |
| 성공 dwell | 0.30초 |
| PPO | rollout 32 steps, 최대 12,000 iterations, 초기 std 0.5 |

속도 profile은 순항 속도의 25%에서 시작해 처음 `0.025 m` 동안 가속하고 마지막
`0.04 m` 동안 정지하도록 감속한다. 성공하려면 endpoint error `< 0.020 m`, normalized
lateral error `< 0.10`, object speed `< 0.020 m/s`를 0.30초 유지해야 한다.

정지한 물체는 velocity reward를 받지 않는다. endpoint running cost와 stall penalty,
안전 실패 시 남은 horizon 비용을 함께 사용해 정지 접촉이나 고의 조기 종료가 유리하지
않도록 구성한다. TensorBoard에서는 `endpoint_error`, `forward_speed`,
`progress_ratio`, success와 안전 종료 비율을 함께 확인한다.

전체 Action/Observation/Reward 수치는
[ConstantVelocity 계약](constant_velocity_action_reward_observation.md)에 정리되어 있다.

## 파생 환경

| 환경 | 추가 기능 |
|---|---|
| `...ConstantVelocity-UprightRandomSize-v0` | 외측 pad 접근, gripper 내부 삽입 실패. 이름은 호환성용 |
| `...ConstantVelocity-UprightRandomSize-HomeReturn-v0` | 목표 정지 후 비접촉 Home 복귀와 물체 위치 보존 |

## 학습

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows에서는 `./IsaacLab/isaaclab.sh` 대신 `.\IsaacLab\isaaclab.bat`를 사용한다.

## 플레이

ConstantVelocity 계열은 force-command 전용 `play_sweep.py`가 아니라 표준 player를
사용한다.

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

문서 내용은 2026-07-19 현재 코드 기준이다.
