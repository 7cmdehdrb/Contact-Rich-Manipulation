# Independent OSC Sweep

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-Independent-v0`

## 목적과 phase

기존 force/constant-velocity 설정을 상속하지 않고, 선반 위에서 긴 거리 sweep과 Home
복귀를 하나의 독립 환경으로 구현한 과제다.

1. `REACH(0)`: 물체 크기를 반영한 pre-contact pose로 접근
2. `SWEEP(1)`: pad 접촉이 시작되면 속도 profile을 따라 목표점까지 밀기
3. `HOME(2)`: endpoint `<0.025 m`, 물체 속도 `<0.020 m/s`를 0.30초 유지한 뒤 물체를
   놓고 Home 복귀

명령은 `[direction_x, direction_y, distance, target_speed]` 4-D다. 방향은 `[-π, π]`,
거리는 workspace 안에서 가능한 `0.12–0.35 m`, 속도는 `0.04–0.12 m/s`다. 에피소드
길이는 20초다.

## Domain randomization

- cube 한 변: environment별 `0.04–0.08 m`
- 질량: episode별 `0.25–2.0 kg`
- 물체와 shelf static friction `0.40–1.10`, dynamic friction `0.25–0.90`
- OSC stiffness calibration `0.95–1.05`, damping `0.95–1.05`, effort `0.97–1.03`
- `replicate_physics=False`

## 관측

현재 물체 pose와 속도는 정책에 주지 않는 56-D 관측이다.

| 항목 | 차원 | 내용과 noise |
|---|---:|---|
| `joint_pos` | 6 | `±0.002` |
| `joint_vel` | 6 | `±0.01` |
| `joint_effort` | 6 | `±0.5` |
| `eef_pose` | 6 | robot base 기준 EEF pose |
| `ft_sensor` | 6 | force `±0.5 N`, torque `±0.02 Nm` |
| `contact_point` | 3 | 접촉점 `±0.002 m`; 비접촉 mask 유지 |
| `initial_target_pose` | 6 | 위치 `±0.003 m`, 회전 `±0.02 rad` |
| `desired_motion` | 4 | 방향 2 + 거리 + 속도 |
| `task_phase` | 1 | REACH 0 / SWEEP 1 / HOME 2 |
| `last_action` | 12 | 직전 Action |
| **합계** | **56** | |

## Action

12-D variable-stiffness OSC다. stiffness `[20, 300]`, 위치 `0.025 m`, 회전 `0.12 rad`,
effort scale 0.9를 사용하고 위 calibration randomization을 적용한다. gripper는 매 physics
step에서 열린 위치 `0.0`을 유지한다.

## Reward와 패널티

의도적으로 네 개의 통합 reward category만 사용한다.

| 항목 | Weight | 활성 phase와 내부 구성 |
|---|---:|---|
| `reaching` | +1.0 | REACH: size-aware pre-contact pose tracking; 접촉 전만 활성 |
| `contact` | +1.5 | SWEEP: 목표 물체 pad 접촉 |
| `push` | +2.0 | SWEEP: `2.5×` 속도 추종 + `0.75×` 전진 + `4×` 목표점 정지 - `2×` lateral - `3×` overshoot |
| `home_return` | +2.0 | HOME: `3×` Home pose - joint 오차 - `4×` 접촉 - `3×` 물체 이동 |

가중치는 category 바깥의 multiplier와도 곱해진다. 이 기준 환경에는 별도 Action rate,
torque 또는 실패 reward term이 없고, 위험 동작은 아래 termination으로 제한한다.

## Termination

- 성공: Home phase에서 joint 오차 `<0.12 rad`, 속도 `<0.15 rad/s`, endpoint
  `<0.030 m`, 물체 속도 `<0.025 m/s`, park 이후 이동 `<0.010 m`, 비접촉을 0.25초 유지
- 접촉 상실: SWEEP 중 한번 접촉한 뒤 0.75초 연속 접촉 없음
- gripper 내부 삽입: EEF-local half extents `(0.040, 0.040, 0.058) m`
- HOME 접촉: 0.30초 release grace 이후 계속/재접촉
- HOME 물체 교란: 이동 `>0.015 m` 또는 속도 `>0.10 m/s`
- 물체 높이 `<1.04 m` 또는 tilt `>0.80 rad`
- F/T force `>100 N`, torque `>15 Nm`, arm 속도 `>6.5 rad/s`
- robot-shelf 또는 비인접 robot self-collision, timeout 20초

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/train_independent_sweep.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p src\sweep_rl\scripts\train_independent_sweep.py `
  --num_envs 2048 --device cuda:0 --headless
```

PPO는 rollout 48 step, 최대 12,000 iteration이며 experiment 이름은
`ur5e_osc_sweep_independent`다.

[환경 목록으로 돌아가기](README.md)
