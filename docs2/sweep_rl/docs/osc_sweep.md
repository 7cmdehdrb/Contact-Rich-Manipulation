# 기본 OSC Sweep

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-v0`

## 목적과 명령

UR5e가 Robotiq gripper의 넓은 pad 면으로 cube를 밀어 목표 위치에 도달시키면서 지정된
접촉력을 유지하도록 학습하는 force-command 기준 환경이다.

명령은 `[direction_x, direction_y, distance, force, tolerance]`의 5-D이며, 방향은
`[-π, π]`, 거리는 `0.10–0.22 m`, 힘은 `8–25 N`, 허용 오차는 `3–6 N`에서 에피소드마다
샘플링한다. 에피소드 길이는 8초다.

## 관측

정책 입력은 아래 순서의 62-D 벡터다.

| 항목 | 차원 | 내용 |
|---|---:|---|
| `joint_pos` | 6 | arm 관절 위치, uniform noise `±0.002` |
| `joint_vel` | 6 | arm 관절 속도, uniform noise `±0.01` |
| `joint_effort` | 6 | arm 관절 effort |
| `eef_pose` | 6 | robot base 기준 EEF `xyz + RPY` |
| `ft_sensor` | 6 | EEF virtual F/T wrench |
| `contact_point` | 3 | 접촉점, robot base 기준; 비접촉 시 0 |
| `initial_target_pose` | 6 | reset 시 물체 pose |
| `current_target_pose` | 6 | 현재 물체 pose |
| `desired_motion` | 5 | 방향 2 + 거리 + 힘 + tolerance |
| `last_action` | 12 | 직전 정책 Action |
| **합계** | **62** | |

## Action

12-D variable-stiffness OSC Action을 사용한다. stiffness 6축은 `[20, 300]`, 상대 위치
scale은 `0.025 m`, 상대 회전 scale은 `0.12 rad`, effort limit scale은 0.9다. 이 기본
환경의 gripper joint는 Action에 포함되지 않는다.

## Reward, 패널티와 termination

Isaac Lab Reward Manager는 각 항의 `raw × weight × step_dt`를 합산한다.

| Reward | Weight | 역할 |
|---|---:|---|
| `reaching` | +1.5 | 물체 뒤 pre-contact pose 접근 |
| `target_contact` | +0.1 | 목표 물체와 pad 접촉 |
| `side_direction` | +1.5 | gripper 넓은 면과 진행 방향 정렬 |
| `side_center_contact` | +2.5 | pad 중앙의 양질 접촉 |
| `force_tracking` | +4.0 | 명령 접촉력과 tolerance 추종 |
| `velocity_progress` | +4.0 | 명령 방향의 물체 속도 |
| `normalized_progress` | +1.0 | 명령 거리 대비 전진량 |
| `direction_alignment` | +1.5 | 실제 이동 방향 정렬 |
| `endpoint_tracking` | +6.0 | 목표점 위치 추종 |
| `success` | +10.0 | endpoint `<0.025 m`, normalized lateral `<0.12` |
| `lateral_error` | -2.0 | 횡방향 이탈 |
| `overshoot` | -4.0 | 목표 거리 초과 |
| `off_center_contact` | -1.5 | pad 중심 밖 접촉 |
| `ft_torque` | -0.02 | `1.5 Nm` 초과 torque |
| `action_rate` | -0.02 | Action 변화량 제곱 |
| `joint_velocity` | -0.002 | arm 관절 속도 제곱 |
| `commanded_effort` | -0.03 | 정규화된 OSC torque 제곱 |
| `torque_saturation` | -0.5 | Action/torque saturation |

종료 조건은 정상 timeout 8초, 위 성공 조건, 물체 높이 `<0.72 m` 또는 roll/pitch
`>0.80 rad`, F/T force `>100 N` 또는 torque `>15 Nm`, arm 관절 속도 `>6.5 rad/s`다.

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Sweep-Object-UR5e-OSC-v0 \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p `
  IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py `
  --task Isaac-Sweep-Object-UR5e-OSC-v0 `
  --num_envs 2048 --device cuda:0 --headless
```

PPO 기본값은 rollout 32 step, 최대 12,000 iteration이며 experiment 이름은
`ur5e_osc_sweep`이다.

[환경 목록으로 돌아가기](README.md)
