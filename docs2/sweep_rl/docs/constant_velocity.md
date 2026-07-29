# Constant Velocity Sweep

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0`

## 목적과 명령

접촉력 명령 대신 물체의 속도 profile을 직접 추종한다. 물체를 출발점에서 가속하고
`0.08 m/s`로 이동한 뒤 목표점 앞에서 감속해 정지시키는 환경이다.

명령은 `[direction_x, direction_y, distance, target_speed]` 4-D다. 방향은 `[-π, π]`,
거리는 `0.10–0.22 m`, 순항 속도는 `0.08 m/s`다. 처음 `0.025 m` 동안 순항 속도의
25%에서 가속하고 마지막 `0.04 m`에서 감속한다. 에피소드는 8초다.

## 관측

정책 관측은 55-D다. F/T wrench와 접촉점은 reward 계산에만 사용하며 정책에는 주지 않는다.

| 항목 | 차원 | 내용 |
|---|---:|---|
| `joint_pos` | 6 | arm 관절 위치, noise `±0.002` |
| `joint_vel` | 6 | arm 관절 속도, noise `±0.01` |
| `joint_effort` | 6 | arm 관절 effort |
| `eef_pose` | 6 | robot base 기준 EEF pose |
| `initial_target_pose` | 6 | reset 시 물체 pose |
| `current_target_pose` | 6 | 현재 물체 pose |
| `object_linear_velocity` | 3 | robot base 기준 물체 속도, noise `±0.005` |
| `desired_motion` | 4 | 방향 2 + 거리 + 목표 속도 |
| `last_action` | 12 | 직전 Action |
| **합계** | **55** | |

## Action

12-D variable-stiffness OSC다. stiffness `[20, 300]`, 상대 위치 `0.025 m`, 상대 회전
`0.12 rad`, effort scale 0.9를 사용한다. gripper Action은 없으며 모든 physics step에서
완전히 열린 위치 `0.0`을 다시 명령한다.

## Reward와 패널티

| 항목 | Weight | 역할 |
|---|---:|---|
| `push_pose_error` | -0.35 | 현재 물체 뒤의 push pose와 EEF 거리 |
| `side_direction_error` | -0.25 | gripper 넓은 면 방향 오차 |
| `target_contact` | +0.50 | pad와 목표 물체 접촉 |
| `side_center_contact` | +0.75 | pad 중앙 접촉 품질 |
| `contact_forward_progress` | +3.0 | 접촉 중 목표 방향 전진 속도 |
| `velocity_tracking` | +10.0 | 가속·순항·감속 profile 추종 |
| `endpoint_error` | -5.0 | 명령 거리로 정규화한 목표점 오차 |
| `stopped_at_goal` | +20.0 | 목표점 위치와 저속 동시 만족 |
| `success` | +40.0 | 위치·횡오차·정지 성공 |
| `failure_termination` | -8.0 | 안전 실패 시 남은 horizon 비용 |
| `lateral_error` | -3.0 | 횡방향 이탈 |
| `overshoot` | -8.0 | 목표 거리 초과 |
| `stall` | -6.0 | grace 0.40초 후 목표 속도의 50% 미만 |
| `object_acceleration` | -0.15 | 큰 물체 가속도 |
| `ft_torque` | -0.02 | `1.5 Nm` 초과 torque |
| `action_rate` | -0.02 | Action 변화량 |
| `joint_velocity` | -0.002 | arm 관절 속도 |
| `commanded_effort` | -0.03 | 정규화된 OSC torque |
| `torque_saturation` | -0.5 | Action/torque saturation |

## Termination

- 성공: endpoint `<0.020 m`, normalized lateral `<0.10`, 물체 속도 `<0.020 m/s`를
  0.30초 유지
- timeout: 8초
- 실패: 물체 높이 `<0.72 m` 또는 tilt `>0.80 rad`, F/T force `>100 N` 또는 torque
  `>15 Nm`, arm 관절 속도 `>6.5 rad/s`

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/train_constant_velocity.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p src\sweep_rl\scripts\train_constant_velocity.py `
  --num_envs 2048 --device cuda:0 --headless
```

PPO는 rollout 32 step, 최대 12,000 iteration, Gaussian 초기 std 0.5를 사용하며
experiment 이름은 `ur5e_osc_sweep_constant_velocity`다.

[환경 목록으로 돌아가기](README.md)
