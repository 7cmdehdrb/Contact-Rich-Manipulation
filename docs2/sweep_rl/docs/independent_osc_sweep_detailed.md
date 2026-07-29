# Independent OSC Sweep Detailed

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0`

## 목적

[`Independent`](independent_osc_sweep.md)와 동일한 scene, command, 관측, Action,
randomization 및 termination을 유지하면서 네 개의 통합 reward를 단계별 항목으로 분해한
환경이다. 학습 신호와 실패 원인을 세밀하게 관찰하거나 weight를 조정할 때 사용한다.

## 관측과 Action

관측은 부모와 같은 56-D다: arm 상태 18-D, EEF 6-D, F/T 6-D, 접촉점 3-D, 초기 물체
pose 6-D, 4-D 명령, phase 1-D, 이전 Action 12-D. 현재 물체 pose/속도는 privileged
reward 및 termination에만 사용한다.

Action도 같은 12-D variable-stiffness OSC이며 stiffness `[20, 300]`, 위치 `0.025 m`,
회전 `0.12 rad`, 열린 gripper와 per-episode OSC calibration randomization을 사용한다.

## Reward와 패널티

| Phase | 항목 | Weight | 역할 |
|---|---|---:|---|
| REACH | `reach_pose_tracking` | +4.0 | size-aware pre-contact pose 추종 |
| REACH | `reach_pose_error` | -1.0 | pre-contact 거리 오차 |
| SWEEP | `sweep_contact` | +1.5 | 목표 물체 pad 접촉 |
| SWEEP | `sweep_velocity_tracking` | +8.0 | 가속·순항·감속 속도 추종 |
| SWEEP | `sweep_forward_progress` | +2.0 | 목표 방향 전진 |
| SWEEP | `sweep_endpoint_error` | -4.0 | 정규화 endpoint 오차 |
| SWEEP | `sweep_lateral_error` | -3.0 | 횡방향 이탈 |
| SWEEP | `sweep_overshoot` | -6.0 | 목표 거리 초과 |
| SWEEP | `sweep_stopped_at_goal` | +15.0 | 목표점 정지 |
| HOME | `home_joint_pose` | +12.0 | Home joint pose 추종 |
| HOME | `home_joint_error` | -2.0 | Home joint 오차 |
| HOME | `home_clearance` | +2.0 | EEF-물체 안전거리 `0.22 m` |
| HOME | `post_goal_contact` | -10.0 | robot-물체 접촉 |
| HOME | `goal_hold_error` | -8.0 | 목표점 이탈 |
| HOME | `post_goal_object_speed` | -2.0 | 물체 속도 |
| HOME | `post_goal_object_displacement` | -6.0 | park 이후 물체 이동 |
| HOME | `home_time` | -0.3 | Home 복귀 시간 |
| HOME | `home_success` | +30.0 | 안정적 비접촉 Home 완료 |
| 공통 | `ft_torque` | -0.02 | `1.5 Nm` 초과 torque |
| 공통 | `action_rate` | -0.01 | Action 변화량 |
| 공통 | `joint_velocity` | -0.001 | arm 관절 속도 |
| 공통 | `commanded_effort` | -0.01 | 정규화된 OSC torque |
| 공통 | `torque_saturation` | -0.5 | Action/torque saturation |
| 실패 | `failure_termination` | -5.0 | 안전 실패 시 남은 episode 시간 비용 |

## Termination

부모 환경과 완전히 같다. Home 성공, 접촉 상실, gripper 내부 삽입, Home 재접촉/물체
교란, 비정상 물체 pose, 과도한 F/T, arm 과속, shelf/self-collision 및 20초 timeout을
사용한다. 정확한 threshold는 [부모 termination 표](independent_osc_sweep.md#termination)를
따른다. 성공에는 실패 패널티를 적용하지 않는다.

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_independent_sweep_detailed.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\train_independent_sweep_detailed.py `
  --num_envs 2048 --device cuda:0 --headless
```

PPO 설정은 부모와 같고 experiment 이름은 `ur5e_osc_sweep_independent_detailed`다.

[환경 목록으로 돌아가기](README.md)
