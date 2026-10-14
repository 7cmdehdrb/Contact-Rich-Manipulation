# OSC Sweep Tactile Localization

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0`

## 목적

현재 물체 pose를 정책에 주지 않고 관절, EEF, F/T와 접촉점으로 물체를 국소화하여 미는
학습 환경이다. [Wide Randomization](wide_randomization.md)을 상속하므로 목표 힘
`8–50 N`, 물체 질량 `0.3–3.0 kg` randomization을 유지한다. simulator의 실제 물체
상태는 reward와 termination 계산에 privileged state로만 사용한다.

## 관측

Wide Randomization의 `current_target_pose` 6-D를 제거한 56-D 벡터다.

| 항목 | 차원 | 내용 |
|---|---:|---|
| `joint_pos`, `joint_vel`, `joint_effort` | 18 | arm 상태; 위치/속도 noise 적용 |
| `eef_pose` | 6 | robot base 기준 EEF pose |
| `ft_sensor` | 6 | virtual F/T wrench |
| `contact_point` | 3 | 감지된 접촉점 |
| `initial_target_pose` | 6 | reset 시 물체 pose |
| `desired_motion` | 5 | 방향, 거리, 힘, tolerance |
| `last_action` | 12 | 직전 Action |
| **합계** | **56** | 현재 물체 pose 없음 |

## Action

12-D variable-stiffness OSC는 기본과 같지만, 무거운 물체가 torque limit에서 막히지 않도록
arm simulator effort limit을 1.5배로 높이고 OSC effort scale을 1.0으로 사용한다.

## Reward, 패널티와 termination

Wide Randomization의 reward 중 다음 weight를 강화한다.

| 항목 | Weight | 변경 목적 |
|---|---:|---|
| `normalized_progress` | +3.0 | 접촉 후 실제 전진 강화 |
| `endpoint_tracking` | +12.0 | 최종 위치를 중심 목표로 설정; coarse std `0.12`, weight `0.35` 추가 |
| `success` | +40.0 | 완주 보상 강화 |

그 외 접근, 접촉, 방향, force tracking 및 lateral/overshoot/off-center/제어 패널티는
[기본 reward 표](osc_sweep.md#reward-패널티와-termination)와 같다. termination도 8초
timeout, endpoint/lateral 성공, 물체 pose, F/T 및 arm 속도 안전 종료를 그대로 쓴다.

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/train_tactile_localization.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p src\sweep_rl\scripts\train_tactile_localization.py `
  --num_envs 2048 --device cuda:0 --headless
```

experiment 이름은 `ur5e_osc_sweep_tactile_localization`이다.

[환경 목록으로 돌아가기](README.md)
