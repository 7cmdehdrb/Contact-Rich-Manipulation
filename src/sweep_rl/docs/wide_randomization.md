# OSC Sweep Wide Randomization

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0`

## 목적과 randomization

기본 force-command 정책을 더 넓은 물체 질량과 접촉력 범위에서 학습해 강건성을 높이는
환경이다. 기본 환경을 상속하며 다음만 바뀐다.

- 목표 힘: `8–50 N`으로 확대
- 물체 질량: 에피소드별 uniform `0.3–3.0 kg`, inertia 재계산
- 모든 actuator의 PD stiffness/damping: 0; arm은 OSC torque로 계속 제어

방향 `[-π, π]`, 거리 `0.10–0.22 m`, force tolerance `3–6 N`, 에피소드 8초는 기본과 같다.

## 관측과 Action

관측은 [기본 환경](osc_sweep.md#관측)과 같은 62-D다: arm 상태 18-D, EEF pose 6-D,
F/T 6-D, 접촉점 3-D, 초기/현재 물체 pose 12-D, 5-D 명령, 이전 Action 12-D.
관절 위치 `±0.002`, 속도 `±0.01` noise가 적용된다.

Action은 같은 12-D variable-stiffness OSC다. stiffness `[20, 300]`, 위치 `0.025 m`,
회전 `0.12 rad`, effort scale 0.9를 사용한다.

## Reward, 패널티와 termination

기본 환경과 동일한 reward/weight를 사용한다. 즉 접근·접촉·pad 중앙·force tracking·진행·
방향·endpoint·success를 보상하고 lateral/overshoot/off-center/torque/Action rate/관절
속도/effort/saturation을 패널티로 준다. 전체 weight는
[기본 reward 표](osc_sweep.md#reward-패널티와-termination)를 따른다.

종료 조건도 8초 timeout, endpoint/lateral 성공, 물체 높이·기울기 이상, F/T
`100 N/15 Nm` 초과, 관절 속도 `6.5 rad/s` 초과로 동일하다.

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/train_wide_randomization.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p src\sweep_rl\scripts\train_wide_randomization.py `
  --num_envs 2048 --device cuda:0 --headless
```

스크립트가 task ID를 자동으로 설정하며 experiment 이름은
`ur5e_osc_sweep_wide_randomization`이다.

[환경 목록으로 돌아가기](README.md)
