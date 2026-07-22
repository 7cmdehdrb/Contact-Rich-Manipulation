# OSC Sweep Play

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-Play-v0`

## 목적

[`Isaac-Sweep-Object-UR5e-OSC-v0`](osc_sweep.md)의 학습 결과를 반복 가능한 고정 명령으로
검사하는 평가 전용 환경이다. 16개 환경을 사용하고 관측 noise를 끄며 다음 명령을
고정한다.

```text
direction = +Y (π/2), distance = 0.16 m,
target force = 15 N, force tolerance = 4 N
```

## 관측과 Action

관측 구성과 순서는 기본 환경과 같은 62-D지만 corruption이 비활성화되어 관절 위치와
속도 noise가 적용되지 않는다. Action도 동일한 12-D variable-stiffness OSC이며,
stiffness `[20, 300]`, 위치 scale `0.025 m`, 회전 scale `0.12 rad`를 사용한다.

## Reward, 패널티와 termination

reward와 weight는 기본 환경의 `reaching`, `target_contact`, `side_direction`,
`side_center_contact`, `force_tracking`, 진행/정렬/목표점/성공 보상과 lateral,
overshoot, off-center, torque, Action rate, 관절 속도, effort, saturation 패널티를 그대로
사용한다. 정확한 weight 표는 [기본 OSC Sweep](osc_sweep.md#reward-패널티와-termination)을
참고한다.

종료도 동일하다: 8초 timeout, endpoint `<0.025 m` 및 normalized lateral `<0.12` 성공,
비정상 물체 pose, F/T `100 N/15 Nm` 초과, 관절 속도 `6.5 rad/s` 초과다.

## 실행 방법

이 환경은 명령 randomization이 없어 별도 학습에 사용하지 않는다. 기본 환경에서 학습한
checkpoint를 `play_sweep.py`로 평가한다.

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_rl/scripts/play_sweep.py \
  --task Isaac-Sweep-Object-UR5e-OSC-Play-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p src\sweep_rl\scripts\play_sweep.py `
  --task Isaac-Sweep-Object-UR5e-OSC-Play-v0 `
  --checkpoint C:\absolute\path\to\model.pt --num_envs 1 --device cuda:0
```

학습이 필요하면 [기본 환경의 학습 명령](osc_sweep.md#학습-실행)을 사용한다.

[환경 목록으로 돌아가기](README.md)
