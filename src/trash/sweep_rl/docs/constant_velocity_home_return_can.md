# Constant Velocity Home Return Can

환경 ID:
`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-Can-v0`

## 목적

[`HomeReturn`](constant_velocity_home_return.md) checkpoint를 cube 대신 `Can_6.usd` 물체에서
평가하는 1-env playback 변형이다. 기본 질량은 `0.35 kg`이며, rigid root가 캔 바닥에
있기 때문에 초기/현재 pose 관측의 Z에 캔 높이의 절반인 약 `0.059565 m`를 더해 중심
pose로 맞춘다. scene에 사설 Omniverse asset 경로가 필요하다.

## 관측과 Action

HomeReturn과 같은 56-D 관측이다. 단, `initial_target_pose`와 `current_target_pose`의 Z만
캔 중심 offset을 반영한다. Action은 동일한 12-D variable-stiffness OSC이고 gripper는
열린 상태다.

## Reward, 패널티와 termination

HomeReturn의 SWEEP/HOME reward, 모든 weight, phase 전환과 termination을 그대로 사용한다.
즉 속도 profile과 endpoint 정지, 외측 pad 접근, Home joint pose, 비접촉과 물체 보존을
평가하며, 성공 기준도 joint 오차/속도, endpoint, 물체 속도·이동, 비접촉 0.25초다.

전체 표는 [HomeReturn 문서](constant_velocity_home_return.md#reward와-패널티)를 참고한다.

## 실행 방법

이 환경은 HomeReturn의 PPO config를 재사용하며 전용 학습 스크립트가 없는 checkpoint
호환 평가 환경이다. 먼저 [HomeReturn](constant_velocity_home_return.md#학습-실행)에서
cube 정책을 학습한 뒤 아래 전용 player로 평가한다. `--object_mass`와
`--target_z_offset`을 필요에 따라 바꾼다.

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_constant_velocity_home_can.py \
  --checkpoint /absolute/path/to/model.pt --object_mass 0.35
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\play_constant_velocity_home_can.py `
  --checkpoint C:\absolute\path\to\model.pt --object_mass 0.35
```

`--checkpoint`를 생략하면 스크립트에 정의된 기본 checkpoint 경로를 사용하므로 다른
시스템에서는 명시적으로 지정하는 것이 안전하다.

[환경 목록으로 돌아가기](README.md)
