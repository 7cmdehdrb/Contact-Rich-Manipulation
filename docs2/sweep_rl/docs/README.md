# Sweep RL 환경 문서

이 디렉터리는 `sweep_rl`에 등록된 UR5e OSC sweep 환경의 기준 문서다. 모든 환경은
Isaac Lab의 Manager-based 환경이며, UR5e와 Robotiq 2F-85를 사용해 물체를 지정한
방향과 거리만큼 미는 정책을 학습하거나 평가한다.

## 공통 실행 전제

명령은 저장소 루트에서 실행한다. 먼저 패키지를 editable mode로 설치한다.

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p -m pip install -e src\sweep_rl
```

문서의 학습 예시는 GPU 0에서 2,048개 환경을 headless로 실행한다. GPU 메모리에
맞춰 `--num_envs`를 줄일 수 있다. 체크포인트는 기본적으로
`logs/rsl_rl/<experiment_name>/<run>/model_*.pt`에 저장된다.

## 환경 목록

### Force-command 계열

| 환경 | 목적 | 관측 |
|---|---|---:|
| [`Isaac-Sweep-Object-UR5e-OSC-v0`](osc_sweep.md) | 접촉력과 목표 위치를 함께 추종하는 기본 sweep 학습 | 62-D |
| [`Isaac-Sweep-Object-UR5e-OSC-Play-v0`](osc_sweep_play.md) | 기본 정책을 고정 명령으로 재현하는 평가 전용 환경 | 62-D |
| [`Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0`](wide_randomization.md) | 물체 질량과 목표 접촉력 범위를 넓힌 강건성 학습 | 62-D |
| [`Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0`](tactile_localization.md) | 현재 물체 pose 없이 촉각·힘 정보로 물체를 찾고 미는 학습 | 56-D |

Force-command는 다음 5-D 명령을 사용한다.

```text
[direction_x, direction_y, distance_m, target_force_N, force_tolerance_N]
```

### Constant-velocity 계열

| 환경 | 목적 | 관측 |
|---|---|---:|
| [`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0`](constant_velocity.md) | 목표 속도 profile을 따라 물체를 밀고 목표점에 정지 | 55-D |
| [`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0`](constant_velocity_upright_random_size.md) | gripper 외측 pad 접근을 유도하고 내부 삽입을 금지 | 55-D |
| [`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0`](constant_velocity_home_return.md) | sweep 완료 후 물체를 건드리지 않고 Home pose로 복귀 | 56-D |
| [`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-Can-v0`](constant_velocity_home_return_can.md) | HomeReturn 정책을 Can_6 물체에서 평가 | 56-D |

Constant-velocity는 다음 4-D 명령을 사용한다.

```text
[direction_x, direction_y, distance_m, target_speed_mps]
```

`UprightRandomSize`는 기존 checkpoint와 실행 경로의 호환성을 위해 남아 있는 이름이다.
현재 이 이름을 사용하는 constant-velocity 환경에는 upright 자세 보상이나 물체 크기
랜덤화가 없다.

### Independent shelf 계열

| 환경 | 목적 | 관측 |
|---|---|---:|
| [`Isaac-Sweep-Object-UR5e-OSC-Independent-v0`](independent_osc_sweep.md) | 선반 장면에서 Reach → Sweep → Home 전체 절차를 독립 구현으로 학습 | 56-D |
| [`Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0`](independent_osc_sweep_detailed.md) | 같은 과제를 단계별 상세 reward로 학습·진단 | 56-D |

Independent 계열은 4-D 속도 명령과 별도의 phase 관측을 사용하고, 물체 크기·질량·마찰
및 OSC calibration을 랜덤화한다.

## 공통 제어 주기와 Action

모든 환경은 physics 120 Hz, `decimation=4`, 정책 30 Hz로 동작한다. 정책 Action은
다음 12-D variable-stiffness OSC 명령이다.

```text
[Kx, Ky, Kz, Kroll, Kpitch, Kyaw,
 dx, dy, dz, droll, dpitch, dyaw]
```

- Action은 `[-1, 1]`로 제한된다.
- stiffness 6축은 `[20, 300]`으로 선형 변환된다.
- 상대 위치는 축별 최대 `0.025 m`, 상대 회전은 축별 최대 `0.12 rad`다.
- damping ratio는 1.0이며 UR5e 관절 torque로 변환된다.
- Constant-velocity와 Independent 계열은 gripper를 정책과 무관하게 완전히 연 상태로
  유지한다.

환경별 차이, 전체 관측 순서, reward weight, termination 조건과 Windows/Ubuntu 실행
명령은 위 표의 개별 문서를 따른다.

문서 기준: 2026-07-22 현재 구현.
