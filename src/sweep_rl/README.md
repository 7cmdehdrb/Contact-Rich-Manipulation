# Sweep RL

UR5e와 Robotiq 2F-85를 사용하는 Isaac Lab Manager-based 강화학습 패키지다.
물체 밀기 환경과 함께 Shelf 내부의 가상 pose를 추종하는 joint-position Reach
환경을 제공한다.

## 환경 선택

| 목적 | 환경 ID | Observation |
|---|---|---:|
| force 기반 기본 sweep | `Isaac-Sweep-Object-UR5e-OSC-v0` | 62-D |
| 기본 정책 소규모 재생 | `Isaac-Sweep-Object-UR5e-OSC-Play-v0` | 62-D |
| 질량·접촉력 범위 확장 | `Isaac-Sweep-Object-UR5e-OSC-WideRandomization-v0` | 62-D |
| 현재 물체 pose 없는 촉각 위치 추론 | `Isaac-Sweep-Object-UR5e-OSC-TactileLocalization-v0` | 56-D |
| 일정 속도 sweep | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0` | 55-D |
| 외측 pad 접근과 gripper 내부 삽입 금지 | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0` | 55-D |
| sweep 후 물체를 건드리지 않고 Home 복귀 | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0` | 56-D |
| 기존 HomeReturn checkpoint로 Can_6 재생 | `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-Can-v0` | 56-D |
| 선반에서 Reach → Sweep → Home 통합 학습 | `Isaac-Sweep-Object-UR5e-OSC-Independent-v0` | 56-D |
| Independent 과제의 상세 reward 학습 | `Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0` | 56-D |
| Shelf 내부 UR5e+Gripper TCP Reach | `Isaac-Reach-Shelf-UR5e-Gripper-v0` | 25-D |
| Shelf Reach 소규모 재생 | `Isaac-Reach-Shelf-UR5e-Gripper-Play-v0` | 25-D |
| Cube 밀기 전 -Y 사전접근 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-v0` | 29-D |
| Cube 사전접근 소규모 재생 | `Isaac-Reach-Shelf-UR5e-Gripper-CubePreReach-Play-v0` | 29-D |

`UprightRandomSize`라는 ID는 기존 실행 경로 호환성을 위해 유지한다. 현재 해당
환경에는 upright 자세 보상이나 물체 크기 랜덤화가 없다.

환경별 목적과 전체 계약은 [환경 문서 허브](docs/README.md)를 기준으로 한다.

## 설치와 학습

Linux:

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/sweep_rl
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size_home.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p -m pip install -e src\sweep_rl
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\train_constant_velocity_upright_random_size_home.py `
  --num_envs 2048 --device cuda:0 --headless
```

각 전용 학습 스크립트는 `--task`를 생략하면 대응 환경 ID를 자동으로 넣는다.

## 재생

Force-command 환경은 목표 pose와 목표/측정 접촉력을 표시하는
`src/sweep_rl/scripts/play_sweep.py`를 사용할 수 있다. ConstantVelocity 계열은
4-D speed command를 사용하므로 Isaac Lab 표준 `rsl_rl/play.py`를 사용한다.

```bash
./IsaacLab/isaaclab.sh -p \
  IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0 \
  --checkpoint /absolute/path/to/model.pt --num_envs 1 --device cuda:0
```

## 문서

환경별 목적, 관측, Action, reward·패널티·termination과 Ubuntu/Windows 실행법은
[Sweep RL 환경 문서](docs/README.md)에 정리되어 있다. Shelf Reach의 세부 계약과
실행법은 [`shelf_reach/README.md`](sweep_rl/shelf_reach/README.md)를 참고한다.
Cube 사전접근 환경은
[`shelf_cube_pre_reach/README.md`](sweep_rl/shelf_cube_pre_reach/README.md)에
정리되어 있다.

문서 내용은 2026-08-07 현재 구현 기준이다.
