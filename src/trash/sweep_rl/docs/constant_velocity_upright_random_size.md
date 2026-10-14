# Constant Velocity UprightRandomSize

환경 ID: `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-v0`

## 목적

[Constant Velocity](constant_velocity.md)를 상속해 gripper 내부로 물체를 끼우는 shortcut을
막고, 열린 gripper의 외측 pad로 접근하도록 유도한다. 환경 ID의 `UprightRandomSize`는
호환성용 이름이며 현재 upright 보상이나 물체 크기 randomization은 없다.

## 관측과 Action

부모 환경과 같은 55-D 관측을 사용한다: arm 상태 18-D, EEF pose 6-D, 초기/현재 물체
pose 12-D, 물체 속도 3-D, 4-D 명령, 이전 Action 12-D. noise와 관측 순서도 같다.

Action은 열린 gripper를 유지하는 12-D variable-stiffness OSC이며 stiffness `[20, 300]`,
위치 `0.025 m`, 회전 `0.12 rad`, effort scale 0.9다.

## Reward와 패널티

[부모 환경의 reward 표](constant_velocity.md#reward와-패널티)를 그대로 사용하되
`push_pose_error`만 교체한다.

| 항목 | Weight | 역할 |
|---|---:|---|
| `push_pose_error` | -1.0 | 물체와 가까운 좌/우 외측 pad의 pre-contact pose 오차; stand-off `0.065 m`, pad center offset `0.055 m`, scale `0.10 m` |

나머지 contact, 속도, endpoint, success 보상과 lateral, overshoot, stall, 가속도 및 제어
패널티는 Constant Velocity와 동일하다. `failure_termination`에는 아래 삽입 실패도 포함해
남은 horizon 비용을 부과한다.

## Termination

부모의 성공·timeout·물체 pose·F/T·arm 속도 조건에 `object_inside_gripper` 실패를
추가한다. 물체 중심이 EEF-local half extents `(0.040, 0.040, 0.058) m`인 box 안에
들어오면 종료한다.

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\train_constant_velocity_upright_random_size.py `
  --num_envs 2048 --device cuda:0 --headless
```

experiment 이름은 `ur5e_osc_sweep_constant_velocity_upright_random_size`다.

[환경 목록으로 돌아가기](README.md)
