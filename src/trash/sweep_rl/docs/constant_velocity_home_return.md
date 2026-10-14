# Constant Velocity Home Return

환경 ID:
`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-UprightRandomSize-HomeReturn-v0`

## 목적과 phase

외측 pad로 물체를 목표점까지 민 뒤 접촉을 끊고 UR5e의 기본 Home joint pose로 복귀하는
2단계 환경이다.

- `SWEEP(0)`: Constant Velocity + gripper exclusion 학습
- `HOME(1)`: endpoint `<0.020 m`, 물체 속도 `<0.020 m/s`를 0.30초 유지하면 전환

전환 후 sweep 전용 reward는 꺼지고, 물체를 그대로 둔 채 비접촉 Home 복귀를 학습한다.
에피소드 길이는 12초다.

## 관측과 Action

부모의 55-D 관측에 `task_phase` 1-D를 마지막 Action 앞이 아닌 상속 정의의 끝에 추가한
56-D 벡터를 사용한다. 구성은 arm 상태 18-D, EEF 6-D, 초기/현재 물체 pose 12-D,
물체 속도 3-D, 명령 4-D, 이전 Action 12-D, phase 1-D다.

Action은 부모와 같은 12-D variable-stiffness OSC이고 gripper는 열린 상태를 유지한다.

## Reward와 패널티

SWEEP phase에서는 [UprightRandomSize reward](constant_velocity_upright_random_size.md#reward와-패널티)를
사용한다. HOME phase에는 다음 항을 사용한다.

| HOME 항목 | Weight | 역할 |
|---|---:|---|
| `home_joint_pose` | +15.0 | 기본 arm joint pose 추종 |
| `home_joint_error` | -3.0 | Home joint 오차 |
| `home_clearance` | +3.0 | EEF-물체 거리 `0.22 m` 확보 |
| `post_goal_contact` | -12.0 | HOME에서 robot-물체 접촉 |
| `goal_hold_error` | -10.0 | 물체 endpoint 이탈 |
| `post_goal_object_speed` | -3.0 | HOME에서 물체 속도 |
| `post_goal_object_displacement` | -8.0 | park 시점 이후 물체 이동 |
| `home_time` | -0.5 | HOME phase 시간 비용 |
| `home_success` | +50.0 | 안정적 비접촉 Home 완료 |

`object_acceleration`, F/T torque, Action rate, 관절 속도, effort, saturation 규제와 안전
실패 패널티는 양 phase에서 유지된다.

## Termination

- 성공: 모든 arm joint 오차 `<0.12 rad`, 속도 `<0.15 rad/s`, endpoint `<0.025 m`,
  물체 속도 `<0.025 m/s`, park 이후 이동 `<0.010 m`, robot-물체 비접촉을 0.25초 유지
- HOME 실패: park 이후 이동 `>0.015 m` 또는 물체 속도 `>0.10 m/s`
- gripper 삽입, 물체 pose, F/T, arm 속도 실패와 12초 timeout은 부모 조건을 유지

## 학습 실행

Ubuntu:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity_upright_random_size_home.py \
  --num_envs 2048 --device cuda:0 --headless
```

Windows PowerShell:

```powershell
.\IsaacLab\isaaclab.bat -p `
  src\sweep_rl\scripts\train_constant_velocity_upright_random_size_home.py `
  --num_envs 2048 --device cuda:0 --headless
```

experiment 이름은 `ur5e_osc_sweep_constant_velocity_upright_random_size_home`다.

[환경 목록으로 돌아가기](README.md)
