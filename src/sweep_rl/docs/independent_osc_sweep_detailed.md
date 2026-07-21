# Independent Detailed Reward 환경

환경 ID:

```text
Isaac-Sweep-Object-UR5e-OSC-Independent-Detailed-v0
```

이 환경은 `UR5eOscSweepIndependentEnvCfg`를 직접 상속하고 reward configuration만
`DetailedRewardsCfg`로 교체한다. 따라서 다음 항목은 부모 환경과 완전히 동일하다.

- 선반 USD, UR5e/Robotiq/F/T/contact sensor 구성과 초기 joint pose
- 56-D observation과 12-D variable-stiffness OSC action
- REACH=`0`, SWEEP=`1`, HOME=`2` phase 전환
- 선반 실측 범위를 사용하는 feasible goal sampling
- 물체 크기·질량·pose·마찰과 OSC calibration randomization
- 성공, contact loss, gripper insertion, Home 재접촉/물체 이동, F/T, arm speed,
  shelf collision, self-collision termination
- 20초 episode와 120 Hz physics / 30 Hz policy

부모 환경의 `reaching`, `contact`, `push`, `home_return` 네 composite reward 대신 각
목적을 별도 term으로 나눈 것이 이 환경의 차이다. 세분화된 TensorBoard 항목을 통해
어느 sub-objective에서 학습이 막혔는지 확인하고 term별 weight를 독립 조정할 수 있다.

## 참고 환경에서 차용한 부분

`constant_velocity_upright_random_size_home_return.md`에서 다음 설계를 차용했다.

- Home joint tracking과 평균 joint error를 분리
- Home 이탈 clearance, 재접촉, 목표 유지, 물체 속도와 변위를 별도 신호로 구성
- Home running cost와 성공 조건 일치 sparse bonus
- action rate, joint velocity, normalized OSC effort, torque saturation의 작은 penalty
- 실패로 episode를 일찍 끝내는 우회 전략을 막는 remaining-horizon penalty

그대로 가져오지 않은 부분은 다음과 같다.

- 기존 문서의 2-phase `SWEEP=0/HOME=1` 대신 현재 3-phase 값을 사용한다.
- `/Robot/.*` wildcard target sensor를 추가하지 않는다. 현재 환경의 18개 explicit
  one-body sensor를 재사용하여 PhysX filter-count 문제를 피한다.
- 기존 테이블 workspace, 높이와 command 범위 대신 현재 선반의 실측 bounds와 feasible
  sampling을 유지한다.
- 부모 환경에 이미 더 강하게 정의된 shelf/self-collision, contact-loss 및 Home grace
  termination을 약화시키거나 대체하지 않는다.
- 참고 환경의 weight를 그대로 복사하지 않고 현재 composite reward 크기와 20초
  horizon을 고려해 낮췄다.

## REACH reward

크기 `s`인 물체와 push 방향 `d`에 대한 EEF 목표는 부모 환경과 같다.

```text
stand_off = s/2 + 0.008
p_reach   = p_object - stand_off * d
p_reach.z = p_reach.z + 0.055
e         = ||p_eef - p_reach||
```

| Term | Weight | Raw value |
|---|---:|---|
| `reach_pose_tracking` | +4.0 | `exp(-(e/0.12)^2)` |
| `reach_pose_error` | -1.0 | `clamp(e/0.12, max=3)` |

두 항은 REACH에서만 활성화된다. 좌·우 pad가 TargetCube에 0.25 N을 초과하는 filtered
contact를 만들면 command가 SWEEP으로 전환하므로 두 항도 즉시 꺼진다.

## SWEEP reward와 penalty

| Term | Weight | 의미 |
|---|---:|---|
| `sweep_contact` | +1.5 | TargetCube와 좌·우 pad의 유효 접촉 |
| `sweep_velocity_tracking` | +8.0 | 접촉 중 accelerate-cruise-stop 속도 profile Gaussian |
| `sweep_forward_progress` | +2.0 | 목표 방향 속도 / commanded cruise speed, `[-1,1]` |
| `sweep_endpoint_error` | -4.0 | endpoint 거리 / sampled command 거리, 최대 2 |
| `sweep_lateral_error` | -3.0 | command 직선에서 벗어난 거리 / command 거리 |
| `sweep_overshoot` | -6.0 | endpoint를 넘은 종방향 거리 / command 거리 |
| `sweep_stopped_at_goal` | +15.0 | 위치 오차 0.030 m와 속도 0.020 m/s의 joint Gaussian |

속도 profile은 부모 `push` 내부와 같은 형태를 유지한다.

```text
처음 0.030 m : cruise speed의 25%에서 smooth acceleration
마지막 0.050 m: speed 0까지 smooth deceleration
```

단, 여기서는 velocity, progress, endpoint, lateral, overshoot와 stop을 별도 manager
term으로 분리한다. 모든 항은 SWEEP에서만 활성화된다.

## HOME reward와 penalty

| Term | Weight | 의미 |
|---|---:|---|
| `home_joint_pose` | +12.0 | canonical Home joint Gaussian, std 0.35 rad |
| `home_joint_error` | -2.0 | 평균 절대 joint error / 0.75 rad, 최대 3 |
| `home_clearance` | +2.0 | EEF-target 거리가 0.22 m까지 증가하는 smoothstep |
| `post_goal_contact` | -10.0 | robot 어느 rigid body든 target과 재접촉 |
| `goal_hold_error` | -8.0 | target endpoint error / command distance, 최대 2 |
| `post_goal_object_speed` | -2.0 | object speed / 0.05 m/s, 최대 4 |
| `post_goal_object_displacement` | -6.0 | Home 진입 pose 대비 변위 / 0.010 m, 최대 4 |
| `home_time` | -0.3 | Home에서 머무르는 매초의 running cost |
| `home_success` | +30.0 | 최종 success termination의 dwell 전 조건과 일치 |

`home_success`는 Home joint error, joint speed, endpoint, object speed, Home 진입 후
변위와 robot-target 비접촉을 동시에 검사한다. 실제 성공 termination은 동일 조건을
0.25초 유지해야 하므로 sparse reward가 termination을 대체하지 않는다.

## 공통 safety/regularization penalty

| Term | Weight | 의미 |
|---|---:|---|
| `ft_torque` | -0.02 | wrist torque norm 중 1.5 Nm 초과분 |
| `action_rate` | -0.01 | 연속 policy action 변화의 L2 제곱 |
| `joint_velocity` | -0.001 | 6개 arm joint velocity L2 제곱 |
| `commanded_effort` | -0.01 | effort limit으로 정규화한 OSC torque L2 제곱 |
| `torque_saturation` | -0.5 | action clipping, NaN/Inf 또는 torque clamp indicator |
| `failure_termination` | -5.0 | 실패 시 남은 episode 시간, 최소 1초 |

`failure_termination`은 timeout과 success에는 적용하지 않는다. contact loss, gripper
insertion, Home 재접촉/물체 이동, invalid target, F/T, arm speed, shelf collision과
self-collision에만 적용한다.

## Termination

새 termination을 중복 정의하지 않고 부모 `TerminationsCfg`를 상속한다.

| 구분 | 부모에서 유지되는 조건 |
|---|---|
| 성공 | HOME, Home joint/저속, endpoint/물체 저속, 비접촉, parked 변위 조건 0.25초 유지 |
| 작업 실패 | SWEEP contact loss, gripper 내부 삽입, HOME 재접촉, parked object 이동 |
| 물체 안전 | 높이/tilt 제한 |
| 로봇 안전 | F/T wrench, arm speed, shelf collision, non-adjacent self-collision |
| 제한 시간 | 20초 timeout |

## 학습

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_independent_sweep_detailed.py \
  --num_envs 2048 --device cuda:0 --headless
```

별도 experiment 이름은 다음과 같다.

```text
ur5e_osc_sweep_independent_detailed
```

관측과 action shape는 부모와 같아 network shape는 호환되지만 reward objective가 크게
달라진다. 기존 부모 checkpoint를 초기화 용도로 사용할 수는 있어도 optimizer와 value
function을 그대로 resume하는 것보다는 새 run 또는 명시적인 fine-tuning으로 다루는 것이
안전하다.
