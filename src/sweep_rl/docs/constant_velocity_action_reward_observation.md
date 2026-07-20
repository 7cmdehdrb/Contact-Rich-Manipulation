# ConstantVelocity 환경의 Action, Observation, Reward

대상 환경은 `Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0`이다. 이 문서는
현재 `env_cfg_constant_velocity.py`와 연결된 MDP 구현을 기준으로 작성되었다.
파생 Gripper Exclusion 환경은 `push_pose_error`와 termination 하나를 바꾸고,
HomeReturn 환경은 phase observation과 HOME 전용 항을 추가한다. 파생 환경 차이는
[등록 환경 문서](registered_osc_sweep_environments.md)의 8–9절을 따른다.

## Task command

에피소드마다 다음 4차원 명령을 한 번 샘플링한다.

```text
[direction_x, direction_y, distance_m, target_speed_mps]
```

| 항목 | 범위 |
|---|---:|
| 평면 진행 방향 | `[-pi, pi]`에서 샘플링한 단위 벡터 |
| 이동 거리 | `0.10 ~ 0.22 m` |
| 목표 순항 속도 | `0.08 m/s` 고정 |

목표 속도 프로파일은 출발 시 순항 속도의 25%에서 시작해 첫 `0.025 m` 동안
부드럽게 가속하고, 목표점 전 마지막 `0.04 m`에서 0으로 감속한다.

## Action

정책 Action은 12차원이며 모든 입력은 `[-1, 1]`로 제한된다.

```text
[Kx, Ky, Kz, Kroll, Kpitch, Kyaw,
 dx, dy, dz, droll, dpitch, dyaw]
```

| 인덱스 | 차원 | 의미 | 변환 |
|---:|---:|---|---|
| `0:6` | 6 | OSC task-space stiffness | `[-1, 1] -> [20, 300]` 선형 변환 |
| `6:9` | 3 | EEF 상대 위치 명령 | 축별 최대 `0.025 m` |
| `9:12` | 3 | EEF 상대 RPY 명령 | 축별 최대 `0.12 rad`, 이후 axis-angle로 변환 |

OSC는 pose relative/variable stiffness 모드이며 damping ratio는 전 축 1.0이다.
팔 관절 effort는 모델의 effort limit 중 90%에서 clamp된다. 그리퍼는 정책 Action에
포함되지 않고 매 physics step마다 완전 개방 위치 `0.0`을 다시 명령한다.

학습 초기의 과도한 wrench 종료를 줄이기 위해 PPO Gaussian policy의 초기 표준편차는
`0.5`이다.

## Observation

정책 Observation은 아래 순서로 concatenate한 55차원 벡터다.

| 순서 | 항목 | 차원 | 좌표계/설명 | Noise |
|---:|---|---:|---|---|
| 1 | `joint_pos` | 6 | 팔 관절 위치 | uniform `±0.002` |
| 2 | `joint_vel` | 6 | 팔 관절 속도 | uniform `±0.01` |
| 3 | `joint_effort` | 6 | 팔 관절 effort | 없음 |
| 4 | `eef_pose` | 6 | robot base 기준 EEF `xyz + RPY` | 없음 |
| 5 | `initial_target_pose` | 6 | reset 시 물체 pose, robot base 기준 | 없음 |
| 6 | `current_target_pose` | 6 | 현재 물체 pose, robot base 기준 | 없음 |
| 7 | `object_linear_velocity` | 3 | 현재 물체 선속도, robot base 기준 | uniform `±0.005` |
| 8 | `desired_motion` | 4 | 방향 2 + 거리 + 목표 속도 | 없음 |
| 9 | `last_action` | 12 | 직전 정책 Action | 없음 |
| | **합계** | **55** | | |

F/T wrench, 접촉점, 목표 접촉력, 접촉력 tolerance는 정책에 제공하지 않는다. 접촉
센서는 보상 계산에만 사용한다.

## Reward

Isaac Lab Reward Manager는 매 step 아래 각 항을 `raw_value * weight * step_dt`로
합산한다. 이 환경은 physics `120 Hz`, decimation 4이므로 정책 step은 `30 Hz`,
`step_dt = 1/30 s`다.

### 접근과 접촉

| Reward term | Weight | 의미 |
|---|---:|---|
| `push_pose_error` | `-0.35` | EEF와 현재 물체 뒤쪽 push pose 사이 거리 오차 |
| `side_direction_error` | `-0.25` | 물체 근처에서 gripper 넓은 면과 목표 방향의 불일치 |
| `target_contact` | `+0.50` | 좌/우 pad가 목표 물체에 접촉했는지 나타내는 작은 binary bridge |
| `side_center_contact` | `+0.75` | 넓은 pad 중앙의 양질 접촉 |
| `contact_forward_progress` | `+3.0` | 접촉 중 물체가 목표 방향으로 실제 전진할 때의 보상 |

`target_contact + side_center_contact`의 최대 합은 초당 `+1.25`로 제한된다. 물체가
정지하면 초당 `-5.0` 수준인 endpoint cost와 stall cost가 더 크므로, 접촉만 하고
멈춰 있는 정책은 이득이 아니다. 큰 접촉 보상 `contact_forward_progress`는 실제
전진 속도에 비례하며 목표점 부근의 감속 프로파일을 따라 0으로 줄어든다.

### 속도와 목표점

| Reward term | Weight | 의미 |
|---|---:|---|
| `velocity_tracking` | `+10.0` | 가속-순항-감속 목표 속도 추종. 실제 전진이 0이면 reward도 0 |
| `endpoint_error` | `-5.0` | 목표점 거리 / 명령 거리, 최대 2로 clamp한 매-step 비용 |
| `stopped_at_goal` | `+20.0` | 위치 오차와 물체 속도가 모두 작은 경우의 연속 Gaussian 보상 |
| `success` | `+40.0` | endpoint `< 0.020 m`, normalized lateral error `< 0.10`, speed `< 0.020 m/s` |

`endpoint_error`는 물체가 목표점에 가까워질수록 직접 감소한다. 따라서 mean reward의
증가가 endpoint 개선과 연결되도록 구성되어 있다.

### 실패와 동작 규제

| Reward term | Weight | 의미 |
|---|---:|---|
| `failure_termination` | `-8.0` | 비정상 물체 pose, 과도한 wrench, arm speed 종료의 남은 horizon 비용 |
| `lateral_error` | `-3.0` | 명령 방향에 수직인 이동 거리 / 명령 거리 |
| `overshoot` | `-8.0` | 목표 이동 거리를 넘긴 종방향 overshoot |
| `stall` | `-6.0` | 시작 0.40 s 후 목표점 전에서 목표 속도의 50%에 못 미치는 속도 부족 |
| `object_acceleration` | `-0.15` | 물체 선가속도 제곱, raw value 최대 25 |
| `ft_torque` | `-0.02` | F/T torque norm이 `1.5 Nm`를 넘는 양 |
| `action_rate` | `-0.02` | 연속 Action 변화량 제곱 |
| `joint_velocity` | `-0.002` | 팔 관절 속도 제곱 |
| `commanded_effort` | `-0.03` | effort limit로 정규화한 OSC 관절 torque 제곱 |
| `torque_saturation` | `-0.5` | Action clipping/비정상 값 또는 OSC torque saturation indicator |

조기 안전 종료 비용은 다음과 같다.

```text
failure_cost = -8.0 * max(남은 에피소드 시간, 1.0 s)
```

에피소드 초반에 스스로 wrench 종료를 일으켜 이후의 endpoint/stall 비용을 회피할 수
없도록, 종료 시 남은 8초 horizon의 비용을 한 번에 청구한다. 정상 timeout과 성공
종료에는 이 항을 적용하지 않는다.

## Termination과 학습 지표

에피소드 최대 길이는 8초다. 성공은 위 성공 조건을 `0.30 s` 연속 유지할 때 발생한다.
안전 종료는 물체 높이/기울기 이상, F/T force `100 N` 또는 torque `15 Nm` 초과,
팔 관절 속도 `6.5 rad/s` 초과다.

TensorBoard command metric은 다음 네 항이다.

| Metric | 의미 |
|---|---|
| `endpoint_error` | 현재 물체 중심과 목표점의 거리 `[m]` |
| `speed_error` | 목표 순항 속도와 전진 속도의 절대 오차 `[m/s]` |
| `forward_speed` | 명령 방향으로의 signed 물체 속도 `[m/s]` |
| `progress_ratio` | 종방향 이동 거리 / 명령 거리 |

수렴 판단에서는 mean reward만 보지 말고 `endpoint_error` 감소,
`progress_ratio -> 1`, success 비율 증가, `excessive_wrench` 종료 비율 감소를 함께
확인해야 한다.

문서 내용은 2026-07-19 현재 코드 기준이다.
