# 로봇과 센서

## Robot assembly

두 환경 계열은
[`make_ur5e_robotiq_ft_cfg()`](../../src/sweep_rl/sweep_rl/osc_sweep/assets.py)가 반환하는 하나의
`ArticulationCfg`를 공유한다. 조립 구조는 다음과 같다.

```text
UR5e tool rigid body
  └─ fixed joint
      VirtualFTSensor
        ├─ fixed joint → Robotiq 2F-85 base
        └─ fixed joint → SweepToolCenter
                              ├─ fixed joint → LeftSweepContactPad
                              └─ fixed joint → RightSweepContactPad
```

UR5e와 Robotiq은 각각 USD에서 불러온 뒤
[`spawn_ur5e_robotiq_ft()`](../../src/sweep_rl/sweep_rl/osc_sweep/assets.py)에서 조립한다.

- UR5e tool frame 후보를 찾아 가장 가까운 rigid-body ancestor를 선택한다.
- Robotiq base rigid body를 찾고 tool frame에 맞게 reference root transform을 조정한다.
- gripper의 중첩 `ArticulationRootAPI`를 제거해 UR5e articulation 안에 편입한다.
- 과거 mount joint를 비활성화하고 새 fixed joint를 만든다.
- geometry가 아닌 prim에 잘못 붙은 CollisionAPI를 제거한다.
- virtual F/T body, OSC center body, 좌/우 pad를 code로 생성해 fixed joint로 연결한다.

기본 USD 경로는 Nucleus를 가리키지만 environment variable로 교체할 수 있다.

| 변수 | 대상 |
|---|---|
| `SWEEP_UR5E_USD_PATH` | UR5e USD |
| `SWEEP_ROBOTIQ_USD_PATH` | Robotiq 2F-85 USD |

```bash
export SWEEP_UR5E_USD_PATH=/absolute/path/to/ur5e.usd
export SWEEP_ROBOTIQ_USD_PATH=/absolute/path/to/robotiq.usd
```

교체한 USD에서는 tool/base 후보 이름과 rigid-body hierarchy가 기존 assembly의 검색 조건과
호환되어야 한다.

## OSC 기준 frame

`SweepToolCenter`는 열린 gripper의 물리적 중심을 나타내는 비충돌 rigid link다.

| 속성 | 값 |
|---|---:|
| body 이름 | `SweepToolCenter` |
| gripper base offset | `(0, 0, 0.16) m` |
| 형상 | `0.008 m` cube |
| 질량 | `1e-3 kg` |
| collision | 비활성 |

Action term은 이 body의 pose/Jacobian을 기준으로 12-D variable-stiffness OSC 명령을 joint
torque로 바꾼다. 따라서 실제 finger link의 origin이 아니라 열린 gripper 중앙이 정책의
EEF가 된다.

## 가상 F/T 센서

`VirtualFTSensor`는 Isaac Lab `ContactSensor`가 아니다. UR5e tool과 gripper 사이에 삽입한
작은 비충돌 rigid body와 fixed joint의 reaction wrench를 읽는 방식이다.

| 속성 | 값 |
|---|---:|
| body 이름 | `VirtualFTSensor` |
| 형상 | `0.025 m` cube |
| 질량 | `1e-3 kg` |
| collision | 비활성 |
| 출력 | `[Fx, Fy, Fz, Tx, Ty, Tz]` |

관측 구현은
[`virtual_ft_wrench_b()`](../../src/sweep_rl/sweep_rl/osc_sweep/mdp/observations.py)다.

```python
wrench = -robot.data.body_incoming_joint_wrench_b[:, body_id, :]
```

부호를 반전해 sensor frame에서 gripper가 받는 wrench convention에 맞춘다. 이 wrench는
target 접촉만 분리한 값이 아니라 sensor 아래쪽 gripper로 전달되는 전체 joint reaction이다.
target-specific 접촉 여부와 접촉점은 아래 ContactSensor를 사용한다.

F/T는 정책 관측, torque penalty, `100 N/15 Nm` 안전 termination에 사용된다. Constant
Velocity 일부 환경은 policy observation에서 F/T를 빼더라도 reward/termination 계산에는
계속 사용한다.

## 좌·우 sweep pad ContactSensor

assembly는 열린 gripper 바깥쪽에 두 개의 collision pad를 만든다.

| 속성 | 값 |
|---|---:|
| body | `LeftSweepContactPad`, `RightSweepContactPad` |
| 크기 | `(0.020, 0.030, 0.055) m` |
| lateral offset | `±0.055 m` |
| 질량 | `0.01 kg` |
| static/dynamic friction | `0.9 / 0.7` |

기본 scene의 `left_contact`, `right_contact`는 각 pad body에 붙고
`{ENV_REGEX_NS}/TargetCube`만 filter한다. `track_pose=True`,
`track_contact_points=True`이므로 다음 데이터를 얻는다.

- `force_matrix_w`: filter target에 대한 접촉 force
- `contact_pos_w`: 접촉점
- pad pose: 접촉점의 local 위치/품질 계산

[`target_contact_data_w()`](../../src/sweep_rl/sweep_rl/osc_sweep/mdp/common.py)는 두 pad의 force를
합하고 force 크기로 가중한 접촉점을 계산한다. 총 유효 force가 기본 `0.25 N`을 넘을 때만
contact로 판정한다.

## HomeReturn의 target-side sensor

HomeReturn은 target object 자체에 `target_robot_contact` sensor를 추가하고 robot의 모든
body를 filter한다. 이 sensor는 sweep 완료 후 arm이나 gripper 어느 부분이라도 물체에
계속 닿는지 검사한다.

- 설정: [`SweepHomeSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg_constant_velocity_upright_random_size_home.py)
- 필수 조건: target USD spawn에 `activate_contact_sensors=True`
- 사용: HOME contact penalty, object disturbance failure, 비접촉 성공 조건

sensor prim이 target이므로 custom USD로 바꿀 때도 target의 rigid body에 PhysX contact
reporter가 활성화되어야 한다.

## Independent 전신 ContactSensor

Independent scene은 base, shoulder, arm links, wrists, Robotiq links와 두 pad에 각각
ContactSensor를 둔다. 각 sensor의 filter 순서는 다음과 같다.

| Filter index | 대상 | 용도 |
|---:|---|---|
| 0 | `{ENV_REGEX_NS}/TargetCube` | target 접촉 |
| 1 | `{ENV_REGEX_NS}/Shelf/rack` | robot-shelf collision |
| 2 이후 | 모든 robot body path | self-collision |

직접 연결되어 원래 겹치거나 fixed/revolute joint로 이어진 link pair는 self-collision
termination에서 제외하고 나머지 비인접 충돌은 실패로 처리한다. 구현은
[`ROBOT_CONTACT_FILTERS`, `make_robot_body_contact_sensor()`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/env_cfg.py)와
[`robot_shelf_collision()`, `robot_self_collision()`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/terminations.py)에
있다.

새 shelf USD를 사용할 때 `Shelf/rack` prim 이름이 바뀌면 filter 1이 아무것도 감지하지
않는다. 실제 collision rigid prim 경로에 맞춰 `ROBOT_CONTACT_FILTERS`를 갱신해야 한다.

## 센서 검증 순서

1. `--num_envs 1`과 GUI로 실행한다.
2. 접촉 전 F/T와 filtered contact가 0 근처인지 확인한다.
3. 좌/우 pad를 target에 각각 접촉시켜 해당 sensor만 반응하는지 본다.
4. target 이외 table/shelf 접촉이 target-filtered pad sensor에 섞이지 않는지 확인한다.
5. Independent에서는 shelf collision과 허용/비허용 self-collision pair를 따로 검사한다.
6. custom USD에서는 `activate_contact_sensors`, rigid root, filter path를 다시 확인한다.

[Environment 구축 문서로 돌아가기](README.md)
