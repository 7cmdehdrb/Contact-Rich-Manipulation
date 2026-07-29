# Wrist 가상 F/T 센서 구현 정리

이 문서는 IsaacLab 환경에서 UR5e Wrist에 가상 6축 Force/Torque 센서를
구현한 방식을 정리한다. 

관련 구현은 아래 소스 코드를 참조.

- [`osc_sweep/assets.py`](../sweep_rl/sweep_rl/osc_sweep/assets.py)
- [`osc_sweep/mdp/observations.py`](../sweep_rl/sweep_rl/osc_sweep/mdp/observations.py)

## 구현 개요

이 구현에서는 Isaac Lab의 `ContactSensor`를 Wrist에 **부착하지 않는다**. UR5e의
tool body와 Robotiq gripper 사이에 작은 rigid body를 삽입하고, 이 body를
연결하는 **fixed joint의 reaction wrench**를 읽는다.

전체 구조는 다음과 같다.

```text
UR5e wrist / tool body
        │
        │ fixed joint
        │ (reaction wrench 측정)
        ▼
VirtualFTSensor
        │
        │ fixed joint
        ▼
Robotiq 2F-85 gripper
```

fixed joint는 부모와 자식 body 사이의 상대 위치와 자세를 고정한다. PhysX는 이
구속조건을 유지하기 위해 joint를 통해 전달되어야 하는 3축 힘과 3축 모멘트를
계산한다. 이 reaction wrench를 가상 F/T 센서의 출력으로 사용한다.

따라서 이 센서는 특정 접촉점의 힘을 직접 읽는 접촉 센서가 아니다. Wrist와
gripper 사이를 통과하는 전체 기계적 하중을 읽는 **6축 로드셀**에 가깝다.

## 센서 구성

### 센서 위치

`VirtualFTSensor`는 UR5e의 tool frame 위치에 배치하고, UR5e와 Robotiq gripper
사이에 직렬로 연결한다. 기존 gripper 연결은 중복 구속이 생기지 않도록 정리한 뒤,
UR5e–센서–gripper가 하나의 articulation을 이루도록 구성한다.

센서 frame은 tool frame과 같은 위치와 방향을 사용한다. 이 때문에 센서 출력은
Wrist 말단에 고정된 local frame의 값으로 해석한다.

### `VirtualFTSensor` body

[`Ur5eRobotiqFtSpawnerCfg`](../sweep_rl/sweep_rl/osc_sweep/assets.py)에서 생성하는
가상 센서 body의 기본 설정은 다음과 같다.

| 항목 | 값 | 의미 |
| --- | ---: | --- |
| 이름 | `VirtualFTSensor` | 센서 body를 식별하는 이름 |
| 크기 | `0.025 × 0.025 × 0.025 m` | 2.5 cm 크기의 cuboid |
| 질량 | `0.001 kg` | articulation에 포함되는 작은 질량 |
| 충돌 | 비활성화 | 센서 body 자체는 접촉을 만들지 않음 |
| 중력 | 활성화 | 일반 rigid body로 동역학 계산에 포함됨 |

센서를 별도의 rigid body로 만든 이유는 articulation 내부에서 incoming joint
wrench를 읽기 위해서다. 질량은 0이 아니므로 센서 body 자체의 미세한 중력 및
관성 하중도 계산에 포함될 수 있다.

### Fixed joint 연결

센서 body의 앞뒤에는 다음 두 fixed joint를 둔다.

1. `UR5e_virtual_ft_parent_joint`: UR5e tool body → `VirtualFTSensor`
2. `VirtualFTSensor_gripper_child_joint`: `VirtualFTSensor` → Robotiq base

측정에는 첫 번째 joint, 즉 `VirtualFTSensor` body로 들어오는 incoming joint
wrench를 사용한다. gripper에 작용한 외력은 UR5e로 전달되는 과정에서 이 joint를
통과하므로, Wrist에 걸리는 합성 하중을 얻을 수 있다.

## 측정 데이터

관측 함수
[`virtual_ft_wrench_b()`](../sweep_rl/sweep_rl/osc_sweep/mdp/observations.py)는
다음 순서의 6차원 wrench를 반환한다.


$$
[Fx, Fy, Fz, Tx, Ty, Tz]
$$


내부적으로는 Isaac Lab articulation의 다음 데이터를 읽는다.

```python
robot.data.body_incoming_joint_wrench_b[:, body_id, :]
```

데이터의 shape은 `(병렬 environment 수, articulation body 수, 6)`이다.
`VirtualFTSensor`의 `body_id`를 선택하면 모든 병렬 environment의 센서 wrench를
한 번에 얻을 수 있다.

최종 출력에서는 부호를 반전한다.

```text
wrench = -body_incoming_joint_wrench_b[VirtualFTSensor]
```

Isaac Lab은 부모 body가 자식 body에 가하는 방향을 양수 기준으로 사용한다. 현재 구현에서는 gripper 측 하중이 Wrist에 작용하는 방향을 센서 출력 기준으로 사용하므로 부호를 반전한다.