# 학습 Scene과 Asset

## Open-table scene: `osc_sweep`

[`OscSweepSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg.py)는 외부 arena USD 없이 Isaac Lab
primitive로 학습 공간을 만든다.

```text
World
├─ Ground
└─ env_N
   ├─ OpenTable
   ├─ Robot
   └─ TargetCube
```

### 구성

| Asset | 구현 | 위치/크기 | Physics |
|---|---|---|---|
| Ground | `GroundPlaneCfg` | world origin | 기본 ground material |
| OpenTable | `CuboidCfg` | center `(0.45, 0, 0.75)`, size `(1.20, 0.90, 0.05) m` | friction `0.8/0.6` |
| Robot | custom `ArticulationCfg` | base `(0, 0, 0.775)` | UR5e + Robotiq + sensor links |
| TargetCube | `CuboidCfg` | center `(0.50, 0, 0.805)`, side `0.06 m` | mass `0.35 kg`, friction `0.65/0.45` |
| Light | `DomeLightCfg` | `/World/Light` | intensity 2500 |

table top은 `z=0.775 m`이고 cube 중심은 top + 반쪽 크기인 `0.805 m`다. target reset은
기본 pose에서 X `±0.06 m`, Y `±0.18 m`, yaw `[-π, π]`를 샘플링한다.

### 장점

- geometry, mass, collision, material을 code에서 완전히 재현할 수 있다.
- USD 내부 hierarchy 의존성이 적다.
- `replicate_physics=True`로 동일 physics scene을 효율적으로 복제할 수 있다.
- 새로운 reward/action을 검증하는 baseline으로 적합하다.

## Shelf USD scene: `osc_sweep_independent`

[`IndependentSweepSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/env_cfg.py)는
`speedrack_shape.usd`를 실제 작업 공간으로 불러온다.

```text
World
├─ Ground
└─ env_N
   ├─ Shelf
   │  └─ rack                # contact filter가 기대하는 prim
   ├─ Robot
   └─ TargetCube
```

### 주요 상수

| 항목 | 값 |
|---|---|
| `SHELF_USD_PATH` | `.../Collected_speedrack_shape/speedrack_shape.usd` |
| shelf world position | `(-0.7, 0, 0)` |
| 작업 shelf 표면 높이 | `1.05 m` |
| robot-root workspace X | `[0.50, 0.90] m` |
| robot-root workspace Y | `[-0.50, 0.50] m` |
| robot base | `(0, 0, 0.79505)`, Z축 180° 회전 |

command sampler는 cube 반쪽 크기와 `0.015 m` margin을 고려해 shelf 경계를 넘지 않는
방향과 거리만 생성한다. 목표 거리는 가능한 범위 안에서 `0.12–0.35 m`다.

### Target와 reset

target은 여전히 procedural cube지만 environment별 한 변 `0.04–0.08 m`, episode별 질량
`0.25–2.0 kg`을 사용한다. reset 시 root Z를 다음처럼 다시 계산한다.

```text
target_root_z = 1.05 + 0.5 × sampled_cube_size
```

scene의 initial state보다 reset event의 이 계산이 실제 episode 시작 pose를 결정한다.

### `replicate_physics=False`

Independent scene은 environment마다 target scale과 material이 다르고, robot body별
filtered ContactSensor가 shelf/self-contact를 구분한다. 이 heterogeneous physics 구성을
위해 `replicate_physics=False`를 사용한다. GPU memory와 초기화 비용은 늘지만 각
environment의 physics property를 독립적으로 유지할 수 있다.

## Basic Object를 추가하는 방법

cube, sphere, cylinder처럼 단순한 물체는 procedural spawner가 가장 안전하다.

```python
target_object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetCube",
    spawn=sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.06),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.003,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.65,
            dynamic_friction=0.45,
            restitution=0.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(...),
)
```

`prim_path`의 이름은 자유롭지만 현재 command, sensor filter, reward, termination이
`target_object` scene key와 `/TargetCube` prim path를 기대한다. 코드를 적게 바꾸려면
형상이 cube가 아니어도 이 key/path를 유지한다.

## OBJ 기반 사용자 USD를 target으로 교체

현재 실제 예시는
[`CanSweepHomeSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg_constant_velocity_upright_random_size_home_can.py)다.

```python
target_object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetCube",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MY_OBJECT_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...),
        collision_props=sim_utils.CollisionPropertiesCfg(...),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(...),
)
```

교체 전에 다음을 확인한다.

| 검사 | 이유 |
|---|---|
| rigid body 1개 | `RigidObject`는 `prim_path` subtree에서 정확히 한 body를 요구 |
| collision shape 존재 | 접촉, 마찰, sweep physics에 필요 |
| convex collision | 움직이는 mesh의 안정적인 PhysX simulation |
| MassAPI 또는 prestartup mass event | default mass/inertia 확정 |
| local origin과 `local_min_z` | 표면 위 reset 높이 계산 |
| meter 단위와 orientation | command 거리 및 observation 의미 유지 |
| contact reporter | target-side sensor를 쓰는 HomeReturn에 필요 |
| sensor filter path | `/TargetCube`를 바꾸면 모든 filter도 수정 필요 |

마찰과 질량 적용은 [마찰 문서](../domain_randomization/friction.md)와
[질량 문서](../domain_randomization/mass.md)를 따른다.

## Shelf USD를 교체하는 방법

새 arena USD를 사용할 때 단순히 `SHELF_USD_PATH`만 바꾸면 충분하지 않을 수 있다.

1. `SHELF_POSITION`과 작업 표면 높이를 측정한다.
2. robot base pose와 Home joint pose를 새 shelf에 맞춘다.
3. 안전 margin을 뺀 workspace X/Y 범위를 다시 측정한다.
4. robot-shelf ContactSensor filter가 가리키는 rigid/collision prim path를 수정한다.
5. shelf를 `RigidObjectCfg`로 관리할 수 있도록 rigid body가 하나인지 확인한다.
6. shelf가 움직이면 안 되므로 USD의 kinematic/static physics 설정을 확인한다.
7. 한 environment에서 reset pose, goal endpoint, shelf collision termination을 시각 검증한다.

현재 filter는 `{ENV_REGEX_NS}/Shelf/rack`을 기대한다. 새 USD가
`Shelf/geometry/collision` 같은 구조라면 `ROBOT_CONTACT_FILTERS`의 두 번째 항목을 실제
collision prim에 맞춰야 한다.

## Scene 검증 checklist

- target이 reset 직후 표면을 관통하거나 공중에 떠 있지 않다.
- 모든 command endpoint가 유효 workspace 안에 있다.
- pad가 target을 접촉할 때만 target-filtered force가 발생한다.
- robot이 shelf에 닿으면 shelf collision termination이 발생한다.
- F/T sensor의 force/torque 단위와 부호가 기대와 일치한다.
- mass와 material을 PhysX view에서 읽었을 때 설정 범위 안에 있다.
- GUI 1-env 검증과 headless multi-env 결과가 동일한 물리 의미를 가진다.

[Environment 구축 문서로 돌아가기](README.md)
