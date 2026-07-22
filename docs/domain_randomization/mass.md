# 질량 Randomization

## Basic cube

절차적으로 생성한 cube는 spawn 시 MassAPI와 기본 질량을 함께 작성한다.

```python
target_object = RigidObjectCfg(
    ...,
    spawn=sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.06),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
        ...
    ),
)
```

기본 코드는
[`OscSweepSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg.py)에 있다. 이후
`randomize_rigid_body_mass`가 PhysX view의 default mass를 기준으로 episode마다 값을
바꾼다.

```python
randomize_target_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("target_object"),
        "mass_distribution_params": (0.25, 2.0),
        "operation": "abs",
        "distribution": "uniform",
        "recompute_inertia": True,
    },
)
```

| 환경 | 범위 | 시점 |
|---|---:|---|
| `WideRandomization` | `0.3–3.0 kg` | reset |
| `Independent` | `0.25–2.0 kg` | reset |

`operation="abs"`는 kg 단위의 최종 질량을 직접 샘플링한다. `scale`은 default mass에
배율을 곱하고 `add`는 offset을 더한다. 넓은 범위에서는 inertia도 함께 바뀌어야 하므로
`recompute_inertia=True`를 사용한다. 이 재계산은 균일 밀도 물체를 가정한다.

## OBJ 기반 USD에서 가장 먼저 확인할 것

OBJ를 USD로 변환할 때 visual mesh만 생성하고 MassAPI를 작성하지 않는 경우가 많다.
`UsdFileCfg(mass_props=...)`는 기존 MassAPI를 수정하는 방식이므로, rigid root에 schema가
없으면 기대한 기본 질량이 설정되지 않을 수 있다.

현재 `Can_6.usd`가 바로 이 경우다.

- scene 정의:
  [`CanSweepHomeSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg_constant_velocity_upright_random_size_home_can.py)
- MassAPI 보완 함수:
  [`define_rigid_object_mass()`](../../src/sweep_rl/sweep_rl/osc_sweep/mdp/events.py)
- `prestartup` 연결:
  `CanSweepHomeEventsCfg.set_target_mass`
- 실행 시 질량 override:
  [`play_constant_velocity_home_can.py`](../../src/sweep_rl/scripts/play_constant_velocity_home_can.py)

## MassAPI가 없는 사용자 USD 적용 절차

### 1. scene에 파일 asset 등록

```python
MY_USD_PATH = "/absolute/path/to/my_object.usd"

target_object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/TargetCube",
    spawn=sim_utils.UsdFileCfg(
        usd_path=MY_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.003,
            rest_offset=0.0,
        ),
        activate_contact_sensors=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(...),
)
```

### 2. simulation 시작 전에 MassAPI 작성

```python
set_target_mass = EventTerm(
    func=mdp.define_rigid_object_mass,
    mode="prestartup",
    params={
        "mass": 0.35,
        "asset_cfg": SceneEntityCfg("target_object"),
    },
)
```

이 helper는 각 environment의 `RigidObjectCfg.prim_path`에
`sim_utils.define_mass_properties()`를 호출해 누락된 MassAPI를 생성한다. 따라서 사용자
USD의 실제 `RigidBodyAPI`가 spawn root와 일치하는 구조가 가장 안전하다. rigid body가
하위 prim에 있다면 helper가 그 실제 rigid-body prim을 찾아 적용하도록 확장해야 한다.

### 3. 필요하면 episode 질량 randomization 추가

고정 기본 질량이 생성된 뒤 Basic cube와 같은 `randomize_rigid_body_mass` reset event를
추가한다. 이 순서가 중요한 이유는 default mass와 default inertia가 environment 초기화
시점에 안정적으로 캡처되어야 하기 때문이다.

## OBJ를 처음 USD로 변환하는 경우

Isaac Lab의 `scripts/tools/convert_mesh.py`에서 `--mass`와 collision approximation을
지정하면 변환 시 rigid/mass/collision schema를 함께 만들 수 있다. 동적 물체에는 보통
`convexHull` 또는 `convexDecomposition`을 사용하고 triangle mesh collision은 피한다.

```bash
./IsaacLab/isaaclab.sh -p IsaacLab/scripts/tools/convert_mesh.py \
  input.obj output.usd --mass 0.35 --collision-approximation convexHull
```

정확한 CLI option 이름은 현재 checkout의 `--help`로 확인한다. 이미 만든 USD에 MassAPI가
없다면 파일을 다시 변환하지 않아도 위 `define_rigid_object_mass()` 패턴을 사용할 수 있다.

## 확인과 문제 해결

PhysX가 실제로 사용하는 값은 다음처럼 확인한다.

```python
masses = env.scene["target_object"].root_physx_view.get_masses()
print(masses[:8])
```

Independent의 `print_reset_physics_info()`도 같은 API를 사용한다.

- 질량 warning: rigid root의 MassAPI 존재 여부 확인
- 값은 바뀌지만 움직임이 이상함: inertia 재계산과 collision approximation 확인
- environment마다 값이 같음: event mode와 `env_ids` 전달 확인
- `RigidObject` 초기화 실패: `prim_path` 아래 RigidBodyAPI가 정확히 하나인지 확인
- 매우 작은 질량: solver 불안정을 피하기 위해 `min_mass >= 1e-6` 유지

[Domain Randomization 문서로 돌아가기](README.md)
