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

## Xform에 MassProperty를 붙였는데 거부되거나 적용되지 않는 이유

USD의 prim type과 physics schema를 구분해야 한다. `Xform`은 transform 계층을 만드는
일반 prim type이고, 물리 body가 되려면 같은 prim에 `UsdPhysics.RigidBodyAPI`가 있어야
한다. 질량은 visual mesh나 단순 wrapper Xform이 아니라 **PhysX가 rigid body로 등록하는
prim**의 `UsdPhysics.MassAPI`에 작성해야 한다.

따라서 “Xform에는 MassAPI를 적용할 수 없다”는 설명은 정확하지 않다. 다음처럼
`RigidBodyAPI`를 가진 Xform은 올바른 질량 대상이다.

```text
TargetCube                  Xform + RigidBodyAPI + MassAPI  ← 올바른 대상
└─ geometry                Xform
   ├─ visual_mesh          Mesh
   └─ collision_mesh       Mesh + CollisionAPI
```

문제가 되는 구조는 spawn root가 물리 body가 아닌 wrapper일 때다.

```text
TargetCube                  Xform                           ← 여기에 MassAPI를 붙이면 안 됨
└─ CanBody                  Xform + RigidBodyAPI            ← 실제 질량 대상
   └─ collision_mesh       Mesh + CollisionAPI
```

위 구조에서 `TargetCube`에 MassAPI를 추가해도 `RigidObject.root_physx_view`는
`CanBody`를 가리킨다. wrapper에 작성한 질량은 실제 body의 질량으로 사용되지 않거나
schema/API 검증 과정에서 거부될 수 있다. visual/collision mesh에 MassAPI를 붙이는 것도
그 mesh 자체가 RigidBodyAPI prim이 아닌 한 올바른 해결책이 아니다.

Isaac Lab의 `RigidObject`는 `RigidObjectCfg.prim_path` 아래를 검색해 다음을 강제한다.

- `RigidBodyAPI` prim이 0개면 초기화 실패
- 2개 이상이면 single rigid body가 아니므로 초기화 실패
- 정확히 1개일 때 그 prim으로 `root_physx_view` 생성

관련 검사는 Isaac Lab의
[`RigidObject._initialize_impl()`](../../IsaacLab/source/isaaclab/isaaclab/assets/rigid_object/rigid_object.py)에
있다.

## `UsdFileCfg.mass_props`와 `define_mass_properties()`의 차이

OBJ를 USD로 변환하면 visual/collision geometry와 RigidBodyAPI는 있지만 MassAPI가 없는
경우가 있다. 이때 두 API의 동작 차이가 중요하다.

| 방법 | 없는 MassAPI 생성 | 대상 |
|---|---|---|
| `UsdFileCfg(mass_props=...)` 내부의 `modify_mass_properties()` | 아니요 | subtree에서 이미 MassAPI가 있는 prim만 수정 |
| `sim_utils.define_mass_properties(path, cfg)` | 예 | 전달한 정확한 prim에 MassAPI를 적용하고 값 설정 |
| `randomize_rigid_body_mass` | USD schema를 직접 만들지 않음 | simulation 시작 후 `root_physx_view`의 mass tensor 수정 |

Isaac Lab의 `modify_mass_properties()`는 대상 prim에 MassAPI가 없으면 `False`를 반환한다.
따라서 `UsdFileCfg(mass_props=...)`만 추가했는데 USD에 기존 MassAPI가 없으면 질량이
설정되지 않고 warning이 발생할 수 있다. 반면 `define_mass_properties()`는 누락된
MassAPI를 먼저 적용한다. 단, 반드시 위에서 찾은 **실제 RigidBodyAPI prim path**를
전달해야 한다.

## 현재 Can 구현의 정확한 흐름

현재 `Can_6.usd`는 spawn root가 실제 rigid-body root이지만 MassAPI가 없는 구조를
전제로 한다. 이 조건 때문에 현재 helper가 안전하게 동작한다.

- scene 정의:
  [`CanSweepHomeSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg_constant_velocity_upright_random_size_home_can.py)
- MassAPI 생성 함수:
  [`define_rigid_object_mass()`](../../src/sweep_rl/sweep_rl/osc_sweep/mdp/events.py)
- `prestartup` 연결:
  `CanSweepHomeEventsCfg.set_target_mass`
- CLI 연결:
  [`play_constant_velocity_home_can.py`](../../src/sweep_rl/scripts/play_constant_velocity_home_can.py)

`play_constant_velocity_home_can.py`의 실행 흐름은 다음과 같다.

1. `--object_mass`를 읽고 0보다 큰지 검사한다.
2. 값을 Hydra override
   `env.events.set_target_mass.params.mass=<kg>`로 변환한다.
3. Can environment의 `set_target_mass` event가 `prestartup`에서 실행된다.
4. `define_rigid_object_mass()`가 각 environment의
   `{ENV_REGEX_NS}/TargetCube`를 찾는다.
5. `sim_utils.define_mass_properties()`가 이 prim에 누락된 MassAPI를 생성하고 kg 값을
   작성한다.
6. 그 다음 simulation/PhysX view가 초기화되므로 지정 질량이 default mass로 캡처된다.

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/play_constant_velocity_home_can.py \
  --checkpoint /absolute/path/to/model.pt \
  --object_mass 0.50
```

이 script는 질량을 randomize하는 것이 아니라 실행 시 하나의 고정 질량을 주입한다.
다만 MassAPI가 없는 파일 asset을 physics 시작 전에 정상화한다는 점에서 episode별 질량
randomization의 준비 단계와 같다.

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

### 2. USD 계층에서 실제 rigid body 확인

먼저 Isaac Sim Stage 또는 code로 `RigidBodyAPI`와 `MassAPI` 위치를 확인한다.

```python
from pxr import Usd, UsdPhysics
from isaaclab.sim.utils.stage import get_current_stage

stage = get_current_stage()
root = stage.GetPrimAtPath("/World/envs/env_0/TargetCube")
for prim in Usd.PrimRange(root):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(
            "rigid body:", prim.GetPath(),
            "has MassAPI:", prim.HasAPI(UsdPhysics.MassAPI),
        )
```

출력되는 rigid body가 정확히 하나인지 확인한다. 경로가 `.../TargetCube` 자체라면 현재
Can helper를 그대로 쓸 수 있다. `.../TargetCube/CanBody`처럼 하위 prim이면 다음 절의
확장 helper가 필요하다.

### 3-A. spawn root가 rigid body인 경우

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
USD의 실제 `RigidBodyAPI`가 spawn root와 일치할 때만 현재 Can 구현을 그대로 사용한다.

### 3-B. rigid body가 wrapper 아래에 있는 경우

현재 `define_rigid_object_mass()`는 `RigidObjectCfg.prim_path` 자체에 MassAPI를 작성하므로
하위 rigid body를 자동으로 찾지 않는다. custom USD 계층을 유지해야 한다면 helper를
다음처럼 확장해 실제 RigidBodyAPI prim에 적용한다.

```python
from pxr import UsdPhysics

import isaaclab.sim as sim_utils


def define_mass_on_actual_rigid_body(env, env_ids, mass, asset_cfg):
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    if env_ids is not None:
        raise ValueError("run this globally during prestartup")

    spawn_paths = sim_utils.find_matching_prim_paths(
        env.scene[asset_cfg.name].cfg.prim_path
    )
    mass_cfg = sim_utils.MassPropertiesCfg(mass=mass)

    for spawn_path in spawn_paths:
        rigid_prims = sim_utils.get_all_matching_child_prims(
            spawn_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI),
            traverse_instance_prims=False,
        )
        if len(rigid_prims) != 1:
            raise RuntimeError(
                f"Expected one rigid body below {spawn_path}, got {rigid_prims}"
            )
        rigid_body_path = rigid_prims[0].GetPath().pathString
        sim_utils.define_mass_properties(rigid_body_path, mass_cfg)
```

이 함수도 `mode="prestartup"` EventTerm으로 연결한다. referenced USD의 rigid body가
instance proxy 내부에 있어 override authoring이 막히면, source USD에 MassAPI를 미리
작성하거나 변환 시 rigid body/MassAPI를 root에 생성하는 편이 안전하다.

### 4. 필요하면 episode 질량 randomization 추가

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

- `modify_mass_properties` warning: 실제 rigid body에 기존 MassAPI가 없으므로
  `define_mass_properties()` 또는 source USD 수정이 필요
- `Failed to find a rigid body`: `prim_path` subtree에 RigidBodyAPI가 없음
- `Failed to find a single rigid body`: subtree에 RigidBodyAPI가 2개 이상임
- MassAPI는 생겼지만 `get_masses()`가 안 바뀜: wrapper Xform이나 visual mesh에 schema를
  작성했는지 확인하고 실제 `root_physx_view` body path에 적용
- instance proxy에 schema 작성 거부: source USD에 MassAPI를 작성하거나 rigid-body
  root에 override할 수 있는 asset 구조로 변환
- 값은 바뀌지만 움직임이 이상함: inertia 재계산과 collision approximation 확인
- environment마다 값이 같음: event mode와 `env_ids` 전달 확인
- `RigidObject` 초기화 실패: `prim_path` 아래 RigidBodyAPI가 정확히 하나인지 확인
- 매우 작은 질량: solver 불안정을 피하기 위해 `min_mass >= 1e-6` 유지

마지막 판정은 USD Inspector에 보이는 attribute가 아니라
`root_physx_view.get_masses()`의 값으로 한다. 이 값이 실제 simulation에 등록된 body의
질량이다.

[Domain Randomization 문서로 돌아가기](README.md)
