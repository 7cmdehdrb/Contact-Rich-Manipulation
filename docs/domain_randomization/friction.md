# 마찰 Randomization

## 현재 구현

Independent 환경은 Isaac Lab의 `randomize_rigid_body_material`을 `startup` event로
사용해 target cube와 shelf의 마찰을 서로 독립적으로 샘플링한다.

| Asset | static friction | dynamic friction | restitution | buckets |
|---|---:|---:|---:|---:|
| `target_object` | `0.40–1.10` | `0.25–0.90` | 0 | 64 |
| `shelf` | `0.40–1.10` | `0.25–0.90` | 0 | 64 |

구현 위치는
[`Independent EventsCfg`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/env_cfg.py)의
`randomize_target_friction`, `randomize_shelf_friction`이다.

```python
randomize_target_friction = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("target_object"),
        "static_friction_range": (0.40, 1.10),
        "dynamic_friction_range": (0.25, 0.90),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
        "make_consistent": True,
    },
)
```

`make_consistent=True`는 dynamic friction을 static friction 이하로 제한한다. bucket은
초기화 시 한 번 생성한 뒤 collision shape에 배정한다. PhysX의 unique material 수
제한과 CPU setter 비용 때문에 매 episode보다 `startup` 사용이 적합하다.

## Basic cube에 적용

`sim_utils.CuboidCfg`로 만든 cube는 collision shape와 기본 physics material이 명시되어
있다. `SceneEntityCfg("target_object")`를 전달하면 PhysX view의 모든 collision shape
material이 random 값으로 바뀐다.

기본값은
[`OscSweepSceneCfg.target_object`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg.py)의 static 0.65,
dynamic 0.45다. event가 없는 환경은 이 값을 계속 사용하고, event가 있는 Independent
환경은 startup에 덮어쓴다.

## OBJ에서 변환한 사용자 USD에 적용

이 event는 `CuboidCfg` 여부나 USD 내부 visual material을 보지 않고
`RigidObject.root_physx_view`의 collision shape material buffer를 수정한다. 따라서
OBJ 기반 USD에도 다음 조건만 충족하면 같은 방법을 쓸 수 있다.

1. USD를 `RigidObjectCfg`로 scene에 등록한다.
2. `prim_path` 아래에 rigid body가 하나만 존재하게 한다.
3. mesh에 collision이 활성화되어 있고 convex collision approximation이 생성되어야 한다.
4. `asset_cfg` 이름을 scene field 이름과 일치시킨다.

실제 file-based 예제가 shelf다.

```python
shelf = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Shelf",
    spawn=sim_utils.UsdFileCfg(usd_path=SHELF_USD_PATH),
    ...
)

randomize_shelf_friction = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={"asset_cfg": SceneEntityCfg("shelf"), ...},
)
```

사용자 object도 `target_object = RigidObjectCfg(...UsdFileCfg...)`로 바꾼 뒤 기존
`randomize_target_friction`을 그대로 재사용할 수 있다. `prim_path`는 ContactSensor
filter 호환성을 위해 `{ENV_REGEX_NS}/TargetCube`로 유지해도 asset의 실제 형상이 cube일
필요는 없다. 현재 `Can_6.usd`가 이 패턴을 사용한다.

## 여러 collision shape를 가진 USD

기본 randomizer는 environment와 collision shape별로 bucket ID를 샘플링한다. cube는
shape가 하나라 문제가 없지만, convex decomposition된 사용자 USD는 한 물체 안의 hull마다
다른 마찰이 배정될 수 있다.

모든 shape에 같은 마찰을 원하면 다음 중 하나를 선택한다.

- collision을 하나의 convex hull로 단순화한다.
- custom event에서 environment당 `(static, dynamic, restitution)` 하나를 샘플링하고
  `materials[env_ids, :, :]`에 broadcast한 뒤
  `root_physx_view.set_material_properties()`로 기록한다.

복잡한 concave 물체는 convex decomposition이 접촉 정확도에 유리하지만, hull별 재질을
허용할지 여부를 실험 설계에 명시해야 한다.

## 확인과 문제 해결

[`print_reset_physics_info()`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/events.py)는
`get_material_properties()`에서 각 shape의 static/dynamic 값을 읽는다.

- 값이 안 바뀜: collision shape 또는 scene asset 이름을 확인한다.
- 초기화 실패: USD 아래 rigid body가 0개 또는 2개 이상인지 확인한다.
- 접촉이 미끄럽지 않음: 물체뿐 아니라 상대 표면의 마찰도 함께 확인한다.
- visual material 변경만 보임: 마찰은 MDL/PreviewSurface가 아니라 physics material이다.

[Domain Randomization 문서로 돌아가기](README.md)
