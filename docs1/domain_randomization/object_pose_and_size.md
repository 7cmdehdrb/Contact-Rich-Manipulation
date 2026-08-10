# 물체 위치와 크기 Randomization

## 위치와 yaw

기본 환경은 episode reset마다 물체의 기본 pose에 XY offset과 yaw를 더한다.

| 구현 | X | Y | Z | yaw |
|---|---:|---:|---:|---:|
| 기본 open-table | `±0.06 m` | `±0.18 m` | 고정 | `[-π, π]` |
| Independent shelf | `±0.05 m` | `±0.14 m` | shelf 높이 + 반쪽 크기 | `[-π, π]` |

기본 구현은
[`reset_target_object()`](../../src/sweep_rl/sweep_rl/osc_sweep/mdp/events.py), Independent 구현은
[`reset_variable_size_target()`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/events.py)다.
두 함수 모두 `env.scene.env_origins[env_ids]`를 더하므로 병렬 scene의 local pose를 world
pose로 올바르게 변환하고, reset 후 linear/angular velocity를 초기화한다.

```python
reset_target = EventTerm(
    func=mdp.reset_variable_size_target,
    mode="reset",
    params={
        "pose_range": {
            "x": (-0.05, 0.05),
            "y": (-0.14, 0.14),
            "yaw": (-math.pi, math.pi),
        },
        "table_top_height": SHELF_SURFACE_HEIGHT,
        "asset_cfg": SceneEntityCfg("target_object"),
    },
)
```

## Procedural cube 크기

Independent 환경은 기본 한 변 `0.06 m` cube를 environment별 `0.04–0.08 m`로 등방
scale한다.

```python
randomize_target_size = EventTerm(
    func=mdp.randomize_target_cube_size,
    mode="prestartup",
    params={
        "size_range": (0.04, 0.08),
        "base_size": 0.06,
        "asset_cfg": SceneEntityCfg("target_object"),
    },
)
```

`randomize_rigid_body_scale()`은 physics 시작 전 prim의 `xformOp:scale`을 변경한다.
샘플링한 실제 한 변 길이는 environment tensor에 저장하며 reset에서 다음 식으로 cube를
shelf 위에 놓는다.

```text
root_z = shelf_top_z + 0.5 × sampled_side_length
```

이 보정이 없으면 큰 cube는 shelf를 뚫고, 작은 cube는 공중에서 떨어지며 episode 초기
접촉 조건이 달라진다.

## OBJ 기반 USD 크기

사용자 USD도 rigid root의 등방 scale이라면 같은 `randomize_rigid_body_scale()`을 사용할
수 있다. 다만 cube용 helper는 “한 변 길이”만 저장하므로 비정형 물체에 그대로 쓰면
바닥 높이와 reward의 size-aware stand-off가 틀린다.

비정형 USD에는 다음 정보를 별도로 정의하는 것이 안전하다.

```python
BASE_HEIGHT = 0.11913
BASE_FOOTPRINT_RADIUS = 0.033

# scale s를 샘플링한 뒤
height = BASE_HEIGHT * s
root_z = support_surface_z - local_min_z * s
```

- origin이 물체 중심이면 `local_min_z = -0.5 × height`
- Can처럼 origin이 바닥이면 `local_min_z = 0`, 따라서 root를 표면 높이에 둔다.
- non-uniform scale은 collision, inertia, 접촉 위치와 정책 관측 의미를 함께 바꾸므로
  별도 검증 없이 사용하지 않는다.

현재 Can 매핑은
[`OPEN_TABLE_TOP_HEIGHT`, `CAN_HEIGHT`, `CAN_OBSERVATION_Z_OFFSET`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg_constant_velocity_upright_random_size_home_can.py)를
분리해 root pose와 center 관측을 맞춘다.

## 크기와 질량의 결합

scale만 바꾸고 질량을 고정하면 밀도는 물체마다 달라진다. 원하는 실험 가정을 먼저
정해야 한다.

- 고정 질량: 형상만 바뀌고 총 질량은 같음
- 고정 밀도: 등방 scale `s`일 때 질량을 `s³`에 비례시킴
- 독립 randomization: 크기와 질량을 각각 샘플링해 더 넓은 domain을 구성

현재 Independent 환경은 크기와 질량을 독립적으로 샘플링한다.

[Domain Randomization 문서로 돌아가기](README.md)
