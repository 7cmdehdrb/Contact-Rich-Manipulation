# Domain Randomization

Domain Randomization은 학습 중 물체와 장면의 조건을 바꾸어, 정책이 하나의 정확한
시뮬레이션 설정에만 과적합되지 않도록 하는 방법이다. 이 프로젝트에서는 Isaac Lab의
`EventTerm`과 관측 noise, command sampler, Action term의 reset hook을 함께 사용한다.

## 문서 구성

| 문서 | 대상 | 현재 구현 위치 |
|---|---|---|
| [물체 위치와 크기](object_pose_and_size.md) | reset pose, yaw, procedural cube 및 USD scale | `mdp/events.py`, `EventsCfg` |
| [마찰](friction.md) | cube, shelf, OBJ 기반 사용자 USD의 collision friction | Independent `EventsCfg` |
| [질량](mass.md) | 기본 cube와 MassAPI가 없는 사용자 USD | Wide/Independent `EventsCfg`, Can event |
| [로봇·명령·관측](robot_command_observation.md) | arm reset, OSC calibration, command, sensor noise | env/action/command config |

## 적용 시점

같은 randomization이라도 적용 시점이 잘못되면 simulation view가 생성되지 않았거나,
이미 시작된 rigid body의 topology를 바꾸게 된다.

| Mode/위치 | 적용 시점 | 현재 용도 | episode마다 변경 |
|---|---|---|---|
| `prestartup` Event | physics simulation 시작 전 | 물체 크기, 파일 기반 USD의 MassAPI 생성 | 아니요 |
| `startup` Event | environment 초기화 중 | 물체·shelf 마찰 bucket 배정 | 아니요 |
| `reset` Event | episode reset | 물체 위치/yaw, 질량, arm 초기 관절 | 예 |
| Command resample | episode 시작 | 방향, 거리, 힘/속도 | 예 |
| Action term `reset()` | episode reset | OSC gain/effort calibration | 예 |
| Observation noise | observation 계산 시 | joint/F/T/contact/pose noise | 매 policy step |

크기와 마찰은 병렬 environment마다 서로 다르지만 한 simulator 실행 중에는 고정된다.
질량과 pose는 각 episode에서 다시 샘플링된다.

## 현재 환경별 적용 범위

| 환경 계열 | 위치/yaw | 크기 | 질량 | 마찰 | 기타 |
|---|---|---|---|---|---|
| 기본 `osc_sweep` | XY/yaw | 고정 0.06 m | 고정 0.35 kg | 고정 | command, arm reset, observation noise |
| `WideRandomization` | XY/yaw | 고정 | `0.3–3.0 kg` | 고정 | 목표 힘 `8–50 N` |
| `Independent` | XY/yaw | `0.04–0.08 m` | `0.25–2.0 kg` | 물체와 shelf random | OSC calibration, command, observation noise |
| `HomeReturn-Can` | XY/yaw | Can 고정 형상 | 기본 0.35 kg | USD에 작성된 값 | CLI로 고정 질량 override |

## 구현 원칙

1. Scene asset에는 재현 가능한 기본값을 둔다.
2. randomization 범위는 `EventsCfg`에 모아 실험 설정으로 읽을 수 있게 한다.
3. geometry topology 변경은 simulation 시작 전에만 수행한다.
4. 질량을 크게 바꾸면 `recompute_inertia=True`를 사용한다.
5. dynamic friction이 static friction보다 커지지 않도록 `make_consistent=True`를 사용한다.
6. 사용자 USD는 visual mesh가 아니라 실제 rigid-body prim과 collision shape를 기준으로
   검증한다.
7. randomization이 적용됐다고 가정하지 말고 PhysX view에서 결과를 읽어 확인한다.

## 사용자 USD 공통 준비 조건

OBJ에서 변환한 USD를 `RigidObjectCfg`로 사용할 때는 다음 조건이 필요하다.

- `RigidObjectCfg.prim_path` 아래에 `RigidBodyAPI`가 적용된 body가 정확히 하나 있어야 한다.
- collision geometry에 `CollisionAPI`와 적절한 convex collision approximation이 있어야 한다.
- 활성화된 `ArticulationRootAPI`가 없어야 한다.
- 단위는 meter이고, local origin과 바닥 높이의 관계를 알고 있어야 한다.
- MassAPI가 없다면 [질량 문서](mass.md)의 `prestartup` 패턴으로 먼저 생성한다.
- 마찰은 visual material이 아니라 collision shape의 physics material에 적용한다.

현재 사용자 USD의 실제 예시는
[`CanSweepHomeSceneCfg`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg_constant_velocity_upright_random_size_home_can.py)다.
Can USD는 rigid root에 MassAPI가 없어 별도 event로 보완하고, rigid root가 물체 바닥에
있어 관측 Z에 center offset을 더한다.

## 결과 확인

Independent 구현에는 질량과 마찰을 PhysX tensor에서 직접 읽는
`print_reset_physics_info()`가 있다.

- 구현: [`osc_sweep_independent/mdp/events.py`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp/events.py)
- 활성화 위치: [`EventsCfg`의 주석 처리된 logger](../../src/sweep_rl/sweep_rl/osc_sweep_independent/env_cfg.py)

logger는 진단 중에만 켜고 대규모 학습에서는 다시 끈다.

[전체 Sweep RL 문서로 돌아가기](../README.md)
