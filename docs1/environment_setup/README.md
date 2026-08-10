# Sweep Environment 구축

이 문서는 `osc_sweep`과 `osc_sweep_independent`가 학습용 가상 공간을 어떻게 구성하는지
설명한다. 두 계열은 같은 UR5e/Robotiq/virtual sensor assembly를 공유하지만, 작업 표면과
contact coverage, physics replication, randomization 범위가 다르다.

## 문서 구성

- [로봇과 센서](robot_and_sensors.md): UR5e, Robotiq 2F-85, 가상 F/T link, EEF frame,
  pad ContactSensor와 전신 collision sensor
- [학습 Scene과 Asset](scenes_and_assets.md): procedural table/cube scene, shelf USD scene,
  OBJ 기반 사용자 USD를 scene에 넣는 방법

## 코드 구조

| 경로 | 역할 |
|---|---|
| [`osc_sweep/assets.py`](../../src/sweep_rl/sweep_rl/osc_sweep/assets.py) | UR5e와 gripper를 조립하고 virtual links/contact pads 생성 |
| [`osc_sweep/env_cfg.py`](../../src/sweep_rl/sweep_rl/osc_sweep/env_cfg.py) | open table, cube, 기본 pad sensor scene |
| [`osc_sweep/mdp`](../../src/sweep_rl/sweep_rl/osc_sweep/mdp) | observation, reward, termination, action, reset 구현 |
| [`osc_sweep_independent/env_cfg.py`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/env_cfg.py) | shelf USD, 전신 contact sensor, independent scene |
| [`osc_sweep_independent/mdp`](../../src/sweep_rl/sweep_rl/osc_sweep_independent/mdp) | 3-phase shelf task와 domain randomization |

## Scene 구성 순서

1. `InteractiveSceneCfg`에 ground, 작업 표면, robot, target object, sensor, light를 선언한다.
2. 모든 environment별 asset path에는 `{ENV_REGEX_NS}`를 사용한다.
3. robot asset을 spawn하면서 UR5e와 Robotiq을 하나의 articulation으로 조립한다.
4. target object는 `RigidObjectCfg`로 등록해 pose, velocity, mass, material tensor에 접근한다.
5. ContactSensor의 sensor prim과 filter prim path를 실제 USD hierarchy에 맞춘다.
6. `EventTerm`으로 reset/randomization lifecycle을 연결한다.
7. command workspace와 object reset 범위가 작업 표면을 벗어나지 않는지 확인한다.
8. 소수 environment에서 collision, F/T sign, contact filter와 origin 높이를 검증한 뒤
   병렬 environment 수를 늘린다.

## 두 Scene의 핵심 차이

| 항목 | `OscSweepSceneCfg` | `IndependentSweepSceneCfg` |
|---|---|---|
| 작업 표면 | code로 만든 open table cuboid | `speedrack_shape.usd` shelf |
| target | 0.06 m procedural cube | 크기 randomization cube |
| robot pose | open table 기준 | shelf 앞 180° 회전 pose |
| ContactSensor | 좌/우 sweep pad → target | robot 전신 → target/shelf/self |
| 기본 environment 수 | 2,048 | 2,048 |
| `replicate_physics` | `True` | `False` |
| episode | 8초, Home 계열 12초 | 20초 |
| 주요 목적 | force 또는 velocity sweep | Reach → Sweep → Home 전체 절차 |

## 관련 문서

- [Domain Randomization](../domain_randomization/README.md)
- [환경별 관측·Action·Reward](../README.md)

[전체 Sweep RL 문서로 돌아가기](../README.md)
