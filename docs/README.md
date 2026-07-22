# 프로젝트 문서

이 디렉터리는 Sweep RL 구현과 확장 방법을 설명하는 프로젝트 수준 문서의 진입점이다.

## 가이드

| 대표 문서 | 설명 |
|---|---|
| [Domain Randomization](domain_randomization/README.md) | 물체 위치·크기·마찰·질량과 로봇·명령·관측 randomization, OBJ 기반 USD 적용법 |
| [Environment 구축](environment_setup/README.md) | UR5e/Robotiq assembly, 가상 F/T, ContactSensor, open table과 shelf USD scene |

## Domain Randomization 세부 문서

- [물체 위치와 크기](domain_randomization/object_pose_and_size.md)
- [마찰](domain_randomization/friction.md)
- [질량](domain_randomization/mass.md)
- [로봇·명령·관측](domain_randomization/robot_command_observation.md)

## Environment 구축 세부 문서

- [로봇과 센서](environment_setup/robot_and_sensors.md)
- [학습 Scene과 Asset](environment_setup/scenes_and_assets.md)

환경별 관측·Action·Reward·실행법은
[`src/sweep_rl/docs`](../src/sweep_rl/docs/README.md)를 참고한다.
