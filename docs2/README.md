# Contact-Rich Manipulation 문서 모음

이 폴더는 Contact-Rich Manipulation 프로젝트의 문서와 문서에서 참조하는 자료를 모아 둔 독립 문서 패키지다.

## 빠른 탐색

| 경로 | 내용 |
| --- | --- |
| [`docs/README.md`](docs/README.md) | 프로젝트 문서의 주제별 목차 |
| [`docs/file_directory.md`](docs/file_directory.md) | 원본 프로젝트의 전체 디렉터리와 핵심 파일 안내 |
| [`docs/domain_randomization/`](docs/domain_randomization/README.md) | Domain Randomization 관련 문서 |
| [`docs/environment_setup/`](docs/environment_setup/README.md) | 로봇, 센서, Scene, Asset 구성 문서 |
| [`docs/lab-meeting/`](docs/lab-meeting/0728.md) | 연구 방향과 제어 설계에 관한 연구실 회의 자료 |
| [`sweep_rl/README.md`](sweep_rl/README.md) | Sweep RL 패키지, 설치, 학습 및 재생 안내 |
| [`sweep_rl/docs/README.md`](sweep_rl/docs/README.md) | Sweep RL 환경별 문서 목차 |
| [`sweep_rl/sweep_rl/`](sweep_rl/sweep_rl/) | 문서에서 직접 참조하는 환경 설정과 MDP 구현 코드 |
| [`img/`](img/) | 문서에서 사용하는 이미지 |
| [`pdf/`](pdf/) | 관련 논문 PDF |
| [`rigid_object/`](rigid_object/) | 질량 Randomization 문서에서 참고하는 Isaac Lab 구현 코드 |

## 프로젝트 공통 문서

### Domain Randomization

| 문서 | 내용 |
| --- | --- |
| [`docs/domain_randomization/README.md`](docs/domain_randomization/README.md) | 문서 구성, 적용 시점, 환경별 적용 범위 |
| [`object_pose_and_size.md`](docs/domain_randomization/object_pose_and_size.md) | 물체 위치, yaw, 크기 Randomization |
| [`friction.md`](docs/domain_randomization/friction.md) | 물체와 Scene의 마찰 Randomization |
| [`mass.md`](docs/domain_randomization/mass.md) | 기본 물체와 사용자 USD의 질량 Randomization |
| [`robot_command_observation.md`](docs/domain_randomization/robot_command_observation.md) | 로봇 초기 상태, OSC calibration, 명령 및 관측 노이즈 |

### Environment 구축

| 문서 | 내용 |
| --- | --- |
| [`docs/environment_setup/README.md`](docs/environment_setup/README.md) | Environment 구축 문서의 목차와 코드 구조 |
| [`robot_and_sensors.md`](docs/environment_setup/robot_and_sensors.md) | UR5e, gripper, 가상 F/T 및 ContactSensor 구성 |
| [`scenes_and_assets.md`](docs/environment_setup/scenes_and_assets.md) | Open-table/Shelf Scene과 Asset 구성 및 교체 방법 |

### 기타

| 문서 | 내용 |
| --- | --- |
| [`docs/file_directory.md`](docs/file_directory.md) | 원본 프로젝트의 디렉터리 구조 |
| [`docs/lab-meeting/0728.md`](docs/lab-meeting/0728.md) | Task definition, force-feedback control, OSC 설계 및 TODO |

## Sweep RL 환경 문서

전체 환경 목록과 공통 실행 전제는 [`sweep_rl/docs/README.md`](sweep_rl/docs/README.md)에 있다.

### Force-command 계열

| 문서 | 대상 환경 |
| --- | --- |
| [`osc_sweep.md`](sweep_rl/docs/osc_sweep.md) | 기본 OSC Sweep |
| [`osc_sweep_play.md`](sweep_rl/docs/osc_sweep_play.md) | 기본 정책 재생 및 평가 |
| [`wide_randomization.md`](sweep_rl/docs/wide_randomization.md) | 넓은 범위의 물성·명령 Randomization |
| [`tactile_localization.md`](sweep_rl/docs/tactile_localization.md) | 촉각·힘 기반 물체 위치 추정 및 Sweep |

### Constant-velocity 계열

| 문서 | 대상 환경 |
| --- | --- |
| [`constant_velocity.md`](sweep_rl/docs/constant_velocity.md) | 목표 속도 기반 Sweep |
| [`constant_velocity_upright_random_size.md`](sweep_rl/docs/constant_velocity_upright_random_size.md) | Upright/Random-size 물체 환경 |
| [`constant_velocity_home_return.md`](sweep_rl/docs/constant_velocity_home_return.md) | Sweep 이후 Home pose 복귀 |
| [`constant_velocity_home_return_can.md`](sweep_rl/docs/constant_velocity_home_return_can.md) | Can 물체 대상 Home-return 평가 |

### Independent shelf 계열

| 문서 | 대상 환경 |
| --- | --- |
| [`independent_osc_sweep.md`](sweep_rl/docs/independent_osc_sweep.md) | Reach → Sweep → Home 과제 |
| [`independent_osc_sweep_detailed.md`](sweep_rl/docs/independent_osc_sweep_detailed.md) | 단계별 상세 Reward 환경 |
| [`osc_sweep_independent/README.md`](sweep_rl/sweep_rl/osc_sweep_independent/README.md) | Independent 환경의 Scene, 관측, Action, Reward, Randomization 상세 명세 |
| [`test.md`](sweep_rl/docs/test.md) | Task 및 Reward formulation 검토 자료 |

## 문서에서 참조하는 구현

| 경로 | 내용 |
| --- | --- |
| [`sweep_rl/sweep_rl/osc_sweep/`](sweep_rl/sweep_rl/osc_sweep/) | 기본 Sweep 환경 설정, Asset 조립 및 MDP 구현 |
| [`sweep_rl/sweep_rl/osc_sweep_independent/`](sweep_rl/sweep_rl/osc_sweep_independent/) | Independent Shelf 환경 설정 및 MDP 구현 |
| [`sweep_rl/scripts/play_constant_velocity_home_can.py`](sweep_rl/scripts/play_constant_velocity_home_can.py) | Can Home-return 환경 실행 스크립트 |
| [`rigid_object/rigid_object.py`](rigid_object/rigid_object.py) | Isaac Lab `RigidObject`의 질량 처리 참고 구현 |

## 이미지와 참고 자료

| 파일 | 사용처 |
| --- | --- |
| [`img/0716.png`](img/0716.png) | `docs/lab-meeting/0728.md`의 Framework 이미지 |
| [`A unified approach for motion and force control...`](pdf/A_unified_approach_for_motion_and_force_control_of_robot_manipulators_The_operational_space_formulation.pdf) | Operational Space Formulation 관련 참고 논문 |
