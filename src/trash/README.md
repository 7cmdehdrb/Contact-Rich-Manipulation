# 과거 패키지와 실험 보관

2026-10-14부터 활성 개발 패키지는 상위 디렉터리의
[`inspire_sweep_baseline`](../inspire_sweep_baseline/README.md)과
[`axia80_feasibility`](../axia80_feasibility/README.md) 두 개다.
이 디렉터리는 과거 코드·모델·실험을 삭제하지 않고 보관한다.

`COLCON_IGNORE`, `AMENT_IGNORE`, `CATKIN_IGNORE`가 하위 트리의 빌드 탐색을 제외한다.
루트 `pytest.ini`도 이 경로를 기본 테스트 탐색에서 제외한다. Python에서 직접 실행하거나
수동으로 import하는 동작까지 차단하는 표시는 아니므로 활성 실행 스크립트는 활성 패키지만 등록한다.

| 디렉터리 | 보관 내용 |
| --- | --- |
| `sweep_rl` | 이전 OSC·선반 Sweep 환경과 실험별 문서 |
| `sweep_jh` | 이전 Sweep 변형 |
| `sweep_shelve_force` | 이전 선반 힘 제어 환경 |
| `sweep_inspire_rl` | 이전 Inspire Sweep 정책 환경 |
| `hand_manipulation_rl` | 이전 Hand 조작 RL 및 로봇 USD·센서 구현 |
| `hand_manipulation_test` | 이전 Hand 조작 검증 환경 |
| `inspire_robot` | Inspire SDK, URDF, ROS 관련 자산 |
| `inspire_tactile` | 이전 촉각 투영·읽기 실험 |
| `inspire_tactile_ros` | 이전 촉각 ROS 2 노드 |
| `Universal_Robots_ROS2_Description` | UR 로봇 Description 서브모듈 |
| `test` | 이전 임시 실행·검증 코드 |
| `workspace_tests` | 비활성 환경용 루트 테스트와 분석 스크립트 |
| `context_artifacts/axia80_feasibility` | 문서 서브모듈에서 옮긴 9월 Axia80 구현 스냅샷. 실험 근거 CSV·metadata·로그는 문서 저장소에 유지 |

옛 `src/__init__.py`와 `src/__pycache__`도 함께 보관했다.
각 패키지 내부 문서와 실행 명령은 당시 상태의 기록이다. 보관으로 경로가 바뀌었으므로
그대로 현재 개발 절차로 사용하지 않는다.

Axia80 실험은 보관된 UR Description의 xacro/mesh를 읽는다.
참고용 Sweep-Policy의 일부 Inspire 환경도 보관된 `hand_manipulation_rl` 자산·센서를 읽는다.
이 자산 참조는 패키지를 ROS 빌드나 활성 정책 등록에 포함하는 것과 구분한다.

[전체 구조와 문서 안내](../../example/Contact-Rich-Manipulation-Context/docs/10_WORKSPACE_GUIDE.md)
에서 현재 개발 흐름을 확인한다.
