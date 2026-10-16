# Contact Rich Manipulation Workspace

UR5e–Axia80 F/T–Inspire Hand의 접촉 기반 조작을 개발하는 Isaac Lab Workspace다.
현재 개발 대상은 **`inspire_sweep_baseline`, `axia80_feasibility`, `inspire_tactile_validation` 세 패키지**다.
전체 연구 흐름·결정·개발 기록은 코드 없는
[Contact-Rich-Manipulation-Context](example/Contact-Rich-Manipulation-Context/README.md)
서브모듈에서 관리한다. 기준일은 2026-10-14다.

## 시작할 문서

| 목적 | 안내 |
| --- | --- |
| 프로젝트 범위와 현재 우선순위 이해 | [인계 브리프](example/Contact-Rich-Manipulation-Context/01_plan/00_HANDOFF_BRIEF.md) |
| 전체 구조·패키지 역할·문서 위치 확인 | [Workspace 안내](example/Contact-Rich-Manipulation-Context/00_guidelines/10_WORKSPACE_GUIDE.md) |
| 환경 상속·등록 ID·실행 명령 확인 | [활성 환경](example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md) |
| Manager 기반 OOP와 개발 로그 작성 | [개발 지침](example/Contact-Rich-Manipulation-Context/00_guidelines/09_DEVELOPMENT_GUIDE.md) |
| 연구·문헌·실험 기록 탐색 | [프로젝트 문서 색인](example/Contact-Rich-Manipulation-Context/README.md) |
| 과거 참고 Sweep 환경 확인 | [Sweep-Policy 환경 문서](example/Sweep-Policy/docs/README.md) |
| SSH·VPN·ROS/Python 일반 지식 확인 | [일반 지식 색인](docs/README.md) |

## 디렉터리 역할

```text
grad/
├── AGENTS.md                        에이전트의 작업 진입 지침
├── src/
│   ├── inspire_sweep_baseline/       활성 Manager 기반 Sweep RL
│   ├── inspire_tactile_validation/   Hand 촉각 벡터 접촉 검증
│   ├── axia80_feasibility/           활성 F/T 정적 하중 검증
│   └── trash/                       과거 패키지·자산·테스트 보관, 빌드 제외
├── example/
│   ├── Contact-Rich-Manipulation-Context/  문서 전용 프로젝트 서브모듈
│   └── Sweep-Policy/                이전 Sweep RL 참고 코드 서브모듈
├── docs/                            프로젝트 밖의 간단한 일반 지식
├── IsaacLab/                        시뮬레이션·Manager 프레임워크 서브모듈
├── scene_randomization/             시뮬레이터 없는 배치 기하 검증 도구
├── tests/                           활성 Axia80 CPU·모델 테스트
├── reports/                         과거 실행·진단 산출물
├── logs/, outputs/, videos/         학습·검증·영상 산출물
└── build/, install/, log/           기존 ROS 빌드 산출물
```

| 활성 패키지 | 수행하는 역할 | 패키지 안내 |
| --- | --- | --- |
| `inspire_sweep_baseline` | 단일 물체 Sweep Baseline, 질량·마찰 랜덤화, 초기 위치 Actor/현재 상태 Critic, 전방향 목표 변형 | [README](src/inspire_sweep_baseline/README.md) |
| `inspire_tactile_validation` | 고정 palm-up Hand의 17개 촉각 영역을 중앙·네 모서리에서 벡터화해 누르는 standalone 검증 | [README](src/inspire_tactile_validation/README.md) |
| `axia80_feasibility` | UR5e–Axia80–tool의 정적 하중, cantilever와 wrist 각도 실험 | [README](src/axia80_feasibility/README.md) |

과거 패키지는 [보관 안내](src/trash/README.md)를 따른다. 활성 Axia80 실험은
보관된 UR Description의 xacro/mesh를 읽기 전용으로 사용한다. Baseline 로봇 USD는
패키지 내부에 포함하며 선반 USD 경로는 별도로 설정한다.

## 문서를 저장하는 기준

프로젝트 전체 Flow, 연구 결정, 개발 원칙, 환경 간 관계와 개발 로그는
`example/Contact-Rich-Manipulation-Context`에 작성한다.
`example/Sweep-Policy/docs`에는 그 참고 프로젝트의 환경·제어·관측·실행 설명을 작성한다.
루트 `docs`에는 두 프로젝트와 무관한 일반 지식만 간략히 작성한다.
패키지 README와 자산 출처 문서는 코드 옆에서 실행·자산 안내를 제공하고 전체 흐름 문서로 연결한다.

## 서브모듈 준비

체크아웃에 기록된 커밋을 기준으로 준비한다.

```bash
git submodule update --init --recursive
git submodule status
```

`Contact-Rich-Manipulation-Context`, `Sweep-Policy`, `IsaacLab`은 기존 등록을 유지한다.
UR Description의 현재 경로는 `src/trash/Universal_Robots_ROS2_Description`이다.
루트 `docs`는 일반 디렉터리이며 서브모듈이 아니다. 서브모듈에서 수정한 파일은 그 저장소의
변경으로 관리하며, 공유할 때 내부 커밋과 상위 저장소의 참조를 함께 갱신한다.

## 활성 환경 실행

저장소 루트에서 Isaac Lab 실행 환경을 사용한다. 아래 명령은 실행 예시이며
전체 환경별 명령·상속 구조는 [활성 환경 안내](example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md)에 있다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Baseline-v0 \
  --num_envs 2048 --sweep-direction both --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Baseline-v0 \
  --num_envs 1 --checkpoint "<checkpoint.pt>"
```

선반 자산은 접근 가능한 USD를 `INSPIRE_SWEEP_SHELF_USD_PATH`로 지정할 수 있다.
체크포인트는 사용자가 선택한 학습 결과를 지정한다.
Axia80의 실행 옵션과 판정 기준은 [패키지 안내](src/axia80_feasibility/README.md)에 있다.

## 빌드와 테스트 탐색

`src/trash`, 참고용 `example`과 빌드·로그·산출물 디렉터리에는 `COLCON_IGNORE`,
`AMENT_IGNORE`, `CATKIN_IGNORE`를 두고 추적한다. 활성 두 패키지는 `pyproject.toml` 기반 Python/Isaac Lab
패키지이므로 현재 `colcon list --base-paths src`의 결과는 비어 있다. ROS 패키지로 빌드하는
구성이 아니며 위 Python 실행 명령을 사용한다.

```bash
colcon list --base-paths src
```

`build/`, `install/`, `log/`는 이전 빌드 산출물을 보존한 경로다. 소스를 옮겨도
예전에 설치하거나 source한 ROS overlay가 제거되는 것은 아니다. 활성 실험은 Isaac Lab
실행 환경으로 시작하고 이전 `install/setup.bash`를 자동 source하는 환경 설정은 별도로 확인한다.

루트 `pytest.ini`는 활성 Axia80 테스트, Baseline 테스트와 Scene 기하 테스트를 탐색한다.
ROS pytest 플러그인과 Python 경로 간섭을 제외한 이 Workspace의 CPU 검증 명령은 다음과 같다.

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/min/miniconda3/envs/env_isaaclab/bin/python -B -m pytest -q -p no:cacheprovider
```

다른 컴퓨터에서는 해당 Isaac Lab Python 경로로 바꾼다. CPU·정적 자산 테스트와
Isaac Sim에서의 실제 학습·재생 검증은 구분해 기록한다.
