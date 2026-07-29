# 프로젝트 디렉터리 구조

이 문서는 저장소 루트(`/`)를 기준으로 주요 디렉터리와 파일의 역할을 설명한다. 실행 중 생성되는 로그, 학습 결과 및 캐시 파일은 대표 경로만 표시한다.

## 전체 구조

```text
.
├── docs/                              # 프로젝트 공통 설계·구현 문서
│   ├── domain_randomization/          # Domain Randomization 항목별 설명
│   ├── environment_setup/             # 로봇, 센서, Scene, Asset 구성 문서
│   ├── img/                           # 문서에서 사용하는 이미지
│   ├── README.md                      # 프로젝트 문서 목차
│   └── file_directory.md              # 현재 문서
├── src/                               # 프로젝트 소스 코드 및 로봇 모델
│   ├── sweep_rl/                      # 주 Sweep RL 학습 환경 패키지
│   ├── sweep_jh/                      # 별도 Sweep 실험 환경 패키지
│   └── Universal_Robots_ROS2_Description/
│                                       # Universal Robots ROS 2 모델 (submodule)
├── tests/                             # 환경 설정과 동작 계약을 검증하는 테스트
├── example/
│   └── Sweep-Policy/                  # 참고용 기존 Sweep Policy (submodule)
├── IsaacLab/                          # 프로젝트 기반 Isaac Lab 포크 (submodule)
├── logs/                              # 학습 로그와 내보낸 정책 파일
├── outputs/                           # Hydra 실행별 설정 및 결과
├── README.md                          # 프로젝트 개요와 정책 프레임워크 설명
├── TODO.md                            # 프로젝트 작업 목록
├── SSH.md                             # 원격 접속 관련 메모
├── .gitmodules                       # Git submodule 경로와 원격 저장소 정의
├── .gitignore                         # Git 추적 제외 규칙
└── LICENSE                            # 라이선스
```

## 핵심 소스 코드

### `src/sweep_rl/`

현재 Sweep 강화학습 환경과 실행 스크립트를 담는 주 Python 패키지다.

```text
src/sweep_rl/
├── sweep_rl/
│   ├── __init__.py                    # Isaac Lab 환경 등록 진입점
│   ├── osc_sweep/                     # 통합 OSC Sweep 환경
│   │   ├── agents/                    # RSL-RL PPO 학습 설정
│   │   ├── mdp/                       # Action, Command, Event, Observation,
│   │   │                               # Reward, Termination 정의
│   │   ├── assets.py                  # 로봇·물체·Scene Asset 설정
│   │   └── env_cfg*.py                # 실험별 환경 설정
│   └── osc_sweep_independent/         # 독립형 OSC Sweep 환경
│       ├── agents/                    # 기본/상세 PPO 설정
│       ├── mdp/                       # 독립 환경의 MDP 구성 요소
│       └── env_cfg*.py                # 기본/상세 환경 설정
├── scripts/                           # 학습 및 정책 실행 진입 스크립트
├── docs/                              # 환경별 관측·Action·Reward·실행 방법
├── pyproject.toml                     # 빌드 시스템과 도구 설정
└── setup.py                           # Python 패키지 설치 설정
```

`osc_sweep/env_cfg*.py`는 constant velocity, tactile localization, wide randomization, home return 등 실험 변형을 정의한다. 대응하는 `scripts/train_*.py`, `scripts/play_*.py`가 학습과 추론의 실행 진입점이다.

### `src/sweep_jh/`

`sweep_rl`과 분리된 OSC Sweep 실험용 Python 패키지다. 기본 구성은 다음과 같다.

```text
src/sweep_jh/
├── sweep_jh/
│   └── osc_sweep/
│       ├── mdp/                       # MDP 구성 요소
│       ├── assets.py                  # Asset 설정
│       ├── env_cfg.py                 # 환경 설정
│       └── rsl_rl_ppo_cfg.py          # PPO 학습 설정
├── scripts/
│   ├── train.py                       # 학습 실행
│   └── play.py                        # 학습된 정책 실행
├── docs/                              # 관측·보상·하이퍼파라미터 문서
├── pyproject.toml
└── setup.py
```

### `src/Universal_Robots_ROS2_Description/`

Universal Robots의 ROS 2 description 패키지를 submodule로 포함한다. 주요 내용은 로봇별 물리·관절 설정(`config/`), 시각 및 충돌 Mesh(`meshes/`), URDF/Xacro 모델(`urdf/`), launch와 RViz 설정이다.

## 테스트

`tests/`에는 Sweep 환경의 설정 및 인터페이스가 의도한 계약을 유지하는지 확인하는 테스트가 있다.

```text
tests/
├── test_learning_envs.py
├── test_constant_velocity_home_can_contracts.py
├── test_independent_sweep_contracts.py
└── test_independent_sweep_detailed_contracts.py
```

## 문서

- `docs/`: 프로젝트 전반에 적용되는 환경 구축 및 Domain Randomization 문서
- `src/sweep_rl/docs/`: `sweep_rl`의 개별 환경과 실행 방법
- `src/sweep_jh/docs/`: `sweep_jh`의 관측, 보상 및 하이퍼파라미터
- 각 submodule 내부의 `README.md` 또는 `docs/`: 외부 프로젝트 자체 문서

## 외부 코드와 Submodule

다음 디렉터리는 `.gitmodules`에서 관리하는 외부 저장소다.

| 경로 | 용도 |
|---|---|
| `IsaacLab/` | 프로젝트가 사용하는 Isaac Lab 기반 프레임워크 |
| `src/Universal_Robots_ROS2_Description/` | Universal Robots의 ROS 2 로봇 모델 |
| `example/Sweep-Policy/` | 구현 참고용 Sweep Policy 프로젝트 |

처음 저장소를 받은 뒤 이 디렉터리가 비어 있다면 다음 명령으로 초기화한다.

```bash
git submodule update --init --recursive
```

## 실행 중 생성되는 디렉터리

| 경로 | 내용 | 관리 원칙 |
|---|---|---|
| `logs/` | RSL-RL 학습 로그, 체크포인트, 내보낸 정책 | 실험 결과로 취급하며 소스 코드와 분리 |
| `outputs/` | Hydra가 실행 시각별로 생성한 설정 및 로그 | 필요 결과만 선별해 보존 |
| `__pycache__/` | Python bytecode 캐시 | 자동 생성되며 Git에서 제외 |
| `.pytest_cache/` | pytest 실행 캐시 | 자동 생성되며 Git에서 제외 |
| `*.egg-info/` | Python 패키지 설치 메타데이터 | 자동 생성되며 Git에서 제외 |

새 기능을 추가할 때 환경 로직은 해당 패키지의 `mdp/`와 `env_cfg*.py`에, 학습·실행 진입점은 `scripts/`에, 검증 코드는 루트 `tests/`에 두는 구성을 따른다.
