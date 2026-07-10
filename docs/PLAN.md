아래는 **상세 구현 명세가 아니라, 개발 방향을 잡기 위한 프로그램 개요 계획서**로 다시 정리한 것이다.

---

# Newton Physics 기반 매니퓰레이터 Reaching 학습 프로그램 개요

## 1. 목표

Newton Physics를 이용하여 매니퓰레이터의 **Point Reaching 작업**을 수행하는 강화학습 환경을 구현한다.

학습에는 `rsl_rl`을 사용하며, 기본 로봇은 **UR5e**로 설정한다. 로봇 모델은 별도 USD asset을 직접 만들기보다는, `Universal_Robots_ROS2_Description` 저장소를 프로젝트의 **Git submodule**로 포함하여 사용한다. 해당 저장소는 공개 저장소이며 기본 브랜치는 현재 `rolling`이다.

---

## 2. 기본 구현 방향

### 2.1 로봇 모델

로봇은 다음 저장소를 submodule로 추가한다.

```bash
git submodule add https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git src/Universal_Robots_ROS2_Description
git submodule update --init --recursive
```

사용 대상은 기본적으로 UR5e이다.

```text
src/
└── Universal_Robots_ROS2_Description/
```

이 저장소의 URDF/Xacro description을 기반으로 Newton 환경에 로봇을 로드한다. 단, 구현 과정에서 Newton이 URDF/Xacro를 직접 처리하지 못하는 경우에는 변환 스크립트를 별도로 둔다.

---

### 2.2 주변 환경

로봇 외의 환경은 복잡한 USD asset을 사용하지 않는다.

다음과 같은 **primitive object**만 사용한다.

| 객체                     | 구현 방식                            |
| ---------------------- | -------------------------------- |
| 바닥                     | Plane 또는 얇은 Cube                 |
| 좁은 테이블                 | Cube                             |
| 로봇 base 고정대            | Cube                             |
| target marker          | Sphere 또는 작은 Cube                |
| workspace boundary 시각화 | optional line / transparent cube |

즉, 환경은 다음 정도로 단순하게 구성한다.

```text
World
├── Ground
├── Narrow Table
├── UR5e
└── Random Target Marker
```

목표는 scene realism이 아니라 **Reaching 학습 환경의 안정적 구현**이다.

---

## 3. Task 정의

Reaching은 **EEF position이 target point에 가까워지는 문제**로 정의한다.

* Orientation은 고려하지 않음
* Target은 3D point
* Target은 reset마다 randomize
* Target 범위는 넓게 잡지 않음
* 일부 target은 base보다 낮은 $z$ 위치를 포함

Task objective는 단순하다.

$$
|p_{eef} - p_{target}|_2 \rightarrow 0
$$

---

## 4. 학습 환경 2종

본 프로그램은 동일한 scene과 reward를 사용하되, action space가 다른 두 환경을 제공한다.

---

## 4.1 Joint Action 환경

첫 번째 환경은 일반적인 방식이다.

Policy는 joint-space action을 출력한다.

```text
Policy action → joint command → Newton simulation
```

UR5e 기준 action dimension은 6이다.

```text
action_dim = 6
```

action은 초기에는 joint position delta 또는 joint velocity command로 사용한다.

예:

$$
q_{cmd} = q + \Delta q
$$

이 환경의 목적은 가장 기본적인 reaching baseline을 구축하는 것이다.

---

## 4.2 Cartesian Action 환경

두 번째 환경은 Cartesian-space action을 사용한다.

Policy는 joint action이 아니라 EEF의 Cartesian 이동 명령을 출력한다.

```text
Policy action → Cartesian command → IK/Jacobian 변환 → joint command → Newton simulation
```

action dimension은 3이다.

```text
action_dim = 3
```

예:

$$
a = [a_x, a_y, a_z]
$$

이를 Cartesian displacement로 해석한다.

$$
\Delta x = s a
$$

내부적으로는 differential IK 또는 Jacobian 기반 변환을 사용하여 joint command로 바꾼다.

이 환경에서 중요한 조건은 하나다.

```text
정책 출력은 반드시 Cartesian-space action이어야 한다.
```

내부 구현에서 joint command로 변환하는 것은 허용되지만, policy가 직접 joint action을 출력하면 안 된다.

---

## 5. 프로그램 구성 개요

대략적인 구조는 다음과 같이 둔다.

```text
/src/learning/
│
├── configs/
│   ├── reach_joint.yaml
│   └── reach_cartesian.yaml
│
├── envs/
│   ├── base_reach_env.py
│   ├── reach_joint_env.py
│   └── reach_cartesian_env.py
│
├── controllers/
│   ├── joint_controller.py
│   └── cartesian_controller.py
│
├── scripts/
│   ├── train_joint.py
│   ├── train_cartesian.py
│   ├── play_joint.py
│   └── play_cartesian.py
│
└── README.md
```

이 정도 구조면 충분하다.
초기 단계에서 과도하게 모듈을 쪼개지 않는다.

---

## 6. 공통 환경 구성

두 환경은 다음 요소를 공유한다.

| 항목           | 내용                                         |
| ------------ | ------------------------------------------ |
| 로봇           | UR5e                                       |
| 물리 엔진        | Newton Physics                             |
| 학습 라이브러리     | rsl_rl                                     |
| 목표           | EEF를 target point에 도달                      |
| observation  | joint state, EEF position, target position |
| reward       | EEF-target 거리 기반                           |
| reset        | target 위치 randomization                    |
| scene object | primitive object 중심                        |

---

## 7. Observation 개요

초기 observation은 단순하게 둔다.

```text
observation =
[
  joint position,
  joint velocity,
  EEF position,
  target position,
  target - EEF
]
```

UR5e 기준으로 대략 다음 차원이 된다.

```text
6 + 6 + 3 + 3 + 3 = 21
```

---

## 8. Reward 개요

초기 reward는 복잡하게 만들지 않는다.

기본은 거리 감소이다.

$$
r = -|p_{eef} - p_{target}|
$$

추가로 다음 정도만 포함한다.

```text
- target 도달 성공 보상
- action penalty
- joint velocity penalty
```

초기 구현에서는 reward shaping을 과도하게 넣지 않는다.
먼저 환경이 제대로 동작하는지 확인하는 것이 우선이다.

---

## 9. 개발 단계

## Phase 1. 기본 실행 환경 구성

* Newton Physics 설치 및 예제 실행
* rsl_rl 설치 및 import 확인
* PyTorch CUDA 확인
* 프로젝트 기본 구조 생성

---

## Phase 2. UR5e 모델 연동

* `Universal_Robots_ROS2_Description`을 submodule로 추가
* UR5e description 위치 확인
* Newton에서 로봇 로드 가능 여부 확인
* joint state와 EEF pose 읽기 확인
* fixed-base 설정 확인

---

## Phase 3. Primitive Scene 구성

* ground 생성
* 좁은 table 생성
* robot base 위치 설정
* target marker 생성
* target randomization 구현

이 단계에서는 USD asset을 추가 제작하지 않는다.
가능한 한 Cube, Plane, Sphere 등 primitive만 사용한다.

---

## Phase 4. Joint Action 환경 구현

* `ReachJointEnv` 작성
* action dimension 6 설정
* joint command 적용
* reward / reset / termination 연결
* rsl_rl 학습 루프 연결
* 간단한 reaching 학습 확인

---

## Phase 5. Cartesian Action 환경 구현

* `ReachCartesianEnv` 작성
* action dimension 3 설정
* Cartesian action을 EEF displacement로 해석
* 내부적으로 IK 또는 Jacobian 변환 적용
* 변환된 joint command를 로봇에 전달
* rsl_rl 학습 루프 연결

---

## Phase 6. 실행 스크립트 정리

최소한 다음 네 개의 스크립트를 둔다.

```text
train_joint.py
train_cartesian.py
play_joint.py
play_cartesian.py
```

필요하면 이후 평가용 스크립트를 추가한다.

---

## 10. 구현 시 우선순위

우선순위는 다음과 같다.

```text
1. Newton에서 UR5e가 정상 로드되는가
2. joint command로 UR5e가 움직이는가
3. EEF position을 정확히 읽을 수 있는가
4. target randomization이 되는가
5. Joint action 환경이 rsl_rl로 학습되는가
6. Cartesian action이 실제 EEF 이동으로 변환되는가
7. Cartesian action 환경이 rsl_rl로 학습되는가
```

---

## 11. 주의 사항

### 11.1 로봇 외 asset은 primitive 중심으로 구성

이 프로젝트에서는 주변 환경의 시각적 완성도가 중요하지 않다.
따라서 테이블, 바닥, target은 USD 파일을 따로 만들기보다는 Newton 또는 사용 중인 scene API에서 제공하는 primitive object로 생성한다.

---

### 11.2 Cartesian 환경의 action 정의를 흐리지 말 것

Cartesian 환경에서 policy output은 반드시 3차원이어야 한다.

```text
[a_x, a_y, a_z]
```

내부적으로 joint command로 바꾸는 것은 괜찮다.
하지만 policy가 6차원 joint command를 출력하면 Joint Action 환경과 구분이 사라진다.

---

### 11.3 초기 target 범위는 좁게 설정

초기부터 넓은 workspace를 쓰면 학습이 안 되는 원인을 찾기 어렵다.

따라서 처음에는 작은 범위에서 시작하고, 학습이 되는 것을 확인한 뒤 확장한다.

---

## 12. 최종 산출물

최종적으로 다음을 구현한다.

```text
1. Newton 기반 UR5e reaching scene
2. Primitive object 기반 table / ground / target 구성
3. rsl_rl용 Joint Action Reaching 환경
4. rsl_rl용 Cartesian Action Reaching 환경
5. 학습 실행 스크립트
6. 학습된 policy 재생 스크립트
```

---

## 13. 요약

이 프로그램은 다음 구조를 갖는다.

```text
UR5e description:
  Universal_Robots_ROS2_Description submodule 사용

Environment object:
  USD asset 대신 primitive object 사용

RL:
  rsl_rl 사용

Task:
  Point reaching

Environment 1:
  Joint-space action

Environment 2:
  Cartesian-space action
```

가장 중요한 구현 순서는 다음이다.

```text
UR5e 로드
→ primitive scene 구성
→ Joint Action 환경 구현
→ rsl_rl 학습 연결
→ Cartesian Action 환경 구현
→ IK/Jacobian 변환 연결
→ 두 환경 모두 play script로 동작 확인
```
