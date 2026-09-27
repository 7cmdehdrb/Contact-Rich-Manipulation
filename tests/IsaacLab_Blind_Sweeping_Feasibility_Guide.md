# Isaac Lab 단일 물체 Blind Sweeping 환경 구축 가이드

작성일: 2026-09-26  
문서 유형: **구현 계획 및 인터페이스 명세**  
상태: **사용자 확정 조건 반영. 구현·실행·학습 결과 보고가 아님.**

## 1. 목적과 적용 범위

이 환경의 목적은 **초기 물체 위치, Binary Tactile, Wrist Wrench와 고유감각을 사용하는 정책이 무접촉 초기 상태에서 접촉을 형성하고, 단일 Cube를 지정 방향·거리만큼 이동시킬 수 있는지 확인하는 것**이다.

학습은 Arm의 6D Cartesian 증분과 Hand의 2D 관절 목표를 함께 출력한다. Arm은 Hand에 고정된 가상 제어점을 기준으로 고정 Stiffness·Damping OSC를 사용한다.

이번 문서는 센서 조합의 비교 실험이 아니라 **제안 메소드 하나를 실행하고 검증하기 위한 환경 구축 순서**를 정의한다. 동작 성공은 이 제한 환경에서의 feasibility 근거이며, 센서별 기여나 실물 전이 성능의 입증과는 구분한다.

### 1.1. 확정된 사양

| 항목 | 구현 사양 |
| --- | --- |
| 작업 공간 | Shelf를 얇은 Cuboid 작업판 1개로 대체 |
| 작업판 | 고정. Rigid Body·Collision·Mass Property 및 Contact Reporter 설정 |
| 로봇 | Base 고정 UR5e → F/T 모사용 Cylinder → Inspire Hand |
| 대상 물체 | Dynamic Rigid Body인 Cube 1개 |
| 초기 상태 | MoveIt 접근이 끝난 상태를 IK로 재현. Robot·Hand는 물체 및 작업판과 **무접촉·비관통** |
| 방향 | 작업판 평면의 연속 각도. 12시와 6시 각각 ±30° 제외 |
| 작업면 | Episode마다 손바닥 또는 손등을 무작위 선택. 두 면 모두 학습 |
| 초기 Variation | EEF 위치·Orientation 및 Hand 초기 자세에 유효 범위의 Variation |
| Actor Observation | 57D. 현재 물체 Pose와 실제 접촉 법선은 입력하지 않음 |
| Action | Arm 6D + Hand 2D = 8D |
| Arm 제어 | 현재 EEF 좌표계의 Cartesian 증분 → 고정 Gain OSC |
| Hand 제어 | 공통 굽힘 1D + 엄지 별도 관절 1D → Joint Position Control |
| 기본 Reward | 목표 오차, 실제 접촉 법선 정렬, 촉각 접촉 유지, Action 변화량, 시간 비용 |
| 추가 Reward | 전도·작업판 접촉력·물체 높이에 대한 점진적 페널티 |
| 성공 | 대상 물체와 목표 위치 사이의 3D 거리 오차 < 0.01 m |
| 실패 종료 | 물체 전도, 과도한 Robot–작업판 접촉력, 물체 높이 상한 초과 |
| 시간 제한 | Timeout으로 별도 분리 |
| 물체 들어 옮기기 | 별도 동작 분류기를 만들지 않음. 물체의 높이 증가에 대한 비용과 상한으로 제한 |
| 시각 | 초기 물체 위치만 제공. Sweep 중 현재 물체 위치를 시각으로 갱신하지 않음 |

**현재 사용자 지시 → 첨부 PDF → 기존 Method 문서 → 구현 예제** 순으로 요구사항을 해석한다. 기존 코드의 Reward·Gripper·초기화·종료 조건은 새 사양에 맞춰 교체할 수 있는 참고 구현이다.

### 1.2. 최초 테스트에서 고정할 조건

물체 형상·크기·질량·마찰, 작업판 물성, Robot Base 배치, OSC Gain은 최초 동작 확인 동안 고정한다. 방향·작업면·EEF 초기 오차는 본 가이드의 핵심 초기 분포에 포함한다.

대상 Cube의 초기 XY 위치와 목표 거리는 작은 유효 범위에서 시작하고, 기본 동작 확인 후 같은 Cube를 유지한 채 범위를 넓히는 구성을 권장한다. 초기 Object Yaw는 첫 단계에서 고정해도 된다. 이 범위의 수치는 구현 설정값이며 현재 확정된 실측값이 아니다.

## 2. 기존 구현에서 가져올 것

다음 내용은 저장소 소스를 읽어 확인했다. 이 세션에서 Isaac Sim 실행이나 학습은 수행하지 않았다.

| 참조 | 확인 내용 | 이번 환경에서의 사용 |
| --- | --- | --- |
| [Sweeping-Policy min 브랜치](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/tree/694c40aca11e38c13a70025ab177784a16e8b4ec) | Manager 기반 Sweeping 환경, OSC 환경, 단일 물체 초기화 변형이 존재 | 환경 등록·설정·MDP 분리 구조 참고 |
| [단일 물체 OSC Offset 설정](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/694c40aca11e38c13a70025ab177784a16e8b4ec/sweeping_policy/config/ur5e/osc_random_single_offset_env_cfg.py) | 단일 물체와 TCP 초기 위치 변형의 출발점 | 단일 물체 환경 구성 재사용. 방향 조건부 초기화로 변경 |
| [기존 Reset](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/694c40aca11e38c13a70025ab177784a16e8b4ec/sweeping_policy/mdp/events.py) | IK 관련 처리·유효성 확인·기존 초기화 보조 기능 | 6축 Pose IK와 무접촉 초기화로 확장. 기존 Wrist 제한을 그대로 복사하지 않음 |
| [Inspire Tactile](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/tree/06431a2f60cebef81fcf8109a8ad4d754b73c541/src/inspire_tactile) | 손바닥 17개 영역, 영역 투영, 촉각 읽기·시각화·Hand 제어 | 원본 Hand, 영역 ID·배치·관절 매핑을 재사용. 손등 17개 영역 추가 |
| [Axia80 Feasibility](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/tree/06431a2f60cebef81fcf8109a8ad4d754b73c541/src/axia80_feasibility) | Cylinder·Fixed Joint 기반 F/T 모사와 하중 검증 구성 | 측정 Joint·Frame 구성 및 정하중 검증 절차 참고 |
| [Method 문서](https://github.com/7cmdehdrb/Contact-Rich-Manipulation-Context/blob/02ad8f560ebcc2b0bcadea58ba8acd6a01db880f/docs/presentation/03_Method.md) | 57D 관측, 8D Action, 양면 촉각 표현 | 사용자가 이번에 수정한 방향 Frame·작업판·종료 조건을 반영 |
| [Isaac Lab 고정 커밋](https://github.com/7cmdehdrb/IsaacLab/blob/4e06a8a516a291e9c948edf4eed6ef4c7242c927/VERSION) | Contact-Rich-Manipulation의 Submodule 버전은 2.3.2 | 실제 설치본과 일치 여부를 구현 시작 시 확인 |

### 2.1. 그대로 복사하면 안 되는 부분

- 기존 Sweeping 환경의 Gripper는 새 Inspire Hand 2D 제어와 다르다.
- 기존 OSC 문서 설명과 실제 설정에 차이가 있다. 실제 설정은 **pose_rel·fixed impedance**를 사용하며, 문서에 적힌 Action 차원과 Gain을 새 사양으로 간주하지 않는다. [기존 OSC 설정](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/blob/694c40aca11e38c13a70025ab177784a16e8b4ec/sweeping_policy/config/ur5e/osc_random_sweep_env_cfg.py)
- **pose_rel이라는 설정만으로 현재 Hand Frame의 증분이 보장되지 않는다.** Frame 변환과 목표 갱신 규칙을 확인해야 한다.
- 손바닥 촉각 리더의 영역 투영 로직에는 CPU 변환과 반복문이 포함되어 있다. 소규모 확인에는 참고할 수 있으나, 다중 환경 학습에서는 GPU Batch 경로로 정리한다. [Projected Tactile 구현](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/blob/06431a2f60cebef81fcf8109a8ad4d754b73c541/src/inspire_tactile/inspire_tactile/projected_tactile.py)
- 기존 외부 USD 경로·Nucleus 주소·로컬 설치 상태는 저장소 텍스트만으로 실재를 확인할 수 없다. 자산 경로·모델 버전·실행 Launcher를 먼저 확인한다.
- Isaac Lab 버전, Isaac Sim 버전, RSL-RL 버전과 실제 적용되는 Runner 설정을 기록한다. 과거 PPO 설정을 근거 없이 현재 실행값으로 선언하지 않는다.

## 3. 좌표계와 Command

### 3.1. 좌표계

| Frame | 정의 | 용도 |
| --- | --- | --- |
| S | 작업판에 고정. +Z는 위, +Y는 Base에서 멀어지는 12시, −Y는 Base 쪽 6시. +X는 오른손 좌표계의 3시 방향 | 방향·거리·물체 목표·높이·작업영역 |
| B | Robot Base | FK·IK·동역학 및 OSC 계산 |
| H | Hand의 강체 기준 링크 | 센서 배치와 가상 제어점의 부모 Frame |
| C | Hand 중앙 부근의 가상 제어점. H에 고정 | Arm 제어, EEF Pose, 보정 Wrench |
| C₀ | 유효한 Episode 시작 때의 C를 고정한 Frame | 현재 EEF 상대 Pose와 초기 물체 상대 위치 |
| F | F/T 측정 기준점 및 축 | Raw Wrench 정의와 보정 |

C는 손가락이 굽혀져도 움직이는 기하학적 중심이 아니라 **Hand 기준 링크에 대한 고정 Transform**이다. 초기화·OSC·관측·Wrench에서 같은 정의를 사용한다.

### 3.2. 방향과 Dead Zone

평면 방향을 다음과 같이 정의한다.

~~~math
\hat{\mathbf{d}}^{S}=(\cos\theta,\sin\theta,0),\qquad
\mathbf{p}_g^{S}=\mathbf{p}_{o,0}^{S}+L\hat{\mathbf{d}}^{S}.
~~~

- 0°: 3시, 90°: 12시, 180°: 9시, 270°: 6시.
- 12시 기준 ±30°와 6시 기준 ±30°의 **Sweep Command를 제외**한다.
- 나머지 방향은 연속 분포로 샘플링한다. 좌우 두 방향만 사용하는 Discrete Task로 축소하지 않는다.
- 각도의 Wrap 경계를 제외 영역인 6시 방향에 두는 방식을 권장한다. 1D 각도 표현의 불연속점이 허용 방향 한가운데 놓이지 않게 하기 위해서다.
- 방향 1D는 **S 기준 각도**다. 초기 EEF Orientation Variation에 따라 기준축이 바뀌지 않는다.
- 초기 물체 위치·목표·방향·거리는 Episode 중 유지한다. 현재 물체 위치로 목표를 계속 재생성하지 않는다.

### 3.3. 유효한 Command의 조건

물체 시작점과 목표점이 작업판의 유효 영역 안에 있어야 한다. Cube의 크기와 안전 여유를 고려하여 가장자리에서 떨어지도록 배치한다. Hand가 이동할 경로와 Arm의 작업 가능 영역도 확인한다.

허용 각도라는 이유만으로 모든 시작점·거리·작업면 조합이 도달 가능하다고 가정하지 않는다. 초기화가 거절된 이유와 최종 채택된 방향·작업면 분포를 기록한다.

## 4. Scene와 물리 자산

### 4.1. 고정 작업판

얇은 Cuboid 하나를 만들고 Collision, Rigid Body, Mass Property, 마찰·반발계수를 설정한다. 고정 방식은 **Kinematic Rigid Body**를 우선 사용하고 위치·회전을 고정한다. 질량 속성은 명시하되, 고정된 판의 운동 응답은 질량으로 결정되지 않는다는 점을 구분한다.

작업판은 환경별로 독립적으로 복제하고, Contact Reporter와 센서를 연결한다. 동적 물체는 판 위에서 중력과 마찰에 의해 지지되고 미끄러져야 한다.

작업판 이동량·회전량 기반 Termination은 이번 환경에 사용하지 않는다.

### 4.2. 대상 Cube

다음 속성을 명시한다.

- 형상·치수 및 Collision 형상.
- 양의 질량과 일관된 관성·질량중심.
- 중력, 동적 Rigid Body, 마찰·반발계수.
- 작업판 위 초기 위치·자세와 초기 선속도·각속도.
- 초기 침투가 없으며 안정적으로 정지하는 배치.
- 성공 계산에 사용하는 위치 기준점. 최초 테스트는 **Cube 기하학적 중심과 질량중심을 일치**시킨다.

가상 높이 제한은 Cube 중심의 초기 높이 대비 증가량으로 계산한다. 별도의 천장 Collider나 Grasp/Lift 분류기를 만들지 않는다.

### 4.3. UR5e–F/T–Hand 조립

하나의 일관된 Articulation·기구학 체인을 구성한다.

1. UR5e Base를 World에 고정한다.
2. 기존 Gripper 자산과 제어 참조를 제거하거나 비활성화한다.
3. Wrist–Cylinder–Hand 장착 Transform을 정의한다.
4. F/T 측정용 Fixed Joint를 명시적으로 유지한다.
5. Hand의 가동·종속 관절과 Collision을 유지한다.
6. 질량·관성·중력을 조립체 전체에 적용한다.
7. C를 H에 대한 고정 Frame으로 정의한다.

**Cylinder의 외형만 추가해서는 F/T 센서가 만들어지지 않는다.** Hand에서 전달되는 반력과 모멘트가 통과하는 측정 Joint와 읽기 경로가 필요하다. 측정용 Fixed Joint가 Import 과정에서 병합되지 않도록 설정한다. [기존 F/T 구성](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/tree/06431a2f60cebef81fcf8109a8ad4d754b73c541/src/axia80_feasibility)

C는 계산·표시용 가상 Frame으로 둔다. 제어점을 표시하기 위해 임의 질량이나 Collision을 추가하지 않는다.

## 5. Event: 전체 Reset과 Manipulator Reset

기본 Reset 동작은 요청대로 **전체 Reset 1개 + Manipulator Reset 1개**로 구분한다. 두 동작은 순차적으로 실행하며, Episode 사양을 중복 샘플링하지 않는다.

### 5.1. 전체 Reset

담당 책임은 다음과 같다.

1. 종료 Episode의 통계와 종료 이유를 먼저 저장한다.
2. 해당 env_ids의 상태·시간·Action 이력·센서 버퍼·종료 플래그를 초기화한다.
3. 작업판과 Robot Base를 정의된 고정 상태로 복원한다.
4. Cube 위치·자세·속도를 초기화한다.
5. 손바닥/손등 Mode, 허용 방향, 목표 거리를 샘플링한다.
6. Cube의 초기 위치와 목표 위치를 저장한다.
7. Hand의 공통 굽힘·엄지 상태를 정한다.
8. 정해진 Episode 사양을 Manipulator Reset에 넘긴다.

표준 Randomization Hook을 추가로 쓰더라도, Robot Base·OSC Gain·물체 형상을 이번 범위와 다르게 임의 변경하지 않는다.

### 5.2. Manipulator Reset의 위치 생성

C의 목표 위치는 물체에서 **밀 방향의 반대쪽**으로 떨어진 위치다.

~~~math
\mathbf{p}_{C,\mathrm{des}}^{S}
=\mathbf{p}_{o,0}^{S}
-\rho\hat{\mathbf{d}}^{S}
+\delta_{\perp}\hat{\mathbf{t}}^{S}
+\delta_z\hat{\mathbf{z}}^{S}.
~~~

여기서 ρ는 Cube의 방향별 외곽, 초기 Hand 형상 및 양의 접촉 여유를 포함한다. 물체 중심에서 고정 거리만 빼는 방식으로 모든 Hand 자세의 비관통을 보장하지 않는다.

Variation은 명령 방향·횡방향·높이의 허용 범위에서 주며, 구체적인 분포와 상한은 설정으로 분리한다. 초기 Hand 자세가 바뀌면 그 자세의 전체 Collision 형상으로 여유를 다시 검사한다.

### 5.3. 손바닥·손등 Mode와 초기 Orientation

Episode Mode는 다음과 같다.

~~~math
m=
\begin{cases}
0,&\text{palm},\\
1,&\text{dorsal}.
\end{cases}
~~~

권장 초기 샘플링 비율은 손바닥:손등 = 1:1이다. 이는 학습 시작용 설정 제안이며, 최종 채택 비율은 IK 유효성 검사 이후에도 확인한다.

H Frame의 손바닥 기준 법선을 $\mathbf{n}_p^H$, 손등 기준 법선을 $\mathbf{n}_d^H$로 정의한다. 실제 장착 형상을 반영하여 두 법선을 별도로 저장한다. 이상적인 대칭 배치에서는 $\mathbf{n}_d^H=-\mathbf{n}_p^H$다.

초기 Orientation의 정렬 대상은 **선택한 면의 기준 법선**이다.

~~~math
\mathbf{R}_{SH}\mathbf{n}_m^H \approx \hat{\mathbf{d}}^{S}.
~~~

이 기준 자세에 Roll·Pitch·Yaw Variation을 적용하고, 선택한 면이 물체를 향하는 조건을 유지한다.

**초기화용 작업면 법선과 Reward용 실제 접촉 법선은 다른 값이다.** 이 절의 법선은 무접촉 상태에서 Orientation을 정하는 데만 사용한다.

### 5.4. Wrist 3의 불필요한 180° 회전 방지

다음 조건을 IK 해 선택에 포함한다.

- 손등 Mode를 손바닥 Mode와 같은 법선으로 처리한 뒤 Wrist 3에 π를 더하는 구현을 하지 않는다.
- 선택한 면의 법선을 기준으로 목표 Orientation을 생성한다.
- 자연스러운 기준 Joint Configuration과 가까운 Seed를 사용하고 여러 IK 해를 비교한다.
- Wrist 3의 허용 관절 구간과 기준값 대비 변화 비용을 명시한다.
- 2π Wrap 차이와 실제 180° Wrist Flip을 구분한다.
- Wrist Flip, Joint Limit 근접, 충돌, 특이점 근접 해는 거절한다.
- 유효한 해가 없으면 방향·초기 위치·작업면 조합을 다시 샘플링한다.

같은 위치·방향에서 양면을 번갈아 쓰는 모든 조합이 같은 Wrist 자세로 가능하다는 뜻은 아니다. **허용된 Wrist 자세에서 가능한 조합으로 초기 분포를 만들면서 양면의 학습 표본을 확보**한다. 손등 표본이 모두 거절되는 상태는 Reset 구현 실패로 취급한다.

### 5.5. IK와 유효성 검사

Arm 6축을 사용하는 Pose IK를 적용한다. 위치 오차뿐 아니라 Orientation 오차도 검사한다.

유효한 Reset은 아래 조건을 모두 만족해야 한다.

| 검사 | 조건 |
| --- | --- |
| IK 수렴 | C 위치·Orientation 오차가 허용 범위 이내 |
| 관절 | Joint Limit과 Wrist 3 조건 만족 |
| 기구학 | 심한 특이점 근접을 피함 |
| 로봇 내부 | 부적절한 Self-Collision 없음 |
| Robot–Object | Arm·Cylinder·Hand의 어느 부위도 물체와 접촉·관통하지 않음 |
| Robot–작업판 | 직접 접촉·관통 없음 |
| Object–작업판 | 정상 지지 접촉은 허용 |
| 이동 가능성 | 목표와 접근 상태가 작업판 및 Robot 작업영역 안에 있음 |

촉각 17비트가 모두 0이라는 것만으로 무접촉을 판정하지 않는다. 센서가 없는 Hand 부위의 접촉도 Collision·Contact 정보로 검사한다.

IK에 실패한 후보를 단순 Joint Clamp로 보정해 채택하지 않는다. 재시도 횟수와 실패 사유를 기록하고, 사전 검증된 유효 상태를 사용하는 경우 그 사용 빈도를 별도 기록한다.

### 5.6. Reset 종료 시 확정할 데이터

FK가 반영된 실제 Pose로 C₀를 저장한다. IK에 요청한 목표 Pose를 실제 시작 Pose 대신 사용하지 않는다.

확정할 항목은 다음과 같다.

- C₀와 초기 Object Position.
- S Frame의 방향·거리와 목표 위치.
- 작업면 Header.
- 초기 Cube 높이와 초기 Upright 기준.
- 초기 실제 Hand 2D 상태.
- 현재 Pose에 맞춘 OSC 목표와 Hand Position 목표.
- 초기 Wrench 보정 상태와 센서 버퍼.

Reset 직후 Arm의 Zero Action은 현재 Pose 유지 명령이다. 초기 Hand 목표도 현재 자세와 일치시켜야 한다. Last Action은 구현상 정한 일관된 초기값을 사용하고, 첫 스텝의 Action 변화량 페널티는 유효한 이전 정책 Action이 없으므로 0으로 처리한다.

### 5.7. Isaac Lab 실행 순서에서 확인할 사항

고정 커밋의 ManagerBasedRLEnv는 Reset Event 뒤에 Command Manager를 Reset한다. 따라서 Event에서 방향을 샘플링하고 이후 Command Manager가 다시 방향을 바꾸면, **IK 시작 자세와 Command가 불일치**한다. Episode 사양을 한 곳에서 생성하고 Command Term은 저장된 값을 읽도록 한다. [Reset 구현](https://github.com/7cmdehdrb/IsaacLab/blob/4e06a8a516a291e9c948edf4eed6ef4c7242c927/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)

또한 일부 환경 Reset을 위해 호출한 내부 sim.step이 나머지 환경을 로그 없이 진행시키지 않도록 한다. 초기화의 물리 갱신·안정화와 Episode 시작 시점을 구분하고, 센서값이 유효하기 전에 학습 Transition을 만들지 않는다.

Action Manager와 센서의 내부 Reset이 초기 목표·버퍼를 다시 지울 수 있으므로, 관리자 Reset과 상태 갱신 이후에 시작 Frame·현재 Pose 유지 목표·센서 유효 상태를 확정하는 마무리 순서를 둔다. 이는 세 번째 무작위 Reset Event를 추가한다는 뜻이 아니라, 두 Reset의 결과를 일관되게 확정하는 처리다.

## 6. Action과 제어

### 6.1. Arm: 현재 C Frame의 6D 증분

~~~math
\mathbf{a}_t^r
=(\Delta p_x,\Delta p_y,\Delta p_z,\Delta\phi_x,\Delta\phi_y,\Delta\phi_z).
~~~

정규화된 Action은 병진과 회전의 서로 다른 Scale로 물리 단위에 매핑한다. 회전 3D는 작은 Rotation Vector로 정의한다.

권장 목표 갱신은 **현재 실제 C Pose를 기준으로 한 증분**이다.

~~~math
\mathbf{p}_{\mathrm{des}}^B
=\mathbf{p}_C^B+\mathbf{R}_{BC}\Delta\mathbf{p}^C,\qquad
\mathbf{R}_{\mathrm{des}}^B
=\mathbf{R}_{BC}\mathrm{Exp}([\Delta\boldsymbol{\phi}^C]_\times).
~~~

목표는 Policy Step마다 한 번 정하고, 여러 Physics Substep 동안 같은 목표를 추종한다. 이전 목표에 Delta를 계속 누적하는 방식과 혼용하지 않는다. 접촉 중 실행되지 않은 이동 명령이 계속 쌓이는 상황을 피한다.

### 6.2. 가상 제어점에 맞춘 OSC

OSC에 제공하는 Pose, Twist, Jacobian은 모두 C를 기준으로 해야 한다. Palm 링크 Pose만 C로 옮기고 Jacobian을 그대로 사용하는 구현은 허용하지 않는다.

부모 링크 원점에서 C로 향하는 벡터를 B Frame으로 표현한 $\mathbf{r}_{HC}^B$에 대해 다음 관계를 사용한다.

~~~math
\mathbf{J}_{v,C}^B
=\mathbf{J}_{v,H}^B-[\mathbf{r}_{HC}^B]_\times\mathbf{J}_{\omega,H}^B,\qquad
\mathbf{v}_C^B=\mathbf{v}_H^B+\boldsymbol{\omega}_H^B\times\mathbf{r}_{HC}^B.
~~~

여기서 H의 선속도는 링크 원점 속도다. 질량중심 속도를 사용한다면 먼저 기준점을 보정한다. 모든 벡터의 표현 Frame을 같게 맞춘다.

고정 커밋의 기본 OSC Action에는 Offset Jacobian·속도의 기준점과 Frame을 검증할 필요가 있는 경로가 있다. 새로운 가상 제어점은 별도 Adapter에서 일관되게 처리하고, 유한차분 FK와 Jacobian·Twist 일치 검사를 통과시킨다. 전체 Isaac Lab을 수정하기 전에 환경 전용 경로로 해결하는 것을 권장한다. [기본 OSC Action](https://github.com/7cmdehdrb/IsaacLab/blob/4e06a8a516a291e9c948edf4eed6ef4c7242c927/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py)

### 6.3. 고정 Gain 및 구동

- Stiffness·Damping은 학습 Action에 넣지 않고 고정한다.
- F/T는 Actor가 다음 Cartesian 증분을 조절하는 입력이다. 이번 기본 제어에 별도의 목표 Wrench Action은 추가하지 않는다.
- Arm은 OSC의 Effort 명령으로 구동하고, 기존 Position Drive와 중복으로 싸우지 않도록 설정한다.
- Hand는 Joint Position Control을 사용한다.
- Arm 동역학과 중력 보상은 F/T Cylinder 및 Hand의 질량을 포함한 조립체 기준으로 확인한다.
- 병진·회전 Action Scale, 토크 상한, 속도 상한, 제어 주기는 개별 설정으로 분리한다.
- Gain 고정은 Zero Action에서 Hand가 공간에 절대 고정된다는 뜻이 아니다. 외력이 있을 때 변위·하중이 생기는 제어 응답을 확인한다.

### 6.4. Hand: 2D 입력과 실제 관절

~~~math
q_j^\ast=(1-u_f)q_j^{\mathrm{closed}}+u_fq_j^{\mathrm{open}},\qquad
u_f\in[0,1].
~~~

- 공통 굽힘: 0 = Fully Closed, 1 = Fully Open.
- 엄지의 별도 관절은 독립적인 0~1 입력으로 매핑한다.
- 원본 모델의 구동 관절·종속 관절 관계를 유지한다.
- 관절마다 Open/Closed 방향과 범위가 다를 수 있으므로 동일 각도를 일괄 적용하지 않는다.
- 입력 변화량과 관절 속도를 제한하여 순간적인 폐쇄·충돌을 방지한다.

Observation의 Hand 2D는 목표값 복사가 아니라 **현재 실제 관절 상태의 저차원 집계값**이다. 공통 굽힘은 각 구동 관절을 동일한 Open/Closed 기준으로 정규화한 뒤 집계하고, 엄지 별도 관절은 개별 정규화한다. 종속 관절을 중복 집계하지 않는다.

## 7. 센서 구성과 정보 경계

### 7.1. 양면 Tactile: 내부 34개 → Actor 18D

기존 손바닥 17개 영역의 ID와 의미를 유지한다. 손등에는 각 영역과 대응하는 17개 영역을 추가한다. 손가락 영역은 해당 링크에 고정하고 관절 운동을 따라가도록 한다.

| 내부 정보 | 구현 |
| --- | --- |
| 손바닥 | 17개 영역. 영역 내 감지 Cell이 하나라도 임계값을 넘으면 1 |
| 손등 | 17개 FSR 대응 영역. 해당 영역 접촉 신호가 임계값을 넘으면 1 |
| Header | Episode에서 선택한 작업면: 손바닥 0, 손등 1 |
| Actor 출력 | Header 1 + 선택한 면의 17비트 |
| 진단 로그 | 양면 34개 Raw/Processed 상태와 선택하지 않은 면의 접촉 |

~~~math
\mathbf{c}_t=(1-m)\mathbf{b}_t^{p}+m\mathbf{b}_t^{d},\qquad
\mathbf{z}_t=[m,\mathbf{c}_t]\in\{0,1\}^{18}.
~~~

Header는 접촉 유무가 아니다. 최초 무접촉 및 접촉 소실 시에도 유지한다. Header를 접촉 개수에 포함하지 않는다.

발표 자료의 0.05 N은 **초기 설정 후보**로 유지할 수 있으나, 새 Collision·영역 투영 구현에서도 적합한지는 센서 검증 단계에서 확인한다. 실제 FSR의 검증된 감도라고 서술하지 않는다.

손바닥의 Cell 최댓값 조건과 영역 전체 힘의 합계 조건은 동일하지 않다. 기존 리더가 영역 전체 힘만 반환하면, Cell 수준 조건을 보존할지 또는 영역 신호로 근사할지를 명시한다. 근사를 택하면 그에 맞춰 임계값을 검증한다.

영역 내부를 촘촘한 독립 Rigid Body로 재구성하는 것을 기본안으로 삼지 않는다. 기존 Hand의 접촉점과 실제 센서 영역을 연결하는 방식을 우선 검토하며, 가상 센서 영역이 전체 Hand 표면을 덮어 비센서 접촉까지 감지하지 않도록 한다.

### 7.2. Tactile의 접촉 대상

센서가 실제로 접촉한 외부 표면에 반응하도록 만든다. Actor용 Tactile에서 대상 Cube와의 접촉만 골라 제공하면 실제 센서보다 유리한 신호가 될 수 있으므로, 가능한 외부 접촉을 반영한다.

반면 Reward용 실제 접촉 법선은 **Hand–대상 Cube** 접촉만 사용한다. 센서 모사와 Reward 정답의 필터 목적을 구분한다.

자체 장착 구조나 내부 인접 링크 사이의 접촉은 물리 모델의 의도와 맞게 Collision Filter로 처리한다. 비선택 면 접촉 및 양면 동시 접촉은 Actor 18D 구조를 유지한 채 진단 로그에 남긴다.

### 7.3. Wrist F/T

측정 Joint를 통과하는 6축 반력·모멘트를 읽는다. 필요한 명세는 다음과 같다.

1. 측정 Joint와 이에 대응하는 Body Index.
2. API가 제공하는 힘·모멘트의 축과 기준점.
3. N, N·m 단위와 작용·반작용 부호.
4. 영점 및 Hand 자중·자세에 따른 중력 성분의 보정.
5. 센서 Frame F에서 현재 C로의 회전·모멘트 기준점 이동.
6. Filtering·Scale·Clip·표본 주기.

기존 Feasibility 측정은 Fixed Joint 인덱스를 가동 Joint 인덱스로 간주하지 않고, 대응 링크의 incoming wrench를 읽는 구성을 사용한다. 새 조립체에서도 Body 이름으로 대응을 확인한다. [F/T 검증 스크립트 모음](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/tree/06431a2f60cebef81fcf8109a8ad4d754b73c541/src/axia80_feasibility/scripts)

측정 구현은 Hand 조립에 맞춰 Frame·기준점을 재검증한다.

F에서 C로의 Wrench 변환은 기준점 이동을 포함한다. $\mathbf{r}_{CF}^C$는 C에서 F로 향하는 벡터다.

~~~math
\mathbf{F}^C=\mathbf{R}_{CF}\mathbf{F}^F,\qquad
\mathbf{M}_C^C=\mathbf{R}_{CF}\mathbf{M}_F^F+\mathbf{r}_{CF}^C\times\mathbf{F}^C.
~~~

초기 영점 한 번만 빼는 처리로 모든 Hand 자세·Orientation 변화의 중력 성분이 제거된다고 가정하지 않는다. 보정에는 로봇 모델·관절 상태를 사용하고, 대상 물체 접촉 정답을 빼서 이상적인 접촉 Wrench를 Actor에 제공하지 않는다.

Actor에는 보정된 6D Wrench를 제공하고 Raw·보정값을 모두 기록한다. Hand 가속·관절 운동에 따른 잔여 동적 하중은 별도로 관찰한다.

### 7.4. 작업판 Contact Sensor

작업판 하나를 센서 Body로 두고, 해당 환경의 Robot 링크들과의 접촉을 필터링한다. 대상 Cube와 작업판의 정상 지지 접촉은 이 페널티에 포함하지 않는다.

기본 측정량은 **작업판–Robot 각 링크의 법선 접촉력 크기의 합**이다.

~~~math
F_{\mathrm{board}}=\sum_{k\in\mathcal R}
\left\|\mathbf{f}_{\mathrm{board},k}^{\,n}\right\|_2.
~~~

$\mathcal R$에는 Arm·Cylinder·Hand의 관련 링크가 포함된다. 전체 벡터를 먼저 합친 뒤 Norm을 취하면 서로 반대 방향 접촉이 상쇄될 수 있으므로, 링크별 크기를 합산한다.

Isaac Lab의 기본 force_matrix_w는 법선 접촉력이다. 전체 접촉력에 마찰력을 포함하려면 별도 Friction 출력과 결합 정의를 검증해야 한다. 한 작업판 Body를 여러 Robot Body에 필터링하는 구성은 센서의 one-to-many 조건에 맞춘다. [Contact Sensor 명세](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sensors.html)

작업판에 가하는 힘을 Wrist F/T로 대체하지 않는다. Arm이 직접 판에 닿는 경우 Wrist에서 같은 힘이 측정된다고 보장할 수 없기 때문이다.

### 7.5. 갱신 주기

Physics Step, OSC 업데이트, Sensor 업데이트와 Policy Step의 관계를 명시한다.

- OSC와 접촉 측정은 Physics Step 단위로 갱신한다.
- Policy Step에는 정해진 시점 또는 구간 집계의 센서값을 전달한다.
- Hard Threshold 감시는 Physics Substep 내 최대값·초과 여부를 보존한다.
- 짧은 작업판 충격이 Policy Step 끝에서 사라졌다고 해서 초과 기록을 지우지 않는다.
- Reset 이후 첫 유효 측정 이전의 잔류값·누락값이 다음 Episode로 넘어가지 않도록 한다.

## 8. Actor Observation: 57D

| 순서 | 관측 | 차원 | 기준·갱신 |
| --- | --- | ---: | --- |
| 1 | Arm Joint Position | 6 | 현재 실제 관절 |
| 2 | Arm Joint Velocity | 6 | 현재 실제 관절 |
| 3 | 현재 C의 상대 Position | 3 | C₀ 기준 |
| 4 | 현재 C의 상대 Orientation | 3 | C₀ 기준 Rotation Vector 제안 |
| 5 | 실제 Hand 공통 굽힘·엄지 상태 | 2 | 현재 관절에서 집계 |
| 6 | 작업면 Header + 선택 면 Binary Tactile | 18 | Header는 Episode 고정, Tactile은 갱신 |
| 7 | 보정 Wrist Wrench | 6 | 현재 C 기준점·축 |
| 8 | 초기 Object Relative Position | 3 | C₀ 기준, Episode 고정 |
| 9 | Command Direction | 1 | **S 기준 평면 각도**, Episode 고정 |
| 10 | Command Distance | 1 | Episode 고정 |
| 11 | Last Action | 8 | 직전 정책 Action |
| **합계** | | **57** | |

현재 C의 상대 Pose는 다음으로 계산한다.

~~~math
{}^{C_0}\mathbf{T}_{C_t}
=\left({}^{W}\mathbf{T}_{C_0}\right)^{-1}{}^{W}\mathbf{T}_{C_t}.
~~~

내부 계산은 Quaternion/Rotation Matrix로 수행하고, Actor에 전달할 때 Orientation 3성분으로 변환한다. 초기 자세에 따른 Quaternion 부호 변화가 관측의 불연속으로 나타나지 않게 한다.

힘·모멘트, 위치·각도·속도는 단위별로 정규화한다. 이진 Header·Tactile은 의미가 바뀌지 않도록 별도 취급한다. 초기 단계에서는 명시적인 Scale·Clip을 먼저 검증한다.

### 8.1. 정보 경계

| 정보 | Actor | Reward·Termination·Reset·로그 |
| --- | --- | --- |
| 초기 Object Position | 허용 | 허용 |
| 현재 Object Pose·속도 | 제외 | 허용 |
| 실제 Hand–Object 접촉점·법선 | 제외 | 허용 |
| 접촉 대상 ID | 제외 | 접촉 필터·검증에 사용 |
| 작업판 전용 접촉력 | 제외 | 판 페널티·종료에 사용 |
| 물체 질량·마찰 정답 | 제외 | 물리 모델·기록에 사용 |
| Tactile·Wrench·Proprioception | 허용 | 허용 |

Critic은 우선 같은 57D 관측으로 시작하는 구성을 권장한다. Privileged Critic은 선택 가능한 후속 설계이며 이번 가이드에서 필수로 추가하지 않는다.

## 9. Reward

### 9.1. 공통 원칙

발표 자료의 다섯 기본 항을 유지하고, 페널티의 부호와 가중치 체계를 통일한다. 아래에서는 **양의 가중치 × 부호가 포함된 항**으로 정의한다.

목표 오차 감소가 주목적이다. 촉각 활성 수, 정상 정렬, 정지 상태 유지가 목표 이동보다 유리해지지 않도록 실제 Episode별 보상 기여와 행동을 함께 확인한다.

### 9.2. 목표 오차

~~~math
r_{\mathrm{goal}}
=\exp\left(-\frac{\left\|\mathbf{p}_o-\mathbf{p}_g\right\|_2}{\sigma_p}\right).
~~~

현재 물체 위치는 Privileged Information이다. 목표 위치는 초기 위치·방향·거리로 정한 고정값이다.

σp는 허용 목표 거리 범위에서 Reward가 유의미하게 변화하도록 설정한다. 성공 문턱 바로 밖에 머무르는 행동도 검증한다.

### 9.3. 실제 Hand–Object 접촉 법선 정렬

사용하는 법선은 **Hand가 대상 물체에 가하는 실제 접촉의 법선 방향**이다. Hand 표면의 사전 정의 법선이나 Wrist Wrench 방향으로 대체하지 않는다.

여러 Contact가 존재하면, 접촉 j의 양의 법선 하중 $f_{n,j}$와 물체에 작용하는 단위 법선 $\hat{\mathbf{n}}_j$를 사용해 합력 법선 방향을 구성한다.

~~~math
\mathbf{f}_n=\sum_j f_{n,j}\hat{\mathbf{n}}_j,\qquad
\hat{\mathbf{n}}_t=\frac{\mathbf{f}_n}{\|\mathbf{f}_n\|_2}.
~~~

유효한 접촉과 충분한 법선 합력이 있는 경우에만 다음을 계산한다.

~~~math
r_{\mathrm{normal}}
=-\frac{1-\hat{\mathbf{n}}_t\cdot\hat{\mathbf{d}}}{2}.
~~~

- 무접촉이면 0.
- 합력 Norm이 너무 작거나 서로 상쇄되어 방향이 정의되지 않으면 0과 유효성 로그.
- 두 벡터는 같은 Frame으로 변환.
- 물체–작업판 접촉과 Hand–작업판 접촉은 제외.
- Hand에 작용하는 반작용을 읽었다면 부호를 반전하여 물체에 작용하는 방향으로 통일.

이 항의 역할은 실제 접촉 방향이 목표 방향과 정렬되는 행동을 보상하고, 정책이 관측 가능한 F/T·Tactile과 행동 결과의 관계를 학습하도록 하는 것이다. Reward만으로 법선 추정기의 학습이나 법선의 유일한 복원이 보장되는 것은 아니며, 별도 법선 예측 Head는 추가하지 않는다.

### 9.4. 촉각 접촉 유지

~~~math
C_t=\mathbf{1}_{\{\sum_{i=1}^{17}c_{t,i}>0\}},\qquad
N_t=\frac{\sum_{i=1}^{17}\alpha_i c_{t,i}}{\sum_{i=1}^{17}\alpha_i},
\qquad
r_{\mathrm{contact}}=\beta C_t+(1-\beta)N_t-1.
~~~

활성 영역 비율은 실제 접촉 면적의 정확한 측정값이 아니라 이진 영역 활성도의 대리값이다. 비센서 부위 접촉은 실제 접촉이 있어도 이 항이 낮을 수 있으며, 임의로 보충하지 않는다.

초기 상태가 무접촉이므로 초기 몇 Step에 낮은 접촉 보상이 나오는 것은 정상이다. 이를 피하려고 Reset에서 물체에 가압하지 않는다.

### 9.5. Action 변화량

~~~math
r_{\mathrm{action}}
=-\lambda_r\|\bar{\mathbf{a}}_{r,t}-\bar{\mathbf{a}}_{r,t-1}\|_2^2
-\lambda_h\|\bar{\mathbf{a}}_{h,t}-\bar{\mathbf{a}}_{h,t-1}\|_2^2.
~~~

각 성분의 단위 영향을 통제하기 위해 정규화된 실행 Action을 사용한다. 위치 m와 각도 rad의 차이를 정규화 없이 단순 합하지 않는다. Hand Action은 목표 Position, Arm Action은 Cartesian 증분이라는 의미 차이를 유지한다.

### 9.6. 시간 비용

한 Policy Step의 비용은 발표식에 맞춰 다음과 같다.

~~~math
r_{\mathrm{time,step}}=-\frac{\Delta t_{\mathrm{policy}}}{T_{\mathrm{ref}}}.
~~~

Isaac Lab RewardManager가 이미 항에 dt를 곱하므로, Manager에 넣는 시간 항의 반환값은 보통 $-1/T_{\mathrm{ref}}$로 두어 dt가 두 번 적용되지 않게 한다. 이 규칙은 아래 연속 위험 페널티와 Terminal 보상에도 일관되게 적용한다. [RewardManager 소스](https://github.com/7cmdehdrb/IsaacLab/blob/4e06a8a516a291e9c948edf4eed6ef4c7242c927/source/isaaclab/isaaclab/managers/reward_manager.py)

### 9.7. 종료 이전에 증가하는 위험 페널티

공통 Ramp 함수를 사용한다.

~~~math
\psi(x;x_s,x_h)
=\left[\mathrm{clip}\left(\frac{x-x_s}{x_h-x_s},0,1\right)\right]^2,
\qquad x_s<x_h.
~~~

| 항목 | 위험량 | Soft 기준 | Hard 기준 | 페널티 |
| --- | --- | --- | --- | --- |
| 전도 | 초기 Upright 대비 기울기 γ | γsoft | γterm | $-\psi(\gamma;\gamma_{\mathrm{soft}},\gamma_{\mathrm{term}})$ |
| 작업판 접촉 | Robot–판 법선 접촉력 Fboard | Fsoft | Fterm | $-\psi(F_{\mathrm{board}};F_{\mathrm{soft}},F_{\mathrm{term}})$ |
| 가상 높이 제한 | Cube 중심의 높이 증가 h | hsoft | hterm | $-\psi(h;h_{\mathrm{soft}},h_{\mathrm{term}})$ |

높이는 다음으로 정의한다.

~~~math
h=p_{o,z}^{S}-p_{o,0,z}^{S}.
~~~

전도는 초기 Upright 축과 현재 물체 Upright 축 사이의 실제 기울기로 계산하는 구현을 권장한다. PDF의 Roll·Pitch 기반 근사와 같은 의도이며, Yaw 회전을 전도로 벌하지 않도록 한다.

**작업판 접촉은 힘이 Soft 기준을 넘으면서 비용이 증가하고, Hard 기준을 넘으면 종료한다.** 첫 접촉 순간을 무조건 실패로 처리하는 구성이 아니다.

높이 항은 어떤 동작이 Grasp/Lift인지 판별하지 않고 물체의 높이에만 작용한다. 기계적 천장을 시뮬레이션하는 것은 아니며, 허용 높이 경계를 비용·종료로 표현한다.

Timeout은 기존 시간 비용과 별도 종료 분류로 다룬다. 남은 시간을 Actor에 제공하지 않는 현재 관측에 별도의 급격한 시간 임박 페널티를 자동 추가하지 않는다.

### 9.8. 합산과 Terminal 항

연속 항은 양의 가중치로 합산한다.

~~~math
r_{\mathrm{dense}}
=w_g r_{\mathrm{goal}}+w_n r_{\mathrm{normal}}
+w_c r_{\mathrm{contact}}+w_a r_{\mathrm{action}}
+w_\gamma r_{\mathrm{tilt}}+w_b r_{\mathrm{board}}+w_h r_{\mathrm{height}}
+w_t r_{\mathrm{time}}.
~~~

성공 Bonus와 실패 Terminal Penalty는 **보완 구현으로 설정 가능하게 제공**할 것을 권장한다. 목표 문턱 직전에서 Reward를 얻거나, 위험 상태에서 조기 종료해 이후 비용을 피하는 행동을 방지하는 데 필요한지 확인한다.

Terminal 항을 채택한다면 성공/실패 시 한 번만 적용하고, RewardManager의 dt 곱셈 때문에 크기가 의도와 달라지지 않게 처리한다. 모든 항의 원시값, 가중값, 시간 적분 후 기여를 구분해 기록한다.

Wrench 입력을 쓴다는 이유만으로 새로운 힘 목표나 Wrench 방향 Reward를 추가하지 않는다. 현재 확정 Reward에서 F/T의 역할은 Actor 관측이다.

## 10. Success, Termination, Timeout

| 종류 | 판정 | 처리 |
| --- | --- | --- |
| Success | Object–Goal 3D 오차 < 0.01 m이며 실패 조건 없음 | 정상 성공 종료 |
| Topple | 기울기 γ가 Hard 기준 초과 | 실패 종료 |
| Board contact | Robot–판 접촉력 Fboard가 Hard 기준 초과 | 실패 종료 |
| Height limit | 높이 증가 h가 Hard 기준 초과 | 실패 종료 |
| Timeout | 제한 시간까지 성공·실패 없음 | Truncation으로 분리 |
| Invalid simulation | NaN·발산·자산 오류 등 | 구현 오류로 별도 기록·중단/복구. 정상 학습 실패와 혼합하지 않음 |

학습 환경에서 Success·Failure Termination과 Timeout을 분리한다. Timeout 여부는 Value Bootstrap 처리에 영향을 주므로 Runner까지 전달되는지 확인한다. [TerminationManager](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/termination_manager.html)

동일 Step에서 성공과 위험 초과가 함께 발생하면 **실패를 우선**한다. Timeout은 같은 Step에 성공 또는 실패가 없는 경우로 통계를 분리한다.

Hard 기준의 초과는 Physics Substep에서 보존하고, Episode 종료까지 Latched Flag로 유지한다. 평균 Filtering이 큰 순간 충격을 지워 종료를 누락시키지 않게 한다.

추가 보호 조건으로 Cube가 판 밖으로 낙하하거나 명백히 유효 영역을 이탈한 경우의 조기 종료를 설정 가능하게 둘 수 있다. 이는 핵심 네 가지 종료 조건을 대체하지 않으며, 채택 여부를 최종 설정에 기록한다.

## 11. MDP 모듈의 책임

아래는 **새로 구성할 모듈의 책임 제안**이다. 이미 해당 파일들이 존재한다는 의미는 아니다.

| 모듈 | 구현할 책임 | 핵심 산출물 |
| --- | --- | --- |
| Scene/Asset 설정 | 고정 판, 단일 Cube, 조립 Robot, Collision·물성 | 재현 가능한 Scene |
| Robot Assembly | UR5e–F/T–Hand 체인, Joint·Body 이름·Frame | 조립 자산과 매핑표 |
| Command | 방향·거리·Mode·초기 위치·Goal의 단일 생성·보관 | Episode 사양 |
| Events | 전체 Reset, 방향·Mode 조건부 IK Reset, 유효성 검사 | 무접촉 초기 분포 |
| Arm Action Adapter | C Frame 증분, Pose·Twist·Jacobian 일치, OSC | 6D 명령→Arm Effort |
| Hand Action Adapter | 2D Synergy→관절 목표·실제 상태 집계 | 2D 제어·관측 |
| Sensor Adapter | 양면 촉각, 보정 Wrist Wrench, 판 접촉, GT Contact | Actor 신호와 평가 신호 |
| Observation | 고정 순서 57D, 좌표계·정규화·초기값 | Actor 입력 |
| Reward | 기본 다섯 항, 위험 Ramp, 선택 Terminal 항 | 분리된 보상 기여 |
| Termination | 성공·세 실패·Timeout·실패 우선순위 | 종료 이유 |
| Runner/Logging | Task 등록, PPO 설정, 영상·상태·센서·종료 로그 | 재현 가능한 학습·실행 기록 |

기존 Task를 직접 덮어쓰기보다 Feasibility 전용 Task와 설정을 추가하는 구성을 권장한다. 이전 Gripper·Homing·온라인 Object 관측이 상속되어 들어오지 않도록 최종 활성 Term 목록을 확인한다.

## 12. 구현 및 검증 순서

### 단계 0. 실행 기반과 자산 확정

**구현:** 실제 버전·Task 등록·Launcher·USD/URDF 경로를 확인하고, 단일 환경을 띄울 최소 구성을 정한다.

**완료 기준:** 사용한 저장소·커밋·자산·버전·실행 경로를 기록하고, 필요한 링크·관절·센서 수를 열거할 수 있다.

### 단계 1. Scene와 조립체

**구현:** 고정 작업판, 고정 Base, UR5e–Cylinder–Hand, 단일 Cube와 물성.

**검증:** Cube가 가만히 놓여 있고, 로봇이 중력 아래에서 유지되며, Fixed Joint와 Hand 관절이 의도대로 남아 있는지 확인한다.

**완료 기준:** 숨은 중력 비활성화·초기 침투·부적절한 관성·잘못된 Joint 연결이 없는 Scene.

### 단계 2. 가상 제어점과 저수준 제어

**구현:** C Pose·Twist·Jacobian, 6D 증분 OSC, Hand 2D Position Control.

**검증:** C 기준 각 병진·회전 축을 개별 움직임으로 확인한다. 다른 Hand Orientation에서도 같은 Local Action이 같은 Local 방향으로 작용하는지 본다. 가상점 회전 시 위치·속도 응답을 확인한다.

**완료 기준:** Zero Action 유지와 작은 증분 추종이 안정적이며, Hand 움직임과 Arm 제어가 충돌하지 않는다.

### 단계 3. 센서와 판 접촉

**구현:** 손바닥·손등 영역, Header 선택, Wrench 보정·변환, 판 접촉 필터.

**검증:** 각 면의 감지 영역과 비센서 영역에 접촉시킨다. 무부하·정하중·알려진 레버암 하중으로 F/T 축과 모멘트 부호를 확인한다. Cube를 판에 올려둔 경우와 Hand/Arm으로 판을 누르는 경우를 구분한다.

**완료 기준:** 영역·부호·기준점·단위가 일치하고, 정상 Object–판 지지력이 Robot–판 페널티로 잡히지 않는다.

### 단계 4. 무접촉 Reset

**구현:** Mode·방향·거리 샘플링, 6축 IK, 초기 Position·Orientation Variation, 유효성 검사.

**검증:** 양면과 허용 각도 영역에서 초기 상태를 반복 생성한다. Wrist 3 분포, 초기 최소 거리, IK 오차, 거절 사유·비율을 본다.

**완료 기준:** 두 Mode가 모두 채택되고, 금지 방향과 초기 접촉·Wrist Flip이 없어야 한다.

### 단계 5. Observation·Reward·Termination

**구현:** 57D 입력과 모든 Reward·종료 조건.

**검증:** 같은 상태에서 Reward 항을 하나씩 읽고, 목표에 가까워질 때 목표 보상이 증가하는지 확인한다. 기울기·판 하중·높이를 단계적으로 증가시켜 Soft→Hard 응답을 확인한다. 성공·실패·Timeout이 각각 올바른 이유로 종료되는지 본다.

**완료 기준:** Actor GT 누출·정규화 오류·부호 오류·dt 중복 적용·종료 우선순위 오류가 없다.

### 단계 6. 소규모 학습과 문제 분리

**구현:** 제안 방식 하나로 제한된 유효 분포에서 학습한다.

**검증:** 초기 무접촉→첫 접촉→목표 이동의 흐름이 만들어지는지 확인한다. 실패는 초기화, 센서, 제어, Reward, 목표 미달로 구분한다.

**완료 기준:** 목표 이동을 보상과 로그로 추적할 수 있고, 반복적인 실패의 원인을 특정할 수 있다. 학습 결과가 없으면 완료로 표시하지 않는다.

### 단계 7. Feasibility 범위 학습

**구현:** 같은 단일 Cube·고정 판·고정 Base·고정 Gain을 유지하고 허용 방향·양면·초기 Variation 범위를 확장한다.

**검증:** 학습과 분리한 초기 상태 집합으로 성공·실패를 측정하고, 방향과 작업면별 실패를 확인한다.

**완료 기준:** 시연 영상 한 개가 아니라 재현 가능한 실행 기록과 성공률·오차·종료 이유를 확보한다. 이 단계의 수치 통과 목표와 Episode 수는 초기 분포 확정 후 사전에 정한다.

위 검증은 센서 조합 간 비교 실험이 아니라 **구현이 의도대로 작동하는지와 제안 정책의 기본 수행 가능성을 확인하는 절차**다.

## 13. 학습 전에 설정표로 확정할 값

아래 값은 빈 상태로 학습을 시작하지 않는다. 구현자가 장면·제어·센서 검증으로 값을 정하고, 설정 파일 및 실험 기록에 남긴다. 현재 문서에서 임의의 숫자를 확정값으로 만들지 않는다.

| 분류 | 파라미터 |
| --- | --- |
| 기하 | 작업판 치수·높이, Cube 치수, Base–판 Transform, F/T 장착 Transform, C Offset |
| 물성 | Cube·Hand·F/T 질량·관성, 정지/동마찰, 반발, Collision Offset |
| Command | Object XY 범위, 거리 범위, 판 가장자리 여유 |
| Mode | 양면 샘플링 비율, 면별 자연스러운 IK Seed·관절 허용 구간 |
| Reset | 무접촉 Gap, 위치·Orientation Variation, IK 허용 오차, 재시도 상한 |
| 제어 | Physics dt, Policy Decimation, OSC Kp/Kd, Action Scale, Effort·속도·Hand 변화량 제한 |
| 촉각 | 영역 배치, Cell/영역 집계 방식, 감지 임계값 |
| F/T | Frame·부호, 영점·중력 보정, Filtering, Scale·Clip |
| Reward | σp, αi, β, 각 항 가중치, Terminal 항 사용 여부·크기 |
| 위험 | γsoft/γterm, Fsoft/Fterm, hsoft/hterm |
| 시간 | Tmax, Tref |
| 실행 | Parallel Env 수, 실제 Runner, Seed, 저장·평가 주기 |

각 위험 변수는 **Soft < Hard**를 만족해야 한다. 판 하중 임계값은 무접촉 수치 잡음과 약한 접촉 측정 후, 높이·전도 임계값은 실제 Cube 치수와 정상 밀기 시 변동을 기준으로 정한다.

## 14. 반드시 남길 로그

| 구분 | 기록 |
| --- | --- |
| 재현 정보 | 저장소·커밋, 자산 버전, 설치 버전, 설정, Seed |
| Episode 초기값 | Mode, θ, L, 초기 Cube·C Pose, 실제 Arm·Hand 상태 |
| Reset 품질 | IK 오차·재시도·거절 사유, 초기 최소 거리, Wrist 3 |
| 수행 | Object 실제 이동량·목표 오차, C 이동량, 첫 접촉 시간 |
| 접촉 | 양면 Tactile, 선택 면·비선택 면 접촉, 접촉 소실 |
| 하중 | Raw/보정 Wrench, Robot–판 하중, Substep 최대값 |
| 위험 | 기울기, 높이 증가, Hard 기준 초과 Flag |
| 학습 | Reward 항별 기여, Action·토크 Clip 빈도, Episode 길이 |
| 결과 | 성공, 전도, 판 하중, 높이, Timeout, 구현 오류 |

**EEF 이동 거리와 Object 이동 거리를 별도로 기록**한다. 성공은 Object 기준으로 계산한다.

## 15. 최종 완료 조건

환경 구축의 완료는 다음을 모두 만족할 때 판단한다.

- UR5e–F/T–Inspire Hand의 조립·질량·관절·Collision이 확인됐다.
- 고정 판 위 단일 Cube가 정상적인 동역학을 보인다.
- 양면 촉각과 Wrist Wrench가 의도한 신호를 생성한다.
- 작업판 센서가 Robot 직접 접촉과 Object 지지 접촉을 구분한다.
- 초기 상태는 무접촉이며, 두 작업면과 허용 각도가 학습에 포함된다.
- C 기준 6D OSC와 Hand 2D 제어가 확인됐다.
- 57D Actor 입력에 현재 Object Pose·실제 접촉 법선이 들어가지 않는다.
- 기본 Reward와 세 위험 Ramp의 부호·크기·시간 단위가 확인됐다.
- 성공·실패·Timeout이 의도한 조건과 우선순위로 종료된다.
- 제안 정책을 학습·실행하고 결과를 기록할 경로가 준비됐다.

학습 후에는 이 조건에 **반복 실행 결과**를 추가하여 feasibility를 평가한다. 시뮬레이터가 GT로 성공 시점을 판정하는 현재 구성은 실물에서의 자율 종료 판단까지 해결한 것으로 해석하지 않는다.

## 부록. 근거와 확인 범위

- 사용자 2026-09-26 지시 및 후속 확정 답변: 이 문서의 최우선 사양.
- 첨부 **졸업발표_ver1(5).pdf**, Method 9–20쪽: 입력·출력·센서·Reward·종료 초안. 실제 렌더링 및 텍스트 확인.
- [프로젝트 지침](https://github.com/7cmdehdrb/Contact-Rich-Manipulation-Context/blob/02ad8f560ebcc2b0bcadea58ba8acd6a01db880f/AGENTS.md): 범위·특권 정보·사실과 제안의 구분.
- [Method](https://github.com/7cmdehdrb/Contact-Rich-Manipulation-Context/blob/02ad8f560ebcc2b0bcadea58ba8acd6a01db880f/docs/presentation/03_Method.md): 57D·8D·기존 Frame와 양면 촉각.
- [Sweeping 학습 저장소](https://github.com/IROL-SSU/Sweeping-Policy-DRL-Sim-Train/tree/694c40aca11e38c13a70025ab177784a16e8b4ec): min 브랜치 소스 확인.
- [Contact-Rich-Manipulation](https://github.com/7cmdehdrb/Contact-Rich-Manipulation/tree/06431a2f60cebef81fcf8109a8ad4d754b73c541): Hand·F/T·학습 예제와 Submodule 확인.
- [Isaac Lab OSC 안내](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_osc.html): 동역학·Frame·구동 경로 참고.
- [Isaac Lab Contact Sensor](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sensors.html): 센서 필터·법선력·Friction 출력 구분.

이번 작업은 소스 및 문서 검토를 기반으로 한 계획 작성이다. 실제 설치 환경, USD 조립 결과, IK 도달률, 센서 측정 오차, PPO 수렴성과 성공률은 앞으로 위 단계에서 확인할 항목이다.
