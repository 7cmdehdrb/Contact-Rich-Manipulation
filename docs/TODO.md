# 후속 연구·개발 TODO

기준일: 2026-10-23. 사용자 요청에 따라 이 파일에 다섯 가지 후속 과제를 정리한다. 현재 상태는 문서와 `src/inspire_sweep_baseline` 코드를 대조한 결과이며, 아래 체크박스는 앞으로 수행할 작업이다. 실험 수치·평가 조건은 실행 전에 확정하고, 실제 결과와 개발 이력은 문서 전용 저장소에 남긴다.

참고: [프로젝트 인계](../example/Contact-Rich-Manipulation-Context/01_plan/00_HANDOFF_BRIEF.md) · [활성 환경](../example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md) · [기존 연구 TODO](../example/Contact-Rich-Manipulation-Context/01_plan/TODO.md).

## 1. Arm 컨트롤러를 OSC로 전환

**현재 상태:** Clutter Fusion의 [Action 설정](../src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_fusion/actions_cfg.py)은 Arm 6D Joint Position과 Hand 2D openness를 사용한다. 별도의 [Dex Renewal OSC v2](../src/inspire_sweep_baseline/inspire_sweep_baseline/dex_tactile_omni_renewal_v2_osc_env_cfg.py)는 이미 구현되어 있으며 stiffness 100, 상대 위치/회전 명령 한도 5 mm/0.025 rad를 사용한다. 그러나 [기존 검증 기록](../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-18_dex_tactile_renewal_v2_osc.md)에는 0 명령에서도 지속 과속 문제가 남아 있다.

- [ ] 비교 기준으로 사용할 Contact Sweep/Clutter Fusion 버전·commit·설정·checkpoint를 고정하고, OSC를 적용할 부모 환경을 명시한다.
- [ ] 기존 [bounded OSC Action](../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/osc_actions.py)을 검토해 Arm만 OSC로 바꾸는 파생 환경을 구성한다. Hand 제어와 비교 대상의 관측·보상·종료 조건을 유지한다.
- [ ] Arm Action을 Cartesian 위치·회전 증분으로 정의하고, 좌표계·EEF 기준점·offset·단위·정책 주기·physics 주기를 문서화한다. 6D라는 차원이 같아도 기존 Joint Position 정책과 행동 의미가 다르므로 기존 checkpoint를 그대로 호환된다고 보지 않는다.
- [ ] Jacobian·질량 행렬·중력 보상·Arm joint PD 비활성화·토크 제한·특이 자세를 점검한다. 명령 크기 제한만으로 실제 관절 속도가 제한된다고 가정하지 않는다.
- [ ] 무접촉 0 명령 유지 → 작은 병진/회전 명령 → 선반 근처 이동 → 단일 물체 접촉 → 다중 물체 접촉 순서로 진동·과속을 재현하고 원인을 분리한다. gain·damping·제어 주기 변경 전후를 기록한다.
- [ ] 동일 초기 조건에서 Joint Position과 OSC의 EEF 추종 오차, 관절 최대 속도, 토크 포화, F/T peak, 물체 목표 오차·전도율을 비교한 뒤 별도 학습·평가한다.
- [ ] Manager 기반 책임 분리와 기존 운동 페널티 계약을 유지한다. Omni 파생에서는 로컬 `goal_rewards.action_rate_l2`·`joint_vel_l2`, Arm/Hand 분리, raw 변화량·실제 속도 제곱합 및 지정 계수(Arm −0.03/−0.03, Hand −0.03/−0.01)를 임의로 바꾸지 않는다.

**완료 기준:** 0 명령 유지와 반복 접촉에서 안정성을 확인하고, 기존 제어 대비 정량 비교표·재현 명령·새 task 등록 및 상속 안내를 남긴다. 짧은 smoke test와 장시간 학습 성과는 따로 기록한다.

## 2. 최대 7개를 넘어 고밀도 물체 스폰으로 확장

**현재 상태:** Fusion v1_0의 [개수 커리큘럼](../src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_fusion_v1_0/population.py)은 기본적으로 타겟 1개 + GRAY/WHITE 각각 1개에서 각각 3개로 증가한다. 따라서 최대 7개는 타겟을 포함한 수다. [Scene](../src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_fusion_v1_0/scene_cfg.py)은 고정 슬롯과 비활성 mask를 사용하며, 현재 40D 장애물 관측은 최대 장애물 10개라는 별도 제한을 가진다.

- [ ] 밀도를 물체 수, 사용 가능한 선반 면적 대비 바닥 투영 점유율, 물체 사이 최소 간격, 타겟 주변 이웃 수로 정의한다. 같은 개수에서도 군집·틈새·타겟 주변 밀집 조건을 분리한다.
- [ ] 기존 7개 조건을 기준선으로 유지하고, 10개·15개·20개 등은 후보 단계로 검토한다. 최종 개수는 실제 에셋 크기·선반 면적·배치 성공률에 따라 정한다.
- [ ] [배치 생성기](../src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter/layout.py)의 RED 제외 영역, 물체 간격, 셀 점유·이웃 수 제한, 접근 경로 여유를 검토한다. 단순히 개수만 올리지 말고 고밀도에서도 가능한 분포와 불가능한 조건을 구분한다.
- [ ] 기존의 여유 있는 배치와 물체끼리 인접해 다중 접촉이 발생하는 배치를 별도 난도로 설계한다. 목표·접근 가능성의 기존 보장을 변경할 경우 그 변경을 명시하고, 생성 실패를 숨기기 위해 목표를 임의로 바꾸지 않는다.
- [ ] 장애물 10개 제한을 넘기기 전에 Scene 슬롯·40D 관측·encoder·접촉 filter·active mask·보상·낙하/전도 판정의 고정 크기 가정을 함께 확장한다. 초기 관측의 가림·노이즈 및 현재 GT의 Actor 유입 여부도 확인한다.
- [ ] 개수와 밀도를 분리한 커리큘럼을 만들고, episode 도중 active mask가 변하지 않는지와 checkpoint 재개 시 단계가 복원되는지 검증한다.
- [ ] seed별 배치 생성 실패율·재시도 횟수·reset 시간·관통 여부를 검사한다. GPU 메모리, 접촉 버퍼 용량, steps/s를 측정해 병렬 환경 수를 결정한다.
- [ ] 밀도별 목표 성공률, 비타겟 변위, 접촉 상실, 전도·낙하율과 실패 영상을 수집한다. 학습에 사용하지 않은 배치 seed를 평가에 사용한다.

**완료 기준:** 7개 초과의 유효한 고밀도 장면을 반복 생성하고, 밀도별 평가표·배치 실패 원인·실행 비용을 제시한다. 개수 증가 자체를 연구 성과로 간주하지 않는다.

## 3. 실사용 물체 에셋 도입

**현재 상태:** [Fusion Scene](../src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter_fusion/scene_cfg.py)은 `RealisticCylinderSpawnerCfg`로 타겟을 만들고 장애물도 복제한다. 시각 mesh 교체뿐 아니라 물리 형상과 배치 가정을 함께 바꿔야 한다.

- [ ] 실제 선반에서 사용할 물체 후보를 정한다. 예: 상자형 포장품, 병·용기, 캔, 비대칭 생활용품. 확보 가능한 실물과 대응되는 소수의 대표 물체부터 시작한다.
- [ ] 에셋별 출처·라이선스·파일 경로·실물 식별자·치수·질량·재질·무게중심 측정 여부를 목록으로 관리한다. 아직 확보하지 않은 에셋은 후보로 표시한다.
- [ ] USD 변환 시 단위·축·원점·스케일을 통일하고, 시각 mesh와 collision mesh를 구분한다. 충돌 근사, 관성, 무게중심 및 바닥 접촉면을 실물과 대조한다.
- [ ] 실린더 공통 반경 기반 배치를 물체별 footprint·방향·높이를 반영하는 방식으로 확장한다. 회전한 상자·비대칭 물체의 초기 관통, 선반 경계, 접근 여유를 검사한다.
- [ ] 타겟/장애물 역할을 에셋 종류와 분리하고, 물체별 질량·마찰을 지정할 수 있게 한다. 현재 실린더 이름·동일 형상·같은 물성 가정에 의존하는 reset, contact, reward, termination 경로를 점검한다.
- [ ] 에셋별 안정된 초기 자세와 허용 자세를 정의한다. 기울어진 외형이나 굴러가는 물체에서도 전도·낙하 판정이 의도대로 동작하는지 확인한다.
- [ ] 단일 에셋 → 혼합 에셋 → 고밀도 혼합 장면 순서로 검증하고, 학습 물체와 미학습 물체를 분리해 형상 일반화를 평가한다.

**완료 기준:** 실물과 대응하는 에셋 목록·물성 설정·reset 검증 결과를 확보하고, 기존 실린더 대비 접촉·미끄러짐·회전·전도 차이를 설명한다.

## 4. 동일 동작의 현실·시뮬레이션 거동 비교와 물성 재조정

**현재 상태:** [물성 randomization](../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/realistic_physics.py)과 [Clutter 물성 적용](../src/inspire_sweep_baseline/inspire_sweep_baseline/contact_sweep_clutter/physics.py)이 존재한다. Clutter에서는 기본 `mass_scale=1.5`를 사용하고 episode에서 샘플링한 질량을 collection 전체에 적용한다. 이 설정이 개별 실물의 동역학을 재현한다는 검증과는 구분한다.

- [ ] 동일 물체·선반 표면·초기 pose·Hand 자세·접촉 높이를 준비하고, 직선 밀기·속도 변화·정지·편심 접촉의 반복 가능한 기준 궤적을 정의한다.
- [ ] 동일 명령 재생과 실제 EEF 궤적 일치를 구분한다. 명령값뿐 아니라 실측 EEF pose·속도·Hand 상태를 저장해 제어 추종 오차와 물체 물성 오차를 분리한다.
- [ ] 실물·시뮬레이션의 시간축, 좌표계, 단위, F/T 모멘트 기준점을 맞춘다. raw/tare/자중·관성 보정 신호를 구분하고 촉각 활성 시점과 지연도 기록한다.
- [ ] 물체 위치·yaw 궤적, 최종 변위, 미끄러짐 시작 시점, 정지 거리, 전도 여부, F/T peak·시간 적분, 촉각 접촉 패턴을 같은 기준으로 비교한다. 외부 영상·추적은 평가용이며 Blind Actor 입력과 분리한다.
- [ ] 질량·치수부터 실측하고, 물체–선반 및 Hand–물체의 정지/동마찰을 구분해 추정한다. 무게중심·관성·반발·접촉 형상도 검토하되 여러 계수를 동시에 임의 조절하지 않는다.
- [ ] 접촉 mesh, solver, timestep, controller 오차를 점검한 뒤 물성 보정을 수행한다. 제어 불안정을 마찰이나 질량 변경으로 상쇄하지 않는다.
- [ ] 보정용 궤적과 검증용 궤적·속도·초기 자세를 분리한다. 반복 실험의 변동 범위를 기록하고, 오차 허용 기준을 사전에 정해 보정 전후를 비교한다.
- [ ] 측정 불확실성과 물체 간 편차를 바탕으로 randomization 범위를 갱신하고, 보정된 물성에서 정책을 재학습·평가한다. 설정 버전·실측 데이터·실행 명령을 함께 보관한다.

**완료 기준:** 같은 동작의 실물/시뮬레이션 궤적 및 센서 비교 그림, 보정 전후 파라미터 표, 미사용 조건에서의 오차 평가를 확보한다. 실물이 준비되지 않으면 준비·시뮬레이션 결과만 기록하고 현실 검증은 미완료로 남긴다.

## 5. 논문 가제본 작성

**현재 상태:** [발표 설계](../example/Contact-Rich-Manipulation-Context/01_plan/presentation/README.md)와 [문헌조사](../example/Contact-Rich-Manipulation-Context/01_plan/literature/README.md)가 있다. 발표의 설계, 현행 환경 구현, 학습 결과, 실물 검증을 구분해 원고로 옮긴다.

- [ ] 가제·연구 질문·기여 후보를 정리한다. 초기 시각 정보 이후 지속 시각 갱신 없이 F/T·촉각으로 지정 물체를 미는 하위 정책이라는 범위와 접근 계획·상위 탐색의 경계를 명시한다.
- [ ] 서론 → 관련 연구 → 문제 정의 → 방법 → 실험 → 결과·분석 → 한계·결론의 전체 초안을 작성한다. 결과가 없는 절도 필요한 실험·표·그림을 표시해 가제본의 구조를 완성한다.
- [ ] 방법 절에 실제 채택한 task 버전, Actor/Critic 정보 경계, Action·OSC, 촉각 표현, F/T 보정, 보상·종료, 개수/밀도 커리큘럼과 물성 설정을 기술한다. 과거 발표 차원을 현행 구현값으로 그대로 쓰지 않는다.
- [ ] 비교 실험표를 설계한다: Joint Position/OSC, 센서 없음/촉각/F/T/둘 다, 밀도, 실린더/실물 대응 에셋, 물성 보정 전후. 한 번에 모든 조합을 확정하지 말고 연구 질문별 필수 비교와 통제 조건을 정한다.
- [ ] 성공률·목표 오차·비타겟 변위·전도/낙하·완료 시간·접촉/F/T 지표를 정의하고, seed·반복 수·학습 예산·평가 분할·변동성 보고 방식을 사전에 명시한다.
- [ ] 시스템 구성도, 관측/제어 흐름, 밀도·에셋 예시, 센서별 비교, 실물–시뮬레이션 비교, 대표 실패 사례의 그림·표 자리를 만든다. 각 주장과 그림에 실제 근거 로그·설정·데이터를 연결한다.
- [ ] 관련 연구의 원문과 서지정보를 확인하고 센서 구성·관측 조건·실물 검증 범위로 비교한다. 기존 실험 보고를 현재 장비에서 얻은 결과로 쓰지 않는다.
- [ ] 확보된 결과만 본문에 채우고 미실험 항목은 명시적으로 표시한다. 지도교수 검토용 PDF를 만든 뒤 논리·용어·그림·인용·재현 정보와 남은 실험을 점검한다.

**완료 기준:** 모든 장이 연결된 가제본 PDF와 원본, 주장–근거 대응표, 미완료 실험 목록을 확보한다. 미확인 수치나 예상 성과를 결과로 기재하지 않는다.

## 권장 진행 순서와 공통 기록

1. 비교 기준 환경을 고정하고 OSC의 안정성 문제부터 확인한다. 논문 목차·관련 연구·문제 정의는 동시에 작성한다.
2. 실물 대응 에셋을 소수 확보해 단일 물체의 거동을 비교하고 물성을 보정한다. 고밀도 배치 생성 검증은 병행할 수 있다.
3. 안정된 제어·에셋·물성을 고밀도 장면에 통합해 학습 및 일반화 실험을 수행한다.
4. 조건별 정량 결과와 실패 분석을 가제본에 반영한다.

- [ ] 구현 시 Manager별 책임을 유지하고 부모 환경·task ID·실행 명령을 [활성 환경 안내](../example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md)에 갱신한다.
- [ ] [루트 README](../README.md)의 실행·검증 안내에 따라 변경에 맞는 CPU 검사와 Isaac Sim 검증을 나누어 수행한다. command·commit·seed·checkpoint·실제 결과·미검증 범위를 개발 로그에 기록한다.
- [ ] 사용자 작업과 독립 submodule의 변경을 보존하고, 학습 로그 정리는 별도 명시 요청이 있을 때만 수행한다.
