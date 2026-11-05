# Revision 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Revision-v0

`RevisionEnvCfg`는 `ManagerBasedRLEnvCfg`의 직접 자식이다. GoalAccuracy-v0_2를 기능 기준으로 재구성했지만 Final/Holding EnvCfg·runtime을 상속하지 않는다.

### 관측

직접 **57D**, 순서는 Arm 실제 위치·속도 12 → Hand 구성 2 → EEF pose 7 → 초기 타겟 XYZ 3 → 목표 극좌표 2 → 이전 행동 8 → 촉각 17 → 보정 F/T 6. Actor는 초기 장애물 16×4=64D를 받아 **121D**, Critic은 직접 57 + 현재 타겟 3 + **현재 활성 장애물 64**로 **124D**다. 인코딩 후 **73/76D**. Critic 장애물은 가시성 누락 없이 실제 활성 물체만 쓰며 좌표 노이즈 없이 매 step 갱신한다(정렬·10% 인접 교환 유지).

### 환경 구성

타겟 1 + Gray 최대 4/White 최대 5, **물리 물체 10개**. 타겟/장애물 모두 8종을 쓰고 중앙 기준 크기에 U(0.9,1.1)을 한 번 적용한다. 종류·크기는 생성 시 고정, reset에서는 활성 구성·배치·물성을 바꾼다. OBB 배치, Gray padding 65mm 고정, 크기를 반영해 바닥을 정렬한다. 기본 **2048개**, 10초·50Hz, 상대 pose OSC 6 + Hand 2(scale 0.03m/0.15rad).

### 보상

접근 최단 기록 30 + 거리 `3.7×exp(−d/0.05)` + Hold 8 + 접촉 1.45 + 성공 timeout 30 − 행동/관절/물체/선반/접촉력/wrench/타겟 속도 비용. Hold는 목표 반경 3cm, 외곽 8cm까지의 위치 점수에 속도 Gaussian(scale 5cm/s)을 곱한다. 접촉은 **실제 타겟 접촉 AND EEF 높이 120mm 미만**(선반 기준)이다.

### 터미네이션

활성 물체 바닥이 선반보다 1cm 이상 낮아짐/기울기 0.9rad 초과, 타겟 속도 1m/s 초과, Arm 5rad/s 초과, 선반 실패. 10초 시점 목표 **3cm 이하**를 성공으로 분류하며 양쪽 timeout 분류 모두 truncation이다. 성공 조기 종료 없음.

### 커리큘럼

장애물 1/1→4/5(White 2k·Gray 2.5k 시작·각 1k 간격). 목표 반경 3cm와 Gray padding 65mm는 고정. 타겟 자유 속도 0.1→0.02m/s, 가속도·wrench·장애물 접촉 비용/문턱은 점차 강화한다. 목표 제외 반각 30°를 3k 이후 500마다 5°씩 감소한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Revision-v1 | Revision-v0 | object 비용을 기울기 −30·변위 −150·가속도 −0.05로 분리. 가속도 허용 20m/s², 장애물 힘 50N/weight −0.05, wrench 변화 20N/2Nm/weight −0.1 등 **7개 일정의 10k 값 고정**. wrench 양의 보상 예산 제거. Hold는 3cm 내부 Gaussian, 속도 5cm/s까지 감쇠 없음. 최초 진입 +30, timeout +40. |
| Revision-Test-v1 | Revision-v1 | 타겟 종류만 원통으로 고정. 크기 및 장애물 랜덤화 유지. |
| Revision-v2 | **Revision-v1** | 거리·Hold·접촉에 EEF 95mm 미만 조건. Hold weight 10, 2.5cm까지 1·3.5cm까지 Smoothstep 감소, 속도 감쇠 없음. 최초 진입 최대 +25×속도 감쇠, timeout +75. 가속도 자유 문턱 15m/s². |
| Revision-Test-v2 | Revision-v2 | 타겟만 원통 고정. 현재 runtime/종료 함수에 Test-v2 전용 임시 종료 계측 print가 있다. |
| Revision-Test-v2-GT | Revision-Test-v2 | EnvCfg 변화 없이 전용 PPO 설정이 Actor에 현재 타겟 XYZ 3D를 추가. **Actor 원시 124D·MLP 76D**, Critic 유지. Test-v2 전용 exact-type print 대상에서는 제외된다. |

v1 이후 개수 일정은 **3/3 → 10k에 3/4 → 12k에 4/4 → 14k에 4/5**이며 v2도 유지한다. v2 최초 진입 보너스는 높이 조건이 없고, 속도 감쇠로 지급액이 작아도 최초 진입 기회는 소진한다. 일반 v0/v1/v2와 Test는 관측 차원이 같다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/revision_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/revision.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/revision.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/revision.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/revision.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/revision.py) · [Actor/Critic 입력](../../src/inspire_sweep_baseline/inspire_sweep_baseline/agent/revision_ppo_cfg.py)
