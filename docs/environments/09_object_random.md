# Holding ObjectRandom 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Holding-OSC-ObjectRandom-v0

`FusionHoldingOSCObjectRandomV0EnvCfg`는 [Holding-OSC-v0](08_holding_osc.md)를 직접 상속한다. 원통 타겟은 유지하고 장애물을 여러 NVIDIA 선반 물체 에셋으로 바꾼 구성이다.

### 관측

직접 **55D**(Arm 12 + 이전 행동 8 + 초기 타겟 3 + EEF 7 + 목표 극좌표 2 + 촉각 17 + F/T 6), 초기 장애물 **14×4=56D**, Critic 현재 타겟 3D. **원시 Actor/Critic 111/114D, MLP 71/74D**.

### 환경 구성

원통 타겟 1 + Gray 7/White 7로 **물리 15개**. 장애물 형상은 생성 시 정해지고 reset에서 배치·질량·마찰을 샘플링한다. 큰 물체의 기하를 반영한 배치 함수로 교체한다. 기본 4096개·10초·50Hz, Arm 상대 OSC 6 + Hand 2는 부모 설정이다.

### 보상

Holding-v0의 접근 30·목표 진전 100·접촉 1.45·정지 Gaussian Hold 2와 비용 구성을 상속한다. 랜덤 장애물에도 기울기·변위·접촉력 비용을 적용한다.

### 터미네이션

Holding-v0와 같다. 5mm·1cm/s·3초의 성공 종료, 물체 낙하/기울기·과속·Arm 속도·선반 실패·10초 timeout.

### 커리큘럼

**v0는 장애물 수 커리큘럼을 끄고 처음부터 7/7**이다. 나머지 Hold·속도·충격·목표 방향 일정은 부모에서 유지한다.

## 파생 구성

동일 파일에 묶여 있지만 다음 환경들은 ObjectRandom-v0의 자식이 아니라 서로 다른 Holding/OSC 부모에서 분기한다.

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Holding-OSC-ObjectRandom-v1 | Holding-OSC-v2 | 랜덤 물체 Scene, 물리 최대 Gray/White 8/8, 관측 16슬롯. 활성 1/1→8/8의 개수 일정. 공통 반경 Hold/성공·접촉 OR 목표 점유 유지. |
| Holding-ObjectRandom-Test-v0 | **Holding-v2** | v1과 같은 랜덤 물체/개수 일정이지만 Arm은 관절 위치 제어. |
| Holding-OSC-ObjectRandom-v2 | Holding-OSC-v3 | 물리 용량 8/8 중 활성 최대 6/6. White 3000·Gray 4000 시작, 각 2000 간격. Gray 영역 안쪽 경계 침범 비율 0→15%(6k→12k). 누적 3초 Hold 예산 유지. |
| Holding-OSC-ObjectRandom-v3 | Holding-OSC-v4 | v2와 같은 물체/개수/padding 구성, 보상·종료는 Holding-v4(무제한 중심 가중 Hold, 목표 점유 Reaching, 성공 조기 종료 없음). |

위 네 구성은 원시 **119/122D**, MLP 71/74D다. [Final-v0](10_final.md)는 ObjectRandom-v3를 설계 기준으로 삼았지만 독립 EnvCfg다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_object_random_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter_fusion.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter_fusion_holding.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter_fusion_holding.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter_fusion_holding_osc_object_random.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/contact_sweep_clutter_fusion_holding_osc_object_random.py)
