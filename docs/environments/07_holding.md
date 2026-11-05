# Holding 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Holding-v0

`FusionHoldingV0EnvCfg`는 [Fusion-v1_0](06_fusion.md)를 직접 상속한다. 목표 부근에서 물체를 멈추고 유지하는 보상과 성공 판정을 추가한 계열이다.

### 관측

부모와 동일: 직접 55D(Arm 12 + 이전 행동 8 + 초기 타겟 3 + EEF 7 + 목표 극좌표 2 + 촉각 17 + 보정 F/T 6), 초기 장애물 10×4=40D, Critic 현재 타겟 3D. **원시 Actor/Critic 95/98D, MLP 71/74D**.

### 환경 구성

원통 타겟 1개와 Gray/White 장애물 각각 최대 5개, 초기 활성 수는 1/1이다. Arm 관절 위치 6 + Hand 개방 2, 4096개·10초·50Hz. 부모의 배치·가시성·초기 장면 관측·reset 물성 랜덤화를 유지한다.

### 보상

접근 최단 기록 30 + 목표 최단 기록 100(허용 거리 0) + 접촉 1.45 + **정지 Hold weight 2**가 중심이다. Hold는 다음 거리·속도 Gaussian이며 `d < 3s_p`에서 지급한다.

```math
H=\exp\left[-\ln 2\left((d/s_p)^2+(v/s_v)^2\right)\right]
```

초기 `s_p=1cm`, `s_v=5cm/s`. 장애물 변위 비중은 부모의 2.5→5.0. 행동 변화·관절 속도·선반/물체/속도 비용은 유지하며 가속도·장애물 힘·wrench 비용은 weight 0에서 커리큘럼으로 켠다.

### 터미네이션

부모 실패 조건과 10초 timeout. 타겟 과속은 1m/s, 성공은 **거리 5mm 이하·속도 1cm/s 이하를 연속 3초 유지**다. Hold의 완만한 보상 범위와 성공 반경은 별개다.

### 커리큘럼

장애물 1/1→5/5(White 3000, Gray 4500 시작·각 3000 간격). 물체/Arm/EEF 자유 속도는 0.1→0.02m/s, 1→0.7rad/s, 1.5→0.2m/s. 가속도·접촉력·wrench 비용을 점진적으로 활성화하고 문턱을 낮춘다. Hold 위치 scale은 1→2→1→0.5cm, 속도 scale은 5→2cm/s. 목표 deadzone은 Fusion-v1_0 일정을 유지한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Holding-v1 | Holding-v0 | 장애물 관측 슬롯 10→16. 물리 최대 수 5/5는 유지. **원시 119/122D**, MLP 71/74D 유지. |
| Holding-v2 | Holding-v1 | Hold weight 4, 반경 내부 Gaussian으로 변경. 반경 2cm(5k까지)→1cm(10k)→0.5cm(20k)를 Hold·성공·접촉 보상에 공유. 접촉 보상은 실제 접촉 **OR 목표 점유**. |
| Holding-v3 | Holding-v2 | Hold 보상 지급 시간을 에피소드 내 **누적 3초**로 제한. 목표 이탈 시 예산 사용을 멈추며 재진입해도 예산은 복구되지 않는다. |
| Holding-v4 | Holding-v3 | 누적 예산 제거, Hold를 2r까지 확장하고 중심에서 최대 2배. 목표 점유 시 Reaching 허용 반경 1m로 완화. 성공 조기 종료 제거, 실패/timeout까지 유지. |

각 버전의 OSC 파생은 [Holding OSC](08_holding_osc.md), 물체 랜덤화 파생은 [ObjectRandom](09_object_random.md)에 정리한다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter_fusion.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter_fusion_holding.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter_fusion_holding.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter_fusion_holding.py)
