# Contact Sweep Clutter Fusion 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Contact-Sweep-Clutter-Fusion-v0

`ContactSweepClutterFusionEnvCfg`는 `ManagerBasedRLEnvCfg`의 직접 자식이다. Clutter의 설계를 참고한 독립 조합이며 Clutter EnvCfg를 상속하지 않는다.

### 관측

직접 관측 **55D**: Arm 상대 위치·속도 12 + 이전 행동 8 + 초기 타겟 XYZ 3 + EEF pose 7 + 초기 목표 극좌표(거리·각도) 2 + 촉각 17 + 보정 F/T 6. 물체 폭과 목표 XYZ 대신 위 구성을 쓴다. 초기 장애물 40D(10×4), Critic 현재 타겟 XYZ 3D를 별도 제공한다. **Actor/Critic 원시 95/98D, 인코딩 후 71/74D**.

### 환경 구성

원통 타겟 1개와 장애물 3개, Arm 위치 6 + Hand 개방 2, 4096개·10초·50Hz. Reset에서 시작 반경 2cm·목표 거리 15cm의 clutter 배치와 물성을 샘플링한다. 초기 장애물 관측은 가시성 문턱 0.35, 가시성 반전 확률 1%, 위치 ±1cm 오차를 반영한다. 초기 ±X 목표 제외 반각은 60°다.

### 보상

접근 최단 기록 30, 목표 최단 기록 100, 목표 Hold 0.9, 접촉 유무/개수 각 0.4. 여기에 행동 변화·관절 속도, 물체 기울기/장애물 변위, 선반 충돌/실패, 물체·Arm·EEF 과속, EEF 방향, wrench 변화율 비용을 합친다. wrench 비용은 양의 누적 보상 5% 예산을 가진다.

### 터미네이션

10초, 물체 낙하/기울기, 타겟 1m/s 초과, Arm 5rad/s 초과, 선반 충돌/변위. 성공 종료 없음. v3 계보에서는 타겟 과속 한계가 0.8m/s로 바뀐다.

### 커리큘럼

물체 자유 속도 0.1→0.05m/s, Arm 1→0.8rad/s, EEF 1.5→0.7m/s, 행동 변화 weight −0.03→−0.06. 목표 제외 반각 60°를 500 iter 유지 후 200 iter마다 1°씩 줄인다.

## 파생 환경

아래 이름은 모두 `Contact-Sweep-Clutter-Fusion-` 뒤의 버전이다. **번호 순서가 상속 순서가 아니다.**

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| v2 | v0 | EEF 방향 비용만 제거. |
| v3 | v2 | 속도/행동 변화 일정을 500 iter 시작·200 간격으로 변경. 목표 deadzone은 1000 유지·100마다 1° 감소. 접촉 보상 weight 증가와 wrench 변화율 문턱 감소 일정 추가. 타겟 종료 속도 0.8m/s. |
| v1_0 | **v3** | 가변 장애물 Gray/White 각 1→3, 비활성 물체 제외, 인접 관측 교환 10%. 접촉을 hand_contact 0.29로 통합하고 Arm/Hand 비용 분리. 장애물 접촉력·물체 가속도 비용 추가, wrench를 변화율 대신 변화량 기반으로 교체. 목표 제외 반각 30°, 3000 이후 500마다 −5°. |
| v1_1 | v1_0 | 장애물 힘 허용 2000→100N, wrench 허용 5000/500→500N/50Nm. 힘·가속도·wrench 문턱을 10k/20k까지 줄이는 staged 일정. |
| v1_1_1 | v1_1 | 가속도 실험: 장애물 힘 2000N, wrench 5000N/500Nm로 완화하고 해당 일정 제거. 가속도 일정 유지. |
| v1_1_2 | v1_1 | 힘 실험: 가속도 허용을 1,000,000m/s²로 올리고 가속도 일정 제거. 장애물 힘·wrench 일정 유지. |
| v1_2 | v1_1 | Hold 반경 2cm→1cm→0.5cm 일정, 목표 부근 접촉 보상 fade 폭 1cm. |
| v1_3 | v1_1 | 힘·wrench·가속도 문턱을 여러 절점의 선형 일정으로 변경. 초기 장애물 힘 200N, wrench 1000N/100Nm. |
| v1_3_1 | v1_3 | Hold 반경 2→1→0.5cm(5k/10k/20k), 접촉 fade 폭 1mm→1cm(5k→10k). |
| v1_4 | **v1_0** | 장애물 각 5개까지, Hold weight 9·접촉 1.45. Hold 초기 비활성 후 반경 축소, 충격/속도 일정 재구성. 목표 3초 유지 성공 종료 추가, 타겟 종료 속도 1m/s. |

[Holding-v0](07_holding.md)는 **v1_0**의 별도 직계 파생이다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter_fusion.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter_fusion.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter_fusion.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/contact_sweep_clutter_fusion.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter_fusion.py)
