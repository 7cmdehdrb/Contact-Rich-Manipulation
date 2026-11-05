# Holding OSC 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Holding-OSC-v0

`FusionHoldingOSCV0EnvCfg`는 [Holding-v0](07_holding.md)의 직접 자식이다. 과제 설정을 유지하고 Arm 제어만 상대 pose OSC로 교체한다.

### 관측

직접 55D = Arm 위치·속도 12 + 이전 행동 8 + 초기 타겟 3 + EEF 7 + 목표 극좌표 2 + 촉각 17 + F/T 6. 초기 장애물 40D와 Critic 타겟 3D를 더해 **원시 95/98D, MLP 71/74D**다. 이전 Arm 행동의 의미는 관절 위치 명령에서 OSC pose 명령으로 바뀐다.

### 환경 구성

원통 타겟 + 최대 Gray 5/White 5, 초기 1/1. 배치/물성 reset과 4096개·10초·50Hz는 Holding-v0와 같다. 행동은 **상대 위치·회전 6 + Hand 개방 2**. v0 OSC는 robot-root 축, 위치/회전 scale 1/1, stiffness 100, damping ratio 1, 관성 분리·중력 보상을 사용하며 경쟁하는 Arm 관절 PD를 해제한다.

### 보상

Holding-v0 그대로: 접근 30 + 목표 진전 100 + 접촉 1.45 + 거리·속도 Gaussian Hold 2 − 행동/관절/물체/선반/과속/충격 비용. OSC라는 이유로 보상식이나 계수를 별도로 바꾸지 않는다.

### 터미네이션

Holding-v0와 동일. 5mm·1cm/s 이하 3초 유지 성공, 낙하/기울기·과속·선반 실패, 10초 timeout.

### 커리큘럼

Holding-v0의 장애물 개수, 자유 속도, Hold scale, 충격 비용·문턱, goal deadzone 일정을 그대로 적용한다.

## 버전별 직접 부모

**OSC-v1이 OSC-v0를 상속하는 직렬 구조가 아니다. 각 버전은 대응 Holding 환경을 직접 상속한다.**

| 환경 | 직접 부모 | 부모 대비 변경점 |
| --- | --- | --- |
| Holding-OSC-v0 | Holding-v0 | 위 기본 OSC 적용. |
| Holding-OSC-v1 | Holding-v1 | 동일 OSC 적용. 16슬롯으로 원시 119/122D. |
| Holding-OSC-v2 | Holding-v2 | Reach OSC 적용: 위치 scale 0.03m·회전 0.15rad, 병진 stiffness 100·회전 50, effort 제한 적용. 부모의 공통 목표 반경 유지. |
| Holding-OSC-v3 | Holding-v3 | v2와 같은 Reach OSC. 부모의 누적 3초 Hold 예산 유지. |
| Holding-OSC-v4 | Holding-v4 | v2와 같은 Reach OSC. 부모의 중심 가중/외곽 Hold, 목표 점유 Reaching, 성공 종료 제거 유지. |

v1~v4 모두 원시 119/122D, MLP 71/74D다. ObjectRandom-v0/v1/v2/v3의 부모는 각각 OSC-v0/v2/v3/v4다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter_fusion.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter_fusion_holding.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter_fusion_holding.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter_fusion_holding.py) · [OSC 설치 함수](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/common/contact_sweep_clutter_fusion_holding_osc_configuration.py)
