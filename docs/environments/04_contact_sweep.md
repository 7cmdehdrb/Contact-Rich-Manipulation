# Contact Sweep 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Contact-Sweep-v0

`ContactSweepEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. Renewal의 실험 구성을 기준으로 삼았지만 EnvCfg 상속은 독립이다.

### 관측

Actor/Critic **57D**: Arm 상대 위치 6 + 속도 6 + 이전 행동 8 + 현재 타겟 XYZ 3 + 폭 1 + EEF pose 7 + 목표 XYZ 3 + 촉각 17 + F/T 6. 타겟 위치·폭 ±1cm, 촉각 접촉 문턱 0.01N이다.

### 환경 구성

UR5e·Inspire·F/T·선반·원통 1개. Arm 관절 위치 6 + Hand 개방 2의 8D 행동. 기본 4096개·10초·50Hz. 시작 반경 2cm, 목표 거리 15cm, ±X 방향 제외 반각 15°. 매 reset 질량(Realistic 분포의 1.5배)·마찰과 배치를 랜덤화한다.

### 보상

`30×새 접근 최단 거리 갱신 + 94×새 목표 최단 거리 갱신`이 중심이다. 접근 반경 10cm·목표 허용 1cm. 비용은 Arm/Hand 행동 변화(각 −0.03), 관절 속도(−0.03/−0.01), 기울기(−2), 선반 충돌(−0.5), 물체 초과속도(−2, 자유 속도 0.2m/s), Arm 초과속도(−1, 자유 속도 1rad/s). 지수 속도 비용의 raw cap은 100이다.

### 터미네이션

10초, 타겟 낙하/기울기(1.04m·0.9rad), 타겟 0.8m/s 초과, Arm 5rad/s 초과, 선반 충돌(손가락 검사 제외·손목 여유 3cm), 선반 변위 5cm 초과. 성공 종료와 선반 밖 위치 종료는 없다.

### 커리큘럼

없음.

## 파생 환경

직접 자식은 `ContactSweepClutterEnvCfg`, 즉 [Clutter-v0](05_clutter.md)다. 장애물 3개·초기 장애물 관측과 인코더를 추가하며, 기울기 비용/낙하 종료를 모든 물체로 확장한다. 그 이후 Clutter 파생들은 해당 문서에 정리했다. Fusion은 이 환경의 Python 자식이 아니다.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/contact_sweep.py)
