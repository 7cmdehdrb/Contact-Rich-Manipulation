# Contact Sweep Clutter 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

## 베이스: Contact-Sweep-Clutter-v0

`ContactSweepClutterEnvCfg`는 [Contact-Sweep-v0](04_contact_sweep.md)의 직접 자식이다. 단일 물체 밀기에 장애물과 초기 장면 인코더를 추가한다.

### 관측

직접 관측 57D는 Contact Sweep과 같다(Arm 12, 행동 8, 타겟 XYZ·폭 4, EEF 7, 목표 3, 촉각 17, F/T 6). 초기 장애물은 **10슬롯 × [존재 여부, XYZ] = 40D**. 따라서 Actor/Critic 원시 입력은 **97D**, 장애물을 16D로 인코딩한 MLP 입력은 **73D**다. 물리 장애물 3개와 관측 슬롯 10개는 구분한다.

### 환경 구성

원통 타겟 1 + 장애물 3. 부모의 8D 관절 위치/손 개방 행동과 10초·50Hz를 유지한다. Reset에서 장애물 배치와 초기 관측(±1cm 잡음), 물체별 물성도 구성한다.

### 보상

Contact Sweep의 접근 30·목표 94와 행동/속도/충돌 비용을 유지한다. 변경점은 기울기 비용을 타겟뿐 아니라 모든 물리 물체에 적용하는 것이다.

### 터미네이션

Contact Sweep의 조건을 유지하되 **어느 물체든** 낙하/기울어지면 종료한다. 성공 조기 종료는 없다.

### 커리큘럼

v0에는 없음. 별도 `Curriculum-v1`에서 추가한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| Clutter-v1 | Clutter-v0 | Actor 타겟 XYZ를 초기값으로 고정, Critic 현재 XYZ 3D 추가. **원시 97/100D, MLP 73/76D**. 목표 진전 94→100, 물체 자유 속도 0.2→0.1m/s. Hold(0.9), 접촉 유무/개수(각 0.16), 장애물 변위(−5), 선반 실패 이벤트(−3) 추가. |
| Clutter-Curriculum-v1 | Clutter-v1 | 물체 자유 속도 0.1→0.05(3000 iter마다 −0.01), Arm 1→0.8(5000마다 −0.05), Arm 행동 변화·관절 속도 weight −0.03→−0.06(3000마다 −0.005). |
| Clutter-FT-Test-v0 | Clutter-Curriculum-v1 | EnvCfg 변경 없음. 등록 runtime에서 보정 F/T 계측을 수행한다. 정책 입력·보상은 부모 계약을 유지한다. |
| Clutter-v2 | **Clutter-v1** | 손목 관측을 보정 wrench로 교체. wrench 변화 비용 weight −0.1 추가(변화 허용 0.5N/0.01Nm, 양의 누적 보상 5% 예산). Curriculum-v1을 상속하지 않는다. |

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/contact_sweep_clutter.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter.py)
