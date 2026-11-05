# Contact Sweep 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

[전체 관측표](#전체-관측-항목) · [전체 보상표](#전체-보상-항목) · [전체 터미네이션표](#전체-터미네이션-항목) · [전체 커리큘럼표](#전체-커리큘럼-항목)

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








<!-- TERM_TABLES_START -->
## 버전별 전체 Manager 항목

이하 표는 **부모 상속과 EnvCfg의 후처리까지 반영한 전체 구성**이다. 동일한 구성을 쓰는 버전은 표 하나를 공유한다. 표의 항목명은 코드/학습 로그의 이름이며 구현 함수 링크로 계산을 확인할 수 있다. 보상 weight는 커리큘럼 적용 전 설정값이다. **weight 0은 설정된 항목이고, `None`은 제거된 항목**이다. 일정은 학습 iteration 기준이다.

[관측](#전체-관측-항목) · [보상](#전체-보상-항목) · [터미네이션](#전체-터미네이션-항목) · [커리큘럼](#전체-커리큘럼-항목)

### 전체 관측 항목

#### 관측 구성 1

적용: `Contact-Sweep-v0`.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 기본 자세 대비 상대 각도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_pos_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 기본 속도 대비 상대 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_vel_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.actions` | 8 | 직전 정책 행동 8개(Arm 6 + Hand 2). | 추가 ObservationTerm 노이즈 없음 | [last_action](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obs_state` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 매 관측 균등 노이즈 -0.01~0.01 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obj_width` | 1 | 타겟의 폭/크기 대표값(target_width). | 매 관측 균등 노이즈 -0.01~0.01 | [object_width](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.goal_pos` | 3 | 목표 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [goal_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.tactile` | 17 | 손의 17개 촉각 부위별 접촉 유무. 힘을 부위 방향에 투영해 문턱과 비교. | 추가 ObservationTerm 노이즈 없음<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [ProjectedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/tactile_observations.py) |
| `policy.wrist_wrench` | 6 | 손목 센서의 힘 XYZ·모멘트 XYZ를 EEF 기준으로 변환한 wrench. | 추가 ObservationTerm 노이즈 없음 | [wrist_wrench_eef](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 전체 보상 항목

#### 보상 구성 1

적용: `Contact-Sweep-v0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 94 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0.01` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object_tilt` | -2 | 타겟 기울기가 free_angle을 넘은 부분의 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -2 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.2`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

### 전체 터미네이션 항목

#### 터미네이션 구성 1

적용: `Contact-Sweep-v0`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 타겟이 기준 높이 아래로 떨어지거나 기울기 한계를 초과하면 실패. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `0.8` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |

`None`으로 비활성: `object_outside_shelf`.

### 전체 커리큘럼 항목

#### 커리큘럼 구성 1

적용: `Contact-Sweep-v0`.

설정된 항목 없음.

<!-- TERM_TABLES_END -->

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/contact_sweep.py)
