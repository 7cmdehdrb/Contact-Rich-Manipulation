# Final GoalAccuracy 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

[전체 관측표](#전체-관측-항목) · [전체 보상표](#전체-보상-항목) · [전체 터미네이션표](#전체-터미네이션-항목) · [전체 커리큘럼표](#전체-커리큘럼-항목)

## 베이스: Final-GoalAccuracy-v0

`FinalGoalAccuracyEnvCfg`는 [Final-v1](10_final.md)을 직접 상속한다. 목표 정밀도 평가를 위해 성공 반경을 고정하고 전용 runtime에서 정확도 지표를 기록한다.

### 관측

Final-v1과 동일한 직접 **57D**: Arm 상대 위치·속도 12 + 이전 명령 8 + 초기 타겟 3 + EEF pose 7 + 목표 극좌표 2 + 촉각 17 + 보정 F/T 6 + Hand 구성 2. 초기 장애물 20×4=80D, Critic 현재 타겟 3D. **원시 Actor/Critic 137/140D, MLP 73/76D**. 모든 Accuracy 파생이 같은 차원을 유지한다.

### 환경 구성

Final-v1의 다양한 타겟/장애물 형상, 생성 시 ±10% 크기, reset 배치·질량·마찰 랜덤화. 물리 용량은 기본 21개(타겟 1 + 장애물 20), 활성 개수는 최대 Gray 4/White 5다. 8D 상대 pose OSC/Hand, 4096개·10초·50Hz.

### 보상

부모의 접근 30·목표 최단 기록 100·접촉 OR 목표 점유 1.45와 행동/관절/물체/선반/접촉력/wrench 비용을 유지한다. **Hold는 weight 8, 반경 1cm 안에서만** 거리·속도 Gaussian과 중심 가중으로 지급하며 외곽 보상은 없다. 속도 scale 5cm/s 고정. 물체 속도 비용의 이름은 `target_speed_penalty`로 바뀐다.

### 터미네이션

부모의 물체 낙하/기울기·타겟 과속·Arm 과속·선반 실패를 유지한다. 10초 시점 XY 거리 **1cm 이하**면 성공 truncation, 아니면 실패 timeout이다. 이 판정에는 정지 속도·연속 유지 시간 조건이 없다.

### 커리큘럼

목표 반경·Hold 속도 scale 일정은 제거한다. 장애물은 1/1→**4/5**(White 2000·Gray 2500 시작, 각 1000 간격, 5000 이후 추가 증가 없음). 부모의 타겟 자유 속도, 가속도, Gray 안쪽 경계 침범 비율(0→15%), 접촉력·wrench, 목표 방향 일정은 유지한다.

## 파생 환경

| 환경 | 직접 부모 | 변경점 |
| --- | --- | --- |
| GoalAccuracy-v1 | GoalAccuracy-v0 | 1cm 반경은 고정, Hold 외곽 2r 범위와 속도 scale 5→2cm/s 일정을 복원. |
| GoalAccuracy-v0_1 | GoalAccuracy-v0 | 반경 2cm·Hold 외곽 6cm, 거리 보상 `3.7×exp(−d/0.05)`, 접촉 대상을 타겟으로 한정(목표 점유 OR 유지). Reaching은 현재 코드 **10cm 고정**, 목표 점유 완화 제거. 큐브 타겟 후보 크기를 6×8×10cm로 변경하며 다른 형상 후보는 유지. Manager 이름을 대상별로 정리. |
| GoalAccuracy-v0_2 | v0_1 | timeout 때 2cm 내부 도달 보너스 +30 추가. 이 보너스 함수는 `< 2cm`, 성공 종료 분류는 `≤ 2cm`라 경계가 다르다. |
| GoalAccuracy-v0_3 | v0_2 | 거리 항을 최단 기록 갱신(weight 100·허용 거리 0)으로 복원, 목표 반경 3cm, 접촉 weight 1.45→0.3625, 성공 timeout 보너스 +100. |
| GoalAccuracy-v0_2_1 | **v0_2** | 실제 타겟 접촉만 보상, 성공 timeout +75(2cm 경계 포함). reset마다 타겟 후보 재선택, OBB 배치·최소 간격 20mm·Gray padding 60→69mm. 후보 풀로 물리 21→36개. |
| GoalAccuracy-v0_2_2 | v0_2_1 | 물리 36→12개, 공유 물성/prototype cache와 텐서 기반 역할 재할당. 보상·관측·활성 최대 4/5·OBB 규칙은 유지. 환경별 보유 형상이 고정되어 reset 간 형상 분포의 상관은 달라진다. |

v0_1의 예전 설명에 있는 Reaching 20cm보다 **현재 코드의 10cm**를 기준으로 읽는다. v0_2_1과 v0_3은 각각 v0_2에서 갈라지는 형제다.





<!-- TERM_TABLES_START -->
## 버전별 전체 Manager 항목

이하 표는 **부모 상속과 EnvCfg의 후처리까지 반영한 전체 구성**이다. 동일한 구성을 쓰는 버전은 표 하나를 공유한다. 표의 항목명은 코드/학습 로그의 이름이며 구현 함수 링크로 계산을 확인할 수 있다. 보상 weight는 커리큘럼 적용 전 설정값이다. **weight 0은 설정된 항목이고, `None`은 제거된 항목**이다. 일정은 학습 iteration 기준이다.

[관측](#전체-관측-항목) · [보상](#전체-보상-항목) · [터미네이션](#전체-터미네이션-항목) · [커리큘럼](#전체-커리큘럼-항목)

### 전체 관측 항목

#### 관측 구성 1

적용: `Final-GoalAccuracy-v0`, `Final-GoalAccuracy-v1`, `Final-GoalAccuracy-v0_1`, `Final-GoalAccuracy-v0_2`, `Final-GoalAccuracy-v0_3`, `Final-GoalAccuracy-v0_2_1`, `Final-GoalAccuracy-v0_2_2`.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 기본 자세 대비 상대 각도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_pos_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 기본 속도 대비 상대 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_vel_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.actions` | 8 | 직전 raw 정책 행동 8개에 tanh를 적용한 관측. 실제 행동·action-rate 계산용 raw 값은 변경하지 않는다. | 추가 ObservationTerm 노이즈 없음 | [previous_action_command](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.target_obs_state` | 3 | reset 직후 타겟 XYZ를 robot-root 좌표계로 저장하여 에피소드 동안 고정. | 초기 ±0.01m 오차를 에피소드 동안 고정<br>추가 수치 설정 없음(생략값은 함수 기본값) | [InitialObjectPositionInRoot](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.goal_pos` | 2 | 초기 타겟에서 목표까지의 평면 거리(m)와 로봇 root 기준 시계 방향각(rad). | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [initial_goal_polar_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.tactile` | 17 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. | 각 bit 1% 확률 반전<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [SharedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.wrist_wrench` | 6 | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. | 힘 σ=0.05N, 모멘트 σ=0.005Nm Gaussian | [PureWristWrench](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.hand_configuration` | 2 | 실제 손 관절에서 역산한 굽힘·대립 개방도. 각 채널을 [0,1]로 제한. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [hand_configuration](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `obstacles.initial_positions` | 80 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 전체 보상 항목

#### 보상 구성 1

적용: `Final-GoalAccuracy-v0`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `hand_reaching` | 30 | 새 최단 접근 진전. 타겟이 목표를 점유하면 접근 허용 반경을 goal_near_radius로 완화. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame`<br>목표 점유 시 접근 반경(m): `1` | [ReachingOrGoal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 8 | 고정 목표 반경 안에서만 거리·속도 Gaussian과 중심 가중을 지급. 외곽 보상 없음. | 목표 반경(m): `0.01`<br>속도 scale(m/s): `0.05` | [goal_accuracy_settling_centered](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `hand_contact` | 1.45 | 손의 실제 접촉 또는 유효한 목표 점유 중 하나라도 성립하면 1회분 접촉 보상. | 접촉 문턱(N): `0.01` | [hand_contact_or_goal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `object_speed_penalty`.

#### 보상 구성 2

적용: `Final-GoalAccuracy-v1`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `hand_reaching` | 30 | 새 최단 접근 진전. 타겟이 목표를 점유하면 접근 허용 반경을 goal_near_radius로 완화. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame`<br>목표 점유 시 접근 반경(m): `1` | [ReachingOrGoal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 8 | 거리·속도 Gaussian, 반경 2r까지 지급하고 r 내부는 중심에서 최대 2배 가중. | 목표 반경(m): `0.01`<br>속도 scale(m/s): `0.05` | [goal_settling_centered](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `hand_contact` | 1.45 | 손의 실제 접촉 또는 유효한 목표 점유 중 하나라도 성립하면 1회분 접촉 보상. | 접촉 문턱(N): `0.01` | [hand_contact_or_goal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding_scheduled.py) |
| `action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `object_speed_penalty`.

#### 보상 구성 3

적용: `Final-GoalAccuracy-v0_1`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | exp(−XY 목표 거리/distance_scale)의 지속 근접 보상. | 거리 scale(m): `0.05` | [exponential_planar_goal_proximity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_goal_hold` | 8 | 목표 반경 내부 중심 가중 Gaussian과 외곽 반경까지의 이차 꼬리를 연결하고 속도 감쇠 적용. | 목표 반경(m): `0.02`<br>외곽 반경(m): `0.06`<br>속도 scale(m/s): `0.05` | [goal_accuracy_outer_settling](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_contact` | 1.45 | 실제 손-타겟 접촉 또는 목표 점유 보상. 장애물 접촉만으로는 인정하지 않음. | 접촉 문턱(N): `0.01` | [target_contact_or_goal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrist_wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `hand_reaching`, `goal_distance`, `goal_hold`, `hand_contact`, `action_rate_penalty`, `joint_velocity_penalty`, `object_speed_penalty`, `wrench_change_penalty`.

#### 보상 구성 4

적용: `Final-GoalAccuracy-v0_2`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | exp(−XY 목표 거리/distance_scale)의 지속 근접 보상. | 거리 scale(m): `0.05` | [exponential_planar_goal_proximity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_goal_hold` | 8 | 목표 반경 내부 중심 가중 Gaussian과 외곽 반경까지의 이차 꼬리를 연결하고 속도 감쇠 적용. | 목표 반경(m): `0.02`<br>외곽 반경(m): `0.06`<br>속도 scale(m/s): `0.05` | [goal_accuracy_outer_settling](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_contact` | 1.45 | 실제 손-타겟 접촉 또는 목표 점유 보상. 장애물 접촉만으로는 인정하지 않음. | 접촉 문턱(N): `0.01` | [target_contact_or_goal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_timeout_bonus` | 30 | 성공 timeout이고 거리도 position_tolerance 미만일 때 일회 지급. 성공 분류의 ≤ 경계와 구분. | 목표 반경(m): `0.02` | [target_goal_timeout_bonus](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `wrist_wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `hand_reaching`, `goal_distance`, `goal_hold`, `hand_contact`, `action_rate_penalty`, `joint_velocity_penalty`, `object_speed_penalty`, `wrench_change_penalty`.

#### 보상 구성 5

적용: `Final-GoalAccuracy-v0_3`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_hold` | 8 | 목표 반경 내부 중심 가중 Gaussian과 외곽 반경까지의 이차 꼬리를 연결하고 속도 감쇠 적용. | 목표 반경(m): `0.03`<br>외곽 반경(m): `0.06`<br>속도 scale(m/s): `0.05` | [goal_accuracy_outer_settling](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_contact` | 0.3625 | 실제 손-타겟 접촉 또는 목표 점유 보상. 장애물 접촉만으로는 인정하지 않음. | 접촉 문턱(N): `0.01` | [target_contact_or_goal](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_timeout_bonus` | 100 | 지정 성공 termination이 발생한 step의 일회 보너스. dt 상쇄. | 성공 term: `target_goal_success_at_timeout` | [success_event](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `wrist_wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `hand_reaching`, `goal_distance`, `goal_hold`, `hand_contact`, `action_rate_penalty`, `joint_velocity_penalty`, `object_speed_penalty`, `wrench_change_penalty`.

#### 보상 구성 6

적용: `Final-GoalAccuracy-v0_2_1`, `Final-GoalAccuracy-v0_2_2`.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | exp(−XY 목표 거리/distance_scale)의 지속 근접 보상. | 거리 scale(m): `0.05` | [exponential_planar_goal_proximity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_goal_hold` | 8 | 목표 반경 내부 중심 가중 Gaussian과 외곽 반경까지의 이차 꼬리를 연결하고 속도 감쇠 적용. | 목표 반경(m): `0.02`<br>외곽 반경(m): `0.06`<br>속도 scale(m/s): `0.05` | [goal_accuracy_outer_settling](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `target_contact` | 1.45 | 최신 손-타겟 접촉력이 문턱 이상일 때만 보상. 목표 점유 대체 없음. | 접촉 문턱(N): `0.01` | [target_contact_only](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_penalty` | -1 | 물체 기울기·장애물 변위·물체 가속도 비용을 내부 계수로 합산. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04`<br>자유 가속도(m/s²): `50`<br>가속도 내부 계수: `0`<br>raw 상한: `100` | [ObjectPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_timeout_bonus` | 75 | 지정 성공 termination이 발생한 step의 일회 보너스. dt 상쇄. | 성공 term: `target_goal_success_at_timeout` | [success_event](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `wrist_wrench_change_penalty` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

`None`으로 비활성: `hand_reaching`, `goal_distance`, `goal_hold`, `hand_contact`, `action_rate_penalty`, `joint_velocity_penalty`, `object_speed_penalty`, `wrench_change_penalty`.

### 전체 터미네이션 항목

#### 터미네이션 구성 1

적용: `Final-GoalAccuracy-v0`, `Final-GoalAccuracy-v1`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `object_drop` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `object_speed` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_termination` | terminated | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `joint_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `success` | truncated | 정확도 지표를 기록하고, 시간 제한 시 목표 반경 이하를 성공으로 분류. 초기 반경 1cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [goal_accuracy_success_at_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `time_out` | truncated | 시간 제한 시 목표 반경 밖이면 미성공 timeout. 초기 반경 1cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [goal_accuracy_unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |

#### 터미네이션 구성 2

적용: `Final-GoalAccuracy-v0_1`, `Final-GoalAccuracy-v0_2`, `Final-GoalAccuracy-v0_2_1`, `Final-GoalAccuracy-v0_2_2`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `object_fall_or_tilt` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `target_speed` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_failure` | terminated | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `arm_joint_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `target_goal_success_at_timeout` | truncated | 정확도 지표를 기록하고, 시간 제한 시 목표 반경 이하를 성공으로 분류. 초기 반경 2cm(반경 일정이 있으면 갱신값 사용). | 목표 반경 참조 term: `target_goal_hold` | [goal_accuracy_success_at_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `target_goal_miss_at_timeout` | truncated | 시간 제한 시 목표 반경 밖이면 미성공 timeout. 초기 반경 2cm(반경 일정이 있으면 갱신값 사용). | 목표 반경 참조 term: `target_goal_hold` | [goal_accuracy_unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |

`None`으로 비활성: `object_drop`, `object_speed`, `shelf_termination`, `joint_velocity`, `success`, `time_out`.

#### 터미네이션 구성 3

적용: `Final-GoalAccuracy-v0_3`.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `object_fall_or_tilt` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `target_speed` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_failure` | terminated | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `arm_joint_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `target_goal_success_at_timeout` | truncated | 정확도 지표를 기록하고, 시간 제한 시 목표 반경 이하를 성공으로 분류. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 목표 반경 참조 term: `target_goal_hold` | [goal_accuracy_success_at_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `target_goal_miss_at_timeout` | truncated | 시간 제한 시 목표 반경 밖이면 미성공 timeout. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 목표 반경 참조 term: `target_goal_hold` | [goal_accuracy_unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |

`None`으로 비활성: `object_drop`, `object_speed`, `shelf_termination`, `joint_velocity`, `success`, `time_out`.

### 전체 커리큘럼 항목

#### 커리큘럼 구성 1

적용: `Final-GoalAccuracy-v0`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→4, 2500부터 / White 1→5, 2000부터 / 각 1000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_gray_zone_padding` | 다음 reset의 Gray 안쪽 경계 침범 비율을 변경. 절대 거리(m)가 아닌 비율. | 4000: 0 → 10000: 0.15; 갱신 간격 1000 iter | [gray_zone_padding_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding_osc_object_random.py) |
| `object_free_speed` | `target_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_free_acceleration` | `object_penalty`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_acceleration_weight` | `object_penalty`의 가속도 내부 계수 (`acceleration_weight`)를 변경. | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change_penalty`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_free_moment_change` | `wrench_change_penalty`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 3000: 5 → 10000: 2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_weight` | `wrench_change_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_free_force` | `obstacle_contact_penalty`의 허용 접촉력(N) (`free_force`)를 변경. | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_weight` | `obstacle_contact_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |

`None`으로 비활성: `goal_speed_scale`, `goal_position_tolerance`.

#### 커리큘럼 구성 2

적용: `Final-GoalAccuracy-v1`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→4, 2500부터 / White 1→5, 2000부터 / 각 1000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_gray_zone_padding` | 다음 reset의 Gray 안쪽 경계 침범 비율을 변경. 절대 거리(m)가 아닌 비율. | 4000: 0 → 10000: 0.15; 갱신 간격 1000 iter | [gray_zone_padding_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding_osc_object_random.py) |
| `object_free_speed` | `target_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_free_acceleration` | `object_penalty`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_acceleration_weight` | `object_penalty`의 가속도 내부 계수 (`acceleration_weight`)를 변경. | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `goal_speed_scale` | `goal_hold`의 속도 scale(m/s) (`speed_scale`)를 변경. | 0: 0.05 → 6000: 0.05 → 10000: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_free_force_change` | `wrench_change_penalty`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_free_moment_change` | `wrench_change_penalty`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 3000: 5 → 10000: 2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_weight` | `wrench_change_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_free_force` | `obstacle_contact_penalty`의 허용 접촉력(N) (`free_force`)를 변경. | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_weight` | `obstacle_contact_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |

`None`으로 비활성: `goal_position_tolerance`.

#### 커리큘럼 구성 3

적용: `Final-GoalAccuracy-v0_1`, `Final-GoalAccuracy-v0_2`, `Final-GoalAccuracy-v0_3`, `Final-GoalAccuracy-v0_2_1`, `Final-GoalAccuracy-v0_2_2`.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→4, 2500부터 / White 1→5, 2000부터 / 각 1000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_gray_zone_padding` | 다음 reset의 Gray 안쪽 경계 침범 비율을 변경. 절대 거리(m)가 아닌 비율. | 4000: 0 → 10000: 0.15; 갱신 간격 1000 iter | [gray_zone_padding_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding_osc_object_random.py) |
| `target_free_speed` | `target_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_free_acceleration` | `object_penalty`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_acceleration_weight` | `object_penalty`의 가속도 내부 계수 (`acceleration_weight`)를 변경. | 3000: 0 → 10000: -0.05; 갱신 간격 1 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `target_goal_direction_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrist_wrench_free_force_change` | `wrist_wrench_change_penalty`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 3000: 50 → 10000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrist_wrench_free_moment_change` | `wrist_wrench_change_penalty`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 3000: 5 → 10000: 2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrist_wrench_change_weight` | `wrist_wrench_change_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_contact_free_force` | `obstacle_contact_penalty`의 허용 접촉력(N) (`free_force`)를 변경. | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_contact_weight` | `obstacle_contact_penalty`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |

`None`으로 비활성: `object_free_speed`, `goal_deadzone`, `goal_speed_scale`, `goal_position_tolerance`, `wrench_free_force_change`, `wrench_free_moment_change`, `wrench_weight`, `obstacle_free_force`, `obstacle_weight`, `target_goal_speed_scale`, `target_goal_position_tolerance`.

<!-- TERM_TABLES_END -->

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/final_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/final.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/final.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/final.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/final.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/final.py) · [보상 함수](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py)
