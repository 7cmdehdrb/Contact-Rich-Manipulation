# Wrench 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

베이스는 전체 항목을, 각 파생 환경은 **직접 부모 대비 변경점만** 적는다. 보상 weight는 커리큘럼 적용 전 값이며, 일정의 단위는 학습 iteration이다.

**환경 바로가기:** [Inspire-Wrench-v0](13_wrench.md#env-wrenchenvcfg) · [Inspire-Wrench-v0_1](13_wrench.md#env-wrenchv0_1envcfg) · [Inspire-Wrench-v0_2](13_wrench.md#env-wrenchv0_2envcfg) · [Inspire-Wrench-v1](13_wrench.md#env-wrenchv1envcfg) · [Inspire-Wrench-v2](13_wrench.md#env-wrenchv2envcfg)

<a id="env-wrenchenvcfg"></a>

## 베이스: Inspire-Wrench-v0

`WrenchEnvCfg`는 `ManagerBasedRLEnvCfg`를 직접 상속한다. Revision-Test-v2를 기준으로 독립 구성했다. **v0의 Arm은 pose OSC**이며 실제 wrench 제어는 v1부터다. Gym ID에는 `Isaac-` 접두사가 없다.

### 관측

직접 **57D**: Arm 실제 위치·속도 12 + Hand 2 + EEF pose 7 + 초기 타겟 XYZ 3 + 목표 극좌표 2 + 이전 행동 8 + 촉각 17 + 보정 F/T 6. Actor 초기 장애물 64D, Critic 현재 타겟 3D·현재 활성 장애물 64D. **원시 Actor/Critic 121/124D, MLP 73/76D**. 모든 Wrench 버전에서 차원은 같다.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 실제 각도. 기본 자세를 빼지 않는다. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 실제 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_velocities](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `policy.hand_configuration` | 2 | 실제 손 관절에서 역산한 굽힘·대립 개방도. 각 채널을 [0,1]로 제한. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [hand_configuration](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.target_obs_state` | 3 | reset 직후 타겟 XYZ를 robot-root 좌표계로 저장하여 에피소드 동안 고정. | 초기 ±0.01m 오차를 에피소드 동안 고정<br>추가 수치 설정 없음(생략값은 함수 기본값) | [InitialObjectPositionInRoot](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) |
| `policy.goal_pos` | 2 | 초기 타겟에서 목표까지의 평면 거리(m)와 로봇 root 기준 시계 방향각(rad). | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [initial_goal_polar_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.actions` | 8 | 직전 raw 정책 행동 8개에 tanh를 적용한 관측. 실제 행동·action-rate 계산용 raw 값은 변경하지 않는다. | 추가 ObservationTerm 노이즈 없음 | [previous_action_command](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/final.py) |
| `policy.tactile` | 17 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. | 각 bit 1% 확률 반전<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [SharedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.wrist_wrench` | 6 | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. | 힘 σ=0.05N, 모멘트 σ=0.005Nm Gaussian | [PureWristWrench](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `obstacles.initial_positions` | 64 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `critic_obstacles.current_positions` | 64 | 매 step 현재 활성 장애물 [존재 여부, x, y, z]. 주차/비활성 물체 제외, 가시성 누락 없이 제공. | 추가 ObservationTerm 노이즈 없음<br>adjacent_swap_probability: `0.1`; Critic 전용 | [current_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/revision.py) |

### 환경 구성

원통 타겟 + 다양한 장애물, 물리 10개(최대 Gray 4/White 5), 기본 2048개·10초·50Hz. 생성 시 종류/크기를 정하고 startup에서 단계별 **layout bank(기본 16개)**와 물성을 만들어 둔다. Reset에서는 bank 배치를 선택하며 물성을 다시 뽑지 않는다. OBB Gray padding 65mm, RED margin 23mm, 뒤쪽 Gray offset 20mm. 행동은 상대 pose OSC 6 + Hand 2, scale 0.03m/0.15rad다.

### 보상

15개 term: 접근 최단 기록 30, 거리 `3.7×exp(−d/0.05)`, Hold 10, 실제 접촉 1.45, 최초 진입 최대 +25, 안정 성공 +75와 비용 9개. 비용은 행동 변화·관절 속도·물체 기울기·장애물 변위·물체 가속도·선반·장애물 접촉력·wrench 변화·타겟 속도다.

거리/Hold/접촉은 **EEF 120mm 미만** 조건. Hold는 2.5cm까지 1, 3.5cm까지 Smoothstep으로 0이 되며 속도 감쇠는 없다. 최초 3cm 내부 진입 보너스에는 `exp(−(max(v−0.01,0)/0.02)²)` 속도 감쇠가 있고 높이 조건은 없다. wrench 비용은 자유 변화 20N/2Nm, weight −0.1이며 양의 보상 예산 제한은 없다.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `target_reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `target_goal_distance` | 3.7 | EEF 높이 제한 미만일 때만 exp(−d/distance_scale) 거리 보상. EEF 높이 제한 120mm. | 거리 scale(m): `0.05` | [goal_distance_v2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_goal_hold` | 10 | EEF 높이 제한 미만에서 full_reward_radius까지 1, outer_radius까지 Smoothstep으로 0으로 감소. 속도 감쇠 없음. EEF 높이 제한 120mm. | 최대 Hold 반경(m): `0.025`<br>외곽 반경(m): `0.035` | [goal_hold_v2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_contact` | 1.45 | 실제 손-타겟 접촉 AND 초기 선반 기준 EEF 높이 제한 미만에서만 보상. EEF 높이 제한 120mm. | 접촉 문턱(N): `0.01` | [target_contact_below_height](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `robot_action_rate_penalty` | -1 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산. | 추가 수치 설정 없음(생략값은 함수 기본값) | [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `robot_joint_velocity_penalty` | -1 | Arm 속도 제곱합×0.03 + Hand 속도 제곱합×0.01. | 추가 수치 설정 없음(생략값은 함수 기본값) | [joint_velocity_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `object_tilt_penalty` | -30 | 활성 타겟/장애물만 대상으로 기울기 초과의 제한된 제곱 비용 합. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9` | [active_object_tilt_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_displacement_penalty` | -150 | 활성 비타겟 장애물의 초기 위치 대비 3D 변위 제곱합. 주차된 물체 제외. | 변위 제곱합 상한: `0.04` | [active_obstacle_displacement_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration_penalty` | -0.05 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `15`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_penalty` | -1 | 선반 충돌/근접 비용×0.5 + 선반 실패 이벤트 3. 이벤트 부분은 dt 상쇄. | 실패 term: `shelf_failure` | [ShelfPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) |
| `obstacle_contact_penalty` | -0.05 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `50`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `target_goal_first_entry_bonus` | 25 | 최초 목표 내부 진입에 속도 감쇠를 곱한 일회 보상. 높이 조건 없음, 지급액과 무관하게 기회 소진. | 자유 속도(m/s 또는 관절 rad/s): `0.01`<br>속도 scale(m/s): `0.02` | [FirstGoalEntryBonusV2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_goal_success_bonus` | 75 | 3초 안정 유지 성공 종료에 한 번 지급하는 보너스. timeout 보너스가 아님. | 추가 수치 설정 없음(생략값은 함수 기본값) | [settled_success_bonus](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py) |
| `wrist_wrench_change_penalty` | -0.1 | 힘·모멘트 변화량 크기의 제한된 비용. 양의 보상 예산 제한은 없음. | 허용 힘 변화(N): `20`<br>허용 모멘트 변화(Nm): `2`<br>포화 비율: `2.5`<br>곡선 기울기: `3` | [IndependentWrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) |
| `target_speed_penalty` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |

### 터미네이션

실패는 Revision과 동일한 낙하/기울기·타겟 1m/s·Arm 5rad/s·선반 실패. 성공은 **XY 거리 3cm 이하·속도 0.005m/s 이하를 연속 3초** 유지하면 조기 종료한다. 성공은 terminated, 10초 미성공은 truncated다. +75는 timeout이 아니라 이 안정 성공 이벤트에 지급한다.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `object_fall_or_tilt` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. 크기/회전을 반영한 물체 바닥 높이로 검사. | bottom_clearance: `0.01`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/revision.py) |
| `target_speed` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_failure` | terminated | 선반 충돌/손목 여유 위반 OR 선반 변위 초과를 하나로 합친 실패. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_termination](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/final.py) |
| `arm_joint_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `target_goal_success` | terminated | 목표 XY 거리·3D 속도 한계를 duration 동안 연속 만족하면 성공 종료. 동시 실패 제외. | 목표 반경(m): `0.03`<br>속도 한계(m/s): `0.005`<br>연속 유지(s): `3` | [SettledGoalSuccess](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/wrench.py) |
| `target_goal_miss_at_timeout` | truncated | 시간 제한 도달 시 성공 조건이 성립하지 않는 timeout. 초기 반경 3cm(반경 일정이 있으면 갱신값 사용). | 추가 수치 설정 없음(생략값은 함수 기본값) | [unsuccessful_timeout](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/wrench.py) |

### 커리큘럼

장애물 3/3→3/4(10k)→4/4(12k)→4/5(14k). 타겟 자유 속도 0.1→0.02m/s. 목표 제외 반각 30°는 **10k에서 처음 5° 감소**, 이후 3k마다 감소한다.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 3→4, 12000부터 / White 3→5, 10000부터 / 각 4000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `target_free_speed` | `target_speed_penalty`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `target_goal_direction_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30° 시작; 10000에서 첫 −5°, 이후 3000마다 감소, 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/wrench.py) |

<a id="env-wrenchv0_1envcfg"></a>

## Inspire-Wrench-v0_1

**직접 부모:** [Inspire-Wrench-v0](13_wrench.md#env-wrenchenvcfg) (`WrenchEnvCfg`). **환경 설정:** [WrenchV0_1EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/wrench_envs.py).

### 이 환경에서 바뀐 점

독립 접촉 보상 제거. 거리 점수에 초기/현재 목표 벡터의 signed cosine과 연속 2step 타겟 접촉 조건을 곱한다. d<3cm에서는 cosine=1. EEF 높이 조건 유지. 목표 제외 반각 40° 고정, 허용 목표 거리 4–7.5cm.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `target_goal_distance` | 수정 | EEF 높이 제한 미만일 때만 exp(−d/distance_scale) 거리 보상. EEF 높이 제한 120mm.<br>구현: [goal_distance_v2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py)<br>접촉 문턱(N): `함수 기본값` | 높이 조건부 거리 점수×signed cosine×연속 2step 타겟 접촉. 거리 3cm 미만에서는 cosine=1. EEF 높이 제한 120mm.<br>구현: [WrenchV0_1GoalDistance](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py)<br>접촉 문턱(N): `0.01` | 높이 조건부 거리 점수×signed cosine×연속 2step 타겟 접촉. 거리 3cm 미만에서는 cosine=1. EEF 높이 제한 120mm. |
| `target_contact` | 삭제 | weight: 1.45<br>접촉 문턱(N): `0.01`<br>구현: [target_contact_below_height](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) | `None` / 미사용 | 실제 손-타겟 접촉 AND 초기 선반 기준 EEF 높이 제한 미만에서만 보상. EEF 높이 제한 120mm. → 이 환경에서는 제거. |

### 커리큘럼 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `target_goal_direction_deadzone` | 삭제 | 30° 시작; 10000에서 첫 −5°, 이후 3000마다 감소, 하한 0°<br>구현: [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/wrench.py) | `None` / 미사용 | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. → 이 환경에서는 제거. |

**직접 부모와 동일:** 관측 · 터미네이션.

<a id="env-wrenchv0_2envcfg"></a>

## Inspire-Wrench-v0_2

**직접 부모:** [Inspire-Wrench-v0](13_wrench.md#env-wrenchenvcfg) (`WrenchEnvCfg`). **환경 설정:** [WrenchV0_2EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/wrench_envs.py).

### 이 환경에서 바뀐 점

거리×signed cosine은 쓰되 연속 접촉 조건 없음, d<3cm에서 cosine=1. 접촉 보상은 최근 3step 중 2회 인정 또는 목표 3cm 내부일 때 누적 count(최대 10)를 지급, 연속 불인정 5회면 초기화. 장애물 기준 −1/−1에서 2k에 White 먼저 증가, 이후 3k마다 교대 증가·reset ±1 샘플 후 0~4/0~5로 제한.

불인정 step에는 접촉 보상을 지급하지 않으며 count는 4회까지 보존한다. 목표 3cm 내부면 접촉력 없이도 인정하며, 실제 접촉 경로에만 EEF 높이 조건을 적용한다. 50Hz에서 접촉 보상 최대값은 `1.45×10×0.02=0.29/step`이다. Arm은 부모 v0의 pose OSC다.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `target_goal_distance` | 수정 | EEF 높이 제한 미만일 때만 exp(−d/distance_scale) 거리 보상. EEF 높이 제한 120mm.<br>구현: [goal_distance_v2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) | 높이 조건부 거리 점수×signed cosine. 접촉 조건 없음, 거리 3cm 미만에서는 cosine=1. EEF 높이 제한 120mm.<br>구현: [goal_distance_cosine](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py) | 높이 조건부 거리 점수×signed cosine. 접촉 조건 없음, 거리 3cm 미만에서는 cosine=1. EEF 높이 제한 120mm. |
| `target_contact` | 수정 | weight: 1.45<br>실제 손-타겟 접촉 AND 초기 선반 기준 EEF 높이 제한 미만에서만 보상. EEF 높이 제한 120mm.<br>구현: [target_contact_below_height](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/revision.py) | weight: 0.1<br>최근 3step 중 2회 접촉 인정 또는 목표 3cm 내부면 누적 count 지급(최대 10). 불인정 5회 연속이면 count 초기화. 실제 접촉 경로에만 EEF 높이 제한 120mm(목표 점유는 예외).<br>구현: [WrenchV0_2ContactMaintenance](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py) | 최근 3step 중 2회 접촉 인정 또는 목표 3cm 내부면 누적 count 지급(최대 10). 불인정 5회 연속이면 count 초기화. 실제 접촉 경로에만 EEF 높이 제한 120mm(목표 점유는 예외). |

### 커리큘럼 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `obstacle_population` | 수정 | Gray 3→4, 12000부터 / White 3→5, 10000부터 / 각 4000 iter마다 +1<br>구현: [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) | −1/−1 시작; 2000에 White 먼저 +1, 이후 3000마다 교대. 상한 4/5; 실제 샘플은 음수를 0으로 제한.<br>구현: [feasibility_obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/wrench.py) | Gray/White 장애물 기준 개수 증가와 reset ±1 랜덤화의 기준값 설정. |

**직접 부모와 동일:** 관측 · 터미네이션.

<a id="env-wrenchv1envcfg"></a>

## Inspire-Wrench-v1

**직접 부모:** [Inspire-Wrench-v0](13_wrench.md#env-wrenchenvcfg) (`WrenchEnvCfg`). **환경 설정:** [WrenchV1EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/wrench_envs.py).

### 이 환경에서 바뀐 점

Arm을 Axia80 센서 중심 6축 closed-loop wrench 제어로 변경. `wrench_abs`, 목표 ±100N/±5Nm, P gain 0.2, 보정 명령 cap ±120N/±6Nm, 중력 보상. Arm action-rate를 목표 범위로 정규화한 변화량으로 계산. EEF 관측/보상 frame과 Hand는 유지.

### 보상 변경

| 항목명 | 변경 | 부모 | 이 환경 | 의미 |
| --- | --- | --- | --- | --- |
| `robot_action_rate_penalty` | 수정 | Arm·Hand 행동 변화 제곱합 비용을 각 0.03 계수로 합산.<br>구현: [action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/final.py) | Arm 목표 wrench를 목표 범위로 정규화한 변화 제곱합×0.03 + raw Hand 변화 제곱합×0.03.<br>구현: [desired_wrench_action_rate_penalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py) | Arm 목표 wrench를 목표 범위로 정규화한 변화 제곱합×0.03 + raw Hand 변화 제곱합×0.03. |

**직접 부모와 동일:** 관측 · 터미네이션 · 커리큘럼.

<a id="env-wrenchv2envcfg"></a>

## Inspire-Wrench-v2

**직접 부모:** [Inspire-Wrench-v1](13_wrench.md#env-wrenchv1envcfg) (`WrenchV1EnvCfg`). **환경 설정:** [WrenchV2EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/wrench_envs.py).

### 이 환경에서 바뀐 점

목표 범위를 ±10N/±0.5Nm, 보정 cap을 ±12N/±0.6Nm로 1/10 축소. 정규화 action-rate와 나머지 구성은 상속.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/wrench_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/wrench.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/wrench.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/wrench.py) · [리셋](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/events/wrench.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/wrench.py) · [제어](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/actions/wrench.py) · [거리·접촉 함수](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/wrench.py)
