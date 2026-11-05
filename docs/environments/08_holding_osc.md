# Holding OSC 계열

[통합 안내](README.md) · 2026-11-05 현재 코드 기준

베이스는 전체 항목을, 각 파생 환경은 **직접 부모 대비 변경점만** 적는다. 보상 weight는 커리큘럼 적용 전 값이며, 일정의 단위는 학습 iteration이다.

**환경 바로가기:** [Holding-OSC-v0](08_holding_osc.md#env-fusionholdingoscv0envcfg) · [Holding-OSC-v1](08_holding_osc.md#env-fusionholdingoscv1envcfg) · [Holding-OSC-v2](08_holding_osc.md#env-fusionholdingoscv2envcfg) · [Holding-OSC-v3](08_holding_osc.md#env-fusionholdingoscv3envcfg) · [Holding-OSC-v4](08_holding_osc.md#env-fusionholdingoscv4envcfg)

<a id="env-fusionholdingoscv0envcfg"></a>

## 베이스: Holding-OSC-v0

`FusionHoldingOSCV0EnvCfg`는 [Holding-v0](07_holding.md)의 직접 자식이다. 과제 설정을 유지하고 Arm 제어만 상대 pose OSC로 교체한다.

### 관측

직접 55D = Arm 위치·속도 12 + 이전 행동 8 + 초기 타겟 3 + EEF 7 + 목표 극좌표 2 + 촉각 17 + F/T 6. 초기 장애물 40D와 Critic 타겟 3D를 더해 **원시 95/98D, MLP 71/74D**다. 이전 Arm 행동의 의미는 관절 위치 명령에서 OSC pose 명령으로 바뀐다.

| 그룹.항목명 | 차원 | 의미 | 처리·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `policy.joint_pos` | 6 | Arm 6개 관절의 기본 자세 대비 상대 각도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_pos_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.joint_vel` | 6 | Arm 6개 관절의 기본 속도 대비 상대 각속도. | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [arm_joint_vel_rel](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.actions` | 8 | 직전 행동을 [−1, 1]로 clamp한 Arm 6 + Hand 2. | 추가 ObservationTerm 노이즈 없음 | [bounded_last_action](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.target_obs_state` | 3 | reset 직후 타겟 XYZ를 robot-root 좌표계로 저장하여 에피소드 동안 고정. | 초기 ±0.01m 오차를 에피소드 동안 고정<br>추가 수치 설정 없음(생략값은 함수 기본값) | [InitialObjectPositionInRoot](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/asymmetric_observations.py) |
| `policy.ee_pose` | 7 | 현재 EEF의 robot-root 기준 XYZ 3 + quaternion 4. | 추가 ObservationTerm 노이즈 없음 | [eef_pose_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |
| `policy.goal_pos` | 2 | 초기 타겟에서 목표까지의 평면 거리(m)와 로봇 root 기준 시계 방향각(rad). | 추가 ObservationTerm 노이즈 없음<br>추가 수치 설정 없음(생략값은 함수 기본값) | [initial_goal_polar_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter_fusion.py) |
| `policy.tactile` | 17 | 정책과 보상이 공유하는 최신 촉각 샘플의 17개 접촉 bit. | 추가 ObservationTerm 노이즈 없음<br>sensor_name: `hand_tactile`<br>접촉 문턱(N): `0.01` | [SharedTactileContactBits](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `policy.wrist_wrench` | 6 | 중력·관성 등을 보정하고 EEF 기준으로 표현한 힘 3 + 모멘트 3. | 추가 ObservationTerm 노이즈 없음 | [PureWristWrench](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `obstacles.initial_positions` | 40 | reset에서 저장한 초기 장애물 [존재 여부, x, y, z] 슬롯. 초기 가시성/위치 오차가 반영된 값을 유지. | 추가 ObservationTerm 노이즈 없음; 16D로 인코딩 | [initial_obstacle_positions](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/contact_sweep_clutter.py) |
| `privileged.target_position` | 3 | 현재 타겟 XYZ, robot-root 좌표계. | 추가 ObservationTerm 노이즈 없음; Critic 전용 | [object_position_in_root](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observation_functions/sweep.py) |

### 환경 구성

원통 타겟 + 최대 Gray 5/White 5, 초기 1/1. 배치/물성 reset과 4096개·10초·50Hz는 Holding-v0와 같다. 행동은 **상대 위치·회전 6 + Hand 개방 2**. v0 OSC는 robot-root 축, 위치/회전 scale 1/1, stiffness 100, damping ratio 1, 관성 분리·중력 보상을 사용하며 경쟁하는 Arm 관절 PD를 해제한다.

### 보상

Holding-v0 그대로: 접근 30 + 목표 진전 100 + 접촉 1.45 + 거리·속도 Gaussian Hold 2 − 행동/관절/물체/선반/과속/충격 비용. OSC라는 이유로 보상식이나 계수를 별도로 바꾸지 않는다.

| 항목명 | weight | 의미 | 설정값·조건 | 구현 함수 |
| --- | ---: | --- | --- | --- |
| `reaching` | 30 | 타겟 접근 반경 밖의 새 최단 거리 감소량에 비례한 보상. 되돌아갔다 재접근해도 중복 지급하지 않음. | 접근 반경(m): `0.1`<br>기준 frame: `ee_frame` | [DexReachProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_distance` | 100 | 허용 반경 밖의 타겟→목표 XY 새 최단 거리 감소량에 비례한 보상. | 목표 반경(m): `0` | [DexPlanarGoalProgress](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/dex_rewards.py) |
| `goal_hold` | 2 | 거리와 타겟 속도의 Gaussian 정지 유지 점수. 위치 scale×cutoff_multiple 미만에서 지급. | 위치 scale(m): `0.01`<br>속도 scale(m/s): `0.05`<br>위치 cutoff 배수: `3` | [goal_settling](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion_holding.py) |
| `hand_contact` | 1.45 | 손 전체의 실제 접촉력 합력이 문턱 이상이면 보상. 목표 근처에서는 설정된 fade 폭에 따라 감쇠. | 접촉 문턱(N): `0.01`<br>접촉 감쇠 폭(m): `0` | [hand_contact_any](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `arm_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `arm`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_action_rate` | -0.03 | 현재와 이전 정책 행동 차이의 제곱합 비용. action_group이 있으면 해당 Arm/Hand만 계산. | 행동 그룹: `hand`<br>raw 상한: `100` | [action_rate_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_joint_velocity` | -0.03 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `hand_joint_velocity` | -0.01 | 지정 관절 각속도의 제곱합 비용. | raw 상한: `100` | [joint_vel_l2](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `object` | -30 | 활성 물체 기울기 + displacement_ratio×활성 장애물 변위 비용. | 무비용 기울기(rad): `0.0872665`<br>실패 기울기(rad): `0.9`<br>변위 비중: `5`<br>변위 제곱합 상한: `0.04` | [combined_active_object_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `shelf_collision` | -0.5 | 선반 움직임 및 손가락/손목의 선반 근접·침범 비용. | 추가 수치 설정 없음(생략값은 함수 기본값) | [ShelfCollisionPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/sweep.py) |
| `pushing_speed` | -5 | 타겟 3D 선속도의 자유 속도 초과에 대한 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `0.1`<br>raw 상한: `100` | [exponential_pushing_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `arm_speed_excess` | -1 | 선택 관절 중 최대 절대 속도의 자유 속도 초과 지수 비용, raw cap 적용. | 자유 속도(m/s 또는 관절 rad/s): `1`<br>raw 상한: `100` | [exponential_joint_speed_cost_limit](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/goal_rewards.py) |
| `shelf_failure` | -3 | 선반 실패 종료 step에 한 번 부과하는 비용. dt 상쇄. | 추가 수치 설정 없음(생략값은 함수 기본값) | [shelf_failure](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter.py) |
| `eef_linear_speed` | -1 | 실제 EEF 지점의 선속도에 대한 제한된 지수 초과속도 비용. | 자유 속도(m/s 또는 관절 rad/s): `1.5`<br>raw 상한: `100` | [eef_linear_speed_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `obstacle_contact` | 0 | 손-장애물 접촉력의 허용 힘 초과 비용, max_cost 적용. | 허용 접촉력(N): `80`<br>raw 상한: `100` | [obstacle_contact_force_cost](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `object_acceleration` | 0 | 활성 물체별 PhysX COM 가속도의 허용값 초과 지수 비용을 각각 상한 처리하여 합산. Reset/활성화 직후 첫 샘플 제외. | 자유 가속도(m/s²): `50`<br>raw 상한: `100` | [ObjectAccelerationPenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |
| `wrench_change` | 0 | 힘·모멘트 변화량의 크기에 대한 제한된 비용. 누적 양의 보상 예산 적용. | 허용 힘 변화(N): `50`<br>허용 모멘트 변화(Nm): `5`<br>포화 비율: `2.5`<br>곡선 기울기: `3`<br>양의 누적 보상 예산 비율: `0.05` | [WrenchMagnitudeChangePenalty](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/reward_functions/contact_sweep_clutter_fusion.py) |

### 터미네이션

Holding-v0와 동일. 5mm·1cm/s 이하 3초 유지 성공, 낙하/기울기·과속·선반 실패, 10초 timeout.

| 항목명 | 종료 유형 | 의미 | 설정값·조건 | 구현 함수 |
| --- | --- | --- | --- | --- |
| `time_out` | truncated | 에피소드 제한 시간(기본 10초) 도달. | 추가 수치 설정 없음(생략값은 함수 기본값) | `time_out` |
| `object_drop` | terminated | 활성 타겟/장애물 중 하나라도 낙하/기울기 조건을 넘으면 실패. 비활성 물체 제외. | 높이 기준(m): `1.04`<br>실패 기울기(rad): `0.9` | [any_active_object_fall_or_tilt](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion.py) |
| `push_fast` | terminated | 타겟 3D 선속도가 허용 속도를 초과하면 실패. | 타겟 종료 속도(m/s): `1` | [push_fast](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_collision` | terminated | 선반 속도 또는 손/손목 선반 여유 조건 위반. check_fingers=False이면 손가락 높이 종료 제외. | threshold: `0.1`<br>손가락 높이 검사: `False`<br>손목 선반 여유(m): `0.03` | [shelf_collision](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `hand_velocity` | terminated | Arm 6개 관절 중 하나라도 절대 각속도가 문턱을 초과하면 실패. hand_velocity라는 이름이어도 Arm 검사. | threshold: `5` | [arm_velocity](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/sweep.py) |
| `shelf_displacement` | terminated | 선반이 초기 위치에서 허용 거리보다 많이 이동하면 실패. | threshold: `0.05` | [shelf_displacement](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/goal_terminations.py) |
| `success` | terminated | 고정 거리·속도 한계를 연속 hold_time 동안 만족하면 성공. 동시 실패 제외. | 목표 반경(m): `0.005`<br>속도 한계(m/s): `0.01`<br>연속 유지(s): `3`<br>제외/검사할 실패 term: `(object_drop, push_fast, shelf_collision, hand_velocity, shelf_displacement)` | [HoldingSuccess](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/termination_functions/contact_sweep_clutter_fusion_holding.py) |

### 커리큘럼

Holding-v0의 장애물 개수, 자유 속도, Hold scale, 충격 비용·문턱, goal deadzone 일정을 그대로 적용한다.

| 항목명 | 의미·변경 대상 | 일정 | 구현 함수 |
| --- | --- | --- | --- |
| `obstacle_population` | 활성 Gray/White 장애물 요청 개수를 단계적으로 증가. | Gray 1→5, 4500부터 / White 1→5, 3000부터 / 각 3000 iter마다 +1 | [obstacle_population_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `pushing_free_speed` | `pushing_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 0.1 → 499: 0.1 → 500: 0.09 → 1900: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `arm_free_speed` | `arm_speed_excess`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 1 → 499: 1 → 500: 0.98 → 3300: 0.7; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `eef_free_speed` | `eef_linear_speed`의 자유 속도(m/s 또는 관절 rad/s) (`free_speed`)를 변경. | 0: 1.5 → 499: 1.5 → 500: 1.48 → 13300: 0.2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_deadzone` | 목표 방향에서 제외하는 ±X 반각을 점차 줄임. | 30°; 3000까지 유지, 이후 500마다 −5° (첫 감소 3500), 하한 0° | [goal_deadzone_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion.py) |
| `wrench_free_force_change` | `wrench_change`의 허용 힘 변화(N) (`free_force_change`)를 변경. | 0: 50 → 3000: 50 → 6000: 30 → 10000: 30 → 12000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_free_moment_change` | `wrench_change`의 허용 모멘트 변화(Nm) (`free_moment_change`)를 변경. | 0: 5 → 3000: 5 → 6000: 3 → 10000: 3 → 12000: 2; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `object_free_acceleration` | `object_acceleration`의 자유 가속도(m/s²) (`free_acceleration`)를 변경. | 0: 50 → 3000: 50 → 6000: 30 → 10000: 30 → 12000: 20; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_free_force` | `obstacle_contact`의 허용 접촉력(N) (`free_force`)를 변경. | 0: 80 → 3000: 80 → 6000: 50 → 10000: 50 → 12000: 30; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `acceleration_weight` | `object_acceleration`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.01 → 6000: -0.03 → 10000: -0.03 → 12000: -0.05; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `wrench_weight` | `wrench_change`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.03 → 6000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `obstacle_weight` | `obstacle_contact`의 보상 weight를 변경. | 0: 0 → 1000: 0 → 3000: -0.02 → 6000: -0.05 → 10000: -0.05 → 12000: -0.1; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_position_scale` | `goal_hold`의 위치 scale(m) (`position_scale`)를 변경. | 0: 0.01 → 1000: 0.02 → 6000: 0.02 → 10000: 0.01 → 12000: 0.01 → 16000: 0.005; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |
| `goal_speed_scale` | `goal_hold`의 속도 scale(m/s) (`speed_scale`)를 변경. | 0: 0.05 → 6000: 0.05 → 10000: 0.02; 갱신 간격 250 iter | [reward_schedule](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curriculum_functions/contact_sweep_clutter_fusion_holding.py) |

## 관련 상속 관계

각 OSC 환경의 직접 부모는 대응 Holding 환경이다. OSC-v1이 OSC-v0를 상속하는 구조가 아니다. 물체 랜덤화 파생은 [ObjectRandom](09_object_random.md)에 정리한다.

<a id="env-fusionholdingoscv1envcfg"></a>

## Holding-OSC-v1

**직접 부모:** [Holding-v1](07_holding.md#env-fusionholdingv1envcfg) (`FusionHoldingV1EnvCfg`). **환경 설정:** [FusionHoldingOSCV1EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_envs.py).

### 이 환경에서 바뀐 점

Arm을 상대 pose OSC로 교체. 16슬롯으로 원시 119/122D.

OSC 설정: robot-root 축, 위치/회전 scale 1/1, stiffness 100, damping ratio 1, 관성 분리·중력 보상, Arm 관절 PD 해제. 원시 119/122D, MLP 71/74D.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

<a id="env-fusionholdingoscv2envcfg"></a>

## Holding-OSC-v2

**직접 부모:** [Holding-v2](07_holding.md#env-fusionholdingv2envcfg) (`FusionHoldingV2EnvCfg`). **환경 설정:** [FusionHoldingOSCV2EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_envs.py).

### 이 환경에서 바뀐 점

Reach OSC 적용: 위치 scale 0.03m·회전 0.15rad, 병진 stiffness 100·회전 50, effort 제한 적용. 부모의 공통 목표 반경 유지.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

<a id="env-fusionholdingoscv3envcfg"></a>

## Holding-OSC-v3

**직접 부모:** [Holding-v3](07_holding.md#env-fusionholdingv3envcfg) (`FusionHoldingV3EnvCfg`). **환경 설정:** [FusionHoldingOSCV3EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_envs.py).

### 이 환경에서 바뀐 점

Reach OSC를 적용. 부모의 누적 3초 Hold 예산 유지.

Arm은 관절 위치 제어에서 상대 pose OSC로 변경한다. 위치 scale 0.03m·회전 0.15rad, 병진 stiffness 100·회전 50, effort 제한을 적용한다. 원시 119/122D, MLP 71/74D.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

<a id="env-fusionholdingoscv4envcfg"></a>

## Holding-OSC-v4

**직접 부모:** [Holding-v4](07_holding.md#env-fusionholdingv4envcfg) (`FusionHoldingV4EnvCfg`). **환경 설정:** [FusionHoldingOSCV4EnvCfg](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_envs.py).

### 이 환경에서 바뀐 점

Reach OSC를 적용. 부모의 중심 가중/외곽 Hold, 목표 점유 Reaching, 성공 종료 제거 유지.

Arm은 관절 위치 제어에서 상대 pose OSC로 변경한다. 위치 scale 0.03m·회전 0.15rad, 병진 stiffness 100·회전 50, effort 제한을 적용한다. 원시 119/122D, MLP 71/74D.

**직접 부모와 동일:** 관측 · 보상 · 터미네이션 · 커리큘럼.

## 코드 위치

[환경 설정](../../src/inspire_sweep_baseline/inspire_sweep_baseline/envs/contact_sweep_clutter_fusion_holding_osc_envs.py) · [관측](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/observations/contact_sweep_clutter_fusion.py) · [보상](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/rewards/contact_sweep_clutter_fusion_holding.py) · [종료](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/terminations/contact_sweep_clutter_fusion_holding.py) · [커리큘럼](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/curricula/contact_sweep_clutter_fusion_holding.py) · [OSC 설치 함수](../../src/inspire_sweep_baseline/inspire_sweep_baseline/mdp/common/contact_sweep_clutter_fusion_holding_osc_configuration.py)
