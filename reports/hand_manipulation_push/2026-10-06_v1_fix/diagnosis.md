**기존 Push-v1 정책은 Cube를 밀기보다 EEF를 위로 이동시키는 동작을 학습했다.** 학습 로그의 접촉·진행 보상이 거의 사라졌고, 저장된 정책의 결정적 실행에서도 첫 동작부터 위쪽 이동을 명령했다. 따라서 진행 보상 가중치만 높이는 것으로는 접촉 이전의 문제를 해결하기 어렵다.

대상은 `2026-10-05_22-17-17` run의 iteration 0–4336과 `model_4300.pt`다. 마지막 100 iteration은 **4237–4336**이다. 아래 수치는 [분석 JSON](summary.json), [보상 CSV](reward_windows.csv), [지표 CSV](metric_windows.csv), [저장 소스 감사](source_findings.json)를 근거로 한다.

| 마지막 100 iteration의 지표 | 관측값 |
|---|---:|
| episode 중 유효 접촉 경험 `contact_seen` | 0.0217593% |
| 실제 pushing 경험 `push_seen` | 0.0069444% |
| 성공 | 0 |
| 종료 시 목표 거리 | 0.249972 m |
| palm 접근 목표와의 거리 | 0.518190 m |
| action_excess의 음수 보상 기여 비중 | 74.7863% |
| episodic approach / progress 기여 | +0.000616672 / +0.000116202 |
| episodic action_excess / failure 기여 | −0.384422 / −0.0962474 |

**분모에 주의해야 한다.** `Episode_Reward`에는 이미 가중치와 `dt`가 적용되어 있으며 최대 episode 시간 **10초**로 나눈 값이 기록된다. 위 episodic 기여는 로그 평균에 10을 곱했다. 지표는 종료되어 reset되는 행들의 평균을 다시 평균한 값이므로 전체 episode 수로 가중한 정확한 모집단 비율이 아니다. `Episode_Termination`은 모든 환경의 최근 종료 상태를 평균하므로 reset 행 지표와 분모가 다르다. 종료 시 palm 높이 오차도 실제 EEF C 높이와 다른 값이다.

**정책 자체가 위쪽으로 이동한다는 사실은 별도 trace로 확인했다.** 저장 당시 기하·reset·controller를 복원한 **32환경 × 500 step**, 실제 64차원 관측 기반 결정적 실행에서 첫 step의 31/32환경이 위쪽 입력을 양의 한계로 clipping했다. 첫 step의 평균 world Z 명령은 +12.01 mm였다. 전체 16,000 transition에서 실제 Cube–palmar 접촉은 0이었고, Table 상면 위 실제 EEF C 높이는 평균 **0.345903 m**, 95백분위 0.468903 m였다. 기록한 관측을 저장 actor에 다시 넣었을 때 최대 출력 차이는 3.77×10⁻⁵로 재현되었다. 이는 확률적 노이즈만으로 설명되지 않는 정책 평균 출력이다. [trace 분석](deterministic_trace/trajectory_analysis.json), [물리 궤적 요약](deterministic_trace/summary.json), [전체 궤적](deterministic_trace/trajectory.csv)

**사실과 원인 추론은 다음과 같이 구분한다.** 저장 소스의 접근 보상은 최초 유효 접촉 이전에 접근 기록을 갱신할 때만 지급되고, 다시 멀어지거나 높이 떠 있는 동안의 지속 비용은 없었다. 누적 위치 target의 6 cm 제한은 현재 C와 target 사이 오차를 제한하며 절대 높이를 제한하지 않았다. 이 구조에서 초기 접근 이득을 얻은 뒤 접촉·충돌을 피하는 동작이 유지될 수 있다는 것은 코드와 궤적에 부합하는 추론이다. 여러 조건이 함께 바뀌므로 각 조건의 독립적인 인과 효과까지 입증한 것은 아니다. 진행 보상은 실제 pad 힘·방향 정렬·Table 지지 조건을 요구하므로 접촉이 없는 episode에서는 가중치가 커도 발생하지 않는다.

확률적 학습 입력의 포화도 별도 문제다. checkpoint 4300의 원시 Gaussian 표준편차는 평균 **5.299**였다. 이 분포의 축별 clipping 확률 평균은 어떤 actor 평균에서도 최소 **85.0006%**라는 해석적 하한을 갖는다. 학습 로그의 종료 행 축별 clipping 평균도 84.9481%였다. 해석적 하한, 종료 행 평균, 결정적 trace의 축별 clipping 26.9039%는 서로 다른 측정이다. 높은 표준편차는 미세 제어를 어렵게 하지만 결정적 정책의 초기 위쪽 동작 원인을 전부 설명하지는 않는다. [checkpoint 분포 분석](checkpoint_action_distribution.json)

**Table과 Cube 경로도 공통 환경에서 달라져 있었다.** robot base 쪽 Table 경계는 `center_x + width_x/2`다. 저장된 V1은 공통 Table보다 23 cm 더 base 쪽까지 뻗어 있고, Cube 경로의 X 중점도 26 cm 더 가까웠다. 이 기하 차이는 확인된 사실이며, 팔·손의 접근 공간과 Table 충돌 가능성에 미치는 학습상의 크기는 로그만으로 분리할 수 없다.

| 기하 | 저장된 V1 | 현재 공통 상속 기하 |
|---|---:|---:|
| Table center X | −0.75 m | −0.75 m |
| Table X 폭 | 0.82 m | 0.36 m |
| base 쪽 Table 경계 X | −0.34 m | −0.57 m |
| Cube 경로 X 중점 | −0.44 m | −0.70 m |

**현재 수정은 접촉 이전의 비용과 실제 접촉 제어를 함께 다룬다.** [현재 V1 설정](../../../src/hand_manipulation_test/hand_manipulation_test/config/ur5e/push_v1_env_cfg.py), [보상](../../../src/hand_manipulation_test/hand_manipulation_test/mdp/push_v1_rewards.py), [높이 판정](../../../src/hand_manipulation_test/hand_manipulation_test/height_math.py)에 다음 내용이 적용되어 있다.

- 실제 접촉 기준점과 현재 Cube 접근 목표의 거리 `−gap`를 매 step 부과한다. 가중치는 1이다. 기존 기록 갱신 보상은 유지하며, 머무르는 것만으로 생기는 양의 보상은 추가하지 않았다.
- 실제 EEF C 높이를 Table 상면 기준으로 계산한다. 0.15 m를 넘으면 `−5 × max(h−0.15, 0)`의 비용을 부과하고, **0.25 m 이상**이면 실패로 처리한다. 실패는 공유 상태에 반영되어 양의 보상·성공·timeout과 일관된 우선순위를 갖는다.
- 원시 입력 초과 비용의 가중치를 0.002에서 **0.02**로 높였다. 입력 clipping 뒤의 물리 action-rate 비용과 구분한다. 위 비용들은 RewardManager가 `dt`를 한 번 적용한다.
- Table과 Cube 경로는 위의 먼 공통 기하를 상속한다. 양측 손가락은 base 바깥을 향하도록 reset하고, episode 중 손가락 방향과 wrist-2의 물리 branch를 검사한다. 오른쪽은 thumb-up, 왼쪽은 thumb-down이다.

carrier에 가려진 palmar pad와 접촉 하중에서 wrist·손 방향이 무너지는 문제는 학습 집계 로그만으로 판별할 수 없는 물리적 병목이다. 현재 코드는 왼쪽 접촉 기준과 해당 기준의 실제 힘 확인을 기존 **Thumb4 palmar pad**로 통일하고, 기존 Thumb3/4 collision pad를 H +Y 방향으로 **12 mm** 노출한다. 실제 17개 palmar pad 센서와 Cube 필터를 사용한다. [접촉 기준](../../../src/hand_manipulation_test/hand_manipulation_test/mdp/push_v1_events.py), [V1 pad 기하](../../../src/hand_manipulation_test/hand_manipulation_test/assets/push_v1_robot.py)

controller는 설정 강성 **200**을 유지하면서 질량을 고려한 implicit 시간 갱신과 실제 Cube–pad 반력 예측을 사용한다. Cube에 작용하는 힘의 반대 부호와 C 기준 모멘트를 robot base 축으로 변환해 예측에 넣는다. 측정된 반력이 실제 하중과 일치하면 설정 강성의 정적 평형을 유지할 수 있으며, 측정되지 않은 하중에는 유한 timestep에 따른 유효 강성 감소가 남는다. 힘·모멘트 제한과 EMA가 있으므로 모든 접촉에서 이상적인 200 평형이 보장된다는 의미는 아니다. F/T 자기 하중을 접촉 반력으로 대체하지 않는다. [controller](../../../src/hand_manipulation_test/hand_manipulation_test/mdp/push_v1_controller.py), [impedance 계산](../../../src/hand_manipulation_test/hand_manipulation_test/impedance_math.py)

**PPO는 요청한 Sweep 예제 설정을 그대로 유지했다.** Gaussian 분포의 강제 std 상한이나 PPO hyperparameter 변경을 사용하지 않았다. [V1 PPO 상속](../../../src/hand_manipulation_test/hand_manipulation_test/agents/rsl_rl_push_v1_ppo_cfg.py)

이 문서는 기존 학습의 실패 진단과 현재 코드 수정 내용을 설명한다. 기존 로그에는 실패 유형별 학습 통계가 없고, 32환경 trace는 전체 학습 모집단을 대표하는 성공률 평가가 아니다. 짧은 물리 검증과 CPU 테스트가 통과해도 PPO 수렴을 증명하지 않는다. 변경된 환경·보상·controller에서 **새 학습을 수행해 접촉, 진행, 성공률, 실제 EEF 높이와 raw clipping이 개선되는지 확인해야 한다.**

최종 [검증 기록](validation.json): CPU 280개 통과·1개 skip, GPU 경계 초기화·500-step timeout, 양방향 실제 측면 접촉·15~31mm 밀기 후 정지, 실제 C 높이 벌점·hard termination, 2,048환경 PPO 2 iteration·새 checkpoint 100-step 재생을 통과했다. 장거리 fixture는 양방향 18cm 이상 움직였지만 좌측 한 환경의 실패 종료 때문에 전체 20cm 목표를 모든 행에서 안정적으로 완료하는 검증은 통과하지 못했다. 실행 경로와 물리적 밀기 가능성을 확인한 결과이며, 새 정책의 학습 성공률은 아직 측정하지 않았다.
