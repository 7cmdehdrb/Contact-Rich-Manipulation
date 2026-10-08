# Sweep Inspire RL

`example/Sweep-Policy`의 선반 환경과 `hand_manipulation_rl`의
UR5e–Axia80–Inspire Hand를 결합한 Isaac Lab 패키지다.
현재 공개 환경은 아래 세 가지이며, 기존 V1/V2/V3의 실험 구성을 재정의한다.

| 환경 ID | 실험 | Reach 목표 |
|---|---|---|
| `Isaac-Sweep-Inspire-Right-OSC-v1` | 물체 접촉: Reach 보상만 사용 | 현재 물체 XY, 시작 물체 Z +7.5 cm |
| `Isaac-Sweep-Inspire-Right-OSC-v2` | example의 단일 물체 Sweeping 보상 로직 | 현재 물체 X −2 cm, Y −width×방향, Z +7.5 cm |
| `Isaac-Sweep-Inspire-Right-OSC-v3` | V2의 Y 접근 offset을 0.5 cm로 축소 | 현재 물체 X −2 cm, Y −0.5 cm×방향, Z +7.5 cm |

세 환경의 기본 물체는 모두 `cup_1`이다. V3도 컵을 사용하며 실린더 실험이 아니다.
`--object-name`으로 `bottle_1`, `cup_1`, `cup_2`, `mug_1`, `mug_2`, `can_1` 중 선택한다.
물체는 실제 `RigidObjectCollection`에 하나만 생성한다.

## 공통 제어와 관측

- EEF/OSC 제어점은 `inspire_base_link` 로컬 **`(0, 0, 0.10)`**이다.
  손의 로컬 Z축을 따라 Wrist 연장선에 놓이며 전방 Y offset은 없다.
  EEF 상대 관측, Reach, OSC, reset IK와 F/T 모멘트 기준점에 같은 위치를 사용한다.
- 초기 관절 설정과 reset IK는 기존 설정을 유지한다. 물체 왼쪽의 높은 위치에서 시작하며,
  reset 제어점 높이는 물체 Z +12 cm다. 보상의 목표 높이인 +7.5 cm와 구분한다.
  미는 방향은 선반의 오른쪽인 world `+Y`, 목표 이동량은 `0.18 m`다.
- Arm은 6D relative pose OSC, fixed impedance, 전 축 stiffness `200`, damping ratio `1`이다.
  중력 보정과 inertial dynamics decoupling을 사용한다. 마지막 3개 회전 action은
  고정된 오른쪽 손 자세로 돌아가는 회전 명령으로 대체한다.
- Action은 Arm 6D + Hand synergy 2D인 **8D**다. Hand 입력은 각각 `[0, 1]`이며
  `openness = 0.8 + 0.2 * clip(action, 0, 1)`로 변환한다. `1`이 완전히 편 상태다.
- Policy 관측은 **48D**다. Arm 6개 + Hand 12개 joint position, Arm joint velocity,
  이전 action, EEF 기준 물체·목표 위치, 물체 width, EEF pose, 실제 Hand synergy를 사용한다.
  **Tactile 17D와 wrist F/T 6D는 정책 관측에 포함하지 않는다.**
- 접촉 센서, F/T reader와 `palm_tactile_bits`, `wrist_wrench_c` 구현은 유지한다.
  이후 센서 관측을 재사용할 때 환경 cfg 생성에 `enable_sensor_observations=True`를 지정하면
  두 관측 항을 다시 활성화해 71D 관측을 사용한다. 기본값은 `False`다.
  F/T는 프레임 회전과 모멘트 기준점 이동만 적용하며 Hand 자중을 제거하지 않는다.
- EEF, Finger, Wrist, 물체/목표 프레임 표시 scale은 기존 값의 **0.02배**다.
- PPO는 Sweep-Policy의 `rsl_rl_ppo_cfg_02.UR5eSweepPPORunnerCfg`를 상속하며,
  기본 학습 횟수 `90000`을 포함한 하이퍼파라미터는 유지한다. 로그는 각 버전별
  `UR5e_shelf_sweep_inspire_right_v1`, `_v2`, `_v3`에 저장한다.

이전 **71D 관측으로 학습한 체크포인트는 현재 48D 정책에 그대로 로딩할 수 없다.**
현재 구성을 비교하려면 새로 학습한다. V0/V4/V5는 학습·재생 스크립트의 선택 대상이 아니다.
기본 task는 V2다.

## 보상 구성

### V1: 물체 중앙에 도달하는 접촉 실험

```text
X = 현재 물체 X
Y = 현재 물체 Y
Z = 에피소드 시작 물체 Z +0.075 m
```

EEF와 목표 사이 **XYZ 거리** `d`에 대해 `3 × exp(-10 × d)`만 보상한다.
자세, action 변화, 관절 속도, 선반 충돌, 물체 충돌, Sweeping, 높이와 복귀 보상은
사용하지 않는다. 종료 조건은 기존 설정을 유지한다. 따라서 선반 충돌·관절 속도·물체 낙하 등의
종료 조건은 계속 적용된다. 제거된 물체 충돌 보상을 참조하는 curriculum도 비활성화한다.

### V2: example의 단일 물체 Sweeping

비교 기준은
`example/Sweep-Policy/sweeping_policy/config/ur5e/osc_random_single_offset_env_cfg.py`다.
기존 로봇 제어·reset은 유지하고, 보상은 example의 단일 물체 로직에 맞춘다.

```text
Reach/게이트 중심 X = 현재 물체 X −0.02 m
Reach/게이트 중심 Y = 현재 물체 Y −width × sign(sweep_dir_y)
Reach/게이트 중심 Z = 현재 물체 Z +0.075 m
```

원본 높이 +9 cm 대신 낮은 +7.5 cm를 사용한다. **V2와 V3의 Z는 현재 물체 Z를 따른다.**
이전 V2의 시작 높이 고정이나 `sweeping_height` 페널티는 적용하지 않는다.

| 보상 | 공식/동작 | 가중치 |
|---|---|---:|
| Reach | `exp(-10 × XYZ 거리)` | 3 |
| 손 자세 | Inspire 로컬 +X와 선반 +Z 내적 `a`에 대해 `sign(a) × a²` | 2 |
| Sweeping | 원본 목표 거리·속도 shaping과 게이트 | 6 |
| Action 변화 | 이전 action과의 차이 제곱합 | −0.03 |
| Arm 관절 속도 | 이름으로 지정한 Arm 6개 관절의 속도 제곱합 | −0.03 |
| 선반 충돌 | 원본 선반 조건 페널티 | −0.5 |
| 다른 물체 충돌 | 원본 비목표 물체 이동 페널티; 단일 물체에서는 0 | −0.5 |

원본 EEF 로컬 Y축 대신 Inspire 로컬 X축으로 위쪽 정렬을 계산한다.
복귀 보상은 비교 기준인 단일 물체 OSC 환경과 같이 사용하지 않는다.

밀기 게이트는 두 조건을 모두 요구한다.

```text
norm(EEF XYZ − 게이트 중심 XYZ) <0.04 m
abs(wrist Y − 게이트 중심 Y) <0.04 m
```

**XY만 검사하는 이전 게이트와 달리, 원본처럼 XYZ norm을 사용한다.**
EEF의 Y offset이 0이므로 wrist 목표에도 별도 −5 cm 보정을 적용하지 않는다.
목표까지 물체의 3D 거리 `d_goal`이 3 cm 이상일 때 밀기 raw 보상은
`gate × (1 − d_goal /0.18 + 속도 보정)`이다.
속도 보정은 원본처럼 `abs(v_y)`가 5 cm/s 초과, 10 cm/s 미만이면 +0.5,
10 cm/s 이상이면 −0.5다.
`d_goal <0.03 m`이면 게이트 없이 `2 × exp(-5 × d_goal)`을 준다.
게이트가 열려도 Reach 보상은 계속 거리로 계산한다.
Tactile·F/T와 접촉 여부는 게이트 조건이나 보상에 사용하지 않는다.
모든 보상은 Isaac Lab reward manager에서 제어 `dt`가 적용된다.

### V3: 극단적으로 가까운 Y 접근 목표

V2를 상속하고 **Y 접근 offset만 `0.005 m`**로 바꾼다.
X offset −2 cm, 현재 물체 Z +7.5 cm, 보상식·가중치와 XYZ/wrist 4 cm 조건은 같다.
Reach와 밀기 게이트는 같은 새 Y 목표를 사용한다.
실제 물체 `width`와 reset 위치는 변경하지 않으며 접근 offset은 width와 분리한다.

## 설치와 자산

저장소 루트에서 Isaac Lab Python으로 세 패키지를 함께 editable install 한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install \
  -e example/Sweep-Policy \
  -e src/hand_manipulation_rl \
  -e src/sweep_inspire_rl
```

선반과 물체는 Sweep-Policy의 원본 USD를 읽는다. 기본 위치는
`omniverse://192.168.0.13/Library/Shelf`이며, 로컬 복사본이나 다른 Nucleus를 사용하려면
패키지를 import하기 전에 root를 지정한다.

```bash
export SWEEP_POLICY_ASSET_ROOT=/path/to/Library/Shelf
```

로컬 root에는 원본과 같은 `Arena/Collected_speedrack_shape/speedrack_shape.usd`와
선택한 물체의 `Objects/...` 경로 및 USD 참조 파일들이 있어야 한다.
선반만 별도 경로에 있으면 `SWEEP_POLICY_SHELF_USD_PATH`를 지정한다.
물체 이름·경로·width는 Sweep-Policy의 `sweeping_policy/src/environment.yaml`에서 읽는다.
자산이 없으면 경로 오류를 보고하며 primitive로 대체하지 않는다.
`SWEEP_POLICY_ROBOT_USD_PATH`는 이 환경의 로봇 선택에 사용하지 않는다.

## 학습과 재생

V1 학습:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v1 \
  --num_envs 2048 --object-name cup_1 \
  --run-name center_reach_no_sensors --headless
```

V2 학습:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v2 \
  --num_envs 2048 --object-name cup_1 \
  --run-name source_sweep_no_sensors --headless
```

V3 학습:

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v3 \
  --num_envs 2048 --object-name cup_1 \
  --run-name close_sweep_0p5cm_no_sensors --headless
```

재생은 학습한 버전의 ID와 해당 체크포인트를 지정한다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/play.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v2 \
  --checkpoint /path/to/v2/model.pt \
  --num_envs 1 --object-name cup_1 --real-time
```

`train.py --checkpoint`는 동일한 현재 구성의 학습을 이어가고,
`play.py --steps`는 재생 스텝 수를 제한한다.

## Sweeping 지표와 게이트 진단

V2/V3의 TensorBoard `Sweep/*`는 episode별 실제 관측 스텝 평균을 기록한다.
보상 가중치나 `dt`를 적용하지 않으며 종료 스텝도 reset 이전 상태를 포함한다.
센서 값은 진단용으로 읽을 수 있지만 정책 관측에는 들어가지 않는다.

| 지표 | 의미 |
|---|---|
| `near_hand_rate`, `near_wrist_rate`, `gate_rate` | EEF, wrist, 두 조건 AND의 스텝 비율 |
| `wrist_blocked_given_near_hand_rate` | EEF 통과 스텝 중 wrist가 차단한 비율 |
| `eef_gate_distance_m`, `wrist_y_distance_m` | EEF XYZ 거리와 wrist Y 거리 |
| `eef_wrist_y_separation_m` | 실제 EEF Y −wrist Y; 목표 자세에서 약 0 m |
| `palm_contact_rate`, `any_pad_contact_rate`, `gate_without_pad_contact_rate` | 유지된 tactile 센서의 접촉 진단 |
| `forward_velocity_m_s`, `forward_motion_rate`, `backward_motion_rate` | 목표 방향 속도와 ±0.005 m/s 기준 이동 비율 |
| `final_progress_m`, `max_progress_m`, `goal_distance_m` | 종료 시 순이동, 최대 전진량, 목표 3D 거리 |
| `height_abs_error_m`, `object_tilt_deg`, `max_object_tilt_deg`, `hand_up_error_deg` | 높이와 기울기 오차 |
| `gate_reached_rate`, `goal_region_reached_rate` | 한 번이라도 게이트/목표 3 cm 영역에 도달한 비율 |

진행량은 물체 origin의 목표 방향 Y 이동량이므로 기울기에 의한 이동도 포함한다.
목표 영역 도달률은 접촉이나 직립을 요구하는 성공률과 다르다.
Pad 접촉률은 그리퍼 전체의 충돌을 모두 검출하지 않는다.
`near_wrist_uncompensated_rate`와 보정 전 거리도 기존 진단 필드로 유지되며,
현재 별도 wrist 보정은 0이므로 보정 후 값과 동일하다.

V2/V3의 `--gate-log`는 각 제어 스텝의 조건과 위치·속도·기울기를 CSV에 기록한다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/play.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v3 \
  --checkpoint /path/to/v3/model.pt --object-name cup_1 \
  --num_envs 1 --steps 500 --gate-log /tmp/sweep_gate.csv
```

`--gate-log-env`는 기록할 vector environment 번호이며 기본값은 0이다.
`reaching_distance_m`은 **밀기 게이트의 XYZ 거리**다.
`near_hand`, `near_wrist`, `gate`, `goal_region`, `sweeping_raw`로 각 보상 조건을 확인한다.
`sweeping_raw`에는 reward 가중치와 `dt`가 적용되지 않는다.
V1은 Sweeping 보상을 사용하지 않아 이 게이트 진단을 사용하지 않는다.

Reset 직후 접촉·F/T는 새 physics step이 생길 때까지 유효하지 않으므로
`sensor_data_fresh`로 이전 에피소드의 센서 값을 마스킹한다.
센서 구현과 실린더 USD 등 기존 자산은 이후 실험에서 재사용할 수 있도록 유지한다.
