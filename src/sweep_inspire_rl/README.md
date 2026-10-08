# Sweep Inspire RL

`Isaac-Sweep-Inspire-Right-OSC-v0`는 `example/Sweep-Policy`의 선반 환경과
`hand_manipulation_rl`의 UR5e–Axia80–Inspire Hand를 결합한 별도 Isaac Lab 패키지다.
선반, 물체 USD, 작업영역, 물리 설정과 기본 MDP는 Sweep-Policy를 따른다.
로봇은 Hand 패키지의 USD와 Axia80 fixed measurement joint를 사용한다.

- 물체는 실제 `RigidObjectCollection`에 하나만 생성한다. 기본 물체는 `cup_1`이며
  `--object-name`으로 `bottle_1`, `cup_1`, `cup_2`, `mug_1`, `mug_2`, `can_1` 중 선택한다.
- 명령은 선반의 오른쪽인 world `+Y`로 `0.18 m` 미는 방향만 생성한다.
  IK로 물체 왼쪽 옆의 약간 높은 위치에서 시작하며, 손바닥이 `+Y`를 향하도록 정렬한다.
  Control point는 물체 origin보다 `0.12 m` 높게 두고, 실제 palm pad 중심이 물체의
  선반 깊이 위치와 맞도록 X offset을 보정한다. Reaching 보상은 실제 palm 표면과
  물체 USD collider의 upstream 표면을 기준으로 계산한다.
  밀기 보상은 원본 `reward_random_sweep.pushing_target`을 직접 호출하며,
  이 환경에서 EEF 거리 한계만 `0.09 m`로 지정한다. 원본 환경의 기본값은 `0.04 m`다.
  원본 offset `(target_x - 0.02, target_y - width * sign(sweep_dir_y), target_z + 0.09)`에
  대해 EEF 3D 거리 `< 0.09 m`, wrist Y 거리 `< 0.04 m`만으로 밀기 게이트를 계산한다.
  목표 거리 `< 0.03 m`에서는 게이트 없이 목표 근처 보상을 준다.
  속도 보정도 원본대로 `abs(v_y)`를 사용한다. Tactile·F/T는 관측이며,
  손바닥 접촉·자세 정렬·upstream AABB 조건은 밀기 게이트에 사용하지 않는다.
- Arm은 6D relative pose OSC, fixed impedance, 전 축 stiffness `200`이다.
  마지막 3개 Arm 회전 action은 독립 회전으로 적용하지 않고, 현재 자세에서 고정된
  오른쪽 손바닥 방향으로 돌아가는 회전 명령으로 대체한다.
  중력 보정은 켠다. 원본의 gravity-off 설정을 새 Hand에 적용한 접근 검사에서 약 36 mm
  하강과 wrist3 속도 제한 위반이 발생했으며, 보정을 켠 비교에서는 시작 높이를 유지했다.
- 총 action은 Arm 6D + Hand 2D인 8D다. Hand 입력 두 값은 각각 `[0, 1]`이며
  `openness = 0.8 + 0.2 * clip(action, 0, 1)`로 변환한다. `1`이 완전히 편 상태다.
  공통 손가락 굽힘과 독립 thumb1을 기존 Inspire synergy로 한 번에 제어한다.
- Policy 관측은 71D다. Sweep-Policy의 관측 순서를 유지하면서 로봇 joint position을
  Arm 6개 + Hand 12개로 바꾸고, 실제 Hand synergy 2D, palm tactile 17D,
  Axia80 wrist F/T 6D를 추가한다. 물체와 목표 위치는 EEF 기준 RELATIVE 관측이다.
  F/T는 frame 회전과 모멘트 기준점 이동만 적용하며 Hand 자중을 제거하지 않는다.
- PPO는 Sweep-Policy의 `rsl_rl_ppo_cfg_02.UR5eSweepPPORunnerCfg`를 상속한다.
  `experiment_name`만 `UR5e_shelf_sweep_inspire_right`로 변경하며,
  기본 학습 횟수 `90000`을 포함한 모든 하이퍼파라미터는 유지한다.

## 설치와 자산

저장소 루트에서 Isaac Lab Python으로 세 패키지를 함께 editable install 한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install \
  -e example/Sweep-Policy \
  -e src/hand_manipulation_rl \
  -e src/sweep_inspire_rl
```

선반과 물체는 Sweep-Policy의 원본 USD를 그대로 읽는다. 기본 자산 위치는
`omniverse://192.168.0.13/Library/Shelf`이며, 로컬 복사본이나 다른 Nucleus를 사용하면
패키지를 import하기 전에 root를 지정한다.

```bash
export SWEEP_POLICY_ASSET_ROOT=/path/to/Library/Shelf
```

로컬 root에는 원본과 같은 `Arena/Collected_speedrack_shape/speedrack_shape.usd`와
선택한 물체의 `Objects/...` 경로 및 USD가 참조하는 파일들이 있어야 한다.
선반 USD만 다른 경로에 있으면 `SWEEP_POLICY_SHELF_USD_PATH`를 지정할 수 있다.
물체 이름·경로·폭은 Sweep-Policy의 `sweeping_policy/src/environment.yaml`을 사용한다.
자산이 없을 때는 경로 오류를 보고하며, 선반이나 물체를 primitive로 대체하지 않는다.
`SWEEP_POLICY_ROBOT_USD_PATH`는 이 환경의 로봇 선택에 사용하지 않는다.

## 실행

### V1: 물체 중심 Reach와 XY 게이트

`Isaac-Sweep-Inspire-Right-OSC-v1`은 높이 고정/페널티 추가 이전 상태다.

- Reach 목표는 `(물체 현재 X, 물체 현재 Y, 물체 현재 Z + 0.075 m)`이며,
  EEF의 XYZ 전체 거리로 `exp(-10 * distance)`를 계산한다.
- 밀기 EEF 게이트는 원본 offset에 대해 `norm(offset_xy - eef_xy) < 0.04 m`다.
  wrist Y 거리 `< 0.04 m`, 목표 근처 보상 분기 및 물체–목표 3D 거리는 그대로다.
  Z는 게이트 계산에서 제외한다. 밀기 offset의 Z는 +7.5 cm지만 XY 게이트에는 영향을 주지 않는다.
- Reset은 기존처럼 물체 Z +12 cm에서 시작한다. 별도 높이 페널티는 없다.

### V2: 고정 높이와 Sweeping 높이 페널티

`Isaac-Sweep-Inspire-Right-OSC-v2`는 V1을 상속하고 고정 높이/높이 페널티 및 Reach offset 정합을 적용한다.

- Reach 목표 XY는 밀기 게이트와 동일한 원본 offset
  `(물체 현재 X - 2 cm, 물체 현재 Y - width * sign(sweep_dir_y))`다.
  Z는 에피소드 시작 물체 Z +7.5 cm로 고정하고, XYZ 거리의 `exp(-10 * distance)`를
  가중치 3으로 보상한다. 밀기 게이트는 기존 XY 4 cm 및 wrist Y 4 cm 조건을 유지한다.
- V2의 wrist 목표 Y는 EEF offset Y -5 cm다. 고정 손 자세에서 control point가
  Axia80 origin보다 +Y로 5 cm 앞에 있으므로 두 점에 서로 다른 목표를 적용한다.
  밀기 보상, 높이 페널티, CSV 진단은 동일한 보정 게이트를 사용한다.
- 손 자세 보상은 Inspire 로컬 +X축과 선반 초기 자세의 +Z축 내적 `a`에 대해
  `sign(a) * a²`를 사용한다. 가중치 2는 유지하며 손바닥 방향 항은 포함하지 않는다.
- `sweeping_height`는 XY/wrist 게이트가 열린 동안 높이 오차 제곱에 대한 페널티다.
  가중치 `-1.0`, 오차 scale `0.015 m`이며 기준 Z는 Reach와 같다.
  물체가 기울어도 기준 높이는 따라 올라가지 않는다. Z는 게이트 조건에 포함하지 않는다.

### V3: V1 보상과 아래쪽이 무거운 실린더

`Isaac-Sweep-Inspire-Right-OSC-v3`는 V1을 상속한다. V2의 높이 고정/페널티는 적용하지 않는다.

- 물체는 패키지에 포함된 `assets/weighted_cylinder.usda` 하나만 사용한다.
- 높이 12 cm, 지름 8 cm, 질량 0.8 kg이다. 아래 2 cm에 질량의 80%가 분포하는
  모델에 맞춰 무게중심을 바닥에서 2.2 cm로, 관성도 함께 지정했다.
- 정지/동마찰 계수는 0.25/0.20이다. 접촉 시 실제 마찰은 선반 재질과의 조합에 따른다.
- 원본 밀기 offset에 쓰는 `width`는 측면 접근 거리이므로 반지름인 4 cm를 사용한다.
  Reset 위치는 기존 선반 배치를 유지한다. 무게중심을 낮춰 기울기보다 미끄러짐을 유도하지만
  실제 밀기 동작은 학습 결과로 확인해야 한다.

### V4: 물체 가까이 접근하는 V2

`Isaac-Sweep-Inspire-Right-OSC-v4`는 현재 V2를 상속하고 Y 접근 offset만 0.5 cm로 바꾼다.

- Reach 목표는 `(현재 물체 X -2 cm, 현재 물체 Y -방향×0.5 cm, 시작 물체 Z +7.5 cm)`다.
- Reach, 밀기 게이트, 높이 페널티 게이트가 동일한 `approach_y_offset=0.005`를 사용한다.
- wrist 목표 Y는 새 EEF offset Y -5 cm이고, EEF XY/wrist Y 거리 한계는 모두 4 cm다.
- 실제 물체 `width`와 reset 위치, 손 자세 보상, 높이 페널티, Metrics 및 PPO 설정은 V2와 같다.
- 전용 experiment 이름은 `UR5e_shelf_sweep_inspire_right_v4`다. 기본 물체는 `cup_1`이다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v4 \
  --num_envs 2048 --object-name cup_1 \
  --run-name close_reach_0p5cm --headless
```

학습·재생 스크립트의 `--task`로 버전을 선택한다. 버전별 PPO 설정은 동일하고,
로그의 experiment 이름은 각각 `UR5e_shelf_sweep_inspire_right_v1`, `_v2`, `_v3`다.
`--object-name`을 생략하면 V0/V1/V2/V4는 `cup_1`, V3는 `weighted_cylinder`를 사용한다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v2 \
  --num_envs 2048 --object-name cup_1 \
  --run-name fixed_height_penalty --headless

./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v3 \
  --num_envs 2048 --object-name weighted_cylinder \
  --run-name weighted_cylinder --headless

./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/play.py \
  --task Isaac-Sweep-Inspire-Right-OSC-v2 \
  --checkpoint /path/to/v2/model.pt --num_envs 1 \
  --object-name cup_1 --real-time
```

V3 재생은 위 명령에서 task를 `Isaac-Sweep-Inspire-Right-OSC-v3`, object-name을
`weighted_cylinder`로 바꾸고 해당 체크포인트를 지정한다.
V1/V2/V3/V4의 `--gate-log`에서 `reaching_distance_m`은 EEF의 XY 게이트 거리이며,
V0에서는 EEF의 3D 게이트 거리다. Reach 보상의 거리와 구분한다.

### 학습 중 Sweep 지표

TensorBoard의 `Sweep/*`에 다음 지표를 자동 기록한다. 별도 CLI 옵션은 필요 없다.
평균은 각 에피소드의 실제 관측 스텝으로 계산하며, reward 가중치나 dt를 적용하지 않는다.
종료 스텝도 reset 이전 상태를 포함한다. 관측/action 차원 및 PPO 설정은 변경하지 않는다.

| 지표 | 의미 |
|---|---|
| `near_hand_rate`, `near_wrist_rate`, `gate_rate` | EEF 조건, 보정 wrist 조건, 두 조건 AND의 스텝 비율 |
| `near_wrist_uncompensated_rate` | 보정 전 wrist 조건 통과율; V2 보정 효과 비교 |
| `wrist_blocked_given_near_hand_rate` | EEF 조건을 통과한 스텝 중 wrist 조건 실패 비율; EEF 통과가 없으면 0 |
| `eef_gate_distance_m`, `wrist_y_distance_m`, `wrist_y_distance_uncompensated_m` | 게이트 거리와 보정 전후 wrist 거리 |
| `eef_wrist_y_separation_m` | 실제 EEF Y - wrist Y; 고정 자세에서 약 0.05 m |
| `palm_contact_rate`, `any_pad_contact_rate`, `gate_without_pad_contact_rate` | tactile 접촉 및 게이트만 열리고 pad 접촉은 없는 스텝 비율 |
| `forward_velocity_m_s`, `forward_motion_rate`, `backward_motion_rate` | 목표 방향 속도와 ±0.005 m/s 기준 이동 비율 |
| `final_progress_m`, `max_progress_m`, `goal_distance_m` | 종료 시 순이동, 에피소드 최대 전진량(0 이상), 평균 목표 3D 거리 |
| `height_abs_error_m`, `object_tilt_deg`, `max_object_tilt_deg`, `hand_up_error_deg` | EEF 높이 오차, 물체 up-axis 기울기와 손 up-axis 오차 |
| `gate_reached_rate`, `goal_region_reached_rate` | 한 번이라도 게이트/목표 3 cm 영역에 도달한 에피소드 비율 |

진행량은 시작 물체 위치에 대한 목표 방향 Y 이동량이다. 물체 origin의 이동이므로
기울기에 의한 이동도 포함한다. 목표 영역 도달률은 접촉/직립을 요구하는 성공률과 다르다.
Pad 접촉률은 tactile pad만 측정하므로 Hand base의 접촉을 모두 검출하지는 못한다.
`--gate-log` CSV에도 보정 전 거리, wrist 목표 Y, 진행량, 방향 속도, 기울기를 기록한다.

```bash
# reset / OSC / 단일 물체 / RELATIVE 관측 / Hand 범위 / tactile / F/T 검사
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/smoke_env.py \
  --num-envs 1 --steps 8 --object-name cup_1 --headless

# 학습: 기본 PPO 설정은 원본과 동일
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --num_envs 4096 --object-name cup_1 --headless

# 짧은 학습 실행
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/train.py \
  --num_envs 4 --max_iterations 2 --object-name cup_1 --headless

# 정책 실행
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/play.py \
  --checkpoint logs/rsl_rl/UR5e_shelf_sweep_inspire_right/<run>/model_89999.pt \
  --num_envs 1 --object-name cup_1
```

`train.py`는 `--checkpoint`로 이어 학습하고, `play.py`는 `--steps`로 실행 길이를
제한할 수 있다. 체크포인트는 이 환경의 71D 관측과 8D action으로 학습한 것을 사용한다.
세 스크립트 모두 `AppLauncher` 실행 후 simulator 모듈을 import한다.

### 물체 옆에서 멈출 때 Sweeping 게이트 진단

학습한 체크포인트를 동일한 물체로 재생하며 각 제어 스텝의 조건을 CSV로 저장한다.

```bash
./IsaacLab/isaaclab.sh -p src/sweep_inspire_rl/scripts/play.py \
  --checkpoint /path/to/model.pt --object-name cup_1 \
  --num_envs 1 --steps 500 --gate-log /tmp/sweep_gate.csv
```

`--gate-log-env`는 기록할 vector environment 번호이며 기본값은 0이다.
로그는 reward 계산 시점에 복사하므로 종료 스텝도 자동 reset 이전 상태를 기록한다.
이 옵션은 보상 가중치나 게이트 조건을 변경하지 않는다. 50스텝마다 현재 조건을
출력하고 실행 종료 시 각 조건이 실패한 횟수를 출력한다. 실패 횟수는 서로 중복될 수 있다.

| CSV 필드 | 해석 |
|---|---|
| `near_hand` / `reaching_distance_m` | 원본 offset과 `ee_frame` 첫 target 사이 3D 거리가 0.09 m 미만이어야 참 |
| `near_wrist` / `wrist_y_distance_m` | wrist 목표와 `wrist_frame` 첫 target 사이 Y 거리가 0.04 m 미만이어야 참. V2의 wrist 목표는 offset Y -0.05 m |
| `gate` / `sweeping_raw` | 위 두 조건의 AND와 가중치·dt 적용 전 밀기 보상 |
| `goal_region` / `goal_distance_m` | 목표 3D 거리 0.03 m 미만이면 게이트 없이 목표 근처 보상 지급 |
| `object_velocity_y_m_s` | 물체 +Y 속도. 원본 속도 보정은 절댓값 사용 |
| `palm_contact` / `palm_force_n` | 손바닥 접촉 비트와 net force 크기. 진단 정보이며 게이트 조건이 아님 |
| `other_pad_contact` / `alignment` | 다른 tactile pad 접촉 및 자세 정렬. 진단 정보이며 게이트 조건이 아님 |
| `sensor_data_fresh` | 현재 physics step의 센서 값인지 |
| `eef_*_w_m` / `palm_*_w_m` / `object_*_w_m` | EEF control point, 손바닥 표면 기준점, 물체 origin의 world 좌표 |
| `done_*` / `episode_step` | 해당 스텝의 종료 원인과 에피소드 스텝 번호 |

게이트는 Inspire의 `ee_frame` control point와 Axia80 `wrist_frame`을 원본 조건에
그대로 대입한다. 거리 게이트가 참이면 tactile 값이 0이어도 밀기 보상을 받을 수 있다.
목표 거리 3 cm 이내에서는 거리 게이트 자체도 요구하지 않는다.

2026-10-06 source USD 형상 감사에서는 손바닥 pad collision mesh의 모든 vertex가
base collision hull 안에 있고, pad 중심의 +Y 방향 선에서 base가 약 5.18 mm
먼저 닿는 것으로 확인되었다. 이 형상은 현재 밀기 보상의 접촉 조건으로 사용되지 않는다.
`scripts/audit_palm_geometry.py`는 pxr·NumPy·SciPy가 있는 USD Python에서 형상을
재검사하며, 결과는 `reports/sweep_inspire_rl/palm_geometry_audit.json`에 있다.

시뮬레이터가 필요 없는 계약 검사는 다음과 같이 실행한다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/min/miniconda3/envs/env_isaaclab/bin/python -m pytest src/sweep_inspire_rl/tests -q
```

이 workspace의 ROS pytest plugin은 `lark`가 없어 자동 로딩을 끈다. 다른 Isaac Lab Python
설치에서는 위 Python 경로를 해당 설치 경로로 바꾼다. 실제 IK 성공 여부, USD 로딩,
접촉 센서와 F/T의 physics 갱신은 위 smoke 실행으로 검증한다. Reset clearance는 Hand
collider의 보수적인 AABB와 물체·선반 표면 높이를 검사한다.

Reset 직후에는 이전 에피소드의 접촉·F/T가 새 자세에 섞이지 않도록 해당 환경의
sensor 관측을 0으로 초기화하고, 첫 physics step부터 live 값을 사용한다.

검증: 기존 수치 테스트와 원본 밀기 보상 회귀 검사를 통과했다. 실제 AppLauncher에서 6종 물체 config 검증과 PPO 전체 설정
비교를 통과했다. `/tmp`의 임시 선반·물체·ground fixture를 사용한 CPU smoke에서도
8개 환경의 reset, 71D 관측, OSC/Hand 제어, 센서, 두 번의 step과 재reset을 검증했다.
원본 Nucleus 자산을 사용하는 smoke는 서버 응답 대기로 완료하지 못했다. 실제 선반의
기둥·다른 단과 물체별 접촉 동작은 원본 자산과 정상 GPU 환경에서 추가 확인해야 한다.
