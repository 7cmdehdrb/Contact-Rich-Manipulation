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
  선반 깊이 위치와 맞도록 X offset을 보정한다. Reaching/pushing 보상은 control point
  대신 실제 palm 표면 위치와 물체 USD collider의 upstream 표면을 기준으로 계산한다.
  실제 palm pad(channel 0) 접촉이 있어야 밀기·목표 도달 보상을 준다. Thumb/base carrier나
  finger pad만 닿는 동작은 이 보상을 얻지 못한다.
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

검증: 수치 테스트 29개, 실제 AppLauncher에서 6종 물체 config 검증과 PPO 전체 설정
비교를 통과했다. `/tmp`의 임시 선반·물체·ground fixture를 사용한 CPU smoke에서도
8개 환경의 reset, 71D 관측, OSC/Hand 제어, 센서, 두 번의 step과 재reset을 검증했다.
원본 Nucleus 자산을 사용하는 smoke는 서버 응답 대기로 완료하지 못했다. 실제 선반의
기둥·다른 단과 물체별 접촉 동작은 원본 자산과 정상 GPU 환경에서 추가 확인해야 한다.
