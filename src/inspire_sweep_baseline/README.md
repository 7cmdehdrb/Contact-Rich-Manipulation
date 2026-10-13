# Inspire Sweep Baseline

환경 ID: `Isaac-Inspire-Sweep-Baseline-v0`

`Isaac-Sweep-Object-UR5e-Random-Single-Inspire-Open-v2-mass`의 구성을 독립 패키지로 옮긴 환경이다. 기존 프로젝트 패키지를 import하거나 상속하지 않는다. MDP 함수, 로봇 스포너, 실린더 스포너, PPO 설정, 실행 스크립트와 로봇 USD를 이 패키지에 포함한다. Isaac Lab, Isaac Sim, RSL-RL은 기존 설치를 사용한다.

## 구성

- UR5e + Axia80을 모티브로 한 원통형 F/T 하우징 + Inspire Robot Hand.
- 단일 실린더: 반지름 4 cm, 높이 10 cm, 질량 1 kg. 물체 link 원점은 바닥이며, 질량 중심은 바닥에서 2 cm 위에 둔다.
- 물체 시작 위치는 `(-0.75, 0, 1.05)`에 XY 각각 ±2 cm와 yaw 변동을 부여한다. 로봇의 초기 관절은 기존 환경과 같다. Reaching을 생략하지 않는다.
- 기본 리셋 이벤트는 좌/우를 환경마다 무작위로 선택한다. `left`는 월드 −Y, `right`는 월드 +Y다. 목표는 초기 물체 위치에서 해당 방향으로 18 cm 떨어진 위치이며, 에피소드 동안 고정된다.
- Reach와 Sweep 게이트의 접근 목표: 물체 X−2 cm, Y−`radius × direction`, **바닥 Z+9 cm**. 높이 10 cm인 실린더의 기하학적 중심을 기준으로는 **+4 cm**다. 질량 중심의 높이를 더하는 계산이 아니다.
- EEF는 손 기준 `(0, 0, 0.13)`이다. F/T 삽입에 따른 2.54 cm 연장을 물리적으로 반영하며, 하우징 중심 기준 `(0, 0, 0.1427)`로 정의해 길이를 중복 가산하지 않는다.
- EEF의 로컬 Y축(Hand의 로컬 X축)과 선반 Z축을 정렬한다. 회전 행렬의 열 벡터를 사용한다.

## 입력과 출력

Actor와 Critic은 같은 `policy` 관측 41D를 사용한다. 별도의 privileged 관측은 없다.

| 관측 순서 | 내용 | 차원 |
| --- | --- | ---: |
| 1 | Arm 상대 관절 위치 | 6 |
| 2 | Arm 상대 관절 속도 | 6 |
| 3 | 직전 액션 | 8 |
| 4 | 현재 물체 위치, 로봇 root 기준 | 3 |
| 5 | 물체 width: 실린더 반지름 0.04 m | 1 |
| 6 | EEF pose: 위치 + quaternion, 로봇 root 기준 | 7 |
| 7 | Sweep 목표 위치, 로봇 root 기준 | 3 |
| 8 | 손바닥 한 영역의 접촉 여부, 0.05 N 임계치 | 1 |
| 9 | EEF 좌표계/기준점의 F/T, N 및 Nm | 6 |
| 합계 | 기존 34D + 센서 7D | **41** |

물체 위치와 width에는 기존의 ±1 cm 관측 노이즈를 유지한다. F/T는 고정 조인트의 반력을 읽어 EEF로 회전하고 모멘트 기준점을 이동한다. Tare를 적용하지 않는다. 리셋 직후에는 stale 센서 데이터를 0으로 마스킹한다. 센서 값을 별도의 보상이나 게이트 조건으로 사용하지 않는다.

| 액션 | 구성 | 차원 |
| --- | --- | ---: |
| Arm | 6개 UR5e 관절의 Joint Position command, 초기 관절 위치 + `0.5 × action` | 6 |
| Hand | 전체 굽힘 openness + 별도 엄지 opposition openness | 2 |
| 합계 | Arm + Hand | **8** |

Hand는 입력을 `[0, 1]`로 제한해 12개 관절에 근사 매핑한다. `1`은 완전히 편 상태, `0`은 굽힌 상태다. 리셋 시 두 입력과 액션 이력은 `[1, 1]`이다. 별도의 액션 필터를 추가하지 않는다.

## 보상과 종료

| 보상 | 가중치 | 내용 |
| --- | ---: | --- |
| reaching | 3.0 | 접근 목표까지의 거리 `d`에 대해 `exp(-10 × d)` |
| orientation | 2.0 | EEF Y축과 선반 Z축의 방향 정렬 |
| sweeping_object | 6.0 | 목표까지 물체 이동 + 기존 물체 Y 속도 보상 |
| homing_after_sweep | 9.0 | 물체 도달 이후 초기 Arm 관절 위치로 복귀 |
| shelf_collision | −0.5 | 선반 이동 및 가상 finger/wrist 프레임의 높이 |
| object_collision | −0.5 | 비대상 물체 이동. 단일 물체이므로 항상 0 |
| action_rate | −0.03 | 8D 액션 변화량의 제곱합 |
| joint_vel | −0.03 | 이름으로 선택한 Arm 6개 관절 속도의 제곱합 |

밀기 보상 게이트는 기존 XYZ 거리 4 cm와 wrist Y 거리 4 cm 조건을 유지한다. 물체가 목표에서 3 cm 이내에 도달하면 기존 도달 보상 분기를 사용한다. Reach 목표는 움직이는 물체를 따라 갱신된다.

- Arm 관절 속도: **6개 Arm 관절 중 하나라도 절댓값이 1 rad/s를 넘으면 종료**. 로그의 기존 이름 `hand_velocity`는 유지하지만 손가락 속도는 이 조건에 포함하지 않는다.
- 물체 바닥 높이 1.04 m 미만 또는 roll/pitch 절댓값 0.9 rad 초과.
- 물체 속도 norm 0.3 m/s 초과.
- 기존 선반 속도 및 가상 finger/wrist 높이 종료 조건.
- 에피소드 10초. 물리 step 0.01초, decimation 2, 제어 50 Hz.

기존의 object collision curriculum은 시작 가중치와 변경 가중치가 모두 −0.5이며, 단일 물체에서 해당 보상은 0이다. 따라서 동작을 바꾸지 않는 이 항목은 따로 구성하지 않았다.

PPO는 기존 설정을 유지한다: Actor/Critic MLP `[256, 128, 64]`, ELU, 관측 정규화 없음, 초기 표준편차 1.0, rollout 36 step, 8 epoch/4 minibatch, learning rate `1e-3` adaptive, gamma 0.98, lambda 0.95, clip 0.2, entropy 0.005, desired KL 0.02, gradient norm 1.0. 체크포인트는 50 iteration마다 저장하며 기본 최대 iteration은 90000이다.

## 실행

저장소 루트에서 실행한다. 스크립트가 새 패키지만 등록하므로 editable 설치나 기존 환경 패키지를 요구하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Baseline-v0 \
  --num_envs 2048 \
  --sweep-direction both \
  --run_name baseline_mass_z9cm_1rad \
  --headless
```

좌로만 밀기는 `--sweep-direction left`, 우로만 밀기는 `--sweep-direction right`로 바꾼다. 해당 옵션을 생략하면 이벤트 기본값인 `both`를 사용한다. 이벤트 설정 파일에서는 `events.object_spawn.params["sweep_direction"]`로 지정한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Baseline-v0 \
  --num_envs 1 \
  --checkpoint "<checkpoint.pt>"
```

녹화:

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Baseline-v0 \
  --num_envs 1 \
  --checkpoint "<checkpoint.pt>" \
  --video --video_length 500 --headless
```

학습 로그는 `logs/rsl_rl/inspire_sweep_baseline/<timestamp>_<run_name>`에 저장된다. 녹화는 해당 체크포인트 디렉터리의 `videos/play`에 저장된다.

로봇 USD는 패키지 내부에 포함하고 물체는 절차적으로 생성한다. 선반 USD는 기존과 동일한 외부 에셋을 사용한다. 기본 주소는 `omniverse://192.168.0.13/Library/Shelf/Arena/Collected_speedrack_shape/speedrack_shape.usd`이며, 다른 컴퓨터에서는 `INSPIRE_SWEEP_SHELF_USD_PATH` 또는 `INSPIRE_SWEEP_ASSET_ROOT`를 설정한다. 기존 `SWEEP_POLICY_SHELF_USD_PATH`/`SWEEP_POLICY_ASSET_ROOT`도 경로 설정값으로 사용할 수 있다.

## 코드 위치

- `inspire_sweep_baseline/env_cfg.py`: scene, 관측/액션, 보상, 이벤트, 종료.
- `inspire_sweep_baseline/env.py`: 리셋 시 Hand 액션 이력과 센서 유효성 처리.
- `inspire_sweep_baseline/assets`: 독립 로봇 USD, F/T 하우징 스포너, 실린더 물성.
- `inspire_sweep_baseline/mdp`: 패키지 자체 MDP 함수와 센서 처리.
- `inspire_sweep_baseline/agents/rsl_rl_ppo_cfg.py`: PPO 설정.
- `scripts`: 복사한 Isaac Lab RSL-RL train/play와 CLI. 기존 프로젝트 import는 제거했다.
- `tests`: CPU 동작 검증 및 USD 정적 검증.
