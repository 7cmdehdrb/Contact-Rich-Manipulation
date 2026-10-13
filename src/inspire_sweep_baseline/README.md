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


## 물체 질량

| 항목                 | 설정값                |
| ------------------ | ------------------ |
| 질량 평균 (Mean)       | 0.365 kg (365 g)   |
| Gaussian Noise (σ) | 0.0675 kg (67.5 g) |
| 예상 95% 범위 (±2σ)    | 0.230 \~ 0.500 kg  |

추천 설정

\\[ \boxed{m\sim\mathcal{N}(0.365,\\;0.0675^2)\ \mathrm{kg}} \\]

- Gaussian Noise는 Episode마다 샘플링
- 약 95%의 샘플이 230\~500 g 범위에 포함
- 범위를 절대 벗어나면 안 된다면 0.230\~0.500 kg으로 Clipping 적용

## 마찰 계수

Friction Combine Mode: `Average`

| 오브젝트 | 정적 마찰 계수 (Mean) | Gaussian Noise (σ) | 동적 마찰 계수 (Mean) | Gaussian Noise (σ) |
| ---- | --------------- | ------------------ | --------------- | ------------------ |
| 바닥   | 0.315           | 0.051              | 0.252           | 0.041              |
| 물건   | 0.315           | 0.040              | 0.252           | 0.032              |
| Tool | 0.585           | 0.145              | 0.468           | 0.116              |

설정 조건

- 동적 마찰 계수의 평균은 정적 마찰 계수 평균의 약 0.8배. 샘플마다 고정 비율을 강제하지 않는다.
- Gaussian Noise 범위: ±2σ ≈ 95%
- Noise는 Episode별 샘플링

접촉쌍별 결과

| 접촉쌍       | 정적 마찰 평균 | 예상 95% 범위    |
| --------- | -------- | ------------ |
| 바닥 ↔ 물건   | 0.315    | 0.25 \~ 0.38 |
| 물건 ↔ Tool | 0.450    | 0.30 \~ 0.60 |

## 실제 질량·마찰 랜덤화 실험 환경

환경 ID: `Isaac-Inspire-Sweep-Realistic-Physics-v0`

`InspireSweepRealisticPhysicsEnvCfg`가 Baseline을 상속한다. 변경 사항은 물체 질량과 선반/물체/Hand의 마찰 및 이를 샘플링하는 Event다. Actor 41D, Action 8D, 초기 자세, 양방향 스폰, EEF/목표 높이, 보상, 게이트, 1 rad/s 종료와 PPO는 유지한다. 학습 로그 폴더만 실험 구분을 위해 분리한다.

- 질량: 매 에피소드 `Normal(0.365, 0.0675²)` kg에서 샘플링한 뒤 `[0.230, 0.500]` kg으로 clipping한다. Spawn 기본 질량은 0.365 kg이다.
- 반지름 4 cm, 높이 10 cm와 바닥 원점/COM +2 cm를 유지한다. 관성은 기존 질량 분포를 유지하면서 새 질량에 비례해 조정한다. 이전 에피소드의 관성을 다시 곱하지 않는다.
- 아래 표의 정적·동적 마찰을 매 에피소드 독립 Gaussian으로 샘플링한다. 동적/정적의 평균 비율은 약 0.8이며, 각 샘플에 고정 비율을 강제하지 않는다.
- 마찰의 ±2σ는 예상 범위이며 clipping 범위가 아니다. 음수 샘플만 0으로 제한한다.

| 접촉면 | 정적 평균 | 정적 σ | 동적 평균 | 동적 σ |
| --- | ---: | ---: | ---: | ---: |
| 선반 | 0.315 | 0.051 | 0.252 | 0.041 |
| 물체 | 0.315 | 0.040 | 0.252 | 0.032 |
| Hand | 0.585 | 0.145 | 0.468 | 0.116 |

선반은 물체를 지지하는 Shelf 충돌 형상에 적용한다. 전역 GroundPlane은 그대로 둔다. Hand는 손바닥 패드, 손가락 패드 및 Hand 링크의 충돌 형상에 적용한다. UR5e Arm 및 Axia80 하우징의 마찰은 유지한다.

Friction Combine Mode는 양쪽 모두 `average`다. 정적 접촉 마찰은 선반↔물체 평균 0.315/약 95% 범위 0.250–0.380, 물체↔Hand 평균 0.450/약 95% 범위 0.300–0.600이다. 기존 restitution, restitution combine mode 및 compliant contact 값은 보존한다.

환경마다 별도 physics material을 만들고, 이후 에피소드에서는 기존 material의 속성만 갱신한다. 따라서 연속 Gaussian 샘플링을 유지하면서 material 수가 에피소드 수에 따라 증가하지 않는다. 질량과 마찰을 정책 관측에 추가하지 않는다. 현재 적용된 값은 디버깅용 `env.unwrapped._realistic_physics`의 `mass`, `static_friction`, `dynamic_friction`에서 확인할 수 있다.

Material 생성·바인딩은 **PhysX 시작 전 `prestartup`**에서 완료한다. `startup`에서는 물리 view의 초기 질량·관성만 저장하고, 에피소드 리셋에서는 질량·관성 및 이미 만들어진 material의 마찰 값만 갱신한다. 실행 중 MaterialBindingAPI를 추가하거나 바인딩을 교체하면 인스턴스 충돌 형상의 shape 연결이 무효화될 수 있어 이 순서로 분리했다. Isaac Lab의 `prestartup` USD 수정 조건에 따라 이 환경과 이를 상속하는 환경은 `scene.replicate_physics=False`를 사용한다.

학습:

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Realistic-Physics-v0 \
  --num_envs 2048 \
  --sweep-direction both \
  --run_name realistic_mass_friction \
  --headless
```

좌측/우측 전용 Event는 기존과 같이 `--sweep-direction left`/`right`로 선택한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Realistic-Physics-v0 \
  --num_envs 1 \
  --checkpoint "<checkpoint.pt>"
```

로그: `logs/rsl_rl/inspire_sweep_realistic_physics/<timestamp>_<run_name>`.

설정: `inspire_sweep_baseline/realistic_physics_env_cfg.py`. 분포 및 reset 처리: `inspire_sweep_baseline/mdp/realistic_physics.py`. 재사용 material 바인딩: `inspire_sweep_baseline/assets/realistic_materials.py`.


## 초기 위치 Actor + 현재 상태 Critic 실험 환경

환경 ID: `Isaac-Inspire-Sweep-Realistic-Physics-Initial-Privileged-v0`

`InspireSweepRealisticInitialPrivilegedEnvCfg`가 실제 질량·마찰 랜덤화 환경을 상속한다. 목적은 정책이 물체를 계속 추적하는 관측 없이, 초기 물체 위치와 로봇/접촉 관측으로 밀기를 수행할 수 있는지 확인하는 것이다. Critic에는 현재 물체 상태를 제공해 가치 추정을 돕는다.

- `policy`의 4번째 항목, 물체 위치 3D를 **에피소드 초기 물체 위치**로 교체한다. 좌표계는 기존과 같은 로봇 root 기준이고, 물체 원점은 실린더 바닥이다.
- 리셋 후 command가 저장한 초기 상태를 사용한다. 선택된 환경만 새 위치를 캡처하며, 에피소드 중에는 물체나 robot root가 움직여도 저장한 측정값을 바꾸지 않는다.
- 기존 ±1 cm 위치 관측 오차도 리셋마다 1회 샘플링하고 유지한다. 다른 policy 관측과 노이즈 설정은 그대로 둔다.
- `privileged`에는 예제처럼 현재 물체의 위치 3D, quaternion 4D, 선속도 3D, 각속도 3D를 넣는다. 모두 로봇 root 좌표계로 표현한 13D이며 관측 노이즈를 적용하지 않는다.
- 리워드는 기존 함수대로 Scene의 현재 물체 정보를 읽는다. Actor의 고정 위치나 privileged 관측 벡터를 보상 입력으로 대체하지 않는다.
- 질량·마찰 랜덤화, 크기·COM, 스폰 방향/위치, Arm/Hand 제어, 보상·게이트·종료 및 PPO 수치 설정을 상속한다.

```python
obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
```

| 네트워크 | 입력 | 차원 |
| --- | --- | ---: |
| Actor | 초기 위치를 포함한 policy | 41 |
| Critic | policy + privileged 현재 물체 상태 | 54 |
| Action | 기존 Arm 6 + Hand 2 | 8 |

학습:

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Realistic-Physics-Initial-Privileged-v0 \
  --num_envs 2048 \
  --sweep-direction both \
  --run_name realistic_initial_privileged \
  --headless
```

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Realistic-Physics-Initial-Privileged-v0 \
  --num_envs 1 \
  --checkpoint "<checkpoint.pt>"
```

좌·우 방향 고정은 기존과 같이 `--sweep-direction left`/`right`를 사용한다. 로그는 `logs/rsl_rl/inspire_sweep_realistic_initial_privileged/<timestamp>_<run_name>`에 저장한다.

설정: `inspire_sweep_baseline/initial_privileged_env_cfg.py`. 관측 처리: `inspire_sweep_baseline/mdp/asymmetric_observations.py`. PPO 입력 경로: `inspire_sweep_baseline/agents/initial_privileged_ppo_cfg.py`.
