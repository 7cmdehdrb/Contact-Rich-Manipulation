# Inspire Sweep Baseline

[활성 환경·상속·실행 안내](../../example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md) · [개발 지침](../../example/Contact-Rich-Manipulation-Context/00_guidelines/09_DEVELOPMENT_GUIDE.md)

2026-10-18 공통 보상 변경: Omni 계열에서 사용하는 `mdp/goal_rewards.py`의 `action_rate_l2`·`joint_vel_l2`는 raw 입력의 제곱합을 계산한 뒤 `max_cost=100.0`으로 출력 상한을 적용한다. 두 함수의 반환 범위는 [0, 100]이며 음의 보상 가중치와 dt는 이후 RewardManager에서 적용한다. 입력·가중치는 바꾸지 않는다. [변경·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-18_omni_squared_cost_caps.md).

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

## Baseline에서 파생한 17채널 Tactile 환경

환경 ID: `Isaac-Inspire-Sweep-Baseline-Tactile-v0`. `InspireSweepTactileEnvCfg`가 `InspireSweepEnvCfg`를 직접 상속한다. [센서 구성·순서·검증 범위](../../example/Contact-Rich-Manipulation-Context/02_implementation/environments/TACTILE_ENVIRONMENT.md)를 참고한다.

기존 Hand의 실제 센서 영역 17개(palm 1, thumb 4, 나머지 네 손가락 각 3)를 각각 0/1로 읽는다. 일부 패드가 부모 링크의 convexHull 안쪽에 가려져 있으므로, 직접 패드 힘과 함께 실제 부모 접촉점을 원래 센서 mesh의 XY 영역에 분류한 값을 사용한다. 서로 다른 영역의 힘을 통합하거나 부모 전체의 힘을 모든 패드에 복제하지 않는다.

2026-10-15 GPU 필터 수정: 부모 접촉점 분류는 GPU에서 지원되는 `mass_cylinder`만 대상으로 한다. 선반·바닥의 미지원 GPU 필터를 제거했으며, 17개 직접 pad 센서는 unfiltered 상태를 유지한다. 따라서 직접 pad의 선반 접촉은 읽지만, 가려진 pad의 선반·바닥 접촉을 부모 접촉점으로 복원하지는 않는다. Tactile 환경의 EEF 좌표축 디버그 표시만 끄고 frame 계산과 F/T 관측은 유지한다.

원본 USD·URDF·mesh, collider 수, 형상·접촉 파라미터·질량·관성·관절·제어·보상·종료 조건은 Baseline과 같다. 새 환경의 Scene에 관측용 ContactSensor view만 추가하며 기존 환경은 1D tactile/41D 관측을 유지한다. 새 환경은 tactile 17D를 포함한 Actor/Critic `policy` **57D**, Action **8D**다. 이 57D는 발표의 OSC 설계와 별개인 Joint Position 기준선이다. 기존 41D checkpoint와 입력 크기가 다르므로 새 정책을 학습한다.

```bash
# Workspace 루트, Isaac Lab 환경에서 새 학습
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Baseline-Tactile-v0 \
  --num_envs 2048 --sweep-direction both \
  --run_name tactile17 --headless

# 새 57D checkpoint 재생
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Baseline-Tactile-v0 \
  --num_envs 1 --checkpoint "<57D_tactile_checkpoint.pt>" --real-time

# 원본 USD와 관측용 영역 metadata를 읽기 전용으로 검사
/home/min/miniconda3/envs/newton/bin/python -B \
  src/inspire_sweep_baseline/scripts/build_tactile_regions.py --check
```

기본 로그 폴더는 `logs/rsl_rl/inspire_sweep_tactile`다. 접촉 Binary의 임계값은 `observations.policy.tactile.params["threshold_n"] = 0.05` N이다. CPU·USD 검사와 GPU 접촉·성능 검증은 구분하며, 실제 GPU 응답과 속도는 아직 측정하지 않았다.


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
- 아래 표의 정적·동적 마찰 Gaussian 분포를 역할별 4096개 재질 풀로 근사한다. 에피소드마다 선반/물체/Hand가 각자 새 재질 ID를 선택한다. 동적/정적의 평균 비율은 약 0.8이며, 각 샘플에 고정 비율을 강제하지 않는다.
- 마찰의 ±2σ는 예상 범위이며 clipping 범위가 아니다. 음수 샘플만 0으로 제한한다.

| 접촉면 | 정적 평균 | 정적 σ | 동적 평균 | 동적 σ |
| --- | ---: | ---: | ---: | ---: |
| 선반 | 0.315 | 0.051 | 0.252 | 0.041 |
| 물체 | 0.315 | 0.040 | 0.252 | 0.032 |
| Hand | 0.585 | 0.145 | 0.468 | 0.116 |

선반은 물체를 지지하는 Shelf 충돌 형상에 적용한다. 전역 GroundPlane은 그대로 둔다. Hand는 손바닥 패드, 손가락 패드 및 Hand 링크의 충돌 형상에 적용한다. UR5e Arm 및 Axia80 하우징의 마찰은 유지한다.

Friction Combine Mode는 양쪽 모두 `average`다. 정적 접촉 마찰은 선반↔물체 평균 0.315/약 95% 범위 0.250–0.380, 물체↔Hand 평균 0.450/약 95% 범위 0.300–0.600이다. 기존 restitution, restitution combine mode 및 compliant contact 값은 보존한다.

생성 단계에서 선반/물체/Hand 충돌 형상에 Average 재질을 바인딩한다. 리셋에서는 PhysX Tensor API의 `set_material_properties()`로 역할별 마찰을 일괄 적용한다. 재질 풀은 Gaussian 분위값으로 평균·표준편차를 근사하며 정적/동적 값의 순서를 각각 조합한다. 풀 크기를 4096개로 제한해 PhysX의 고유 재질 수 제한을 피한다. 매 에피소드에 연속 Gaussian 값을 새로 작성하는 이전 방식과 달리, 마찰은 유한한 풀에서 재샘플링한다. 질량은 기존대로 연속 Gaussian 샘플링과 clipping을 유지한다. 질량과 마찰을 정책 관측에 추가하지 않는다. 현재 적용된 값은 디버깅용 `env.unwrapped._realistic_physics`의 `mass`, `static_friction`, `dynamic_friction`에서 확인할 수 있다.

Material 생성·바인딩은 **로봇·선반·물체 스포너에서, PhysX 시작 및 환경 복제 전에** 완료한다. `startup`에서는 물리 view의 질량·관성·마찰 버퍼와 Hand shape 인덱스를 한 번 저장한다. 리셋에서 USD 속성을 변경하거나 물성 버퍼 전체를 다시 읽지 않는다. 업데이트는 질량·관성 2회, 선반/물체/Hand 마찰 3회의 Tensor API 호출로 처리한다. `prestartup` 이벤트와 `replicate_physics=False` 설정을 제거해 Baseline의 physics replication을 사용한다. Initial-Privileged 상속 환경에도 같은 경로가 적용된다.

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

설정: `inspire_sweep_baseline/realistic_physics_env_cfg.py`. 분포 및 reset 처리: `inspire_sweep_baseline/mdp/realistic_physics.py`. 초기 material 바인딩: `inspire_sweep_baseline/assets/realistic_materials.py`. 생성 시 바인딩을 수행하는 스포너: `inspire_sweep_baseline/assets/realistic_spawners.py`.


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

## 전방향 목표 위치 실험 환경

환경 ID: `Isaac-Inspire-Sweep-Omni-v0`

Scene, Action, Event, Observation, Reward, Termination과 각 파라미터의 의미는 [Omni 환경 상세 구성](../../example/Contact-Rich-Manipulation-Context/02_implementation/environments/OMNI_ENVIRONMENT.md)에 정리되어 있다.

`InspireSweepOmniEnvCfg`가 `InspireSweepEnvCfg`를 직접 상속한다. 로봇·선반·실린더 형상, Initial Joint, Arm 6 + Hand 2 제어, 현재 물체 위치를 사용하는 policy 41D 관측과 노이즈, 50 Hz 제어 및 10초 제한은 Baseline을 사용한다. `InspireSweepPPORunnerCfg`도 동일 클래스를 그대로 등록한다. 질량·마찰은 `apply_realistic_physics()`와 `RealisticPhysicsEventsCfg`의 startup/reset 이벤트를 재사용하므로 Realistic Physics 환경과 동일하다.

### 스폰과 목표

- 물체 바닥 원점을 선반 상판 정중앙 `(-0.7, 0, 1.05)` 기준 **R=2 cm 원판**에서 면적에 대해 균일하게 샘플링한다. 현재 좌표계의 Z가 높이이며, 수평 평면은 XY다. 실린더 yaw는 무작위이고 높이는 상판에 고정한다.
- 목표는 초기 물체 위치에서 **무작위 수평 방위로 15 cm** 이동한 점이다. 방위는 `[−π, π)` 전체에서 균일 샘플링한다. 에피소드 동안 목표와 초기 물체 위치는 고정된다.
- 같은 선반 USD의 중간 상판 영역은 X `[-0.9, -0.5]`, Y `[-0.5, 0.5]`다. 목표에서는 실린더 반지름 4 cm에 여유 2 cm를 더해 가장자리와 간격을 확보한다. 목표가 이 안전 영역을 벗어나면 **방위와 스폰을 유지하고 목표 이동 거리만 줄인다**. 해당 값은 `events.object_spawn.params`에서 조정할 수 있다.

### 보상과 종료

[Omni 보상 제안 초안](../../example/Contact-Rich-Manipulation-Context/01_plan/environments/OMNI_REWARD_PLAN.md)의 식을 적용하며 다음 수치는 확인한 초기 시험값이다. 목표 비용에는 접근 게이트를 두지 않는다. 기존 EEF 방향 정렬, Y 속도 보상, Homing 보상과 단일 물체에서 0인 비대상 물체 보상을 제거한다.

| 항목 | 초기 가중치 | 계산 |
| --- | ---: | --- |
| 목표 평면 거리 | +6.0 | `exp(-d / 0.10) - 1` |
| 접근 진척 | +1.0 | 고정 `ee_frame`에서 초기 물체 바닥 원점까지의 새 최단 거리 진척. 근접 반경 10 cm, 에피소드 합계 최대 +1 |
| Arm Action Rate | −0.03 | Arm 6개 명령 변화량의 제곱합. Baseline의 Arm 비용과 동일 |
| Hand Action Rate | −0.03 | Hand 2개 raw 명령 변화량의 제곱합. 보상 내부 clamp 없음 |
| Arm Joint Velocity | −0.03 | 이름으로 선택한 Arm 6개 실제 관절 속도의 제곱합. Baseline과 동일 |
| Hand Joint Velocity | −0.01 | Hand 12개 실제 관절 속도의 제곱합 |
| 물체 기울기 | −2.0 | 상판 법선과 물체 위쪽 축 사이 각도. 5° 이하 비용 0, 0.9 rad까지 정규화한 제곱 비용 |
| 선반 페널티 | −0.5 | 기존 `ShelfCollisionPenalty` 계산 유지 |
| 시간 비용 | −0.1 | 초당 비용 |
| 성공 / 실패 이벤트 | +30 / −80 | 종료 전이에 한 번 지급. 여러 실패가 동시에 발생해도 −80 한 번 |

목표 오차 **1 cm 미만**, 물체 평면 속도 **2 cm/s 미만**, 기울기 **10° 이하**, 상판 지지 상태와 실패 없음이 **0.3초** 연속 유지되면 성공 보상을 지급하고 종료한다. 지지 상태는 pose 기반 기하학적 추정으로, 바닥 최저점과 상판 높이 오차가 1 cm 이내이고 물체 원점이 반지름 여유를 적용한 상판 영역 안에 있는지 검사한다. 실제 접촉력을 검사하거나 기울어진 실린더 전체의 투영 영역을 계산하지 않는다. 손 자세와 Homing은 성공 조건에 포함하지 않는다.

실패는 기존 Arm >1 rad/s, 물체 속도 norm >0.3 m/s, 물체 바닥 높이 <1.04 m 및 기존 선반 속도·finger/wrist 높이 검사를 유지한다. roll/pitch 검사는 상판 법선 기준 기울기 **≥0.9 rad**로 교체한다. 물체 바닥 원점의 수평 위치가 상판 영역을 벗어나거나 선반 위치가 초기 위치에서 **5 cm 초과** 이동해도 종료한다. 실패가 발생하면 성공 유지 시간을 초기화하고 성공 보상을 지급하지 않는다. 시간 제한만 도달한 경우에는 실패 이벤트 페널티를 지급하지 않는다.

Isaac Lab `RewardManager`가 모든 항에 `step_dt`를 곱하므로 접근 진척과 성공·실패 함수는 내부에서 이 곱셈을 상쇄한다. 나머지 항은 계획서의 `Δt / T_ref` (`T_ref=1초`)로 누적된다. 접근 기록과 성공 유지 카운터는 리셋된 환경만 초기화한다. PPO의 `gamma=0.98`도 그대로 유지하며, 보상 가중치와 종료 임계값은 새 Cfg에서 조정한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-v0 \
  --num_envs 2048 \
  --experiment_name inspire_sweep_omni \
  --run_name centered_spawn_goal15cm \
  --headless
```

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-v0 \
  --num_envs 1 \
  --checkpoint "<checkpoint.pt>"
```

새 환경은 항상 전방향 목표를 생성하므로 `--sweep-direction`을 사용하지 않는다. PPO 클래스의 기본 로그 폴더는 Baseline과 같으며, 위 학습 명령처럼 `--experiment_name`으로 구분할 수 있다.

설정: `inspire_sweep_baseline/omni_env_cfg.py`. 스폰·목표: `mdp/goal_events.py`. 보상: `mdp/goal_rewards.py`. 종료: `mdp/goal_terminations.py`.

### Omni v1: Arm 속도 종료 기준 변형

환경 ID: `Isaac-Inspire-Sweep-Omni-v1`. `InspireSweepOmniV1EnvCfg`는 `InspireSweepOmniEnvCfg`를 상속하며 `terminations.hand_velocity.params["threshold"]`를 변경한다. 현재 코드 값은 3.0 rad/s다. 이름이 `hand_velocity`인 종료 항목은 Arm 6개 관절을 검사한다. 절대 속도 3.0 rad/s는 허용하고, 초과하면 종료한다. v0는 1.0 rad/s를 유지한다.

관측 41D, Action 8D, 스폰·목표·물성·보상·나머지 종료 조건과 PPO는 v0에서 상속한다. Arm 속도 보상은 실제 관절 속도의 제곱합이며 actuator 속도 제한도 그대로다. v0 체크포인트를 지정해 v1에서 재생할 수 있지만, 변경한 종료 조건에서 성공하거나 첫 스텝 종료가 해결됐다는 뜻은 아니다.

```bash
# v1 새 학습
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-v1 --num_envs 2048 \
  --experiment_name inspire_sweep_omni_v1 \
  --run_name centered_spawn_goal15cm_arm3rad --headless

# v0 체크포인트를 v1 조건으로 재생
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-v1 --num_envs 1 \
  --checkpoint /home/min/7cmdehdrb/grad/logs/TEMP/2026-10-14_16-16-28_centered_spawn_goal15cm/model_4100.pt \
  --real-time
```

### Omni v2: 현재 물체 접근과 목표 이동 진전

환경 ID: `Isaac-Inspire-Sweep-Omni-v2`. `InspireSweepOmniV2EnvCfg`는 v1을 상속한다. 기존 v0/v1의 동작은 유지한다. [활성 환경 안내](../../example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md)와 [개발·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_omni_v2_rewards.md)에 계산식과 검증 범위를 기록했다.


| 보상 | v2 계산과 기본값 |
| --- | --- |
| `reaching` | 현재 물체 바닥 중심과 EEF의 XYZ 거리에서 10cm를 뺀 거리가 episode 최소 기록을 갱신한 만큼 지급. Reset 거리로 정규화하고 누적 최대 +30 |
| `goal_distance` | 물체의 XY 목표 거리를 새롭게 줄인 비율만 지급. Reset 시 목표 거리에서 허용 오차 1cm를 뺀 값으로 정규화하고 누적 최대 +40 |
| Arm action rate / joint velocity | 각각 −0.03, 기존과 같은 sum 계산 |
| Hand action rate / joint velocity | 각각 −0.03 / −0.01, raw Action 변화량 / 실제 관절 속도의 sum 계산 |
| `object_tilt` / `shelf_collision` | v1과 같은 −2 / −0.5 |
| `time` / `success` / `failure` | 보상 설정을 `None`으로 제거 |

두 진전 보상은 함수 내부에서 `step_dt`로 나눠 RewardManager의 시간 곱셈을 상쇄한다. 접근 전에는 Reaching이 주요 양의 신호이고, 접근 기록을 소진한 뒤에는 물체의 목표 접근이 추가 task 보상을 만든다. 가만히 기다리기, 같은 거리에서 함께 이동하기, 후퇴 후 기존 최고 기록으로 복귀하기에는 추가 진전 보상이 없다. 목표 보상은 v1의 음의 거리 비용과 의미가 다르므로 TensorBoard 값을 직접 비교하지 않는다.

이 계수는 학습 성능을 검증하기 전의 실험 기본값이다. Reaching은 상대 거리만 검사하므로 물체가 EEF 쪽으로 움직여도 지급될 수 있고, 목표 이동 보상만으로 물체의 직립·접촉을 보장하지 않는다. 직립 비용은 유지한다. v2에서는 `terminations.success=None`으로 성공 종료도 제거하므로 목표 도달 후에도 episode가 계속된다. 시간 제한 10초, 실패 종료와 Arm 속도 임계값 3 rad/s는 v1과 같다. PPO·관측·action·물성은 변경하지 않는다.

```bash
# v2 새 학습: v1 체크포인트 resume 없이 보상 변경 효과 확인
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-v2 --num_envs 2048 \
  --experiment_name inspire_sweep_omni_v2 \
  --run_name live_reach30_sweep40 --headless

# 생성된 v2 체크포인트 재생
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-v2 --num_envs 1 \
  --experiment_name inspire_sweep_omni_v2 \
  --checkpoint "<v2-checkpoint.pt>" --real-time
```

전방향 목표를 사용하므로 `--sweep-direction`은 지정하지 않는다. Runner 기본 클래스는 기존과 같으므로 위 `--experiment_name`으로 v2 로그와 체크포인트를 구분한다.

### Omni v3: 높이 조건을 만족하는 XY 접근

`Isaac-Inspire-Sweep-Omni-v3`의 `InspireSweepOmniV3EnvCfg`는 현재 v2를 상속하고 Reaching의 함수와 높이 파라미터만 변경한다. 이번 작업 시 코드의 가중치는 Reaching 30, Goal 40이며 그대로 상속한다. 다른 보상·종료·관측·action·PPO 설정은 v2와 같다.

- 목표 높이: **현재 물체 바닥 Z + 0.09m**. `example/Sweep-Policy/sweeping_policy/mdp/reward_random_sweep.py`의 기본 Reaching 목표 높이를 따른다.
- 지급 조건: EEF의 목표 높이 오차가 **±0.02m 이내**이면서, 현재 물체와의 XY 거리가 episode 최소 기록을 갱신할 때. 2cm는 v3에 추가한 실험 허용 오차다.
- 거리: `max(XY 거리 − 0.10m, 0)`. Reset 시 같은 XY 거리로 정규화하며 기존의 일회성 진전 보상 구조와 `step_dt` 상쇄를 유지한다.
- 높이 조건 밖에서도 XY 최소 기록은 갱신한다. 따라서 낮은 높이에서 접근한 뒤 Z만 조정해 누적 진전을 보상받거나, 높이 범위 출입으로 반복 보상받을 수 없다. 높이 밖에서 소진한 접근 진전은 다시 지급하지 않는다.

참고 코드의 원래 Reaching은 Z 오프셋을 둔 3D 지수 보상이며, 위 hard gate와 2cm 허용 오차는 이번 요청에 맞춰 추가했다. EEF 높이 조건이 손 전체의 선반 충돌 방지를 보장하는 것은 아니며, 실제 학습·재생 검증은 별도다. [v3 개발 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_omni_v3_height_gated_reaching.md).

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-v3 --num_envs 2048 \
  --experiment_name inspire_sweep_omni_v3 \
  --run_name height_gated_xy_reach --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-v3 --num_envs 1 \
  --experiment_name inspire_sweep_omni_v3 \
  --checkpoint "<v3-checkpoint.pt>" --real-time
```

### Dex Omni: 과거 최고 상태를 갱신한 거리만큼 보상

환경 ID: `Isaac-Inspire-Dex-Omni-v0`. [InspireDexOmniEnvCfg](inspire_sweep_baseline/dex_omni_env_cfg.py)는 `InspireSweepOmniV2EnvCfg`를 상속하고, `DexOmniRewardsCfg(OmniV2RewardsCfg)`에서 접근·목표 이동 보상 두 항만 교체한다. Runtime은 `InspireSweepEnv`, PPO는 `InspireSweepPPORunnerCfg`를 그대로 사용한다.

DexTouch 논문의 IV-B, 식 (1)과 식 (2)의 목표 거리 항(PDF 4쪽, 인쇄 페이지 10775)에서 **에피소드의 과거 최고 상태를 갱신할 때 그 개선량만 지급하는 구조**를 차용했다. 논문의 손가락별 접근 거리 대신 기존 고정 EEF 프레임과 현재 물체 바닥 원점 사이 거리를 사용하고, 목표 거리는 Sweeping의 XY 평면에서 계산한다. 논문의 grasp·lift·rotation 단계나 L1 관절 속도 비용은 추가하지 않는다. 출처와 구현·검증 범위는 [개발 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_dex_omni_rewards.md)에 기록한다.

| 보상 | 남은 거리 `gap` | 계수 |
| --- | --- | ---: |
| `reaching` / `DexReachProgress` | `max(EEF–현재 물체 바닥 원점 XYZ 거리 − 0.10 m, 0)` | +30 reward/m |
| `goal_distance` / `DexPlanarGoalProgress` | `max(물체–목표 XY 거리 − 0.01 m, 0)` | +300 reward/m |

각 term의 보상 기준은 리셋 상태의 `gap`이다. Step마다 `max(직전 최소 gap − 현재 gap, 0)`만큼 개선량을 구하고 최소 기록을 갱신한다. 함수가 이 거리 차를 `step_dt`로 나누므로 RewardManager의 시간 곱셈 후 실제 지급값은 **계수 × 새롭게 줄인 거리(m)**다. 대기·후퇴·기존 최소 거리로의 복귀는 추가 보상이 없고, 목표 허용 오차 안에서도 추가 보상은 없다. 선택된 환경의 리셋마다 기록을 초기화한다.

Omni v2도 과거 최고 기록을 사용하지만, 초기 거리로 정규화해 +30/+40의 에피소드 보상 예산을 갖는다. Dex Omni는 초기 거리로 나누지 않으므로 초기 거리에 관계없이 새로 1 cm를 줄일 때 접근 +0.3, 목표 이동 +3을 지급한다. **30/300은 논문에 제시된 계수가 아니라 이 환경의 실험 기본값**이며 학습 성능은 아직 검증하지 않았다.

Arm action rate/velocity 비용은 각각 −0.03의 제곱합, Hand action rate/velocity 비용은 각각 −0.03/−0.01의 제곱평균으로 유지한다. 물체 기울기 −2, 선반 충돌 −0.5 비용도 상속한다. `time`, `success`, `failure` 보상과 `terminations.success`는 v2와 같이 `None`이다. 선반 이동·충돌을 포함한 모든 실패 종료, Arm 속도 3 rad/s 종료, 10초 시간 제한도 v2와 같다.

Scene·스폰·목표 이벤트를 상속하므로 물체는 기존 반지름 4 cm·높이 10 cm의 단일 실린더이며, 선반 중앙 반지름 2 cm 원판 스폰과 최대 15 cm 전방향 목표를 사용한다. 질량·마찰 랜덤화, 현재 물체 위치를 포함한 Actor/Critic 41D, Action 8D, 제어 50 Hz도 동일하다. 접근 보상은 상대 거리만 보므로 물체가 EEF 쪽으로 움직여도 지급될 수 있으며, 보상 자체가 접촉이나 직립을 증명하지는 않는다.

```bash
# Workspace 루트에서 Dex Omni 학습
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Omni-v0 --num_envs 2048 \
  --experiment_name inspire_dex_omni \
  --run_name dex_best_distance --headless

# 생성된 Dex Omni 체크포인트 재생
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Dex-Omni-v0 --num_envs 1 \
  --experiment_name inspire_dex_omni \
  --checkpoint "<dex-omni-checkpoint.pt>" --real-time
```

`--sweep-direction`은 생략한다. 기본 PPO 로그 이름은 Baseline과 같으므로 위 `--experiment_name`을 사용하면 `logs/rsl_rl/inspire_dex_omni`에 저장한다. 위 명령은 실행 안내이며, 이번 구현의 실제 train/play 수행이나 학습 성공 결과를 뜻하지 않는다.

Omni 베이스와 모든 파생 환경(Dex Omni 포함)은 로컬 `mdp/goal_rewards.py`의 `action_rate_l2`·`joint_vel_l2` 네 항을 반드시 상속한다. raw Action 변화량과 이름으로 선택한 실제 관절 속도를 제곱합하며, 보상 내부 clamp·평균·정규화는 사용하지 않는다. 계수와 필수 사용 규칙은 [개발 지침](../../example/Contact-Rich-Manipulation-Context/00_guidelines/09_DEVELOPMENT_GUIDE.md)을 따른다.

## Omni v4: 종료 기준 완화와 pushing 속도 비용

`Isaac-Inspire-Sweep-Omni-v4`는 `InspireSweepOmniV4EnvCfg(InspireSweepOmniV3EnvCfg)`로 등록한다. Runtime은 `InspireSweepEnv`, PPO는 `InspireSweepPPORunnerCfg`다. V3의 41D Actor/Critic 관측, 8D Joint Position/Hand action, 높이 조건부 XY reaching(+30), 목표 진행(+40), 물체·목표 스폰, 질량·마찰 및 0.02 s 정책 주기를 상속한다.

| 설정 | V4 값 |
| --- | --- |
| `hand_velocity` | Arm 6개 중 절대 관절속도 **5.0 rad/s 초과** 시 종료. 이름과 달리 Arm 대상 |
| `object_outside_shelf` | `None`, 비활성 |
| `success` | `None`, 비활성 |
| `push_fast` | 대상 물체 3D 선속도 norm **0.5 m/s 초과** 시 종료 |
| `rewards.pushing_speed` | `goal_rewards.pushing_speed_cost`, weight **−0.05**, `free_speed=0.3`, `reference_speed=0.5` |

`object_drop`, 선반 충돌·선반 이동·timeout 종료는 유지한다. 선반 외곽 종료를 제거해도 물체 낙하·전도에 대한 종료는 남는다. 이벤트의 안전 목표 영역과 스폰은 그대로다. Arm/Hand action 변화와 실제 관절속도 네 비용은 기존 함수·계수(−0.03/−0.03/−0.03/−0.01)를 유지한다.

2026-10-15 후속 요청으로 `2026-10-15_16-53-58_height_gated_xy_reach/params/env.yaml`의 공통 파라미터를 대조해 Arm 속도 threshold **5.0**을 코드에 적용했다. 기존 관측·action·actuator·접근/목표 보상·물성 설정은 저장 run과 같아 상속을 유지한다. V4 전용 물체 속도 0.5·외곽/성공 종료 비활성·속도 비용은 유지한다. [적용·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_omni_v4_saved_parameters.md).

물체 속도 `v`는 `push_fast`와 같은 월드 선속도 **3D norm(m/s)**이다. 새 비용 함수는 `max(v−0.3, 0)² / (0.5−0.3)²`를 반환한다. 가중치 적용 후 값은 다음과 같다.

```math
c(v)=-0.05\left(\frac{\max(v-0.3,0)}{0.2}\right)^2.
```

0.3 m/s 이하는 비용이 없고 초과 구간에서는 제곱으로 증가한다. 함수는 상한 clamp를 두지 않으므로 종료 threshold를 넘는 transition도 실제 속도의 비용을 받는다. 요청의 −0.05는 **weight 적용 후, RewardManager의 시간 곱셈 전**으로 해석했다. 기존 dense 비용과 같이 `/step_dt`로 상쇄하지 않아 실제 step 기여는 `c(v) × step_dt`다.

| 속도(m/s) | 가중치 적용 후 `c(v)` | 기본 0.02 s step 기여 |
| --- | --- | --- |
| 0.30 이하 | 0 | 0 |
| 0.35 | −0.003125 | −0.0000625 |
| 0.40 | −0.0125 | −0.00025 |
| 0.45 | −0.028125 | −0.0005625 |
| 0.50 | **−0.05** | **−0.001** |

Workspace 루트에서 실행한다. Omni에는 `--sweep-direction`을 지정하지 않는다. 다음은 실행 안내이며 이번 구현에서 train/play를 수행한 것은 아니다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-v4 --num_envs 2048 \
  --experiment_name inspire_sweep_omni_v4 \
  --run_name quadratic_push_speed --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-v4 --num_envs 1 \
  --experiment_name inspire_sweep_omni_v4 \
  --checkpoint "<v4-checkpoint.pt>" --real-time
```

상속·등록·CPU 검증과 미실행 범위는 [V4 개발 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_omni_v4_speed_penalty.md)에 기록했다.

### 추가 Arm 속도 페널티: `arm_speed_excess`

V4는 기존 `joint_vel_l2` 등 모든 비용을 유지하면서 `goal_rewards.joint_speed_cost`를 추가한다. `terminations.hand_velocity`와 같은 Arm 6개 관절의 실제 절대속도를 검사하며, 어느 관절이든 빠르면 비용이 발생하도록 **최대 절대속도**를 사용한다. Hand 관절은 검사 대상이 아니다.

```math
v_{\max}=\max_{j\in\mathrm{Arm}}|\dot{q}_j|,\qquad c=-0.03\left(\frac{\max(v_{\max}-1.0,0)}{2.0-1.0}\right)^2.
```

설정은 `rewards.arm_speed_excess`, weight **−0.03**, `free_speed=1.0`, `reference_speed=2.0`, `asset_cfg=_arm()`이다. 1.0 rad/s 이하는 비용 0이며 초과 구간에서 제곱으로 증가한다. 음수 속도도 절댓값으로 처리하고, 2.0 이후에도 상한 없이 증가한다.

| Arm 최대 절대속도(rad/s) | 가중치 적용 값 | 기본 0.02 s step 기여 |
| --- | --- | --- |
| 1.0 이하 | 0 | 0 |
| 1.5 | −0.0075 | −0.00015 |
| 2.0 | **−0.03** | **−0.0006** |
| 3.0 | −0.12 | −0.0024 |
| 5.0 | −0.48 | −0.0096 |

`pushing_speed_cost`와 같이 dense 비용이므로 RewardManager의 시간 곱셈을 상쇄하지 않는다. 위 −0.03은 **weight 적용 후, 시간 곱셈 전**이다. 기존 `arm_joint_velocity`는 Arm 속도 전체의 제곱합에 비용을 주고, 이 새 항은 가장 빠른 관절의 1.0 초과 구간에 추가 비용을 준다. 두 항은 함께 적용된다. 종료 기준은 기존 **5.0 rad/s 초과**로 유지하며 이 새 항 자체는 종료시키지 않는다.

[추가·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_omni_v4_arm_speed_excess.md).

## Omni Renewal 환경

Renewal은 Baseline 위에 장면·보상·종료·이벤트를 구성하고 현재 지수 속도 비용을 사용한다. V4와 환경 상속 체인은 별개다.

| Gym ID | 환경 설정 | 보상 설정 |
| --- | --- | --- |
| `Isaac-Inspire-Sweep-Omni-Renewal-v0` | `InspireSweepOmniRenewalEnvCfg` | `OmniRenewalRewardsCfg` |
| `Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0` | `InspireDexSweepOmniRenewalEnvCfg` | `DexOmniRenewalRewardsCfg` |

환경 상속은 `InspireSweepEnvCfg → InspireSweepOmniRenewalEnvCfg → InspireDexSweepOmniRenewalEnvCfg`다. Renewal은 Baseline 장면·관측·action·command·주기 위에 `OmniRenewalEventsCfg`, `OmniRenewalRewardsCfg`, `OmniRenewalTerminationsCfg`를 구성하며, V1→V2→V3→V4의 설정 변경 체인을 상속하지 않는다. 선반 기하 상수와 기존 로컬 함수, `RealisticPhysicsEventsCfg` 및 `apply_realistic_physics`를 재사용한다. Dex Renewal은 Renewal을 직접 상속하며 Dex 접근·목표 보상과 공통 지수 속도 비용 두 항을 사용한다. 보상 Cfg 상속은 `OmniRenewalRewardsCfg → DexOmniRenewalRewardsCfg`다.

두 환경의 runtime은 `InspireSweepEnv`, PPO는 `InspireSweepPPORunnerCfg`다. Actor/Critic은 현재 물체 GT를 포함하는 같은 41D, action은 Arm Joint Position 6D + Hand 2D, 정책 주기는 0.02 s, episode 제한은 10 s다. 물체·목표 스폰과 질량·마찰, 센서·초기 상태는 현재 V4와 동일하다.

| 보상·종료 | Sweep Renewal | Dex Sweep Renewal |
| --- | --- | --- |
| `reaching` | `HeightGatedPlanarReachProgress`, +30; XY·초기 거리 정규화·Z +9 cm / ±2 cm gate | `DexReachProgress`, +30 reward/m; 3D 실제 기록 개선 거리, 높이 gate 없음 |
| `goal_distance` | `PlanarGoalProgress`, +40; 초기 목표 거리 정규화 | `DexPlanarGoalProgress`, +300 reward/m; 실제 XY 기록 개선 거리 |
| 네 필수 운동 비용 | Arm/Hand raw action 변화 제곱합 −0.03/−0.03, 실제 관절속도 제곱합 −0.03/−0.01 | 동일 |
| `object_tilt`, `shelf_collision` | 기존 함수·계수 −2.0 / −0.5 | 동일 |
| `pushing_speed` | free_speed 0.3의 지수 비용, weight −1.0 | 동일 |
| `arm_speed_excess` | 최대 Arm 절대속도, free_speed 1.0의 지수 비용, weight −1.0 | 동일 |
| Arm 속도 종료 | 최대 절대속도 5.0 rad/s 초과 | 동일 |
| 물체 속도 종료 | 3D 선속도 0.8 m/s 초과 | 동일 |
| 외곽·성공 종료 | 비활성 | 동일 |
| 기타 종료 | 물체 낙하·전도, 선반 충돌·변위, timeout 유지 | 동일 |

두 환경은 활성 reward 10개를 사용한다. 시간 비용·성공/실패 terminal reward는 비활성이다. Dense 비용에는 RewardManager가 `weight × step_dt`를 적용하며, 진행 보상 함수는 기존 구현대로 시간 곱셈을 상쇄한다. Dex Renewal의 접근·목표 두 항은 `DexOmniRewardsCfg`의 함수·가중치·params와 동일하다. 이후 부모의 `__post_init__`가 이 두 항을 높이 gate 보상으로 다시 바꾸지 않는 구조다.

Workspace 루트에서 실행한다. `--sweep-direction`은 생략한다. 기본 PPO 로그 폴더를 공유하므로 명령의 `--experiment_name`으로 구분한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-Renewal-v0 --num_envs 2048 \
  --experiment_name inspire_sweep_omni_renewal --run_name renewal_v0 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-Renewal-v0 --num_envs 1 \
  --experiment_name inspire_sweep_omni_renewal \
  --checkpoint "/home/min/7cmdehdrb/grad/2026-10-15_22-04-03_renewal_v0/model_24000.pt" --real-time

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0 --num_envs 2048 \
  --experiment_name inspire_dex_sweep_omni_renewal --run_name dex_renewal_v0 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0 --num_envs 1 \
  --experiment_name inspire_dex_sweep_omni_renewal \
  --checkpoint "/home/min/7cmdehdrb/grad/logs/TEMP2/2026-10-15_22-04-26_dex_renewal_v0/model_30600.pt" --real-time
```

위 명령은 실행 안내이며 이번 추가 작업에서 새 학습·재생을 수행하지 않았다.

코드: [Renewal 설정](inspire_sweep_baseline/omni_renewal_env_cfg.py), [Dex Renewal 설정](inspire_sweep_baseline/dex_omni_renewal_env_cfg.py). [구현·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-15_omni_renewal_environments.md).

### Reaching 이탈 페널티 정정 (2026-10-15)

`HeightGatedPlanarReachProgress`는 현재 물체와 EEF의 XY 거리에서 10cm를 뺀 비음수 거리를 이전 episode 최소 기록과 비교한다. 새 기록에는 높이 gate 안에서 보상, 기록보다 먼 상태에는 높이와 관계없이 매 step 페널티를 적용한다. 10cm 이내 이동과 과거 기록 복귀에는 재보상이 없다. 물체 이동으로 인한 이탈도 포함한다. 초기 남은 거리로 정규화하되 초기부터 반경 안이면 10cm로 나눈다. RewardManager의 dt 곱셈은 함수 내 나눗셈으로 상쇄된다.

물체 link 원점은 바닥 중심이며 Z +9cm가 맞다(직립한 높이 10cm 물체의 기하 중심 +4cm). 이 변경은 공통 함수를 쓰는 Omni v3/v4/Sweep Renewal에 적용되고 Dex reaching에는 적용되지 않는다.

## Renewal OSC 환경

`Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0`는 `InspireSweepOmniRenewalOSCEnvCfg(InspireSweepOmniRenewalEnvCfg)`다. Arm을 Isaac Lab `OperationalSpaceControllerActionCfg`로 교체하고 부모 `OmniRenewalRewardsCfg`의 지수 속도 비용 두 항을 상속한다. 제어점은 `scene.ee_frame`의 첫 target과 같은 body(`axia80_link`), 위치 및 quaternion offset을 사용한다.

- Arm action: relative pose 6D (`pose_rel`), fixed impedance. 위치 scale 0.05m, axis-angle 회전 scale 0.1rad. task frame 미지정으로 robot root 좌표계를 사용한다.
- OSC: 모든 6축 stiffness 200, damping ratio 1, 관성 보정 및 중력 보상 사용.
- 토크 출력과 중복되지 않도록 Arm implicit actuator stiffness/damping은 0. 노력·속도 제한은 부모 설정을 상속한다.
- Hand 2D와 관측 41D, 속도 비용 두 항 외 reward, termination/event, 물성, reset, PPO는 부모와 같다. raw Arm action 페널티는 Cartesian 6D 입력 변화량에 적용된다.

Workspace 루트에서 실행:

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0 --num_envs 2048 \
  --experiment_name inspire_sweep_omni_renewal_osc --run_name renewal_osc_v0 --headless


./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Sweep-Omni-Renewal-OSC-v0 --num_envs 1 \
  --experiment_name inspire_sweep_omni_renewal_osc \
  --checkpoint "/home/min/7cmdehdrb/grad/logs/rsl_rl/inspire_sweep_omni_renewal_osc/2026-10-15_22-12-35_renewal_osc_v0/model_6300.pt" --real-time
```

Joint Position 정책과 action 의미가 달라 새 정책으로 학습한다. 현재 지수 비용 적용과 상속을 CPU에서 검증했으며 실제 실행 및 장시간 학습 검증 범위는 최신 개발 로그를 따른다.

### 재생 배속

`play.py --real-time --playback-speed 0.25`는 물리 dt와 제어 주기를 유지하고 step 사이 대기만 늘려 1/4 속도로 재생한다. `--playback-speed` 기본값은 1.0이며 양의 유한수만 허용한다. `--real-time` 없이 실행하면 기존처럼 최대 처리 속도로 실행한다. 배속은 실시간 화면 pacing에 적용되며 녹화 영상 FPS를 변경하지 않는다.

### Renewal 계열 선반 종료와 목표 시각화 (2026-10-16)

`OmniRenewalTerminationsCfg`에서 `terminations.shelf_collision.params["check_fingers"] = False`를 설정한다. Renewal과 OSC·Dex Renewal을 포함한 모든 파생 환경에 적용된다. 선반 root 6D 속도 norm > 0.1 또는 손목 frame Z가 선반 root Z +1.05m 기준 7cm 미만이면 종료한다. 손가락 frame 높이는 종료에 사용하지 않는다. Renewal 계열 밖의 환경은 기본값 True로 기존 판정을 유지하며 선반 보상 항은 변경하지 않았다.

Renewal 부모의 `commands.target_goal_pos.debug_vis=True`를 모든 파생 환경이 상속하여 현재 목표 월드 위치에 반지름 2.5cm 초록색 구형 마커를 표시한다. reset 이후 command 위치를 따라 갱신되며 충돌용 형상이 아니다. 현재 사용자 설정인 OSC 위치/회전 scale 1.0/1.0은 유지했다.

### Renewal 17채널 Tactile 환경 (2026-10-16)

각 부모를 직접 상속하며 Scene의 센서 view와 tactile 관측을 교체하고 공통 지수 속도 비용 두 항을 사용한다. Arm/Hand 제어, 물성·중력, 다른 보상·종료·이벤트, 나머지 관측과 노이즈는 부모 설정을 유지한다.

| 새 Gym task ID | 부모 Cfg | 새 Cfg | 기본 실험 폴더 |
| --- | --- | --- | --- |
| `Isaac-Inspire-Tactile-Sweep-Omni-Renewal-v0` | `InspireSweepOmniRenewalEnvCfg` | `InspireTactileSweepOmniRenewalEnvCfg` | `logs/rsl_rl/inspire_tactile_sweep_omni_renewal` |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v0` | `InspireDexSweepOmniRenewalEnvCfg` | `InspireDexTactileSweepOmniRenewalEnvCfg` | `logs/rsl_rl/inspire_dex_tactile_sweep_omni_renewal` |

Standalone `run_probe_test.py --pad-layout original`과 같은 원본 Hand 패드와 `ProjectedTactileContactBits`를 사용한다. 직접 패드 접촉의 NetForce 크기 또는 센서 영역에 매핑한 부모 링크 접촉력의 벡터합 크기가 **0.01 N 이상**이면 `1.0`, 아니면 `0.0`이다. 두 경로의 판정은 OR로 결합한다. 센서 간 힘을 합산하지 않으며, reset 직후 아직 갱신되지 않은 행은 0이다. 직접 경로는 모든 충돌 상대, 부모 영역 투영은 기존 GPU 지원 방식에 따라 `mass_cylinder` 접촉을 읽는다.

채널 순서는 손바닥 1개 → 엄지 1–4 → 검지·중지·약지·소지 각각 1–3으로 **17개**다. 기존 policy의 촉각 위치에 1D 대신 17D를 넣으므로 Actor/Critic 관측은 `34 + 17 + 6 = 57D`, Action은 **8D**다. 기존 41D 정책은 입력 크기가 다르므로 새 정책으로 학습한다. 패드를 옮기는 `exposed` 검증 배치나 standalone의 무중력·고정 자세는 적용하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Tactile-Sweep-Omni-Renewal-v0 \
  --num_envs 2048 --run_name tactile_renewal_v0 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v0 \
  --num_envs 2048 --run_name dex_tactile_renewal_v0 --headless
```

PPO는 부모와 같고 기본 실험 폴더 이름만 분리한다. 두 task 모두 GPU 8개 환경에서 1 iteration 학습을 완료해 57D 입력과 8D 출력을 확인했다. 장시간 학습 성능과 2048개 환경 처리 속도는 미검증이다.

[환경 설정](inspire_sweep_baseline/tactile_omni_renewal_env_cfg.py) · [PPO 설정](inspire_sweep_baseline/agents/tactile_omni_renewal_ppo_cfg.py) · [구현·검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-16_tactile_renewal_environments.md).

### Renewal 네 환경의 공통 지수 속도 페널티

현재 아래 네 환경 모두 `pushing_speed`, `arm_speed_excess`에 Renewal 부모의 지수 비용을 적용한다. OSC를 포함한 Renewal 계열의 해당 두 비용은 exp다. Renewal 밖의 Omni v4는 기존 제곱 속도 비용을 유지한다.

| Gym task ID | 적용 |
| --- | --- |
| `Isaac-Inspire-Sweep-Omni-Renewal-v0` | Renewal 기본 지수 rewards |
| `Isaac-Inspire-Dex-Sweep-Omni-Renewal-v0` | Dex rewards가 공통 지수 rewards 상속 |
| `Isaac-Inspire-Tactile-Sweep-Omni-Renewal-v0` | Renewal 부모에서 상속 |
| `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v0` | Dex 부모에서 상속 |

두 양의 비용 함수는 `expm1(2 * max(x - t, 0)) / 2`를 반환하며 weight는 모두 -1.0이다. 물체 항은 대상 물체의 3D 선속도 norm과 임계값 0.3m/s, Arm 항은 6개 Arm 관절의 최대 절대 속도와 임계값 1.0rad/s를 사용한다. 임계값 이하에서 0이고 초과하면 지수적으로 감소한다. reference_speed 정규화나 상한은 없으며 RewardManager의 step_dt 곱셈을 유지한다.

당시 두 Tactile v1 환경은 중복이므로 등록·Cfg·전용 PPO를 삭제했다. 이후 아래 변경을 갖는 Dex Tactile v1을 새로 추가했다. 위의 v0 실행 명령을 사용한다. 촉각 관측 57D, 행동 8D, 센서·제어·물성·이벤트·나머지 보상과 종료는 유지한다. 종료 기준은 물체 속도 >0.8m/s, Arm 관절 절대 속도 >5.0rad/s다. 기존 제곱 함수는 보존하지만 일반 Sweep Renewal v0도 지수 속도 비용을 사용한다.

[촉각 v0 설정](inspire_sweep_baseline/tactile_omni_renewal_env_cfg.py) · [공통 rewards](inspire_sweep_baseline/omni_renewal_env_cfg.py) · [새 비용 함수](inspire_sweep_baseline/mdp/goal_rewards.py) · [정리·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-16_tactile_v0_exponential_cleanup.md).

[Renewal 계열 속도 초과 비용의 exp 통일·검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-16_renewal_only_exponential.md). `joint_vel_l2` 등 별도 운동 제곱합 페널티는 기존 규칙대로 유지한다.


### Contact Sweep v0 — 독립 기준 환경

2026-10-20 사용자 요청으로 Renewal v3의 최종 동작을 독립된 기준 환경 **`Isaac-Inspire-Contact-Sweep-v0`**로 구성했다. 기존 v3의 클래스와 설정은 보존한다. 새 환경은 기존 Baseline/Omni/Renewal 환경 Cfg를 import하거나 상속하지 않는다.

[`contact_sweep/`](inspire_sweep_baseline/contact_sweep)에서 Manager별 클래스를 직접 정의한다.

| 파일 | 클래스·책임 |
| --- | --- |
| `env.py` | `ContactSweepEnv(ManagerBasedRLEnv)`: 센서 freshness·reset Action 이력 |
| `env_cfg.py` | `ContactSweepEnvCfg(ManagerBasedRLEnvCfg)`: Manager 조합·물리·제어 주기 |
| `scene_cfg.py` | `ContactSweepSceneCfg(InteractiveSceneCfg)`: 자산·frame·촉각 view |
| `observations_cfg.py` | `ContactSweepObservationsCfg`: Actor/Critic의 동일한 57D policy 순서 |
| `actions_cfg.py` | `ContactSweepActionsCfg`: Arm Joint Position 6D + Hand 2D |
| `commands_cfg.py` | `ContactSweepCommandsCfg`: 에피소드 고정 목표 |
| `events_cfg.py` | `ContactSweepEventsCfg`: reset·스폰·물성 이벤트 |
| `rewards_cfg.py` | `ContactSweepRewardsCfg`: 10개 활성 보상 전체 선언 |
| `terminations_cfg.py` | `ContactSweepTerminationsCfg`: 실패·timeout 전체 선언 |
| `ppo_cfg.py` | `ContactSweepPPORunnerCfg(RslRlOnPolicyRunnerCfg)`: 독립 PPO 설정 |
| `common.py` | 기하·질량 상수와 Arm 관절 선택 |

로봇 USD·spawn 유틸리티·MDP 계산 함수는 공통 모듈을 재사용한다. `goal_rewards.action_rate_l2`·`joint_vel_l2`의 Arm/Hand 네 운동 페널티와 계수도 그대로 사용한다. 목표 보상 94, 접근 보상 30, pushing_speed free_speed=0.2 m/s·weight=−2, 질량 중심 0.5475 kg·범위 0.345–0.750 kg, 0.02초 제어·최대 10초를 명시했다. 현재 물체 GT를 포함한 관측이므로 최종 Blind 환경으로 해석하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Contact-Sweep-v0 --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Contact-Sweep-v0 --num_envs 1 \
  --checkpoint "<checkpoint.pt>" --real-time
```

실험 폴더는 `logs/rsl_rl/inspire_contact_sweep`다. CPU 관련 검사 **131개 통과**, GPU 8 env·1 iteration 완료. v3 smoke와 저장 환경 YAML은 log_dir 외 동일하고 PPO는 experiment/run 이름 외 동일했다. 같은 seed의 1 iteration 후 Actor/Critic tensor도 전부 일치했다. 장시간 학습·실물 성능은 미검증이다. [구현·검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_contact_sweep_reference.md).

### Contact Sweep Clutter v0 — 장애물 초기 관측과 MLP 임베딩

`Isaac-Inspire-Contact-Sweep-Clutter-v0`는 `ContactSweepEnv`·`ContactSweepEnvCfg`·`ContactSweepPPORunnerCfg`를 상속한다. 구현은 [`contact_sweep_clutter/`](inspire_sweep_baseline/contact_sweep_clutter)에 있다. 부모의 중앙 반경 2 cm 타겟 스폰, 최대 15 cm 목표와 선반 ±X 각각 15° 방향 제외, 고정 초기 관절 자세를 유지한다.

목표를 먼저 확정하고 [`src/scene_randomization`](../scene_randomization/LOGIC.md)의 장애물 배치 규칙을 적용한다. 장애물은 **3개 고정**이며 뒤쪽 GRAY 1개, WHITE 2개다. RED 제외 반경은 **10.3 cm**(물체 반지름 4 cm × 2 + 여유 2.3 cm)로, 최초 v0의 8.3 cm에서 2 cm 확대했다. 형태와 전면 방향 확장 규칙은 유지한다. RED 회피, Tool/Wrist 전면 점유, 물체 간격, 셀·국소 밀도와 손가락 접근 경로 조건을 재사용한다. 장애물 배치 재시도에서도 타겟·목표는 바꾸지 않으며, 시도 상한에 도달하면 오류로 명시한다. 참조 생성기의 360°·5–20 cm 목표 및 넓은 타겟 스폰 분포는 사용자 선택에 따라 적용하지 않는다.

원본 cm 좌표 `(x, y)`는 선반 로컬 m 좌표 `(0.20-y/100, x/100)`로 변환한다. 원본 −Y 전면 방향은 로봇 쪽인 선반 +X가 된다. 타겟은 파란색, 장애물은 주황색이다. 모든 실린더는 반지름 4 cm·높이 10 cm와 같은 질량·관성·마찰·충돌 설정을 사용한다. 매 reset에서 타겟의 물성 샘플을 모든 장애물에 동일하게 적용한다.

| 구성 | 부모 대비 변경 |
| --- | --- |
| `object_tilt` | 기존 물체별 비용 `clip((angle−5°)/(0.9−5°), 0, 1)²`를 타겟과 장애물 총 4개에 합산. weight −2 유지 |
| `object_drop` | 어떤 물체든 기존 높이 1.04 m 미만 또는 선반 법선 대비 tilt ≥0.9 rad이면 종료 |
| 17D 촉각 | 직접 패드 접촉에 더해 손가락 본체 투영 필터에 타겟·장애물 총 4개 모두 포함 |
| `obstacles` 관측 | `[HEADER, x, y, z] × 10` = 40D. 실제 장애물 3개는 HEADER=1, 나머지 7개는 전부 0 |
| 초기 위치 | 타겟과 같은 robot-root frame, 단위 m, 실린더 밑면 중심. reset 때 각 축 ±1 cm 노이즈를 한 번 추가하고 episode 동안 고정 |
| 정책·가치망 | 별도 학습 가능한 `40 → 32(ReLU) → 16` 인코더를 각각 사용, 기존 57D에 연결해 73D 입력 |

기존 타겟 위치는 현재값이며 장애물만 초기 관측이다. rollout에는 `policy` 57D와 `obstacles` 40D를 저장하고 PPO 미니배치에서 임베딩을 다시 계산하므로 인코더까지 학습된다. Checkpoint는 인코더·optimizer를 포함하며 JIT/ONNX export도 인코더를 포함한다. Export의 외부 입력은 `[policy57, obstacles40]` 순서의 **97D**, 내부 정책망 입력은 **73D**다.

나머지 보상·종료, Action 8D, Arm/Hand 네 운동 페널티와 물리적 관절 속도 제한, 제어 주기·PPO hyperparameter는 상속한다. 물체 속도 비용·`push_fast`는 기존처럼 타겟만 검사한다. 기존 57D 단일 물체 checkpoint와 입력·모델 구조가 다르므로 이 task로 학습한 checkpoint를 사용한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-v0 --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-v0 --num_envs 1 \
  --checkpoint "<clutter-checkpoint.pt>" --real-time
```

로그는 `logs/rsl_rl/inspire_contact_sweep_clutter`에 저장한다. 배치 계산은 checkout의 `src/scene_randomization/generator.py`를 읽기 전용으로 사용하므로 두 패키지를 함께 유지한다. 3개 장애물·RED 확장 변경은 [후속 검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_clutter_red_expansion.md)에 기록했다. 최초 구현의 검증 명령·결과·미검증 범위는 [개발 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_contact_sweep_clutter.md)에 기록한다.

### Contact Sweep Clutter v1 — 초기 타겟 관측과 보상 균형

2026-10-20 후속: `shelf_failure`는 `shelf_collision` 실패의 마지막 step 보상에서 **−3을 한 번** 차감한다. 실패 판정 후 reset 전에 계산하며 `dt`를 상쇄한다. Timeout 또는 다른 실패만 발생한 경우에는 0이다. v2도 상속한다. [구현·검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_clutter_v1_shelf_failure.md).

`Isaac-Inspire-Contact-Sweep-Clutter-v1`는 Clutter v0의 Env·EnvCfg·관측·보상·PPO 설정을 상속한다. 구현은 [`contact_sweep_clutter_v1/`](inspire_sweep_baseline/contact_sweep_clutter_v1)에 있으며 runner는 `ContactSweepClutterV1PPORunnerCfg`, 로그 경로는 `logs/rsl_rl/inspire_contact_sweep_clutter_v1`다.

Actor의 `policy.target_obs_state` 3D는 현재 타겟 위치에서 **episode 시작 위치**로 바뀐다. `InitialObjectPositionInRoot`가 reset 때 Command가 저장한 타겟 link 원점을 robot-root frame으로 변환해 고정한다. Initial Privileged 환경처럼 각 좌표에 reset당 한 번 ±1 cm 오차를 더하며 매 step 새로 측정하지 않는다. Critic 전용 `privileged.target_position` 3D는 현재 타겟 위치를 같은 frame에서 매 step 읽고 관측 노이즈를 추가하지 않는다. 기존 `agents/initial_privileged_ppo_cfg.py`의 비대칭 라우팅을 따르되, Critic 추가 정보는 요청한 위치 3D로 한정한다.

| 경로 | 원시 입력 | 장애물 인코더 이후 |
| --- | --- | --- |
| Actor | `policy` 57D + `obstacles` 40D = 97D | 57D + 16D = 73D |
| Critic | `policy` 57D + `privileged` 3D + `obstacles` 40D = 100D | 57D + 3D + 16D = 76D |

장애물 입력은 v0의 `[HEADER,x,y,z] × 10`, 실제 3칸·더미 7칸과 초기 위치 노이즈를 상속한다. Actor/Critic은 각각 `40 → 32(ReLU) → 16` 인코더를 학습한다. Actor export 입력은 97D이며 현재 타겟 GT를 요구하지 않는다. 보상 계산용 `_initial_object_positions_w`는 모든 reset 배치가 끝난 뒤 실제 월드 XYZ를 별도로 저장한다. 이 정확한 위치에는 Actor의 타겟·장애물 관측 오차를 섞지 않는다.

| 보상 항 | 계수 | v1 식·조건 |
| --- | ---: | --- |
| `reaching` | 30.0 | 기존 최선 접근 거리의 개선량 유지 |
| `goal_distance` | 100.0 | 기존 최선 목표 XY 거리의 개선량; v0 계수 94.0에서 조정 |
| `goal_hold` | 0.9 | 월드 XY 목표 반경 0.01 m 이내이면 1, 이외 0 |
| `contact_any` | 0.16 | 17개 tactile bit 중 하나 이상 ON이면 1 |
| `contact_count` | 0.16 | `min(ON 개수, 5) / 5` |
| `obstacle_displacement` | −5.0 | 타겟을 제외한 초기 대비 XYZ 이동 거리 제곱합, 최대 0.04 m² |

신규 항은 RewardManager가 `weight × raw × step_dt`로 누적한다. `goal_hold`는 조건 안에서 초당 +0.9로 일정하며 5초 유지하면 +4.5다. 목표의 월드 Z·속도·연속 유지 이력은 조건에 사용하지 않는다. Contact 두 항은 현재 정책과 같은 tactile 판정을 공유하며 각각 최대 초당 +0.16이다. 5개 이상을 7초 유지하면 +1.12 +1.12 = +2.24다. 장애물 2개가 시작 위치에서 각각 10 cm 떨어진 채 10초 있으면 −1.0, 모든 장애물 이동 비용의 상한은 초당 −0.2·10초당 −2.0이다. 타겟 이동과 관측 노이즈는 이 비용에 포함되지 않는다.

`pushing_speed.free_speed`는 0.2→1.0 m/s, `arm_speed_excess.free_speed`는 1.0→0.5 rad/s다. 기존 가중치 −2/−1과 출력 상한 100은 유지한다. Arm/Hand action rate·joint velocity 네 항은 로컬 `goal_rewards.action_rate_l2`·`joint_vel_l2`, raw 제곱합과 −0.03/−0.03/−0.03/−0.01 계수를 그대로 사용한다.

부모의 장애물 배치·물성·8D Joint Position action·0.02초 제어·10초 timeout·실패 종료를 상속한다. 목표 도달로 종료하지 않는다. 특히 `push_fast`의 0.8 m/s 종료는 그대로이므로 보상의 free speed 1.0은 종료 속도 완화를 의미하지 않는다.

계수는 `logs/TEMP/2026-10-20_14-04-31`의 9,655개 iteration을 기준으로 정했다. 목표 유지 5초·5개 tactile 접촉 7초를 가정하면 **양의 네 보상 그룹 합**에서 Reaching/Goal distance/Goal 유지/Contact가 각각 **34.92/34.61/20.35/10.13%**, 마지막 1,000개 기준 **33.88/36.91/19.50/9.71%**다. 이는 기존 progress를 재가중한 시나리오이며 새 정책에서 실제 측정된 비중은 아니다. 음의 페널티는 이 분모에서 제외해 별도로 비교한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-v1 --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-v1 --num_envs 1 \
  --checkpoint "<clutter-v1-checkpoint.pt>" --real-time

python src/inspire_sweep_baseline/scripts/analyze_clutter_rewards.py \
  --log-dir logs/TEMP/2026-10-20_14-04-31 --last-n 1000 \
  --output /tmp/clutter_v1_calibration_last1000.json
```

분석 스크립트는 TensorBoard가 설치된 Python으로 실행한다. `--last-n 0` 또는 생략 시 전체 iteration을 사용한다. `--log-dir`를 새 v1 run으로 바꾸면 기존 가정치와 실제 신규 보상 그룹 비중을 함께 확인할 수 있다. 저장된 `params/env.yaml`의 계수를 읽으며 simulator를 시작하지 않는다. 재생은 관측 의미와 Critic 차원이 맞는 v1 checkpoint를 사용한다. [개발·검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_contact_sweep_clutter_v1.md)에 계산 근거와 실제 검증 범위를 기록한다.

### Contact Sweep Clutter Curriculum v1 — Learning iteration 기반 보상

`Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1`는 현재 Clutter v1의 Env·Cfg·보상·PPO 설정을 상속한다. 구현은 [`contact_sweep_clutter_v1_curriculum/`](inspire_sweep_baseline/contact_sweep_clutter_v1_curriculum), 일정은 [`curriculum_cfg.py`](inspire_sweep_baseline/contact_sweep_clutter_v1_curriculum/curriculum_cfg.py)에서 관리한다.

| 변경 대상 | 초기값 | 감소 주기 | 회당 감소량 | 최종값 |
| --- | --- | --- | --- | --- |
| `pushing_speed.params["free_speed"]` | 0.1 m/s | 3,000 iteration | 0.01 | 0.05 |
| `arm_speed_excess.params["free_speed"]` | 1.0 rad/s | 5,000 iteration | 0.05 | 0.8 |
| `arm_action_rate.weight` | −0.03 | 3,000 iteration | 0.005 | −0.06 |
| `arm_joint_velocity.weight` | −0.03 | 3,000 iteration | 0.005 | −0.06 |

로그의 0-based **Learning iteration**을 사용한다. 첫 변경은 각각 3,000·5,000 iteration의 rollout 첫 step 전에 적용한다. 이후 최종값에 도달하면 고정한다. `CurriculumOnPolicyRunner`가 `learn()` 시작 iteration과 실제 `num_steps_per_env`로 rollout 경계를 추적하므로 병렬 환경 수·physics step·episode reset 횟수에 의존하지 않는다. 보상은 `CurriculumManager`에서 절대값으로 계산해 실행 중인 `RewardManager`에 반영한다. Reset을 반복해도 감소가 누적되지 않으며 진행 중인 모든 episode에도 같은 값이 적용된다.

재개는 checkpoint 내부 `iter`를 사용한다. 설치된 RSL-RL 5.0.1이 저장 index부터 다시 시작하는 원래 동작을 유지한다. 재생은 checkpoint 단계에 고정한다. 직접 환경만 실행하면 iteration 0이며, 다른 실행기는 `set_learning_iteration(i)`를 연결해야 한다. 기본 train/play는 등록된 전용 runner를 자동 선택한다.

사용자 명시 요청에 따라 이 환경의 Arm 두 가중치만 변화한다. Hand 가중치 −0.03/−0.01, 로컬 raw 제곱합 함수·출력 상한·dt 처리, v1 관측·Action·장면·reset·종료·나머지 보상은 상속한다. 별도 v2의 F/T 보정은 포함하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1 --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1 --num_envs 2048 --headless \
  --resume --load_run "<run-directory-name>" --checkpoint "model_15000.pt"

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-Curriculum-v1 --num_envs 1 \
  --checkpoint "<curriculum-checkpoint.pt>" --real-time
```

로그 폴더는 `logs/rsl_rl/inspire_contact_sweep_clutter_v1_curriculum`이다. TensorBoard의 `Curriculum/pushing_free_speed`, `Curriculum/arm_free_speed`, `Curriculum/arm_action_rate_weight`, `Curriculum/arm_joint_velocity_weight`에 reset 시점의 현재값을 기록한다. [실행한 검사와 GPU 미검증 범위](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_clutter_v1_curriculum.md)를 확인한다.

### Contact Sweep Clutter v2 — F/T 하중 보정과 변화 비용

`Isaac-Inspire-Contact-Sweep-Clutter-v2`는 Clutter v1의 Env·EnvCfg·관측·보상·PPO runner를 상속한다. [`contact_sweep_clutter_v2/`](inspire_sweep_baseline/contact_sweep_clutter_v2)에 구현하며, 기본 로그 폴더는 `logs/rsl_rl/inspire_contact_sweep_clutter_v2`다.

기존 `policy.wrist_wrench` 6D를 **Hand 중력·관성 하중을 차감한 EEF 기준점·EEF 축의 6축 반력**으로 교체한다. `axia80_feasibility`의 tool-only 차감 원리를 참조하되, 현재 Hand 자세·손가락 구성·가속도에 맞춘 모델 기반 보정을 사용한다. 하류 30개 강체의 실제 PhysX 질량·관성과 운동 상태를 사용하며 물체 GT·접촉 대상 ID를 보정 입력으로 사용하지 않는다. 고정 tare를 수집하지 않아 초기 접촉을 영점으로 지우지 않으며, 수치·solver 오차에 따른 잔여값은 남을 수 있다.

새 `wrench_change` 비용은 같은 보정 표본을 월드 축으로 표현한 뒤 제어 step 사이 힘·모멘트 변화량을 계산한다. 각각 0.5 N·0.01 N·m까지 무비용이며, 초과분을 5 N·0.1 N·m로 나눈 제곱합의 절반을 최대 1로 제한한다. weight는 −0.1로 **초당 최대 −0.1·10초당 최대 −1**이다. 추가로 각 시점의 누적 비용을 **그때까지 실제로 얻은 양의 보상 합의 5%** 이하로 제한한다. 첫 유효 표본은 이력 초기화에만 사용하며 부분 reset 시 해당 환경의 이력·한도만 초기화한다.

Actor 초기 타겟·Critic 현재 타겟, 원시 Actor 97D·Critic 100D와 인코더 후 73D·76D, 8D action·0.02초 제어·10초 timeout·실패 종료, 기존 14개 보상과 Arm/Hand 네 운동 비용의 함수·계수는 유지한다. F/T 관측 의미가 바뀌므로 v2로 새로 학습한 checkpoint 사용을 기본으로 한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-v2 --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Contact-Sweep-Clutter-v2 --num_envs 1 \
  --checkpoint "<clutter-v2-checkpoint.pt>" --real-time
```

[보정·비용 계산식과 검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_contact_sweep_clutter_v2.md)에 실제 실행 결과와 미검증 범위를 구분해 기록한다. 보정 모델과 비용 상한의 검증은 장기 학습의 밀기 성공률·충돌 회피 편향 개선을 입증하지 않는다.

### Dex Tactile Renewal v3 — v1 상속, 질량·속도 비용·목표 보상 조정

`Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v3`는 `InspireDexTactileSweepOmniRenewalV1EnvCfg`를 직접 상속한다. Actor/Critic 57D, Joint Position 6D + Hand 2D, 0.02초 제어 주기와 기존 종료 조건을 사용한다.

| 항목 | v1 | v3 |
| --- | --- | --- |
| pushing_speed free_speed | 0.3 m/s | 0.2 m/s |
| pushing_speed weight | −1.0 | −2.0 |
| goal_distance weight | 300.0 | 94.0 |
| reaching weight | 30.0 | 30.0 |
| 물체 질량 분포 중심 | 0.365 kg | 0.5475 kg |
| 물체 질량 범위 | 0.230–0.500 kg | 0.345–0.750 kg |

Goal 계수는 `logs/2026-10-19_12-15-47`의 마지막 10 iteration(39027–39036)에서 측정한 TensorBoard reaching 평균 0.778318, goal_distance 평균 3.314545를 사용했다. `300 × (4/3) × 0.778318 / 3.314545 = 93.9276`을 94로 반올림했다. 기존 궤적을 재가중하면 누적 기여가 약 +7.783 : +10.386으로 3:4가 된다. 이론상 최대 보상으로 산정하지 않았다. 질량·속도 비용을 바꾼 뒤 새로 학습한 정책에서 같은 기여 비율이 유지되는지는 별도 확인해야 한다.

생성 질량과 reset 샘플을 모두 1.5배로 적용한다. 질량 노이즈도 비례 확대하며 관성은 질량에 맞게 갱신한다. 마찰 분포와 Arm/Hand 네 운동 페널티는 부모 설정을 상속한다. v2의 success/goal_hold 보상은 포함하지 않는다. PPO는 v1과 같고 실험 폴더는 `logs/rsl_rl/inspire_dex_tactile_sweep_omni_renewal_v3`다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v3 \
  --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v3 \
  --num_envs 1 --checkpoint "<v3-checkpoint.pt>" --real-time
```

[환경 설정](inspire_sweep_baseline/dex_tactile_omni_renewal_v3_env_cfg.py) · [구현·검증 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-20_renewal_v3_reward_mass.md).

### Dex Tactile Renewal v1

2026-10-19: Arm actuator의 `velocity_limit_sim`을 shoulder pan/lift·elbow는 2.0 rad/s, wrist 1/2/3는 4.0 rad/s로 설정했다. v2·OSC v2도 상속한다. 관절 위치 범위는 유지하며 Arm 과속 종료 기준은 >5.0 rad/s 그대로다. PhysX가 적용하는 속도 제한으로, 순간 초과를 완전히 보장하는 상태 clamp는 아니다. [변경·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-19_v1_arm_velocity_limits.md).

2026-10-19 후속 요청으로 아래 종료 진단 블록은 전체 주석 처리했다. 현재 v1·v2·OSC를 포함한 모든 환경에서 종료 진단 print가 비활성화되어 있다.

2026-10-19 종료 진단: 공통 `InspireSweepEnv`가 reset 직전에 발생한 모든 종료 항목을 `[TERMINATION]`으로 print한다. 출력은 종료 이름·함수·timeout 여부·환경 번호·episode step이며, 동시 발생도 각각 표시한다. v1·v2·OSC 등 모든 파생 환경에 기존 명령으로 새로 실행하면 적용된다. [종료 출력 구현](inspire_sweep_baseline/env.py) · [변경·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-19_v1_termination_prints.md).

이후 파생 작업의 기준 환경은 사용자 지정 `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v0`다. 이번 `Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v1`은 해당 v0를 직접 상속한다.

- 물체·Arm 지수 속도 비용에 각각 `_limit` 함수를 사용하고 반환 비용을 최대 100으로 제한한다. weight −1.0에서 시간 곱셈 전 −100, 0.02 s 스텝의 항별 기여 하한은 −2다. 지수 계산 전에 포화시켜 overflow를 방지한다.
- `shelf_collision`의 손목 상대 높이 기준을 0.07 m에서 0.03 m로 낮춰 더 가까운 접근을 허용한다. 선반 속도·변위 검사는 유지한다.
- 목표 방향은 선반 로컬 ±X 각각 ±15°를 제외하고 나머지 300°에서 균등 샘플링한다. 기존 좌·우 ±Y 방향은 허용한다.

57D/8D 관측·행동과 센서, Dex 진행 보상, 네 필수 운동 비용·계수 및 나머지 설정은 상속한다. 기본 로그 폴더는 `logs/rsl_rl/inspire_dex_tactile_sweep_omni_renewal_v1`다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v1 \
  --num_envs 2048 --headless
```

[v1 환경 설정](inspire_sweep_baseline/dex_tactile_omni_renewal_v1_env_cfg.py) · [활성 환경 계약](../../example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md) · [개발·검증 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-16_dex_tactile_renewal_v1.md).

### Dex Tactile Renewal v2: 목표 유지와 성공 보너스

`Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2`는 **v1을 직접 상속**한다. v1의 지수 물체/Arm 속도 비용 상한 100·손목 높이 종료 기준 3 cm·선반 로컬 ±X 각각 ±15° 목표 방향 제외를 적용한다. 물체 위치의 반경 2 cm 분포는 그대로이며 v1의 스폰 관련 변경은 목표 방향 제한이다. 기존 57D Actor/Critic 관측(현재 물체 GT 포함), 8D Joint Position/Hand 행동·센서·물성·reset·보상·실패 종료 위에 다음 양의 보상 두 항을 추가한다. **성공으로 종료하지 않으며 기존 실패 또는 10초 timeout까지 유지한다.** 별도의 새 속도·위치 오차 비용과 Joint Position 액션 제한은 없다.

- `success`: 목표 XY 오차 <1 cm, 물체 XY 속도 <0.02 m/s, 선반 법선 대비 기울기 ≤10°, 물체 바닥의 선반면 오차 ≤1 cm, 물체 반경을 고려한 선반 내부 지지를 **0.5초 연속** 만족하면 달성 보상을 지급한다. 기존 실패 조건이 동시에 발생하면 지급하지 않는다. 성공 termination은 비활성이다.
- 성공 보상: `StableGoalAchievement`가 최초 달성을 기억해 **에피소드당 +100 한 번** 지급한다. 목표 이탈·재진입으로 다시 지급하지 않으며 실제 episode reset에서만 재지급 가능해진다. dt를 상쇄하므로 제어 주기에 따라 보너스가 달라지지 않는다.
- `goal_hold`: 자격 조건을 만족한 연속 시간을 `h`라 할 때 보상률은 `20 × min(h / 0.5, 1)`/초다. 첫 스텝부터 증가하며 이탈하면 0으로 초기화되고 재진입하면 다시 증가한다. 성공 보상 term이 먼저 갱신한 카운터를 읽는다. 0.02초 주기에서는 +0.016부터 +0.4/스텝까지 증가한 뒤 **성공 이후에도 +0.4/스텝을 계속 지급**한다. 처음부터 10초 동안 자격을 유지하면 체류 보상 합계 +195.2, 성공 보너스 +100이다.

+100과 초당 최대 +20은 이번 구현의 초기 실험값이며 학습 성능으로 튜닝한 값은 아니다. 기본 로그 폴더는 `logs/rsl_rl/inspire_dex_tactile_sweep_omni_renewal_v2`다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2 \
  --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-v2 \
  --num_envs 1 --checkpoint "<v2-checkpoint.pt>" --real-time
```

[v2 설정](inspire_sweep_baseline/dex_tactile_omni_renewal_v2_env_cfg.py) · [초기 구현 기록](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-18_dex_tactile_renewal_v2_goal_hold.md) · [성공 후 timeout까지 유지하도록 수정](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-18_v2_goal_hold_until_timeout.md).

현행 상속은 `v0 → v1 → v2 → OSC v2`이며 PPO runner도 같은 버전 상속을 따른다. 환경 ID·실행 명령·보너스 계수는 그대로다. [v1 상속 전환·검증](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-18_v2_inherits_v1.md).

### Dex Tactile Renewal OSC v2

`Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2`는 v2를 직접 상속하고 Arm 제어만 OSC로 바꾼다. Hand 제어, 57D 관측·8D 행동 크기, 센서·물성·스폰·기존 비용·종료, +100 성공 보너스·증가하는 유지 보상은 v2에서 상속한다. 기존 Arm 속도 종료 기준 5.0 rad/s는 유지하며 1.5 rad/s는 명령 범위 선택의 대략적인 목표다.

- 기준은 `omni_renewal_osc_env_cfg.py`의 fixed-impedance `pose_rel` OSC다. 제어점은 기존 EEF와 같은 `axia80_link` 및 pose offset이고 로봇 root 좌표계를 사용한다. 중력 보상·관성 decoupling·damping ratio 1.0을 유지한다.
- `motion_stiffness_task=100.0`으로 낮춘다. OSC effort와 경쟁하지 않도록 기존 OSC처럼 Arm implicit PD의 stiffness/damping만 0으로 바꾼다.
- Arm raw 6D는 위치 scale 0.005 m, 회전 scale 0.025 rad를 곱한 뒤 **이동 벡터 norm ≤0.005 m, 회전 벡터 norm ≤0.025 rad**로 각각 제한한다. 각 성분별 제한이 아니며 방향을 보존한다. 큰 Gaussian 출력에도 실제 명령의 상한이 적용된다. Hand나 raw Action 제곱합 보상 입력은 이 제한으로 바꾸지 않는다.
- 현재 pose를 기준으로 매 0.02초 정책 스텝에 새 상대 목표를 생성한다. 한 명령당 증분 제한이며 누적 가동 범위를 제한하지 않는다. 관절 속도 보장은 아니며 자세·접촉과 controller 동역학에 따라 달라진다. upstream OSC에서 실제 적용되지 않는 `clip` 필드에 의존하지 않는다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/train.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2 \
  --num_envs 2048 --headless

./IsaacLab/isaaclab.sh -p src/inspire_sweep_baseline/scripts/play.py \
  --task Isaac-Inspire-Dex-Tactile-Sweep-Omni-Renewal-OSC-v2 \
  --num_envs 1 --checkpoint "<osc-v2-checkpoint.pt>" --real-time
```

기본 로그 폴더는 `logs/rsl_rl/inspire_dex_tactile_sweep_omni_renewal_v2_osc`다. Joint Position v2와 action 의미가 다르므로 OSC 정책으로 학습·평가한다. 초기 0.1초의 명령 응답은 낮아졌지만, 이후에는 0 명령 대조군에서도 과속이 발생했다. 짧은 실행을 장시간 안정성이나 학습 성공으로 해석하지 않는다. [OSC v2 설정](inspire_sweep_baseline/dex_tactile_omni_renewal_v2_osc_env_cfg.py) · [명령 제한 구현](inspire_sweep_baseline/mdp/osc_actions.py) · [실측 속도·검증 및 한계](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-10-18_dex_tactile_renewal_v2_osc.md).
