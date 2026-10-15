# Inspire Sweep Baseline

[활성 환경·상속·실행 안내](../../example/Contact-Rich-Manipulation-Context/02_implementation/08_ACTIVE_ENVIRONMENTS.md) · [개발 지침](../../example/Contact-Rich-Manipulation-Context/00_guidelines/09_DEVELOPMENT_GUIDE.md)

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
