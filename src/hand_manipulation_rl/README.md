# Hand Manipulation RL — Blind Sweeping

`Isaac-Blind-Sweep-Inspire-v0`는 UR5e–가상 Axia80–Inspire Hand로 고정 선반판 위의
단일 Cube를 시각 갱신 없이 미는 Isaac Lab 2.3.2 학습 환경이다. 이를 그대로 보존하면서
낮고 안전한 시작 자세와 dense 접근 보상을 추가한 상속 환경
`Isaac-Blind-Sweep-Inspire-Approach-v0`도 제공한다. 해당 환경의 실제 학습에서 확인된
중력 하강·선반 접촉 reward shortcut을 수정한 2차 상속 환경
`Isaac-Blind-Sweep-Inspire-Approach-v1`과, v1의 제어·관측·접근 보상을 유지하면서 미는
행동의 우선순위를 높인 계수 조정 환경 `Isaac-Blind-Sweep-Inspire-Approach-v2`도 함께
제공한다. 환경 코드는
`src/sweep_rl`, `example/Sweep-Policy`, `src/inspire_tactile`,
`src/axia80_feasibility`를 import하지 않는다. 참고 자산은 이 패키지 안의 상대 경로 USD로
복사했으며 Nucleus, ROS, 절대 경로에도 의존하지 않는다.

## 구현된 계약

- 고정 Base UR5e → Cylinder 강체 → 명시적 Fixed measurement joint → Inspire Hand의 한
  Articulation. Arm drive는 effort OSC와 경쟁하지 않도록 stiffness/damping이 0이고 Hand만
  position drive를 사용한다.
- Sweep-Policy의 shelf root·표면 높이·작업영역을 바탕으로 만든 폭 1.00 m(Y) × 깊이
  0.36 m(X)의 얇은 kinematic 등가판과 양의 질량·중력·마찰을 가진 dynamic Cube 하나.
  로봇 root와 초기 arm 자세도 reference 값을 따른다.
- 명령은 선반 폭 방향인 world `+Y/-Y` 좌우 두 방향만 동일 확률로 샘플링하며 거리는
  reference와 같은 0.18 m다. 깊이 `+X/-X` 방향 명령은 생성하지 않는다. 초기 물체 위치와
  목표는 episode 동안 고정된다.
- palm/dorsal 1:1 mode와 mode별 면 법선 정렬. 활성 reset은 Cube 기준
  `(반대 sweep 방향 backoff, shelf tangent, world-up)` position offset과 local orientation
  offset의 좁은 안전 영역만 사용한다. dorsal은 고정 H→C 변환을 뒤집을 때도 실제 Hand 중심
  standoff가 palm과 같도록 backoff를 보정한다. Sweep-Policy reaching-pose 방식의 6축 DLS를
  쓰되 큰 pose 오차는 작은 Cartesian 단계로 나누고, IK 내부 physics `sim.forward()` 없이 첫
  유효 seed에서 중단한다. Robot–Cube/Board의 빠른 clearance와 보수적 OBB 인증을 모두 통과한
  pose만 시작 상태로 채택한다. 기존 `ConditionalPoseIKReset` 구현은 비교·감사를 위해 남아
  있지만 event에는 등록되지 않는다.
- 현재 실제 C pose 기준 6D local Cartesian delta와 full inertia-decoupled fixed-gain OSC.
  Sweep-Policy OSC처럼 전 축 stiffness 100, damping ratio 1, OSC gravity compensation off다.
  PhysX의 COM 기준 Jacobian을 C까지 정확히 이동하고 `J·q̇`와 C twist 일치를 smoke에서
  검사한다. Hand는 2D Inspire synergy(공통 굽힘 + 독립 thumb1)이며 총 8D action이다.
- 정확한 57D 관측 순서와 런타임 차원 검사. 현재 물체 pose/속도, 실제 접촉 법선,
  작업판 접촉력은 actor/critic 입력에 포함하지 않는다.
- 17개 palmar physical contact body + 17개 dorsal channel, episode header를 결합한
  actor tactile 18D. 내부 진단은 양면 34개를 유지한다.
- child body의 incoming joint wrench를 그대로 읽고 F→C 회전과 moment-arm 좌표변환만
  적용한다. 중력 cancellation이나 reset tare/bias를 전혀 적용하지 않으므로 Hand 자체 무게의
  힘과 모멘트가 모든 live F/T 표본에 계속 포함된다.
- 작업판 sensor는 Cube 지지 접촉을 제외하고 Robot link별 normal-force norm을 먼저 구한
  뒤 합산한다. hard 기준은 physics substep마다 latch된다.
- 목표, 실제 Hand–Cube 합력 법선, tactile 유지, action 변화, 시간 비용과 tilt/board/height
  soft→hard ramp reward. 성공·세 core failure와 out-of-bounds/invalid-reset guard·timeout을
  분리하고 같은 step에서는 실패가 성공보다 우선한다.
- 기존 PPO 설정은 보존하고, 기본 학습은 Sweep-Policy `rsl_rl_ppo_cfg_02`를 독립적으로 옮긴
  새 설정을 사용한다. reference와 달리 `max_iterations`만 요청대로 10,000을 유지한다.
- Approach 환경은 손의 비대칭 collision envelope 때문에 단일 높이를 일괄 하강시키지 않는다.
  Hand +X가 위/아래인 자세에 따라 Cube 중심 위 `0.075 m`/`0.100 m`를 사용한다. 기존
  `0.100 m` 대비 여유가 있는 자세는 25 mm 낮추고, 비대칭 형상 때문에 하강 여유가 없는
  반대 자세는 기존 높이를 유지한다. 두 경우 모두 기존 proxy clearance와 전체 collider OBB
  인증을 통과해야 시작된다.
- Approach 환경에는 선택된 palm/dorsal 면을 대칭화한 control-point proxy가 Cube의 upstream
  face로 접근하는 bounded reward(`weight=0.10`)가 추가된다. 목표 높이는 위의 안전 높이로
  고정되므로 손을 선반 쪽으로 내리는 동작은 이 보상을 늘리지 못한다. face 바깥쪽 capture
  radius에서만 보상이 포화되고 Cube 안쪽으로 들어가면 감소하므로 관통 유인도 만들지 않는다.
- Approach-v1은 zero action에서도 발생하던 수직 하강을 막기 위해 OSC gravity compensation을
  켠다. 병진 action은 mode와 명령 방향에 관계없이 `[world-up, toward-object, shelf-depth]`로
  해석하며 각각 `4/12/4 mm`로 제한한다. reset backoff는 보수적 OBB 인증을 유지한 채
  `0.140 -> 0.100 m`로 줄인다. 작은 relative target만으로는 1 kg Cube의 정지마찰을 안정적으로
  넘지 못하므로, 실제 Hand--Cube 접촉 중 양의 접근 action에만 최대 `10 N`의 bounded push
  preload를 더한다. free-space, 후퇴 action, Cube가 아닌 접촉에는 이 힘을 적용하지 않는다.
- v1 접근 신호는 실제 접촉 probe로 구한 palm/dorsal × Hand-X up/down별 전진 거리
  `105/55/105/165 mm`와 안전 하강량 `14/6/20/35 mm`를 따르는 signed progress
  potential(`weight=2.0`)이다. 목표 높이는 전진량에 비례해 낮아지므로 수직 하강만으로는 보상을
  얻지 못한다. 모든 분기의 접촉 전 potential 몫은 동일한 `0.35`이며, free frontier 이후에는
  손 전진과 물체의 명령 방향 이동이 함께 발생해야 나머지 보상이 열린다. 따라서 손만 물체 옆을
  통과하는 우회 행동은 전체 shaping return을 얻지 못한다. 현재 접촉 경로 아래에는 별도
  one-sided 높이 penalty가 선반 충돌 전에 작동한다.
- 소스 자산의 palm physical pad는 이 접근 경로에서 접촉하지 않고 base/thumb carrier link가 먼저
  닿는 반면 dorsal 채널은 parent-link 근사이므로, 비대칭인 tactile reward는 v1에서 끈다. 17채널
  tactile 관측은 원형 그대로 유지한다. 양 mode 공통 접촉 bonus는 `target_hand_contacts`에서 확인한
  모든 Hand body–Cube 최초 접촉에 episode당 한 번만 주므로 정지 접촉으로 누적할 수 없고, 선반
  단독 접촉 보상은 정확히 0이다.
- v1 actor에는 episode-fixed 접촉 frontier 오차 3D를 action 축 순서로 추가해 관측은 60D다.
  PPO는 초기 표준편차 `0.5`, fixed learning rate `3e-4`를 사용해 기존 adaptive LR이 첫
  iteration부터 `0.01`로 상승하던 현상을 차단한다.
- Approach-v2는 v1 전체를 상속하며 goal reward를 `4.0 -> 12.0`, 실제 접촉 법선 정렬
  reward를 `0.75 -> 4.0`, success terminal bonus를 `1.0 -> 4.0`으로 높인다. goal이 가장
  크고 정렬은 그보다 작도록 하여 목표까지 계속 미는 행동을 우선하면서 접촉 방향을 유지한다.
  초기 탐색에 필요한 접근 progress `2.0`과 최초 접촉 bonus `0.5`는 v1 값을 그대로 유지한다.
- 독립형 RSL-RL PPO train/play 진입점, vectorized palm/dorsal smoke probe, simulator 독립
  pure-torch tests.
- `extras["episode_diagnostics"]`에 명령·초기/현재 pose, reset 표본/IK/보수적 live-FK 충돌 인증 결과,
  first-contact/loss, 양면 tactile, raw-F/measured-C wrench, board/tilt/height peak,
  OSC/Hand 포화 및 term별 raw·weighted-rate·integrated reward를 반환한다. 이 중 핵심
  scalar와 모든 reward term은 `extras["log"]`를 통해 RSL-RL/TensorBoard에 기록한다.

## 설치와 실행

저장소 루트에서 Isaac Lab Python 환경을 사용한다.

```bash
./IsaacLab/isaaclab.sh -p -m pip install -e src/hand_manipulation_rl
```

현재 쉘에서 Conda가 활성화되어 `isaaclab.sh`가 잘못된 Python을 고르면 먼저
`conda activate env_isaaclab`처럼 Isaac Lab 환경을 활성화한다.

```bash
# 구조/차원/짧은 dynamics smoke
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/smoke_env.py \
  --headless --device cuda:0 --steps 4

# 마커 생성/갱신까지 포함한 smoke (GUI로 보려면 --headless 제거)
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/smoke_env.py \
  --headless --device cuda:0 --steps 4 --debug-vis

# 양면을 각각 고정해 vectorized reset/무보정 F/T/짧은 OSC dynamics 확인
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/smoke_env.py \
  --headless --device cuda:0 --num-envs 8 --surface palm --steps 4 --seed 42
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/smoke_env.py \
  --headless --device cuda:0 --num-envs 8 --surface dorsal --steps 4 --seed 42

# C-frame 병진/회전 6축의 작은 양의 증분을 실제 pose 응답으로 확인
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/probe_control_axes.py \
  --headless --device cuda:0 --surface palm --steps 4 --seed 42
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/probe_control_axes.py \
  --headless --device cuda:0 --surface dorsal --steps 4 --seed 42

# 작은 학습 확인
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --headless --device cuda:0 --num_envs 4 --max_iterations 2

# Approach 상속 환경 smoke (palm/dorsal 및 양 sweep 방향의 reset 안전성)
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/smoke_env.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v0 \
  --headless --device cuda:0 --num-envs 8 --steps 4

# Approach 상속 환경 작은 학습 확인
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v0 \
  --headless --device cuda:0 --num_envs 4 --max_iterations 2

# 수정된 Approach-v1: 장시간 zero-action hold 및 작은 학습 확인
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/smoke_env.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v1 \
  --headless --device cuda:0 --num-envs 8 --steps 50
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v1 \
  --headless --device cuda:0 --num_envs 4 --max_iterations 2

# 본 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --headless --device cuda:0 --num_envs 4096

# Approach 환경 본 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v0 \
  --headless --device cuda:0 --num_envs 4096

# 수정된 Approach-v1 본 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v1 \
  --headless --device cuda:0 --num_envs 4096

# 미는 행동을 우선한 Approach-v2 본 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/train.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v2 \
  --headless --device cuda:0 --num_envs 4096

# checkpoint 재생
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/play.py \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/model.pt

# Approach checkpoint 재생
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/play.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v0 \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/model.pt

# Approach-v1 checkpoint 재생(v0 checkpoint와 관측/action 계약이 달라 호환되지 않음)
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/play.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v1 \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/model.pt

# Approach-v2 checkpoint 재생(v1과 동일한 60D 관측/8D action 계약)
./IsaacLab/isaaclab.sh -p src/hand_manipulation_rl/scripts/play.py \
  --task Isaac-Blind-Sweep-Inspire-Approach-v2 \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/model.pt
```

재생 화면에는 현재 Cube 중심의 반투명 주황색 `Target`, 초록색 `Goal`, Hand 중앙의
가상 C/EEF 원점과 RGB 축(X/Y/Z)이 기본으로 표시된다. 이 마커는 충돌·질량·관측·보상에
영향을 주지 않는다. 성능 비교 등으로 숨길 때는 위 명령에 `--disable-markers`를 추가한다.
대규모 headless 학습에서는 `BlindSweepEnvCfg.debug_vis=False`가 기본이라 생성되지 않는다.

Pure-torch 계약 테스트는 Isaac Sim을 띄우지 않는다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q src/hand_manipulation_rl/tests
```

현재 검증 기준은 repository의 Isaac Lab `2.3.2` (`4e06a8a516a2`)와 그 환경의
Isaac Sim 5.1이다. 다른 Isaac Lab/Isaac Sim 조합은 API와 dynamics를 다시 smoke해야 한다.

## 파일 구성

- `hand_manipulation_rl/assets/robot.py`: package-local USD spawn과 F/T chain 조립
- `hand_manipulation_rl/env_cfg.py`: scene, action/observation/reward/termination, 모든 수치 설정
- `hand_manipulation_rl/env.py`: episode spec, reset 순서, C0 확정, substep latch, sensor adapter
- `hand_manipulation_rl/mdp/actions.py`: C-frame OSC와 2D Hand synergy
- `hand_manipulation_rl/mdp/events.py`: 전체 reset, 활성 stable-offset reset, 비활성 보존 legacy reset
- `hand_manipulation_rl/mdp/observations.py`: 고정 순서 57D policy vector
- `hand_manipulation_rl/mdp/rewards.py`, `terminations.py`: MDP 식과 manager adapter
- `hand_manipulation_rl/sensors.py`: bilateral tactile, fixed-joint wrench, board/normal 집계
- `hand_manipulation_rl/agents/rsl_rl_ppo_cfg_02.py`: Sweep-Policy reference 기반 기본 PPO 설정
- `hand_manipulation_rl/agents/rsl_rl_ppo_cfg.py`: 보존된 기존 PPO 설정
- `scripts/train.py`, `scripts/play.py`: 외부 task package를 import하지 않는 standalone RSL-RL 실행
- `scripts/smoke_env.py`: 양면 reset contact, 무보정 zero-delta dynamics, Jacobian/twist, Axia80 extent,
  모든 38개 Robot body의 의도적 Cube overlap 거절 검사
- `scripts/probe_control_axes.py`: 동일 초기 상태의 zero-action pair를 빼서 C-frame 6축의 실제 응답 검사

## 반드시 교정할 PROPOSED 값

`BlindSweepTaskCfg`의 수치는 최초 bring-up을 위한 제안값이지 실측 확정값이 아니다. 특히
board/base transform, `c_offset_h`, 물체/판 마찰, C standoff, action scale,
tactile threshold, uncancelled wrench scale/sign, 세 soft/hard 위험 기준은 scene probe와
정하중/lever-arm 시험 후 교정해야 한다. 설정은 모든 `soft < hard` 관계와 관측/action 차원을
시작 시 검증한다.

현재 dorsal 17채널은 원본 자산에 물리 dorsal pad가 없어서 대응 parent link의 외부 접촉을
읽는 명시적 bring-up 근사다. 같은 distal link의 두 region은 같은 신호를 내며 접촉 면의
앞/뒤를 완전히 분류하지 못한다. API와 17개 channel 순서는 고정되어 있으므로 추후 검증된
GPU contact-point region projection으로 sensor backend만 교체할 수 있다. 이 한계를 숨긴 채
독립적인 17개 FSR을 검증했다고 해석하면 안 된다.

Stable-offset reset 후보는 빠른 clearance prefilter 뒤에, 각 rigid body의 전체 collision mesh를 감싸는
body-local box와 현재 tensor FK pose로 만든 world OBB를 사용해 Robot–Cube와 Robot–Board를
15축 SAT로 검사한다. 이 검사는 실제 mesh보다 보수적이어서 false positive는 후보 재샘플링만
늘리고, tensor teleport 직후 stale한 PhysX scene-query broadphase에 의존하지 않는다. IK 반복은
joint write가 invalidation한 link-pose buffer의 kinematic refresh만 사용하며 physics scene을 반복
forward하지 않는다. 실패 후보는 Episode 시작 전에 좁은 offset 범위 안에서 다시 샘플링된다.
Reset 직후 observation은 이전 episode의 backend
contact/wrench를 재사용하지 않고 첫 정상 physics substep 전까지 18D tactile과 6D wrench를
0으로 mask한다. 첫 live F/T 표본부터는 Hand 자중을 포함한 uncancelled reaction이 들어간다.
센서 warm-up용 학습 transition이나 부분 적용 action 없이 첫 정책 명령부터 전체 decimation에
적용된다.

원본 Inspire collision mesh는 정상적인 인접 knuckle끼리도 겹치므로 package-local
articulation은 source hand 설정과 같이 self-collision을 끈다. Robot–Cube 및 Robot–Board
충돌은 그대로 활성화되어 있고 reset IK는 두 외부 충돌에 빠른 proxy clearance와 전체
collision mesh를 포함하는 보수적 live-FK OBB 검사를 모두 적용한다. Hand 내부 self-collision까지 검증하려면 인접
pair별 collision filtering을 갖춘 자산으로 교체해야 한다.

학습 결과, IK 채택률, reset 무접촉률, force 부호/오차, PPO 수렴성은 smoke/학습을 실제로
실행해 별도로 기록해야 한다. 구현 존재 자체를 feasibility 성공으로 간주하지 않는다.
