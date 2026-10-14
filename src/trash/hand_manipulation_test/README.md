# Hand Manipulation Test — Reaching

`Isaac-Hand-Manipulation-Test-v0`는 UR5e–Axia80–Inspire Hand를 지정한 초기 관절
상태에서 시작시켜 Target에 도달하도록 학습하는 Isaac Lab 환경입니다.
Cube와 선반은 없으며, Target은 episode 동안 고정된 위치와 충돌 없는 마커로 표현합니다.
F/T와 palm 측 17개 물리 접촉 센서를 유지하고 손등 센서는 사용하지 않습니다.

`Isaac-Hand-Manipulation-Contact-v0`는 이 환경을 상속한 table–Cube 접촉 환경입니다.
같은 `train.py`, `play.py`, `smoke_env.py`에 `--task`로 선택합니다. 기본 task는 기존
`Isaac-Hand-Manipulation-Test-v0`입니다.

`Isaac-Hand-Manipulation-Push-v0`는 Contact 환경을 상속해 Cube를 Table 위에서
지정한 방향과 거리만큼 미는 환경입니다. `train.py`와 `play.py`는 같은 `--task`
선택을 사용하며, Push smoke는 `smoke_push_env.py`로 실행합니다.

## 환경 계약

- 초기 팔 관절: `shoulder_pan=0`, `shoulder_lift=-2.2`, `elbow=2.2`,
  `wrist_1=0`, `wrist_2=1.57`, `wrist_3=0.785` rad. 초기 속도는 0입니다.
- 초기 Hand synergy는 `(0.5, 0.5)`이며, 실제 Inspire 12개 관절과 제어 목표에 함께
  적용합니다. 두 값은 공통 굽힘과 독립 thumb1의 openness입니다.
- EEF는 `inspire_base_link`에서 `(0, 0.05, 0.10)m` 이동한 기존 가상 제어점 C입니다.
- 팔 action은 **현재 EEF 좌표축 기준 6D Cartesian delta**, Hand action은 2D synergy로
  총 8차원입니다. OSC 중력 보상을 켜고 기존 stiffness `100`, damping ratio `1`,
  병진 scale `(0.012, 0.012, 0.008)m`, 회전 scale `(0.06, 0.06, 0.06)rad`를 유지합니다.
- Target은 환경 원점 기준 `X=[-0.77,-0.58]`, `Y=[-0.22,0.22]`, `Z=1.08m`에서
  reset마다 균등 샘플링합니다. 로봇 초기 자세를 Target 옆으로 이동하는 IK는 없습니다.
- Physics dt는 `0.01s`, decimation은 `2`, episode는 `10s`입니다. 도달 후에도 계속
  제어하며 timeout에서 다음 episode를 시작합니다.

EEF 프레임과 Target 위치 마커는 기본으로 표시합니다. `scene.ee_frame`의
`FrameTransformerCfg`는 `base_link`를 기준으로 `inspire_base_link`에 C offset을
적용해 실제 제어점의 좌표축을 추적합니다. RGB 축은 각각 X·Y·Z이며 scale은
`(0.05, 0.05, 0.05)`입니다. Target은 주황색 구로 표시합니다.
`--disable-markers`를 사용하면 두 마커를 모두 끌 수 있습니다.
좌표축 USD도 패키지에 포함되어 외부 자산 서버 연결 없이 표시할 수 있습니다.

## 관측과 리워드

Actor와 critic은 동일한 55차원 policy 관측을 사용합니다. 방향·Angle·명령 Distance
관측은 없습니다.

| 순서 | 항목 | 차원 | 표현 |
|---|---|---:|---|
| 1 | 팔 관절 위치 | 6 | 기존 `q / π` 표현 |
| 2 | 팔 관절 속도 | 6 | 기존 `dq / 3.14` 표현 |
| 3 | EEF 위치 변화 | 3 | 초기 EEF 좌표축 기준 현재 EEF 변위, m |
| 4 | EEF 회전 변화 | 3 | 초기 EEF 기준 rotation vector / π |
| 5 | 실제 Hand 상태 | 2 | synergy openness |
| 6 | Palm 헤더 + tactile bits | 18 | 고정 헤더 `0` + palm 17채널 |
| 7 | F/T | 6 | C 좌표계 force / 40N, moment / 4Nm |
| 8 | Initial Target 상대 위치 | 3 | 실제 로봇 **`base_link` 좌표계** 기준 고정 Target 위치, m |
| 9 | 이전 action | 8 | 기존 정규화 action |

두 위치 관측은 미터 값을 그대로 반환합니다. Target 관측은 실제 `base_link`의
world 위치를 빼고, `base_link`의 회전을 역변환한 3차원 위치입니다. 예를 들어
`base_link`의 X축 방향으로 10cm 떨어진 Target의 X 관측값은 `0.1`입니다.
고정된 베이스와 Target에서는 EEF가 이동하거나 회전해도 Target 관측은 일정합니다.
EEF 위치 변화 관측은 초기 EEF 좌표계를 기준으로 계산하고, 팔 action은 현재 EEF
좌표축을 기준으로 적용합니다.

Palm 센서는 손바닥 중앙 1개, 엄지 4개, 나머지 손가락 3개씩으로 총 17채널입니다.
기존 채널 순서와 `0.05N` binary threshold를 유지합니다. F/T는 measurement joint의
incoming wrench를 읽어 F→C 좌표와 모멘트 기준만 변환합니다. Hand 자중을 포함하며
tare나 중력 제거를 적용하지 않습니다. Reset 관측은 촉각·F/T를 0으로 반환하고 첫
정상 physics step부터 실제 값을 사용합니다.

리워드는 두 항만 사용합니다. `d`는 현재 EEF와 Target 사이의 3D 거리(m)입니다.

```text
eef_distance = 4 × exp(-d / 0.05)
action_rate  = -0.06 × (ΣΔa_arm² / 6 + ΣΔa_hand² / 2)
step_reward  = (eef_distance + action_rate) × 0.02
```

Action Rate는 실제 실행된 정규화 action을 비교합니다. Hand rate limit을 반영하고,
reset 후 첫 action의 벌점은 0입니다. RewardManager가 시간 적분을 한 번 수행합니다.
Target 1cm 이내 도달 여부는 `Metrics/target_position/reached`, 거리는
`Metrics/target_position/distance_m`에 기록하며 별도의 성공 보상·종료를 추가하지 않습니다.

## Table–Cube 접촉 환경

`Isaac-Hand-Manipulation-Contact-v0`는 기존 55D 관측·8D action·F/T·palm 센서·EEF
FrameTransformer·10초 episode를 상속합니다. Table은 기존 선반 판과 같은
크기 `(0.36, 1.00, 0.04)m`, 중심 `(-0.75, 0, 1.03)m`이며 윗면 높이는 `1.05m`입니다.
이 중심은 선반이 있는 공통 부모 `ContactTaskCfg.table_center`에서만 지정하고
Push와 이후 파생 환경이 상속합니다. Cube 범위는 선반 중심에 대한 offset으로
계산하므로 중심을 바꾸면 함께 이동합니다.
한 변 `0.06m`, 질량 `1kg`의 dynamic Cube를 reset마다
`X=[-0.82,-0.63]`, `Y=[-0.22,0.22]m`에서 배치합니다.
Cube 중심의 초기 Z는 `1.08m`이며 Cube는 중력과 접촉에 따라 움직일 수 있습니다.

Target은 **reset 때의 Cube 중심**입니다. Cube가 움직여도 Target의 `base_link` 관측과
거리 보상은 이 초기 위치를 계속 사용합니다. 주황색 Target 마커도 초기 위치에 남습니다.
Cube를 배치한 뒤 bounded IK로 hand를 안전한 object-relative pose에 배치합니다.
실제 중앙 palm 표면점의 `H +Y` 법선이 초기 Cube 중심을 3D에서 바라보도록 하고,
허용 각도는 `0.05rad`입니다. Robot–Cube와 Robot–Table collision bounds를 검사하며
안전한 해가 없으면 reset을 명시적으로 실패시킵니다. 초기 Hand openness는 두 synergy
각각 `[0.90, 0.98]`에서 샘플링합니다.

Contact variant는 중앙 palm collider를 `H +Y` 방향으로 `35mm` 노출하여 실제 물리
접촉을 측정합니다. 접촉 threshold는 `0.01N`입니다. Table의 robot-filtered 센서는
38개 robot body를 검사하고 Cube의 정상 지지력은 제외합니다. 두 physics substep 중
어느 robot body pair라도 threshold 이상이면 episode를 종료합니다. Table failure는
timeout보다 우선하며 접촉 보상을 받지 않습니다.

리워드는 아래 세 항입니다. `d`는 EEF에서 **초기 Cube 중심**까지의 거리이고,
`palm_cube_contact`는 17개 물리 palm pad 중 하나라도 Cube와 접촉하는 동안 1입니다.
Carrier hand-link 접촉만으로는 접촉 보상을 받지 않습니다.

```text
eef_distance      = 4 × exp(-d / 0.05)
action_rate       = -0.001 × (ΣΔa_arm² / 6 + ΣΔa_hand² / 2)
palm_cube_contact = 0.02 × contact
step_reward       = (eef_distance + action_rate + palm_cube_contact) × 0.02
```

Hand action은 기존 absolute synergy 의미를 유지합니다. 초기 열린 손을 유지하려면
팔 action을 0으로 하고 Hand action을 `2 × 초기 실제 synergy − 1`로 설정합니다.
8D action을 모두 0으로 보내면 Hand는 openness `(0.5, 0.5)`로 이동합니다.
현재 EEF 기준 OSC는 매 policy step의 실제 자세에 Cartesian delta를 더해 목표를
설정합니다. 팔 action 0은 현재 자세를 다시 목표로 설정하므로 초기 world 자세를
계속 고정하는 명령은 아닙니다. Contact smoke는 매 step의 OSC 목표에 대한 실제
추적 오차가 `2cm`, `0.05rad` 이내인지 검사하고, 초기 자세 대비 누적 변화도 출력합니다.

```bash
# 네 sampling corner와 random reset, 부분 reset, 실제 Cube 이동, 500-step timeout
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_env.py \
  --task Isaac-Hand-Manipulation-Contact-v0 \
  --headless --device cuda:0 --num_envs 4 --steps 50 --debug-vis --check-timeout

# 접촉 센서 검증용 physics fixture: Contact task와 2개 환경으로 고정
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_contact_sensors.py \
  --headless --device cuda:0

# Contact PPO 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/train.py \
  --task Isaac-Hand-Manipulation-Contact-v0 \
  --headless --device cuda:0 --num_envs 4 --max_iterations 2

# Contact checkpoint 재생
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/play.py \
  --task Isaac-Hand-Manipulation-Contact-v0 \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/contact_model.pt
```

Contact PPO는 기존 runner를 상속하고 로그를 `logs/rsl_rl/hand_manipulation_contact`에
분리합니다. Smoke는 reset 시간과 검증을 포함한 일반 step 시간을 따로 출력하며,
일반 rollout에서 IK·OBB 검사 counter가 늘지 않는지도 확인합니다.

`smoke_contact_sensors.py`는 robot 상태를 고정하고 Cube에 제어된 하중을 적용하는
**센서 검증용 physics fixture**입니다. 학습된 정책의 물체 유지 성능은 별도로
평가해야 합니다. 실제 물리 접촉으로 아래 항목을 검증합니다.

- Cube의 Table 지지력 약 `9.825N`이 Robot–Table 종료 필터에서 제외됨
- 중앙 palm pad의 실제 접촉력과 관측 bit, Cube-filtered 접촉 항이 모두 ON이고,
  2회 연속 physics step에서 유지됨. 측정 접촉력은 약 `1.151N`, `1.180N`임
- Cube를 분리하면 palm bit와 접촉 항이 OFF/0으로 복귀함
- 실제 Robot–Table 충돌에서 `terminated=True`, `truncated=False`로 종료하며,
  timeout과 동시에 발생해도 Table failure가 우선함

`--verbose`를 추가하면 raw palm 17채널, Cube-filtered 17채널과 자세 진단을 출력합니다.

## Table 위 Cube Push 환경

Push 환경은 부모 Contact의 Table 중심 `(-0.75, 0, 1.03)m`와 크기
`(0.36, 1.00, 0.04)m`를 상속합니다. 이전 Push 중심보다 베이스에서 `25cm` 더
떨어져 있습니다. Cube는 reset마다 `X=[-0.71,-0.69]`, `Y=[-0.06,0.06]`, `Z=1.08m`에서
샘플링합니다. 방향 `θ`는 `[-10°,10°]` 또는 `[170°,190°]`, 거리 `L`은 `[0.20,0.30]m`
입니다. 기본 베이스 자세에서 `θ=0`은 world `+Y`, `θ=180°`는 world `−Y`이며,
world 방향은 `d=(-sinθ, cosθ, 0)`입니다. Goal은 `초기 Cube 중심 + L×d`로 episode
동안 고정합니다. Cube의 회전까지 고려한 반경과 `1cm` 여유를 적용해 시작점부터
Goal까지의 전체 직선 경로가 Table 안에 들어오는 명령만 샘플링합니다.

초기 pose는 실제 노출된 중앙 palm 표면점 기준으로 Cube 뒤 `14cm`, Cube 중심 위
`10cm`입니다. 높이도 부모의 안전 초기 offset을 상속하여 바깥쪽 손가락 자세에서
엄지가 선반에 겹치지 않도록 합니다. Hand의 `H +Y`는 수평 push 방향을 향하며,
손가락 `H +Z`는 양방향 모두 world `−X`를 법선의 직교 평면에 투영한 방향을
향합니다. 실제 FK에서도 손가락이 베이스에서 선반 쪽을 향하는지 검사합니다.
이 초기 자세에서는 오른쪽 밀기의 엄지는 위쪽, 왼쪽 밀기의 엄지는 아래쪽을
향합니다. 이후 정책 action은 기존처럼 손 자세를 제어할 수 있습니다.
초기 Hand openness는 `[0.90,0.98]`이며, Push의 OSC stiffness는
병진·회전 6축 모두 고정 `200`, damping ratio는 `1`입니다. 명령을 먼저
샘플링한 뒤 bounded IK와 모든 robot body의 Cube·Table `4mm` clearance 검사를
수행합니다. IK 시도 중에 방향이나 Goal을 다시 샘플링하지 않습니다.

Push의 Hand 두 synergy는 **`1=완전히 펴짐`** 기준 `[0.8,1.0]`으로 제한합니다.
제어 목표뿐 아니라 12개 실제 관절의 PhysX 위치 제한에도 해당 범위를 적용합니다.
정규화 Hand action `a∈[-1,1]`은 `u=0.9+0.1a`로 변환합니다. 따라서 action `−1`,
`0`, `+1`은 각각 openness `0.8`, `0.9`, `1.0`을 명령합니다. 초기 손을 유지하려면
`hand_action.synergy_to_action(hand_action.actual_synergy)`를 사용합니다. 실제 Hand
관측은 원래의 `[0,1]` openness 표현입니다. 이전 action 관측은 정규화한 입력값이며,
Action Rate는 rate limit 후 실행된 목표를 제한된 action 범위로 다시 정규화합니다.

관측은 기존 55D 항목을 모두 유지하고 아래 6D를 마지막에 추가한 **61D**입니다.

| 인덱스 | 항목 | 표현 |
|---|---|---|
| `55:58` | Push command | `[sinθ, cosθ, L]`, `L`은 m |
| `58:61` | 현재 Cube 위치 | 실제 `base_link` 좌표계의 현재 Cube 중심, m |

기존 Initial Target 관측은 초기 Cube 중심으로 고정되고, 마지막 현재 Cube 위치만
실제 물리 이동에 따라 변합니다. 별도의 위치 scaling이나 clipping은 적용하지 않습니다.
초기 Cube는 주황색, Goal은 녹색, Push 방향은 노란 화살표로 표시합니다.
EEF 좌표축을 포함한 모든 task 마커는 `--disable-markers`로 끌 수 있습니다.

유효한 side contact는 17개 palm pad 중 개별 pad에 `0.01N` 이상의 힘이 발생하고,
그 pad가 Cube에 가하는 힘과 palm 법선이 각각 push 방향에서 `30°` 이내이며,
같은 physics substep에 Cube–Table의 위쪽 지지력이 측정된 경우입니다.
Cube–Table 센서도 두 substep을 보존합니다. 서로 약한 pad 힘을 합쳐 threshold를
넘기지 않으며, 다른 pad의 반대 힘이 유효한 접촉을 지우지 않습니다.

리워드는 아래 항목으로 교체합니다. 거리 record는 접촉 여부에 관계없이 갱신하므로
같은 접근이나 전진을 왕복해 중복 보상을 받을 수 없습니다.

| 항목 | 가중치 | 동작 |
|---|---:|---|
| Approach | `0.05` | 첫 접촉 전 palm–Cube 측면 접근 record의 개선량 |
| Progress | `12` | 지지된 유효 접촉 중 아래 progress potential record의 증가량 |
| Backslide | `6` | 첫 접촉 이후 XY Goal 거리가 증가한 양 / `L`을 벌점으로 적용 |
| First contact | `0.02` | episode의 첫 유효 side contact에 한 번 지급 |
| Contact | `0.005` | 지지된 유효 side contact를 유지하는 동안의 rate |
| Alignment | `0.3` | live step에서 `−(1−dot(H +Y,d))/2` |
| Action rate | `0.001` | 기존 arm·Hand 정규화 action 변화 벌점 |
| Success | `2` | 성공 종료에 한 번 지급 |
| Failure | `2` | 실패 종료에 한 번 벌점 |

Progress는 명령 거리로 정규화한 전진량을 포화 곡선으로 변환합니다. `D`는 현재
Cube에서 Goal까지의 XY 거리, `L`은 명령 거리이며 sigma는 무차원입니다.

```text
p = clamp(1 - D / L, 0, 1)
sigma = 0.5
phi(p) = (1 - exp(-p / sigma)) / (1 - exp(-1 / sigma))
progress = 12 × max(phi(p_current) - phi(p_record), 0)
```

이는 manager 적분까지 반영한 step의 progress 기여도입니다. 25cm 명령에서 첫
1mm를 유효하게 밀면 기존 `0.024` 대신 약 `0.11058`을 받습니다. 작은 초기 전진에
더 민감하며, episode 전체 progress 보상은 최대 `12`입니다. 접촉 유지의 10초
최대 보상은 `0.05`로 유지합니다.

개선량과 일회성 보상은 manager의 시간 적분을 고려해 한 번만 반영합니다.
Contact·Alignment·Action rate는 `step_dt=0.02s`로 적분하는 rate입니다.
성공하려면 실제 유효 접촉 기록이 있어야 하며, 지지된 Cube가 Goal에서 3D 거리
`1cm` 이내, 속도 `0.02m/s` 이하를 `0.20s` 연속 유지해야 합니다.
Robot–Table 접촉, 실제 회전된 Cube footprint의 Table 이탈, 낙하, 비정상 수치는
실패입니다. 종료 우선순위는 Failure → Success → 10초 Timeout입니다.

```bash
# 24개 방향·Cube corner 조합, random/부분 reset, 센서, 500-step timeout
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_push_env.py \
  --headless --device cuda:0 --num_envs 4 --steps 50 --debug-vis --check-timeout \
  --skip-physical-fixture

# 초기 manager/reset 검증만 실행
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_push_env.py \
  --headless --device cuda:0 --num_envs 4 --skip-physical-fixture

# 실제 Push fixture만 재실행하며 Cube 자세·지지력·접촉 진단 출력
# 아래 기본 재질에서의 힘 제한을 함께 확인할 수 있음
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_push_env.py \
  --headless --device cuda:0 --num_envs 2 --physical-only --verbose

# 별도 prescribed-robot geometry fixture: actuator 힘 검증과 구분
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_push_env.py \
  --headless --device cuda:0 --num_envs 2 --physical-only --prescribed-robot-fixture

# Push PPO 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/train.py \
  --task Isaac-Hand-Manipulation-Push-v0 \
  --headless --device cuda:0 --num_envs 4 --max_iterations 2

# Push checkpoint 재생
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/play.py \
  --task Isaac-Hand-Manipulation-Push-v0 \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/push_model.pt
```

`smoke_env.py --task Isaac-Hand-Manipulation-Push-v0`도 전용 smoke로 연결합니다.
Scripted fixture는 현재 EEF 좌표계의 실제 OSC action으로 palm을 이동시켜 물리
접촉·Cube 이동·정지를 검사합니다. 초기 방향과 palm 높이는 같은 action으로
servo합니다. `--prescribed-robot-fixture`는 별도의 IK 경로로 실제 robot 관절 상태를
지정해 PhysX가 Cube 접촉과 이동을 계산하도록 합니다. 이 모드는 fixture의 IK·OBB
작업을 따로 출력하며 actuator의 추진력을 검증하지 않습니다. 두 모드 모두 외부
Cube 추진력이나 합성 접촉값을 사용하지 않습니다. 학습된 정책의 성공률은 별도로
평가해야 합니다. Fixture는 `1.5cm` 전진 후 정착을 검사하며, 명령의 전체 거리 `L`에
대한 정책 평가와 구분합니다.
이 수평 fixture는 초기 높이를 유지하므로 새 `10cm` 시작점에서 Cube로 내려가는
접근 동작을 검증하지 않습니다. 새 배치의 초기화·관측·센서 검증에는
`--skip-physical-fixture`를 사용합니다.
Push 로그는 `logs/rsl_rl/hand_manipulation_push`에 저장합니다. 기존 55D checkpoint는
61D Push 환경과 호환되지 않으므로 새로 학습해야 합니다.
이전 61D Push checkpoint는 로드할 수 있지만, 선반 위치와 초기 roll이 바뀌었으므로
새 배치에서의 밀기 정책은 다시 학습합니다.

2026-10-05 공통 배치 변경 후 Contact 44개와 Push 424개 초기 자세에서 실제
손가락 방향·IK·전체 로봇의 `4mm` 충돌 여유를 확인했습니다. 2048개 환경에서
부분 reset 4,000회와 경계 초기화 192회, 500-step timeout·손 제어 범위·마커,
PPO 2 iteration 및 기존 `model_4850.pt`의 50-step 재생을 통과했습니다.
CPU 테스트는 111개 통과했고, 결과는
`reports/hand_manipulation_push/2026-10-05_far_workspace`에 저장합니다.
이 재생 검증은 체크포인트 실행 경로를 확인하며 새로운 배치의 밀기 성공률을
검증하지는 않습니다.

Push PPO의 기준은
`example/Sweep-Policy/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py`입니다.
`36` rollout steps, 최대 `90000` iteration, `50`마다 저장, ELU `[256,128,64]`,
초기 Gaussian std `1.0`, entropy `0.005`, gamma `0.98`, lambda `0.95`, PPO epochs
`8`, minibatches `4`, adaptive learning rate `0.001`, desired KL `0.02`입니다.
환경·로그 이름은 Push 이름을 사용하고, 설치된 RSL-RL의 actor/critic API에 맞춰
기존 `init_noise_std`를 Gaussian `init_std`로 옮기는 등 같은 의미의 값을 선언합니다.
PPO 설정을 바꿀 때도 이 기준을 유지하며, `tests/test_push_ppo.py`가 기준 파일과
수치 및 API 기본값을 비교합니다. 별도 std 상한은 추가하지 않습니다.

Headless 학습에서는 Target·Goal·방향·EEF 마커 업데이트를 기본으로 끕니다.
재생 화면의 기본 마커는 유지하며, headless 디버깅에는 `--enable-markers`를 쓸 수
있습니다. Reset IK는 시작 시 로봇 joint frame을 캐시합니다. 작은 reset 배치는
CPU의 NumPy에서 FK/Jacobian과 IK를 한 번에 계산하며, iteration마다 GPU와 PhysX에
관절을 쓰지 않습니다. 큰 배치는 기존 PhysX 계산을 사용합니다. 수렴한 후보는 실제
PhysX 자세와 전체 Robot–Cube/Table OBB 검사로 인증합니다. 정상 step에서는 IK와
collider mesh 순회를 실행하지 않습니다.

작은 reset 배치의 cached IK는 float64 계산 결과를 PhysX의 float32 관절 상태에
저장하는 경계 오차를 피하도록 내부에서 `2.85mm / 0.0475rad`까지 수렴시킵니다.
저장한 실제 자세는 기존 `3mm / 0.05rad`로 검사하고, 필요한 경우 seed당 80회
예산 안에서 한 번만 실제 FK 기반 보정을 수행합니다. 충돌 여유는 계속 `4mm`이며,
Cube·목표·Hand 샘플을 재추첨하지 않습니다. 실패 진단에는 각 seed의 수렴 오차와
충돌 검사 실행 여부·겹친 body를 별도로 기록합니다.

학습과 같은 2048개 환경의 큰 월드 좌표에서 반복 부분 reset을 검증할 수 있습니다.
기존 실행의 `source/hand_manipulation_test/mdp/push_events.py`를
`--baseline-source`로 지정하면 최초 실패의 동일한 목표·Hand 상태를 두 solver로
비교하고 fixture를 출력 경로 옆에 저장합니다. 실제 PhysX 좌표에서 `3mm` 수치
경계 사례도 별도로 검증합니다. `--fixed-random-resets`를 추가하면 이 경계 비교
이후 무작위 reset에는 수정된 solver만 사용합니다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/stress_push_reset.py \
  --headless --device cuda:0 --num_envs 2048 --samples 1000 --batch-size 8 \
  --output /tmp/push_reset_stress.json
```

2026-10-04 검증에서 실제 env 1927의 원점 `(-48.75,37.5,0)m`을 사용한 경계
사례는 기존 cached 오차 `2.99964mm`에서 실제 PhysX 오차 `3.00036mm`로 바뀌어
거부됐고, 수정 후 실제 오차 `1.12970mm`와 전체 로봇 OBB 검사를 통과했습니다.
수정된 solver의 부분 reset 8,000회, 경계 조합 192회, PhysX backend로 전환되는
33행 배치와 `model_50.pt` 재개 학습 10 iteration을 통과했습니다. CPU 회귀
테스트는 103개 통과했습니다. 결과는 `reports/hand_manipulation_push/2026-10-04_reset_fix`에
저장합니다. 이 수정은 기존 최신 Push 체크포인트와 호환됩니다.

검증 assertion을 제외한 step/reset 성능은 다음 명령으로 분리해 측정합니다.
`--legacy-reset`으로 기존 PhysX FK 기반 IK와 같은 조건에서 비교할 수 있습니다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/profile_push_env.py \
  --headless --device cuda:0 --num_envs 2048 --steps 100 --reset-repeats 3 \
  --output /tmp/push_profile.json
```

실제 학습처럼 종료 시점을 분산하려면 `--stagger-timeouts`를 추가합니다. RTX 3090,
2048개 환경, 마커 OFF의 같은 36-step hold rollout에서 138개 reset을 포함한 수집
시간은 기존 `38.39s`에서 최종 최적화 후 `11.42s`로 줄었습니다(약 `3.36배`). Reset 없는
rollout은 약 `2.85s`입니다. 이는 제어된 성능 측정이며 policy 행동과 실제 접촉·실패
빈도에 따라 학습 시간은 달라집니다.

관측은 계속 61D지만 Hand action의 물리적 의미와 리워드가 바뀌었으므로, 이전 Push
checkpoint로 기존 실험을 이어가는 대신 새 run으로 학습합니다.

아래 검증 결과는 stiffness `100`을 사용한 이전 구현의 측정입니다.
4개 환경의 GPU smoke로 최대 거리의 24개 방향·XY corner 조합, 최소 거리 2개,
random reset 4회, 부분 reset, 61D 관측, 마커와 500-step timeout을 확인했습니다.
최대 OSC 추적 오차는 `0.048mm`, `0.00098rad`였으며, 4개 환경의 reset은 약
`0.5–2.7s`, 검증을 포함한 일반 step은 평균 `63ms`였습니다.
Push PPO 2 iteration 학습과 checkpoint의 2개 환경·50-step 재생도 통과했습니다.

기본 재질을 유지한 OSC 직선 push fixture에서는 지지된 side contact와 방향 정렬을
확인했지만, 접촉력이 약 `1.3N`으로 약 `7.4N`의 정지 마찰을 넘지 못해 `1.5cm`
이동 검사를 통과하지 못했습니다. 아래 임시 대조 실험에서는 두 방향 모두 실제
OSC push 후 35 step 동안 정착했습니다.

| 임시 변경 | 두 방향의 실제 이동 | Push step 수 |
|---|---|---:|
| Table·Cube 마찰만 static `0.10`, dynamic `0.08` | `1.51cm`, `1.55cm` | `136` |
| 병진 stiffness만 `100→800`, 회전 stiffness `100`·기존 마찰 유지 | `1.56cm`, `1.60cm` | `51` |

두 대조 실험의 마찰·gain 변경은 기본 환경에 반영하지 않았습니다.
기본 재질을 유지한 prescribed-robot geometry fixture는 130 step에서 두 방향의 실제
Cube 이동 `1.52cm`, `1.54cm`를 확인했고, hand를 `6mm` 물린 뒤 35 step 동안
정착했습니다. 이 결과는 실제 접촉·전진 이력·정착 경로의 검증이며 OSC 추진력
검증 결과와 구분합니다.

현재 stiffness `200`과 제한된 Hand 범위로 다시 수행한 350-step 직선 OSC 시험에서도
양방향 side contact는 확인됐으나, 최대 전진은 각각 약 `0.0022mm`, `0.0289mm`였고
`1.5cm` 이동 검사는 통과하지 못했습니다. 이번 설정의 밀기 힘이 충분하다는 검증은
아직 확보되지 않았으며, 위 마찰·질량은 기존 값을 유지합니다.

현재 설정의 GPU manager smoke는 24개 방향·XY corner, 최소 거리 2개, random/부분
reset, cached FK/Jacobian과 실제 PhysX 비교, 500-step timeout을 통과했습니다.
두 Hand action endpoint를 50 step씩 실행해 12개 PhysX 관절 제한과 실제 openness
범위도 확인했습니다(측정 범위 약 `0.8000~0.9965`). 2048개 환경의 PPO 2 iteration은
rollout 수집 `12.19s`, `12.08s`, 전체 학습 호출 `27.95s`였으며 checkpoint를 저장했습니다.
이는 실행 경로 검증이며 학습된 밀기 성공률을 입증하는 실험은 아닙니다.

## Push v1 상속 환경

`Isaac-Hand-Manipulation-Push-v1`은 기존 Push v0 환경과 설정을 상속하는 별도
환경입니다. 환경 클래스는 Manager의 기본 step·reset 흐름을 사용합니다.
관측은 기존 61D 뒤에 누적 병진 목표 오차 3D를 추가한 **64D**, action은 기존
**8D**입니다. 관측 차원이 달라 Push v0 체크포인트와 호환되지 않으므로 새로
학습해야 합니다.

병진 action은 현재 EEF 제어점 C의 좌표축으로 목표 위치를 누적하고, 실제 C에서
목표까지의 오차 norm을 `0.06m`로 제한합니다. OSC stiffness는 `200`을 사용합니다.
v1은 실제 Cube→palmar pad 접촉 반력을 예측하는 implicit Cartesian impedance를
사용합니다. `200`은 operational inertia를 곱하는 가속도 gain 대신 Cartesian
spring gain으로 적용하며, measured 반력은 `20N`·moment `2Nm` norm 제한과
EMA `0.5`를 거쳐 제어에만 사용합니다. 접촉 필터는 Cube–palm pair로 제한하고
F/T 관측의 Hand 자중과 실제 tactile·reward 측정값을 유지합니다. 부분 reset은
해당 행의 controller 접촉 이력도 초기화합니다.
회전 action은 reset 때 실현한 C 자세에 대한 제한된 잔차로 해석합니다. 회전
action이 0이면 부하 중에도 reset에서 실현한 손 자세를 목표로 유지합니다. 이는 v0의
현재 자세 기준 회전 증분과 의미가 다릅니다. 추가된 마지막 3D 관측은 현재 C
좌표축에서 표현한 실제 C→누적 목표 위치 오차이며 미터 값을 그대로 전달합니다.

기존 progress `12`, 접촉 유지 `0.005` 가중치를 사용하고, palm 법선 정렬
`0.3`과 reset roll 유지 `0.1` 항은 모두 0 이하의 벌점으로 계산합니다. 원본
policy 입력이 `[-1,1]`을 벗어난 양의 제곱 평균에는 `0.02` 가중치로 벌점을
줍니다. PPO 수치 설정은 유지하면서 범위 밖 action 입력을 억제하는 환경 항입니다.

기존 접근 record 보상에 더해, 현재 palmar 접촉 목표까지의 거리를 매 step
`-거리(m)`로 계산하는 가중치 `1`의 연속 벌점을 적용합니다. 접근 후 다시
멀어지거나 접촉을 잃어도 비용이 생기며, 가까이 머무르는 양의 보너스는 없습니다.
실제 EEF 제어점 **C**의 Table 상판 기준 높이가 `0.15m`를 넘으면
`-5 × max(높이−0.15, 0)`의 선형 벌점을 적용하고 `0.25m` 이상이면 실패합니다.
RewardManager가 각 rate에 dt를 한 번 적용하며 실패는 성공과 timeout보다 우선합니다.

v1은 공통 테이블 중심 `(-0.75, 0, 1.03)m`과 크기 `(0.36, 1.0, 0.04)m`를
상속합니다. 베이스 쪽 Table 가장자리는 `X=-0.57m`입니다. Cube 배치 bounding box는
`X∈[-0.73,-0.67]m`, `Y∈[-0.03,0.03]m`입니다. 방향 `±10°` 또는 `170~190°`와
거리 `0.20~0.30m`를 각각 한 번 균등 샘플링한 뒤 경로의 X 중점을 `-0.70m`로 두고
초기 X에 `±0.002m` jitter를 적용합니다. 명령 방향과 거리를 재추첨하지 않습니다.

손바닥 법선 H+Y는 수평 밀기 방향을 향하고, 손가락 축 H+Z는 world−X를 그 법선의
수직 평면에 투영한 방향을 사용합니다. 양쪽 모두 손가락이 베이스 반대쪽을 향하며
오른쪽의 엄지는 위, 왼쪽의 엄지는 아래를 향합니다. 초기 palmar 기준점은 Cube 뒤로
`0.14m`, 위로 `0.10m`에 두고 공통 위치 jitter `±(0.004, 0.004, 0.003)m`를 사용합니다.
접촉 접근 높이 설정 `0.025m`는 reset 높이와 별개입니다. 오른쪽 접촉 기준점은
기존 central palm, 왼쪽 접근 기준점은 실제 좌측 접촉력이 확인된 palmar **thumb4**
센서의 노출 면을 사용합니다. 접촉 판정은 기존 17개 palmar pad의 실제
Cube-filtered force를 사용합니다.
mesh 정점은 시작 시 한 번 cache하고 reset에서 실제 손 자세에 맞는 면을 선택합니다.
정상 step은 해당 점을 실제 sensor body pose로 변환하므로 thumb joint 변형도
접근 거리와 C 목표 변환에 반영되며 mesh를 다시 읽지 않습니다. C 높이 검사는
이 접촉 기준점과 분리하여 실제 제어점 C로 계산합니다. v1 전용 spawner는 carrier에
가려진 기존 palmar thumb3·thumb4의 collision 면만 H+Y 방향으로 `12mm` 노출합니다.
body·joint·mass·visual과 기존 task 자산을 유지하고, 실제 filtered thumb4 접촉력도
물리 probe에서 확인했습니다. 센서값을 만들어 접촉을 판정하지 않습니다. 실제 wrist_2의
`sin(q)>0.15`와 actual outward cosine `>0.25`를 reset 최종 검사에 적용하며,
동일한 물리 방향과 wrist 분기를 정상 rollout에서도 검사합니다. 각도의 `2π`
표현 차이는 같은 분기로 취급합니다. 최대 3개 seed, seed당 80 iteration,
위치 `3mm`·회전 `0.05rad` 허용오차와 전체 로봇의 `4mm` Cube·Table 여유를 유지합니다.

IK의 nominal arm seed는 v1의 먼 작업영역에 맞춰 방향별로 선택합니다.
`(shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)` 순서로 오른쪽은
`(0.179, 0.154, -1.560, -4.876, 1.750, 1.571)`, 왼쪽은
`(-0.615, 0.159, -1.625, -4.818, 0.955, -1.571)`을 사용합니다. 기존 3개 seed
offset 목록과 실패 행만 재시도하는 정책은 유지합니다. 이 anchor는 reset 대상 행의
팔 기본값에만 임시 적용하고 성공·실패 후 원래 모델 기본값을 복원합니다.
Cube 배치와 angle·distance는 고정된 채 IK를 수행합니다.

현재 먼 배치와 outward 자세의 GPU core smoke는 sampling corner와 500-step
timeout을 통과했습니다. 최신 Thumb4 기준점·옆방향 중심 정렬·Cube 이동 추종
fixture에서 실제 OSC로 **양방향 18cm 이상** 밀기를 확인했습니다. 다만 좌측
한 행의 실패 종료로 전체 `0.20m` 목표를 모든 행에서 안정적으로 완료하는
검증은 실패했습니다. 별도 fixture의 오른쪽 두 행은 `0.20m` 목표와 정지 후
성공 종료를 확인했지만, 좌측의 전체 목표 안정성은 확인하지 못했습니다. 최신 Thumb4
fixture의 별도 짧은 밀기 검증은 양쪽의 `15~31mm` 실제 변위와 `25~35 step`
연속 정지를 통과했습니다. 이때 일반 step에서 IK 계산 횟수도 증가하지 않았습니다.

높이 종료는 4개 환경·58 step GPU probe를 통과했습니다. 실제 C의 상판 기준
높이 `0.25129~0.25232m`에서 `terminated=True`, `truncated=False`였고,
높이 벌점 `-5 × max(높이−0.15, 0) × 0.02`와 terminal 양의 task 보상 0을 확인했습니다.

2048개 환경의 fresh PPO 2 iteration과 checkpoint 생성도 통과했습니다.
첫 rollout 수집은 `6.54s`였으며 checkpoint는
`logs/rsl_rl/hand_manipulation_push_v1/2026-10-06_12-42-54_v1_fixed_validation/model_1.pt`입니다.
이 checkpoint의 100-step 재생도 통과했습니다. 이 결과는 실행 경로와 실제
스크립트 물리 검증이며, 학습 수렴이나 학습된 정책의 성공률을 증명하지 않습니다.
이전의 가까운 배치 결과를 현재 설정의 검증 결과로 사용하지 않습니다. 설정과
리워드 의미가 달라졌으므로 기존 64D v1 checkpoint도 새 설정으로 다시 학습해야 합니다.

PPO는 `PushPPORunnerCfg`를 상속하며 Sweep-Policy `_02`의 모든 수치와 모델
설정을 유지합니다. 로그 이름만 `logs/rsl_rl/hand_manipulation_push_v1`로
분리합니다.

```bash
# Push v1 관측·부분 reset·센서·timeout 계약 검증
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_env.py \
  --task Isaac-Hand-Manipulation-Push-v1 \
  --headless --device cuda:0 --num_envs 4 --steps 50 \
  --skip-physical-fixture --check-timeout --debug-vis

# Push v1 학습: 최대 iteration 등은 Sweep-Policy _02 기본값 사용
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/train.py \
  --task Isaac-Hand-Manipulation-Push-v1 \
  --headless --device cuda:0 --num_envs 2048

# 새 Push v1 checkpoint 재생
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/play.py \
  --task Isaac-Hand-Manipulation-Push-v1 \
  --device cuda:0 --num_envs 1 --checkpoint /absolute/path/to/push_v1_model.pt
```

`smoke_env.py`는 v1 선택 시 전용 `smoke_push_v1_env.py`를 실행합니다.
`--check-boundaries`는 각 명령 방향·길이에 조건화된 실제 sampling 범위의 XY
corner를 검증하며 경계값을 출력합니다. 초기·최종 Cube footprint가 테이블
안에 남는지도 확인합니다.
전용 스크립트의 물리 fixture는 좌우의 실제 접촉 기준점과 live Hand offset을 사용해
OSC action으로 뒤쪽에서 접촉 높이까지
접근한 뒤 Cube를 밀고 멈춥니다. `--physical-only --physical-steps 350`으로
별도 실행할 수 있으며 Cube에 외력을 가하거나 센서값을 덮어쓰지 않습니다.
접근 시작점은 실제 pad의 normal 간격을 유지하고 옆방향 위치를 Cube 중심에
맞춥니다. 이후 Cube의 실제 옆방향 이동을 따라가며 모서리만 스치는 경로를 피합니다.
이 fixture는 학습된 정책의 성공률 평가와 별개입니다.

전체 명령 목표까지 밀고 실제 성공 종료를 확인하는 fixture는 다음과 같이
실행합니다. 자동 reset 전의 Cube 위치·속도·접촉 이력·지지와 정지 시간을
기록하며, 실패 또는 timeout은 검증 실패로 처리합니다. `--fixture-verbose`는
fixture 상태만 출력합니다.

```bash
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_push_v1_env.py \
  --headless --device cuda:0 --num_envs 2 \
  --physical-only --physical-goal --physical-goal-distance .20 \
  --physical-steps 450 --fixture-verbose
```

`--physical-angle-offset-deg -10` 또는 `10`을 추가하면 양방향 명령에 해당
각도를 적용하고 초기 Cube X를 실제 조건부 배치 공식으로 계산합니다. 기본값은
`0°/180°`이며, `--physical-goal-distance .30`으로 최대 목표 거리를 검사할 수 있습니다.

과거 Table 폭 `0.82m`·경로 X 중점 `-0.44m`·양방향 엄지 위 배치에서는
1,184개 reset, 64D/8D manager smoke와 500-step timeout, 실제 OSC `0.20~0.30m`
목표 fixture, 2048개 환경 PPO 2 iteration과 checkpoint 재생을 통과했습니다.
이 기록은 **이전 배치**의 실행 경로 검증이며, 현재 먼 배치나 학습된 정책의
성공률을 증명하지 않습니다. 당시
[`reset 기록`](../../reports/hand_manipulation_push/2026-10-05_push_v1/thumb_up_boundaries_certified.json)과
[`validation.json`](../../reports/hand_manipulation_push/2026-10-05_push_v1/validation.json)을 보존합니다.

## 설치와 실행

저장소 루트에서 Isaac Lab Python 환경을 사용합니다.

```bash
conda activate env_isaaclab
./IsaacLab/isaaclab.sh -p -m pip install -e src/hand_manipulation_test

# 실제 physics, 부분 reset, 센서, 마커, 500-step timeout 확인
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/smoke_env.py \
  --headless --device cuda:0 --num-envs 4 --steps 50 --debug-vis --check-timeout

# 짧은 학습 검증
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/train.py \
  --headless --device cuda:0 --num_envs 4 --max_iterations 2

# 본 학습
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/train.py \
  --headless --device cuda:0 --num_envs 4096

# checkpoint 재생: Target 및 EEF 마커가 기본 표시됨
./IsaacLab/isaaclab.sh -p src/hand_manipulation_test/scripts/play.py \
  --device cuda:0 --num_envs 1 --checkpoint /home/min/7cmdehdrb/grad/logs/rsl_rl/hand_manipulation_test/2026-10-02_19-08-51/model_1300.pt

# Isaac Sim 없이 math·sensor·MDP 계약 검증
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./IsaacLab/isaaclab.sh -p -m pytest -q \
  src/hand_manipulation_test/tests
```

학습 로그는 `logs/rsl_rl/hand_manipulation_test`에 저장됩니다. 기본 PPO 설정은 기존
`hand_manipulation_rl`의 기본 `rsl_rl_ppo_cfg_02`와 같고 실험 이름만 분리했습니다.
기존 57D/60D 환경의 checkpoint는 관측 차원이 달라 새 환경과 호환되지 않습니다.
이 환경에서 Target을 현재 EEF 기준으로 관측하던 기존 55D checkpoint도 관측의
의미가 달라졌으므로 `base_link` 기준 관측으로 새로 학습해야 합니다.

Isaac Lab 2.3.2 / Isaac Sim 5.1 / RSL-RL 5.0.1 / RTX 3090에서 최신 단위 테스트는
280개를 통과했고 `pxr`가 필요한 1개는 CPU 실행에서 skip했습니다. Reaching 환경은
2개 환경의 GPU smoke로 실제 `base_link` 변환,
EEF 이동·회전 중 Target 관측 유지, 부분 reset, 센서와 마커를 검증했습니다.
50-step zero-action hold의 최대 위치 변화는 약 `0.55mm`였습니다. Contact 환경은
4개 환경의 GPU smoke로 네 sampling corner, random reset, 부분 reset, 실제 Cube
이동 후 초기 Target 유지, 센서·마커와 500-step timeout을 검증했습니다.
50-step 열린 손 유지 시 최대 OSC 추적 오차는 `0.089mm`, `0.00169rad`였으며,
초기 자세 대비 누적 변화는 `4.19mm`, `0.0775rad`였습니다. 4개 환경의 reset은
약 `0.09–0.68s`, 검증을 포함한 일반 step은 평균 `55ms`였고 rollout 중 IK·OBB
counter는 변하지 않았습니다. PPO 2 iteration 학습과 생성한 checkpoint의
2개 환경·50-step 재생도 확인했습니다. 위 접촉 센서 fixture도 실제 GPU physics에서
ON→연속 ON→OFF와 Table 종료 우선순위 검증을 통과했습니다.

## 코드 구조

- `env.py`: 기본 stepping/reset을 사용하는 `ManagerBasedRLEnv` 환경 클래스
- `env_cfg.py`: Scene·Commands·Actions·Observations·Events·Rewards·Terminations 공통 설정
- `config/ur5e/reach_env_cfg.py`: UR5e–Inspire 자산·palm 센서·EEF FrameTransformer·action 설정
- `config/ur5e/contact_env_cfg.py`: Reach 설정을 상속한 Table·Cube·접촉 보상과 종료 설정
- `config/ur5e/push_env_cfg.py`: Contact 설정을 상속한 수평 Push·61D 관측·성공/실패 설정
- `push_v1_env.py`, `config/ur5e/push_v1_env_cfg.py`: Push v0를 상속한 얇은 환경 클래스와 64D v1 설정
- `mdp/commands.py`: episode-fixed Target, 초기 EEF pose, 도달 지표와 Target marker
- `mdp/observations.py`: 개별 관측 term과 센서 reset masking
- `mdp/events.py`: 초기 root·joint·제어 목표 복원
- `mdp/contact_events.py`: Cube 배치와 cached collision bounds를 사용하는 안전한 palm IK reset
- `mdp/push_commands.py`, `mdp/push_events.py`: reset 고정 Push 명령과 수평 palm pose 초기화
- `mdp/push_v1_commands.py`, `mdp/push_v1_events.py`: v1 조건부 배치·접촉 진단과 실제 outward 손가락·wrist 분기 검사
- `mdp/push_v1_actions.py`, `accumulation_math.py`: 누적 위치 목표와 reset 기준 회전 잔차, 목표 오차 관측
- `mdp/push_v1_controller.py`, `impedance_math.py`: 실제 Cube–pad 반력 예측과 implicit Cartesian impedance
- `mdp/push_v1_rewards.py`, `push_v1_math.py`, `height_math.py`: 법선·roll·접근 거리·입력 범위·EEF 높이 벌점과 양방향 outward palm 회전
- `push_state.py`, `mdp/push_rewards.py`: 접촉·전진 record·정착 이력을 공유하는 Push 상태와 보상
- `mdp/rewards.py`: 거리 보상과 부분 reset을 지원하는 Action Rate term
- `mdp/actions.py`: current-EEF OSC와 Hand synergy ActionTerm
- `geometry.py`, `sensors.py`, `action_math.py`: EEF 계산, palm/F/T reader와 pure Torch math
- `assets/`: 패키지 내부 상대 경로 USD와 F/T chain 조립
- `assets/push_v1_robot.py`: v1에서만 기존 palmar thumb3·thumb4 collision 면을 노출하는 spawner
- `agents/rsl_rl_push_v1_ppo_cfg.py`: 기존 Push PPO를 상속하고 v1 로그 이름을 분리
- `scripts/smoke_push_v1_env.py`: 관측·reset 계약과 실제 OSC 목표 도달 fixture

기존 task나 `example/Sweep-Policy`를 runtime import하지 않습니다. 로봇 자산과 필요한
저수준 구현을 이 패키지 안에 포함하며 Nucleus·ROS·절대 자산 경로에 의존하지 않습니다.
