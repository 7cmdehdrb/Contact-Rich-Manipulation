# UR5e–Axia80 wrist3 +10° 경사 cantilever 실험

이 문서는 폭을 줄인 thin-face-mounted tool을 UR5e wrist3에서 `+10°` 회전하고,
payload–tool 사이의 마찰을 높인 추가 실험을 설명한다. 이 실험은 기존 수평
cantilever 실험을 대체하지 않으며, 기존 CSV·로그·이미지를 수정하지 않는 별도
모델·실행 스크립트·결과 경로를 사용한다.

최종 10–500 g, 50환경 실행 결과는 `PASS`이다. 최종 판정의 기준 결과는 다음
디렉터리다.

```text
src/axia80_feasibility/results/
  tilted_cantilever_10deg_10g_to_500g_pose_checked/
```

## 형상과 경사 설정

조립 순서는 기존 cantilever 구성과 같다.

```text
UR5e wrist_3/tool0
  └─ Axia80 cylinder
       └─ fixed measurement joint (Axia80 중심)
            └─ 얇은 끝면으로 장착된 cantilever tool
                 └─ tool 넓은 면 중앙의 payload cube
```

- Axia80 근사 형상: 반지름 `0.041 m`, 높이 `0.0254 m`
- Tool 크기 `(X, Y, Z)`: `(0.012, 0.120, 0.240) m`
- Tool 질량: `0.20 kg`
- Payload: 한 변 `0.040 m`인 cube
- Axia80 중심에서 payload 중심까지의 길이 `d`: `0.1337 m`
- Payload 중심의 tool 법선 방향 오프셋 `h`: `0.0260 m`

기존 cantilever tool의 폭 `Y=0.160 m`를 `0.120 m`로 줄였다. 얇은 장착면은
`0.012 × 0.120 m`, 넓은 하중면은 `0.120 × 0.240 m`이다. 40 mm payload를
정중앙에 놓았을 때 초기 좌우 edge margin은 각각 40 mm이므로, 폭을 줄인 뒤에도
payload가 놓일 충분한 면적이 남는다.

기존 수평 자세에서 `wrist_3_joint`만 `+10°` 회전한다. 이 joint의 회전축은
Axia80/tool의 `+Z`, 즉 cantilever 길이축이다. 따라서 길이축은 계속 수평이지만,
넓은 면은 tool의 `Y` 방향으로 10° 기울어진다.

센서 좌표계는 다음과 같다.

- `+X`: tool 넓은 면의 법선
- `+Y`: 경사면의 내리막 방향
- `+Z`: 센서 축이자 cantilever 길이축

저장되는 wrench 순서는 `[Fx, Fy, Fz, Mx, My, Mz]`이며 기준점은 Axia80
실린더 중심이다.

## 마찰 및 미끄럼 안전 여유

경사 전용 scene은 payload material에 다음 값을 사용하고 friction combine mode를
`max`로 설정한다.

```text
static friction  = 2.0
dynamic friction = 1.5
```

10° 경사에서 정지에 필요한 이론상 최소 정지 마찰계수는 다음과 같다.

```text
mu_min = tan(10°) = 0.176327
mu_static / mu_min = 11.3426
```

따라서 정적 마찰 조건에는 충분한 여유가 있다. 다만 마찰계수만으로 성공을
가정하지 않고, 실행 중 실제 payload의 이동과 자세를 별도로 검사한다.

## 기대하는 힘과 모멘트 분산

정확히 10°인 강체 자세를 가정하면 질량 `m`, 중력가속도 `g`, lever arm `d`,
payload 법선 오프셋 `h`에 대해 tare-corrected wrench는 다음과 같다.

```text
Fx =  m g cos(theta)
Fy = -m g sin(theta)
Fz =  0

Mx =  d m g sin(theta)
My =  d m g cos(theta)
Mz = -h m g sin(theta)
```

즉, 수평 실험에서 주로 `Fx`와 `My`에 나타나던 하중이 경사 실험에서는
`+Fx/-Fy` 및 `+Mx/+My/-Mz`로 분산되어야 한다. `Fz`는 거의 0이어야 한다.
설정값 `theta=10°`, `g=9.81 m/s²`에서 nominal 기대 기울기는 다음과 같다.

| 성분 | 질량당 nominal 기울기 |
|---|---:|
| Fx | `+9.660964 N/kg` |
| Fy | `-1.703489 N/kg` |
| Fz | `0 N/kg` |
| Mx | `+0.227756 N·m/kg` |
| My | `+1.291671 N·m/kg` |
| Mz | `-0.044291 N·m/kg` |

## 실제 자세 기반 expected wrench

위 식은 설정된 joint angle이 하중과 관계없이 정확히 유지된다는 nominal 기준이다.
실제 시뮬레이션에서는 implicit arm drive가 tool 및 payload 하중을 받으면서 수십 분의
일 도 정도 자세가 변할 수 있다. 이를 센서 오차로 잘못 해석하지 않도록 최종 판정은
각 240 Hz sample의 실제 자세로 기대 wrench를 다시 계산한다.

각 환경과 각 sample에서 다음 계산을 수행한다.

```text
r_sensor = R_tool^T (p_payload - p_sensor)
F_sensor = R_tool^T [0, 0, m g]
M_sensor = r_sensor × F_sensor
W_expected = [F_sensor, M_sensor]
```

여기서 `R_tool`과 payload 위치는 해당 physics sample에서 읽은 값이다. sample별
`W_expected`를 측정 구간에서 평균하여 summary CSV의 `expected_Fx_N`부터
`expected_Mz_Nm`까지 기록한다. 따라서 판정은 단순한 설정각 10°가 아니라 실제
tool 방향, 실제 payload 위치 및 실제 lever arm을 반영한다.

최종 실행에서 하중에 따른 actual-pose expected 기울기와 측정 기울기는 다음과
같았다.

| 성분 | actual-pose expected | 측정 |
|---|---:|---:|
| Fx [N/kg] | `+9.671895` | `+9.667977` |
| Fy [N/kg] | `-1.640284` | `-1.635749` |
| Fz [N/kg] | `-0.005524` | `-0.007040` |
| Mx [N·m/kg] | `+0.219313` | `+0.218706` |
| My [N·m/kg] | `+1.293317` | `+1.292884` |
| Mz [N·m/kg] | `-0.043479` | `-0.043528` |

설정각만으로 환산한 측정 force distribution angle은 `9.6031°`였고, 실제 자세
모델의 기대각은 `9.6254°`였다. 두 값의 차이는 `-0.0223°`이다. Bending
distribution은 측정 `9.6013°`, 실제 자세 기대 `9.6243°`로 차이는
`-0.0230°`였다. 즉, 실제 자세를 기준으로 힘과 모멘트가 의도한 축들로 분산됐다.

## 안정화, 미끄럼 및 전도 검사

payload를 올린 뒤 모든 환경이 연속 안정 구간을 만족해야만 측정을 시작한다.
측정 구간 전체에서도 다음 조건을 매 sample 검사한다. 운동·위치·edge 조건을
하나라도 위반하면 새 CSV를 쓰지 않고 실행을 중단하며, tool 방향 조건은 최종
수용 판정에서 검사한다.

- payload–tool 상대 선속도 `<= 0.005 m/s`
- payload–tool 상대 각속도 `<= 0.05 rad/s`
- 목표 중심에 대한 위치 오차 `<= 0.004 m`
- 경사면 내리막 방향 변위 `<= 0.004 m`
- payload와 tool의 상대 자세각 `<= 2.0°`
- payload side edge가 tool edge 안쪽에 위치
- tool 넓은 면과 cantilever 축이 기대한 방향을 유지

최종 실행의 관측 결과는 다음과 같다.

- 모든 50개 환경 안정화: `1.495833 s`
- 최대 위치 오차: `0.002542 m`
- 최대 내리막 변위: `0.002539 m`
- 최대 payload–tool 상대 자세각: `0.176939°`
- 최소 side-edge margin: `0.037461 m`
- 최대 cantilever 축 world-Z 성분: `0.000561`

모든 안정성·미끄럼·전도·edge 조건을 통과했다.

## Headless 전체 실험 실행

저장소 루트에서 다음 명령을 실행한다. 아래 안전한 재실행 예시는 최종 기준 결과를
덮어쓰지 않도록 별도의 `tilted_cantilever_10deg_rerun` 디렉터리에 저장한다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-tilted-cantilever-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_tilted_cantilever_payload_test.py \
  --headless \
  --device cuda:0 \
  --strict \
  --output_dir src/axia80_feasibility/results/tilted_cantilever_10deg_rerun
```

기본값으로 `10, 20, ..., 500 g`의 50환경을 240 Hz에서 동시에 실행한다. Tool
형상이나 생성 URDF를 변경한 직후라면 위 명령에 `--force_conversion`을 한 번
추가해 USD를 다시 생성한다.

최종 기준 실행에 실제로 사용된 주요 옵션은 기본 시간 설정과 `--headless`,
`--device cuda:0`, `--strict`이며 출력 디렉터리만 다음과 같았다.

```text
src/axia80_feasibility/results/
  tilted_cantilever_10deg_10g_to_500g_pose_checked
```

이 기준 디렉터리에는 이미 검증된 결과가 있으므로 같은 `--output_dir`로 다시
실행하지 않는 것을 권장한다.

## Isaac Lab 화면으로 보기

`--headless`를 넣지 않으면 Isaac Sim 창이 열린다. 다음 명령은 50개 환경 대신
10–50 g의 5개 환경만 띄우고 관찰 시간을 늘리며, 기존 결과와 겹치지 않는 별도
디렉터리에 로그를 쓴다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-tilted-cantilever-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_tilted_cantilever_payload_test.py \
  --device cuda:0 \
  --strict \
  --mass_start_g 10 \
  --mass_stop_g 50 \
  --mass_step_g 10 \
  --warmup_seconds 3 \
  --min_settle_seconds 3 \
  --stable_window_seconds 2 \
  --sample_duration 10 \
  --output_dir src/axia80_feasibility/results/tilted_cantilever_10deg_visual
```

창은 측정 종료 후 자동으로 닫히고, 중간 종료는 터미널에서 `Ctrl+C`를 사용한다.
50개 환경을 화면에서 그대로 확인하려면 세 개의 `--mass_*` 옵션만 제거하면 된다.

## 최종 산출물

최종 `PASS` 디렉터리에는 다음 파일이 있다.

- `axia80_tilted_cantilever_summary.csv`: 질량별 actual-pose expected,
  raw/tare/tare-corrected 6축 평균·표준편차와 안정성 필드
- `axia80_tilted_cantilever_samples.csv`: 측정 구간의 240 Hz raw 및
  tare-corrected 6축 시계열
- `axia80_tilted_cantilever_response.png`: raw와 tare-corrected force/moment 및
  actual-pose expected 선
- `axia80_tilted_cantilever_metadata.json`: 모델, 인자, 판정 기준, metrics,
  `PASS`/`FAIL`
- `axia80_tilted_cantilever_run.log`: 실행 시각, 형상·마찰 설정, 핵심 metrics 및
  산출물 경로

그래프를 새 이름으로 다시 만들려면 다음을 실행한다. 기존 PNG를 보존하기 위해
출력 파일에 `_regenerated` 접미사를 사용한다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-tilted-cantilever-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/plot_tilted_cantilever_results.py \
  src/axia80_feasibility/results/tilted_cantilever_10deg_10g_to_500g_pose_checked/axia80_tilted_cantilever_summary.csv \
  --output src/axia80_feasibility/results/tilted_cantilever_10deg_10g_to_500g_pose_checked/axia80_tilted_cantilever_response_regenerated.png
```

## 기존 결과 보존

다음 기존 결과는 모두 그대로 남아 있다.

| 경로 | 내용 | 판정 |
|---|---|---:|
| `results/10g_to_500g/` | 최초 broad-face 장착, tool `0.240 × 0.160 × 0.012 m` | PASS |
| `results/cantilever_10g_to_500g/` | 기존 폭 0.160 m의 수평 cantilever | PASS |
| `results/cantilever_narrow_10g_to_500g/` | 폭 0.120 m의 수평 cantilever | PASS |
| `results/tilted_cantilever_10deg_10g_to_500g/` | 실제 자세 보정 전 초기 경사 진단 실행 | FAIL |
| `results/tilted_cantilever_10deg_10g_to_500g_pose_checked/` | sample별 실제 자세 기반 최종 경사 실행 | PASS |

초기 경사 결과의 `FAIL` 로그도 삭제하거나 덮어쓰지 않았다. 이는 최종 구현 전후의
판정 방식과 결과를 추적하기 위한 진단 기록이다. 최종 해석에는
`tilted_cantilever_10deg_10g_to_500g_pose_checked/`를 사용한다.

## 해석 범위

이 실험은 fixed joint와 rigid tool을 사용한다. 따라서 화면에서 tool의 탄성 처짐이
보이지는 않지만, Axia80 중심으로 전달되는 force/moment와 경사에 따른 축별 분산을
검증할 수 있다. 실제 센서의 노이즈, 대역폭, 포화, 온도 드리프트나 실제 그리퍼의
탄성 변형까지 재현하는 시험은 아니다.
