# UR5e–Axia80 고정 질량 Wrist Roll × Tilt 실험

이 실험은 기존 수평 cantilever, 질량 sweep, wrist3 단일 10° 실험을 변경하지
않고 추가된 독립 실험이다. 모든 환경에 동일한 `500 g` cube를 올리고 Wrist의
Roll과 Tilt만 바꾸면서 Axia80 좌표계의 `[Fx, Fy, Fz, Mx, My, Mz]` 변화를
측정한다.

최종 기준 결과는 다음 디렉터리에 있으며 판정은 `PASS`이다.

```text
src/axia80_feasibility/results/wrist_angle_sweep_500g_roll_tilt_10deg/
```

## 실험 구성

- Tool: thin-face-mounted cantilever, `(X,Y,Z)=(0.012,0.120,0.240) m`
- Payload: 모든 환경에서 동일한 `500 g`, 한 변 `0.040 m`인 cube
- Roll: 수평 기준 자세에 대한 `wrist_3_joint` 오프셋
- Tilt: 수평 기준 자세에 대한 `wrist_2_joint` 오프셋
- Roll/Tilt 각각: `-10°, -5°, 0°, 5°, 10°`
- 벡터화: `5 × 5 = 25`개 환경 동시 실행
- 정지/동마찰계수: `2.0 / 1.5`, combine mode `max`
- Axia80 근사 형상: 반지름 `0.041 m`, 높이 `0.0254 m`

여기서 Roll은 tool/cantilever 길이축인 sensor `+Z` 주위 회전이고, Tilt는
`wrist_2`를 움직여 길이축 자체를 위아래로 기울이는 회전이다. 기존 단일 10°
실험에서 wrist3 회전을 `tilt`라고 부른 명칭은 유지하지만, 이 신규 실험에서는
관례를 명확히 하여 wrist3를 **Roll**, wrist2를 **Tilt**로 정의한다.

## 기대되는 센서 출력

Roll을 `rho`, Tilt를 `tau`, 고정 payload 질량을 `m`이라고 하면 이상적인 센서
좌표계 반력은 다음과 같다.

```text
Fx =  m g cos(tau) cos(rho)
Fy = -m g cos(tau) sin(rho)
Fz = -m g sin(tau)
```

센서 중심에서 payload 중심까지의 위치를 `r=[h,0,d]`,
`h=0.0260 m`, `d=0.1337 m`로 두면 `M=r×F`이므로 다음 관계가 성립한다.

```text
Mx = -d Fy
My =  d Fx - h Fz
Mz =  h Fy
```

따라서 다음 패턴이 나타나야 한다.

- Roll 부호가 바뀌면 `Fy`, `Mx`, `Mz`의 부호가 바뀐다.
- Tilt 부호가 바뀌면 주로 `Fz`가 반대 방향으로 변하고 `My`도 변한다.
- `Fx`는 0° 부근에서 가장 크고 Roll/Tilt 절댓값이 증가하면 감소한다.
- 전체 힘의 크기는 모든 각도에서 거의 `m·g = 4.905 N`으로 유지된다.

## 실제 자세 기반 판정

Joint command만으로 계산한 위 식은 그래프 해석용 nominal 값이다. 최종 판정은
implicit drive의 미세한 변형과 payload의 실제 위치를 반영하기 위해 240 Hz의
각 sample마다 다음을 다시 계산한다.

```text
F_payload = R_tool^T [0, 0, m g]
r_payload = R_tool^T (p_payload - p_sensor)
M_payload = r_payload × F_payload
```

Tare 이후 tool 자세도 미세하게 달라질 수 있으므로 기대 tare-corrected 값에는
다음 보정까지 포함한다.

```text
W_expected = W_payload(sample)
           + W_tool(sample)
           - mean(W_tool during tare)
```

즉, CSV의 `expected_*`는 각 sample의 실제 tool 자세, payload 위치, tool 자중
변화를 반영한다. `nominal_expected_*`에는 command angle 기반 이상식을 별도로
기록한다.

## 미끄럼과 전도 검사

Roll과 Tilt가 함께 적용되면 미끄럼 방향이 tool 좌표계의 Y/Z 양쪽에 걸칠 수 있다.
따라서 단일 축 변위가 아닌 접선 변위 `sqrt(delta_y² + delta_z²)`를 매 sample
검사한다. 다음 항목 중 하나라도 한계를 벗어나면 측정 CSV를 쓰지 않고 실행을
중단한다.

- payload–tool 상대 선속도 `<= 0.005 m/s`
- 상대 각속도 `<= 0.05 rad/s`
- 목표 중심 위치 오차 `<= 0.006 m`
- Y/Z 접선 변위 `<= 0.006 m`
- payload–tool 상대 자세각 `<= 2°`
- payload가 tool 폭 방향과 길이 방향 edge 안쪽에 위치
- tool의 넓은 면 법선이 여전히 위쪽을 향함

실제 자세의 합성 경사에서 필요한 정지마찰계수도 계산하여 설정값 `2.0`보다 작은지
최종 판정에 포함한다.

## 최종 실행 결과

25환경이 모두 `1.495833 s`에 안정화됐고 전체 판정은 `PASS`였다.

| 지표 | 결과 |
|---|---:|
| 최대 힘 성분 오차 | `0.002369 N` |
| 힘 성분 RMSE | `0.001356 N` |
| 최대 모멘트 성분 오차 | `0.000445 N·m` |
| 모멘트 성분 RMSE | `0.000182 N·m` |
| 최대 힘 방향 오차 | `0.00762°` |
| 최대 모멘트 방향 오차 | `0.01571°` |
| 최대 접선 이동 | `0.000364 m` |
| 최대 payload–tool 상대각 | `0.07913°` |
| 최소 폭 방향 edge margin | `0.039636 m` |
| 최소 길이 방향 edge margin | `0.099880 m` |
| 실제 최대 합성 경사 | `14.3987°` |
| 실제 최대 필요 정지마찰계수 | `0.256732` |

측정된 전체 angle grid 범위는 다음과 같다.

| 성분 | 최솟값–최댓값 차이 |
|---|---:|
| Fx | `0.154312 N` |
| Fy | `1.703096 N` |
| Fz | `1.702803 N` |
| Mx | `0.227708 N·m` |
| My | `0.055322 N·m` |
| Mz | `0.046445 N·m` |

Roll/Tilt 변화가 의도한 축으로 분산되며 실제 자세 기반 정역학 예측과 일치했다.

## Headless 재실행

기존 결과를 덮어쓰지 않도록 새로운 출력 디렉터리를 지정한다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-wrist-angle-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_wrist_angle_sweep.py \
  --headless \
  --device cuda:0 \
  --strict \
  --output_dir src/axia80_feasibility/results/wrist_angle_sweep_rerun
```

기본값은 500 g, Roll/Tilt 각각 `-10 -5 0 5 10`이다. 다른 고정 질량이나 각도를
사용하려면 다음처럼 지정한다.

```text
--payload_mass_g 300
--roll_angles_deg -15 -10 -5 0 5 10 15
--tilt_angles_deg -15 -10 -5 0 5 10 15
```

안전을 위해 각 축은 ±30°로 제한된다. 결과 파일이 이미 있는 출력 디렉터리는
기본적으로 거부하며, 정말 교체하려는 경우에만 `--overwrite`를 명시해야 한다.

## Isaac Lab 화면으로 보기

`--headless`를 넣지 않으면 Isaac Sim 창이 열린다. 보기 편하도록 3×3 환경과 긴
관찰 시간을 사용하는 명령은 다음과 같다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-wrist-angle-visual-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_wrist_angle_sweep.py \
  --device cuda:0 \
  --strict \
  --roll_angles_deg -10 0 10 \
  --tilt_angles_deg -10 0 10 \
  --warmup_seconds 3 \
  --min_settle_seconds 3 \
  --stable_window_seconds 2 \
  --sample_duration 10 \
  --output_dir src/axia80_feasibility/results/wrist_angle_sweep_visual
```

## 생성 산출물

- `axia80_wrist_angle_summary.csv`: 25개 angle pair별 raw/tare/corrected,
  nominal/actual-pose expected 6축 값과 안정성 지표
- `axia80_wrist_angle_samples.csv`: 240 Hz raw/corrected/expected 6축 시계열
- `axia80_wrist_angle_response_heatmaps.png`: Roll×Tilt에 따른 측정 6축 heatmap
- `axia80_wrist_angle_error_heatmaps.png`: 측정−expected 6축 오차 heatmap
- `axia80_wrist_angle_center_slices.png`: Tilt=0° Roll sweep 및 Roll=0° Tilt
  sweep의 측정값과 expected 비교
- `axia80_wrist_angle_metadata.json`: 모델, 인자, 판정 기준과 전체 metrics
- `axia80_wrist_angle_run.log`: 실행 시각, 설정, 판정, 핵심 지표와 산출물 경로

그래프만 새 이름으로 다시 생성하려면 다음을 실행한다.

```bash
cd /home/min/7cmdehdrb/grad

/usr/bin/python3 \
  src/axia80_feasibility/scripts/plot_wrist_angle_results.py \
  src/axia80_feasibility/results/wrist_angle_sweep_500g_roll_tilt_10deg/axia80_wrist_angle_summary.csv \
  --output_prefix src/axia80_feasibility/results/wrist_angle_sweep_500g_roll_tilt_10deg/regenerated
```

## 해석 범위

본 결과는 rigid tool과 fixed measurement joint에서 좌표계, 부호, 정역학적
force/moment 전달이 의도와 일치하는지 검증한다. 실제 Axia80의 캘리브레이션 행렬,
노이즈, 포화, 대역폭, 온도 드리프트와 실제 그리퍼의 탄성 변형을 재현하는 시험은
아니다.
