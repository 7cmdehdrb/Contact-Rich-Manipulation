# UR5e–Axia80 thin-face cantilever 하중 실험

이 문서는 기존 broad-face 장착 실험을 대체하지 않고 추가된
`UR5e → Axia80 → thin-face-mounted tool` 실험을 설명한다. 기존 코드와
`results/10g_to_500g/axia80_payload_*` 결과는 그대로 유지되며, cantilever
실험은 별도 URDF·CSV·그래프·로그를 생성한다.

## 조립 형상과 좌표계

조립 순서는 다음과 같다.

```text
UR5e wrist_3/tool0
  └─ Axia80 cylinder
       └─ fixed measurement joint (Axia80 중심)
            └─ 0.012 m 두께의 얇은 끝면으로 장착된 tool
                 └─ tool 넓은 면 중앙의 payload cube
```

Axia80은 반지름 `0.041 m`, 높이 `0.0254 m`의 실린더로 근사한다.
Tool 크기는 센서 좌표계의 `(X, Y, Z)` 순서로
`(0.012, 0.120, 0.240) m`이며 질량은 `0.20 kg`이다. 초기 `0.160 m`
폭 모델보다 폭을 줄여 UR wrist 주변의 형상 겹침 여유를 확보했다.

- `+X`: tool 넓은 `YZ` 면의 법선. 기본 로봇 자세에서 world `+Z`, 즉 위쪽이다.
- `+Z`: Axia80 실린더 축이자 tool이 뻗는 cantilever 축. world에서 수평이다.
- `+Y`: `+X`와 `+Z`에 수직인 센서 축이다.

센서 끝면과 tool은 `0.001 m` 간격을 둔다. Tool의 얇은 장착면은
`0.012 × 0.120 m`, 넓은 하중면은 `0.120 × 0.240 m`이다. 센서
중심에서 tool 및 중앙 payload의 중심까지의 cantilever 길이는 다음과 같다.

```text
d = sensor_height / 2 + gap + tool_length / 2
  = 0.0254 / 2 + 0.001 + 0.240 / 2
  = 0.1337 m
```

Payload는 한 변이 `0.040 m`인 cube이며 기본 실험은 `10 g`, `20 g`, …,
`500 g`의 50개 환경을 동시에 실행한다.

## 기대하는 Fx/My 반응

저장하는 wrench의 순서는 `[Fx, Fy, Fz, Mx, My, Mz]`이고, 기준점은
Axia80 실린더 중심이다. 이 장착에서 수직 하중은 센서 `+X` 반력과
`+Y` 축 굽힘 모멘트를 만든다.

```text
Fx = m · g
My = m · g · d,  d = 0.1337 m
```

| Payload | 기대 Fx | 기대 My |
|---:|---:|---:|
| 10 g | 0.0981 N | 0.013116 N·m |
| 500 g | 4.905 N | 0.655799 N·m |

`raw_*`에는 0.20 kg tool 자중이 함께 포함되므로 payload가 없어도 약
`Fx=1.962 N`, `My=0.262319 N·m`이다. 기본 `Fx_N`, `My_Nm` 및 나머지
축 결과는 payload를 올리기 전 tool-only 평균을 뺀 tare-corrected 값이다.

## Headless 전체 실험

저장소 루트 `/home/min/7cmdehdrb/grad`에서 다음을 실행한다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-cantilever-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_cantilever_payload_test.py \
  --headless --device cuda:0 --strict \
  --output_dir src/axia80_feasibility/results/cantilever_narrow_10g_to_500g
```

처음 실행하거나 URDF가 바뀌어 USD를 반드시 다시 변환해야 하면
`--force_conversion`을 추가한다. 전체 240 Hz 시계열 CSV가 필요 없으면
`--no_sample_csv`를 추가할 수 있다.

## Isaac Lab 화면으로 보기

`--headless`를 빼면 Isaac Sim 창이 열린다. 아래 명령은 시각적으로 확인할
시간을 주기 위해 warm-up, 안정 구간, 측정 구간을 늘린 실행이다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-cantilever-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_cantilever_payload_test.py \
  --device cuda:0 --strict \
  --warmup_seconds 3 \
  --min_settle_seconds 3 \
  --stable_window_seconds 2 \
  --sample_duration 10 \
  --output_dir src/axia80_feasibility/results/cantilever_visual
```

50개 환경 대신 소수의 환경을 크게 보고 싶으면 위 명령에 예를 들어
`--mass_start_g 10 --mass_stop_g 50 --mass_step_g 10`을 추가한다. 실험이 끝나면
창은 자동으로 닫히며 중간 종료는 터미널에서 `Ctrl+C`를 사용한다.

## 별도 산출물과 그래프 재생성

Headless 명령은 기존 broad-face 결과와 겹치지 않는 다음 파일을
`results/cantilever_narrow_10g_to_500g/` 아래에 생성한다. 폭이 `0.160 m`였던
이전 결과 `results/cantilever_10g_to_500g/`는 비교용으로 그대로 보존된다.

- `axia80_cantilever_summary.csv`: 질량별 raw/tare/corrected 6축 평균·표준편차,
  `expected_Fx_N`, `expected_My_Nm`, 오차, 실측 lever arm, 안정성
- `axia80_cantilever_samples.csv`: 측정 구간의 240 Hz raw/tare-corrected 6축 시계열
- `axia80_cantilever_response.png`: 질량에 따른 raw/tare-corrected force·moment 그래프
- `axia80_cantilever_metadata.json`: 실행 인자, 형상·좌표계, 수용 지표, `PASS`/`FAIL`
- `axia80_cantilever_run.log`: 실행 시각, 안정화 시간, 지표, 판정, 산출물 경로

생성된 합성 URDF와 USD도 기존 모델을 덮어쓰지 않고
`src/axia80_feasibility/generated/cantilever_narrow/` 아래에 별도로 저장된다.

기존 summary CSV에서 cantilever 그래프만 다시 생성하려면 다음을 실행한다.

```bash
cd /home/min/7cmdehdrb/grad

env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-cantilever-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/plot_cantilever_results.py \
  src/axia80_feasibility/results/cantilever_narrow_10g_to_500g/axia80_cantilever_summary.csv \
  --output src/axia80_feasibility/results/cantilever_narrow_10g_to_500g/axia80_cantilever_response.png
```

## 해석 한계: 굽힘 모멘트와 탄성 변형은 다르다

이 실험의 Axia80–tool 연결은 `fixed joint`이고 tool은 rigid body이다. 따라서
센서 중심에서 `My = m·g·d`에 해당하는 **굽힘 모멘트 전달**은 측정하지만,
Isaac Sim 화면에서 tool이 실제로 휘어지는 **탄성 처짐·변형**은 나타나지 않는다.

실물 그리퍼의 변형량까지 재현하려면 재료의 굽힘 강성과 감쇠를 정하고,
적절한 compliant joint, 분할 link 모델, 또는 deformable-body 모델을 별도로
추가해야 한다. 현재 결과는 Axia80 fixed-joint wrench의 좌표계·부호·정역학적
하중 전달을 검증하는 feasibility test로 해석해야 한다.
