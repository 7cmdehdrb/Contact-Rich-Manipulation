# UR5e–Axia80 정적 하중 feasibility test

이 패키지는 Isaac Lab에서 `UR5e → Axia80 F/T sensor → 얇은 직육면체 tool`을 한
articulation으로 구성하고, tool 중앙에 10 g부터 500 g까지의 물체를 올렸을 때의
6축 반력(`Fx, Fy, Fz, Mx, My, Mz`)을 검증한다. 기본 실행은 질량 하나당 환경 하나를
사용하므로 50개 환경을 동시에 시뮬레이션하고, 안정화 뒤 원시값과 무부하 tare 보정값을
CSV 및 그래프로 저장한다.

추가 thin-face cantilever 및 각도 실험은 기존 기본 실험과 별도 코드·결과 경로를
사용한다.

- [CANTILEVER.md](CANTILEVER.md): 폭 0.120 m의 thin-face 수평 cantilever 실험
- [TILTED_CANTILEVER.md](TILTED_CANTILEVER.md): wrist3 단일 +10° 질량 sweep
- [WRIST_ANGLE_SWEEP.md](WRIST_ANGLE_SWEEP.md): 고정 500 g Wrist Roll × Tilt 2D sweep

## 모델과 좌표계

조립 순서는 다음과 같다.

```text
UR5e tool0
  └─ fixed: ur5e_to_axia80_joint
       └─ axia80_link
            └─ fixed: axia80_measurement_joint  ← 측정하는 joint
                 └─ payload_tool_link
                      └─ centered payload cube (접촉 물체)
```

| 요소 | 형상 및 파라미터 |
|---|---|
| Axia80 근사 모델 | 실린더, 반지름 `0.041 m`, 높이 `0.0254 m` |
| 시험용 tool | 직육면체 `0.24 × 0.16 × 0.012 m`, 질량 `0.20 kg` |
| sensor–tool 간격 | `0.001 m` |
| payload | 한 변 `0.040 m`인 정육면체, 환경별 질량 `0.010–0.500 kg` |
| 기본 중력 | `9.81 m/s²` |

tool의 넓은 면은 local XY 평면이며 local `+Z`가 위를 향하도록 UR5e 자세를 고정한다.
센서 link의 중심과 tool link의 원점·방향은 정확히 일치하고, tool 형상과 질량중심만 센서
앞쪽으로 오프셋된다. 따라서 저장되는 wrench의 원점은 **Axia80 실린더 중심**, 축은
**Axia80/tool local frame**, 순서는 `[Fx, Fy, Fz, Mx, My, Mz]`이다. 힘 단위는 N,
모멘트 단위는 N·m이며 `+Z`는 tool 넓은 면의 위쪽 법선이다.

측정값은 tool body의 `body_incoming_joint_wrench_b`에서 읽는 parent-to-child 고정-joint
반력이다. 이 조립과 부호 규약에서는 중앙에 놓인 물체의 아래쪽 하중이 **양의 `Fz`**로
나타난다. `raw_*` 값에는 0.20 kg tool의 자중이 포함되고, 기본 결과 열은 payload를
올리기 전 측정한 무부하 평균을 뺀 tare 보정값이다. 센서 본체에는 PhysX 표현의 안정성을
위해 `0.001 kg`을 부여하지만 측정 joint의 상류에 있으므로 결과에는 포함되지 않는다.

결합 URDF/USD는 실행할 때 로컬
`src/Universal_Robots_ROS2_Description`에서 `generated/` 아래에 자동 생성한다. 외부
Nucleus asset은 필요하지 않으며 `merge_fixed_joints=False`를 유지해야 측정 joint가
사라지지 않는다.

## 실행

저장소 루트(`/home/min/7cmdehdrb/grad`)에서 실행한다. 현재 설치된 Isaac Lab 전용
Python을 명시적으로 쓰고 ROS 환경 변수를 제거하는, 이 워크스페이스에서 검증한 명령은
다음과 같다.

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/run_payload_test.py \
  --headless --device cuda:0 --strict \
  --output_dir src/axia80_feasibility/results/10g_to_500g
```

스크립트가 패키지 경로를 직접 추가하므로 editable install은 필수가 아니다. 별도 환경에서
의존성을 설치해야 한다면 다음을 한 번 실행한다.

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  /home/min/miniconda3/envs/env_isaaclab/bin/python -m pip install -e \
  src/axia80_feasibility
```

기본 질량 범위는 `--mass_start_g 10 --mass_stop_g 500 --mass_step_g 10`이다. 즉
`env_0=10 g`, `env_1=20 g`, …, `env_49=500 g`으로 매핑된다. 다른 범위를 사용할 때도
끝값이 step으로 정확히 도달해야 한다. 창으로 확인하려면 `--headless`를 빼면 된다.
URDF/USD를 다시 변환해야 할 때는 `--force_conversion`, 전체 시계열 CSV가 필요 없을
때는 `--no_sample_csv`를 추가한다.

실험 순서는 다음과 같다.

1. payload의 중력을 끄고 멀리 둔 상태에서 로봇을 안정화한다.
2. 기본 `0.75 s` 동안 tool-only wrench를 측정해 tare를 만든다.
3. 모든 payload를 각 tool의 중앙 위에 놓고 중력을 켠다.
4. 접촉 위치·상대 선속도·상대 각속도가 기본 `0.5 s` 동안 연속으로 기준을 만족할 때까지
   기다린다.
5. 기본 `1.0 s`, 240 Hz의 wrench를 기록하고 그 전체 구간에서 안정성이 유지됐는지 다시
   검사한다.

기본 안정 기준은 상대 선속도 `≤ 0.005 m/s`, 상대 각속도 `≤ 0.05 rad/s`, 목표 높이
오차 `≤ 0.004 m`이며, 최소 1초 이후부터 최대 8초까지 기다린다. 하나의 환경이라도
안정화되지 않거나 측정 중 안정성을 잃으면 잘못된 CSV를 남기지 않고 실행을 중단한다.
`--strict`는 CSV를 만든 뒤의 물리량 수용 기준까지 실패한 경우 exit code를 non-zero로
만든다.

## 출력 파일

위 명령은 아래 파일을 만든다.

- `axia80_payload_summary.csv`: 질량별 한 행의 평균·표준편차와 안정화 정보
- `axia80_payload_samples.csv`: 240 Hz 개별 샘플(기본 1초 × 50환경)
- `axia80_payload_response.png`: 질량–힘/모멘트 그래프
- `axia80_payload_metadata.json`: 실행 인자, 모델 치수, 프레임 규약, 수용 지표와
  `PASS`/`FAIL` 판정

`summary.csv`의 식별/상태 열은 `env_id`, `mass_g`, `mass_kg`, `expected_Fz_N`,
`settled`, `settle_time_s`, sample 수, 측정 중 최대 상대 속도, `tool_normal_z`이다.
각 축에는 다음 세 종류의 평균과 표준편차가 기록된다.

- `Fx_N` … `Mz_Nm`: 무부하 tare 보정값(주 결과)
- `raw_Fx_N` … `raw_Mz_Nm`: tool 자중을 포함한 원시값
- `tare_Fx_N` … `tare_Mz_Nm`: payload가 없는 구간의 기준값

추가로 `Fz_error_N = Fz - mass × gravity`와 `Fz_error_percent`를 저장한다.
`samples.csv`에는 `sample_index`, `time_s`, `env_id`, `mass_g`와 각 샘플의 보정/원시
6축 값이 들어간다.

PNG는 위쪽에 raw force/moment, 아래쪽에 tare-corrected force/moment를 배치한 2×2
그래프다. 보정 force 그래프에는 이상적인 `m·g` 선도 함께 표시한다. 기존 summary
CSV에서 그림만 다시 만들려면 다음을 실행한다.

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  MPLCONFIGDIR=/tmp/axia80-mpl \
  /home/min/miniconda3/envs/env_isaaclab/bin/python \
  src/axia80_feasibility/scripts/plot_payload_results.py \
  src/axia80_feasibility/results/10g_to_500g/axia80_payload_summary.csv \
  --output src/axia80_feasibility/results/10g_to_500g/axia80_payload_response.png
```

판정 결과는 metadata와 콘솔의 `[RESULT]` 줄에서 확인한다. 기본 수용 기준은 다음과 같다.

- `Fz` 기울기가 중력가속도의 ±5%, `R² ≥ 0.995`
- `|Fz intercept| ≤ 0.02 N`, `Fz RMSE ≤ 0.01 N`, 최대 `|Fz-mg| ≤ 0.02 N`
- 질량 증가에 따라 tare-corrected `Fz`가 엄격히 증가
- tool-only tare `Fz` 오차 `≤ 0.02 N`
- 최대 `|Fx|`/`|Fy| ≤ 0.10 N`, 최대 절대 모멘트 `≤ 0.01 N·m`
- 모든 payload 안정화 및 tool normal의 world-Z 성분 `≥ 0.999`

## 해석 범위와 한계

이 모델은 Axia80의 기계 외형을 지정된 실린더로 근사하고, 이상적인 fixed-joint 반력을
F/T 센서 출력으로 사용하는 **정적 정상하중 feasibility test**이다. 실물 Axia80의
strain-gauge 구조, 캘리브레이션 행렬, 축간 결합, 노이즈, bias drift, 샘플링/필터 지연,
대역폭, 포화, 온도 특성, 과부하 한계는 모델링하지 않는다.

또한 하중을 tool 정중앙에 수직으로 놓으므로 이 시험이 직접 검증하는 것은
`Fz ≈ mass × gravity`의 선형성과 `Fx`, `Fy`, `Mx`, `My`, `Mz`가 거의 0인지 여부다.
따라서 이 결과만으로 실물 센서의 완전한 6축 감도나 보정을 검증했다고 볼 수 없다.
6축 검증에는 알려진 오프셋 위치의 하중, 수평력, 양·음 방향의 독립 모멘트, 동적 입력과
실물 데이터 비교가 추가로 필요하다.
