# Inspire Tactile 벡터 검증

Isaac Lab Standalone으로 기존 `InspireSweepTactileEnvCfg`의 촉각 계산이 실제 시뮬레이션 접촉에 반응하는지 확인한다. 기본 `--pad-layout original`은 원래 패드 배치를 유지한다. `--pad-layout exposed`는 직접 패드 검증을 위해 시험용 Hand에서 패드를 부모 hull 바깥에 노출하는 별도 옵션이다. 노출 배치의 결과를 원본 배치의 정확도로 해석하지 않는다. Arm과 정책 학습 없이, 손바닥이 위를 향하는 고정 Hand와 얇은 동적 막대를 사용한다.

## 실행

Workspace 루트에서 Isaac Lab 환경을 활성화하고 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_tactile_validation/scripts/run_probe_test.py \
  --num-envs 85 --repeats 3 --headless
```

17채널 × (중앙 + 모서리 4개) × 3회 = 255회다. 85개 Hand/막대를 병렬 실행하며, 한 번의 시퀀스에서 서로 다른 85개 위치를 검사한다. `--num-envs`가 작으면 자동으로 여러 batch에 나누고, 마지막 batch의 여분 환경은 보고서에서 제외한다. 각 batch마다 손을 완전히 편 자세로 reset한 뒤 안정화한다.

손바닥 한 채널을 화면으로 확인하려면:

```bash
./IsaacLab/isaaclab.sh -p src/inspire_tactile_validation/scripts/run_probe_test.py \
  --channels 0 --num-envs 1 --repeats 1 --real-time
```

`--channels 0 5`처럼 채널을 선택할 수 있다. 채널 순서는 아래 `metadata.json`과 기존 정책의 17채널 순서가 같다. 기존 결과를 덮어쓰지 않으며 기본 출력은 `outputs/inspire_tactile_validation/<날짜_시각>/`이다. `--output-dir`로 새 경로를 지정할 수 있다.

## 재사용과 시험용 변경

- `inspire_sweep_baseline`의 USD Hand 몸체·충돌 mesh·질량·관절·17개 패드와 패드 좌표 metadata를 읽기 전용으로 사용한다. exposed 배치는 패드 몸체를 로컬 +Z로 이동하고 부착 fixed joint를 재설정한다. 원본 USD는 저장하거나 수정하지 않는다. Arm은 시뮬레이션에서 비활성화하며 Hand의 12개 관절만 남긴다.
- `ProjectedTactileContactBits`를 수정 없이 호출한다. 직접 패드 접촉과 부모 링크 접촉점의 센서 영역 매핑을 OR로 결합한다.
- 원래의 17개 직접 ContactSensor와 12개 부모 ContactSensor 구조, 0.05N 이진 검출 임계값을 사용한다. 부모 센서의 접촉 상대만 학습 물체에서 막대로 바꾼다.
- Hand의 fully-open 관절 목표, stiffness/damping, effort/velocity 제한을 재사용한다. 뿌리만 고정하고 손가락의 접촉에 따른 작은 변형은 허용·기록한다.

시험 장치에서 바꾸는 부분은 palm-up 자세, Arm 제거, 선택적 패드 노출 배치, 명시적인 world gravity=(0,0,0), 막대의 위치/자세 servo다. 패드 mesh의 로컬 좌표와 촉각 계산은 유지한다. 손과 막대의 sleeping은 정지 시험을 위해 비활성화한다. 물리 dt 기본값은 기존 환경과 같은 0.01초다. 매 호출에서 `sim.step(render=False)`로 물리 한 step만 진행하고 실제 경과 시간이 dt인지 검사한 뒤 센서를 갱신한다. GUI 렌더링은 별도로 수행하며 물리 시간이 늘어나지 않는지 검사한다. 기존 환경에서는 정책 관측이 0.02초마다 이뤄진다. TGS의 외력 반복 적용은 시험에서 기본 활성화하며 `--no-external-forces-every-iteration`로 대조할 수 있다. 기존 RL 코드는 변경하지 않는다.

## 누르는 위치와 시퀀스

패드의 실제 convex footprint에서 중앙과 네 대각선 방향의 경계를 구한다. 모서리 목표는 경계에서 중앙 쪽으로 20% 들어온 점이다(`--corner-inset 0.2`). 사각 bounding box의 모서리가 실제 패드 바깥인 경우도 있어 polygon 경계를 사용한다. 목표의 기준은 패드 몸체 로컬 좌표이며 각 막대는 그 패드의 바깥 법선 반대 방향으로 접근한다. 손바닥은 하늘 방향이고, 엄지 등 다른 패드에는 각자의 법선으로 접근한다.

일부 패드는 부모 손 링크의 큰 충돌 hull 안에 묻혀 있다. 막대의 시작점은 그 부모 표면 바깥까지 띄우고, 접촉은 직접 패드 또는 부모 링크에서 일어날 수 있다. 이를 구분 없이 버리면 기존 투영 촉각을 검증할 수 없다.

| 단계 | 기본 시간 | 행동 |
| --- | ---: | --- |
| baseline | 0.15초 | 표면 밖에서 무접촉 신호 확인 |
| approach | 0.5초 | 목표 패드 표면의 3mm 안쪽을 향해 이동 |
| hold | 0.3초 | 위치 목표를 유지하며 하중 확인 |
| retract | 0.5초 | 표면 밖으로 복귀 |
| recovery | 0.2초 | 신호 해제 확인 |

막대는 반지름 0.75mm, 중심 원통 길이 30mm의 capsule이다. 질량은 50g이며 중력이 꺼져 있다. 위치 servo는 100N/m, critical damping, 명령 힘 상한 1N이다. 막대를 목표 위치로 매 step 순간이동시키지 않고 물리적 외력으로 이동한다. 실제 접촉력은 명령 힘과 다르므로 따로 기록한다. 1N은 servo 명령의 상한이며 충돌 순간의 실제 접촉력이 이보다 클 수 있다. 위치 목표는 각 trial 시작 시 정한 위치이고, 센서 판정·접촉 위치 변환에는 움직인 손의 현재 frame을 사용한다.

## 판정과 출력

`probe_contact`는 촉각 bit와 독립적으로 막대의 실제 접촉력을 읽는다. 목표 패드 또는 그 부모 몸체에 전달된 힘과 접촉점의 XY footprint를 확인한다. 접촉 위치는 body별 접촉점 평균을 힘의 크기로 가중한 centroid이므로 개별 접촉점 정답과 구분한다. 기존 센서 코드의 Z/normal gate는 정답 판정에 재사용하지 않는다.

검출 평가는 approach/hold의 유효 접촉에서만 한다. 복귀 중 우연히 켜진 bit로 누르기 성공을 인정하지 않는다. 무접촉, 하중 부족, 다른 부위 접촉은 센서 미검출과 구분한다. 정지 hold에서 막대의 외력·가속도 평형과 접촉 보고 힘이 크게 다르면 `reference_mismatch`로 표시하고 정확도 확정을 보류한다.

| 파일 | 내용 |
| --- | --- |
| `metadata.json` | 실행 옵션, 17채널·85개 목표, 자산 metadata 해시, commit, 실제 Hand 구성 |
| `samples.csv` | 매 step의 17개 bit와 직접 패드 힘, 막대 접촉력·목표 몸체 하중, 접촉 로컬 좌표, 명령 힘, 위치 오차, 손 관절 오차, 막대 속도·힘 평형 |
| `trials.csv` | 위치별 검출 여부·유효 하중 중 검출 비율, 오채널, baseline 오검출, 복귀 잔류 bit, 기준 힘 불일치 |
| `confusion.csv` | 목표 채널별 관측 채널의 trial-any 검출 비율. 유효 접촉이 없는 행은 빈 값 |
| `summary.json` | 전체 완료 여부와 판정, 검출·미검출·무접촉·하중 부족·오접촉·기준 불일치 개수 |

`target_detection_rate_valid_trials`는 유효한 접촉을 만든 trial만의 비율이다. 반드시 `invalid_trial_count`와 채널별 coverage를 같이 읽는다. 1회라도 검출된 trial의 비율과 모든 loaded frame에서의 검출 비율(`detection_fraction`)은 다르다. 직접 패드 힘이 0인데 최종 bit가 켜지면 부모 접촉 투영 경로로 검출된 것이다.

`--strict`는 모든 trial에 유효 검출이 있고 오채널·baseline 오검출·복귀 잔류·기준 불일치가 없을 때만 성공한다. 그 외에는 exit code 2다. 실물 센서의 압력 분포·정량 힘 측정 정확도와 동일한 시험으로 해석하지 않는다.

## CPU 검증

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/min/miniconda3/envs/env_isaaclab/bin/python -B -m pytest -q -p no:cacheprovider \
  src/inspire_tactile_validation/tests
```

## 초기 실행 결과 — 시간 동기화 오류가 포함된 과거 기록

`outputs/inspire_tactile_validation/validation_255_01/`에서 85개 환경, 3회 반복, 총 255회 / 42,075 frame을 실제 RTX 3090 GPU 시뮬레이션으로 수집했다. 검출 66회, 기준 힘 불일치 186회, 오접촉 3회이며 전체 판정은 `inconclusive`다. 검출 trial 중에도 별도 정지 하중 불일치가 있어 깨끗한 통과는 0회다. 유효 접촉 중 검출률 100%만 인용하면 안 된다.

손바닥 중앙의 정지 hold에서는 외력·가속도 평형 약 0.73N에 비해 막대 접촉 보고 힘이 약 0.00002N인 예가 있었다. TGS 외력 반복 적용, PGS, sleeping 비활성화 대조에서도 불일치가 해소되지 않았다. 물리 접촉 보고·시험 하중 기준의 불일치 원인은 확정하지 않았다. 기존 RL 센서를 수정하거나 실물 정확도를 검증한 결과가 아니다. 재현 세부는 프로젝트 개발 로그에 기록한다.

## 결과 시각화

시뮬레이터 실행 없이 결과 CSV/JSON에서 PNG 5개와 통합 PDF를 생성한다. Matplotlib·NumPy·Pandas가 필요하다.

```bash
/home/min/miniconda3/envs/env_isaaclab/bin/python -B \
  src/inspire_tactile_validation/scripts/plot_results.py \
  --input outputs/inspire_tactile_validation/2026-10-16_13-11-03
```

`--input`을 생략하면 결과 루트에서 최신 날짜 형식의 run을 선택한다. 디버그 run은 혼합하지 않는다. 출력은 선택한 run의 `visualizations/`다. 전체·채널별 상태, 센서×위치×반복, 조건부 혼동행렬, 예시 시계열, hold 힘 비교를 그린다. 유효 접촉이 없는 혼동행렬 행은 NA로 표시한다. 파란 detected는 한 번의 검출이며 정상 유지·clean pass를 뜻하지 않는다. 힘 평형 값은 무접촉 baseline에서도 0이 아닐 수 있어 접촉력 정답으로 해석하지 않는다.

## 수정된 시험과 결과 (2026-10-16)

`--pad-layout exposed`일 때 패드의 최하단이 원래 부모 hull의 로컬 +Z 최상단보다 1mm 위에 오도록 원래 패드 mesh와 몸체를 이동한다. 패드 로컬 footprint는 같고, 고정 부착 joint만 새 pose에 맞게 재설정한다. `--pad-clearance`는 이 여유 높이다. 실제 이동량은 `metadata.json`의 `pad_lifts_m`에 기록된다(이번 손바닥 약 20.40mm, 다른 패드 약 2.56～12.41mm). 이는 원래 메쉬의 매몰·측면 가림을 피하는 검증 fixture이며 물리적으로 동일한 원본 Hand의 인증은 아니다.

원래 배치에서 시간 동기화와 무중력 조건을 바로잡은 기본 시험은 다음과 같이 실행한다.

```bash
./IsaacLab/isaaclab.sh -p src/inspire_tactile_validation/scripts/run_probe_test.py \
  --pad-layout original --num-envs 85 --repeats 1 --headless
```

선택적 exposed 배치의 `2026-10-16_13-30-20`에서 17채널 × 5위치 × 3반복 = 255회, 42,075 frame을 GPU에서 수집했다. detected 255, clean pass 255, 기준 불일치·오채널·오접촉·무접촉·복귀 잔류 0, strict exit code 0이다. 유효 하중 frame에서의 target 검출률도 100%였다. 일부 모서리에서 hold 하중 자체가 임계값 아래로 떨어질 때 bit가 꺼지는 것은 별도로 표시하며, hold 내내 모든 frame에 하중이 유지됐다는 뜻은 아니다.

기존 결과와 같은 그림 및 전후 비교:

```bash
/home/min/miniconda3/envs/env_isaaclab/bin/python -B \
  src/inspire_tactile_validation/scripts/plot_results.py \
  --input outputs/inspire_tactile_validation/2026-10-16_13-30-20 \
  --compare-with outputs/inspire_tactile_validation/2026-10-16_13-11-03
```

출력 `visualizations/`에 원래 5종과 전후 비교, hold 하중/bit coverage 그림 및 통합 PDF를 저장한다. 전후 비교는 시간 동기화·중력·배치를 모두 변경한 시험 프로토콜 비교다. `inspire_sweep_baseline` 코드와 자산은 수정하지 않았다.

### 원본 배치와 노출 배치의 수정 후 비교

| 수정 후 시험 | 원본 배치 (기본값) | 노출 배치 |
| --- | ---: | ---: |
| Run | `2026-10-16_13-33-55` | `2026-10-16_13-30-20` |
| trial 검출 / 시도 | 255 / 255 | 255 / 255 |
| 유효 하중 frame 검출 | 8430 / 8475 (99.47%) | 8130 / 8130 (100%) |
| 기준 불일치·오채널·오접촉·복귀 잔류 | 0 | 0 |

기본값을 원본 배치로 유지하므로 기존 센서 geometry와 부모 접촉 투영을 그대로 검증한다. 원본 배치에서 검지 3의 X−Y+ 모서리는 유효 30 frame 중 15 frame만 검출됐고 이 현상이 3회 반복돼 총 45 frame 미검출이었다. trial-any 검출 100%와 frame 검출률을 구분한다. 노출 배치에서는 이 미검출이 없었다. 어느 경우도 실물 압력 calibration 결과는 아니다.

원본 배치의 최신 결과와 비교 그림:

```bash
/home/min/miniconda3/envs/env_isaaclab/bin/python -B \
  src/inspire_tactile_validation/scripts/plot_results.py \
  --input outputs/inspire_tactile_validation/2026-10-16_13-33-55 \
  --compare-with outputs/inspire_tactile_validation/2026-10-16_13-11-03
```

패드 매몰을 제거한 시험을 선택하려면 실행 명령에 `--pad-layout exposed`를 추가한다. `summary.json`은 trial 집계와 별도로 `valid_loaded_sample_count`, `target_detected_loaded_sample_count`, `target_detection_rate_valid_samples`를 제공한다. frame 비율은 trial별 비율의 단순 평균이 아니라 전체 유효 frame 수로 가중한다.
