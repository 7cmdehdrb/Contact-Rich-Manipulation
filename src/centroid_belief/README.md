# OBB 관측 → 단일 가우시안 중심 추정

현재 기본 API는 관측 OBB 목록을 받아 **단일 2차원 가우시안**을 반환한다. 가리기 전 전체 OBB의 XY 기하 중심을 대상으로 하며, 질량중심이나 메쉬 체적 중심은 아니다. Isaac Sim·물리 시뮬레이션·이미지 렌더링·학습 없이 CPU에서 실행한다.

## 한 번 추정한 뒤 5개 값만 사용

```python
import json
from observed_obb import Box, Observation
from centroid_belief import Config, GaussianBelief, estimate

snapshot = (
    Observation(8, Box((0.0, -0.02, 0.05), (0.08, 0.04, 0.10))),
    Observation(31, Box((0.20, 0.04, 0.03), (0.035, 0.08, 0.06))),
)
belief = estimate(snapshot, target_id=8, config=Config(seed=17))
saved = json.dumps(belief.parameters)

# 이후에는 물체 목록, 후보 표본, 형상 사전분포 없이 사용한다.
belief = GaussianBelief.from_parameters(json.loads(saved))
mean_xy = belief.mean                   # MAP도 동일하다.
covariance_xy = belief.covariance
ellipse90 = belief.ellipse(mass=0.90)
probability = belief.probability_in_rectangle((-0.01, -0.01), (0.01, 0.01))
inside90 = belief.contains([[0.0, 0.0]], mass=0.90)
```

저장 순서는 `(mu_x, mu_y, cov_xx, cov_xy, cov_yy)`다. 평균의 단위는 m, 공분산은 m²다. XY 상관관계를 보존한다. 저장된 5개 값으로 밀도, 타원, 영역 적분을 직접 계산하며 후보 탐색을 다시 하지 않는다. 입력은 관측별 독립 ID, 중심 XYZ, 크기 XYZ, yaw다. 정답 종류·크기·중심·원본 ID는 입력하지 않는다. 관측 방향은 +X에서 −X, 좌우는 Y다.

`density()`는 m⁻² 단위의 확률밀도다. 한 점의 확률은 0이며 특정 범위의 확률은 적분으로 구한다. 공분산은 양의 정부호여야 한다. 90% 타원은 이 가우시안의 확률 질량 90%를 포함하며, 실제 데이터에서의 정답 포함률을 보장하는 수치가 아니다.

## 먼저 두 후보군을 평가한다

1. **완전 관측 후보군**: 관측된 상자 자체가 물체 전체일 수 있다. 수평 치수와 높이를 관측 OBB에서 정하고, 중심·yaw 후보를 생성해 주변 물체와 함께 다시 관측한다. 기존 8종 비율에 맞지 않는 작은 물체도 이 후보군에서 허용한다.
2. **부분 관측 후보군**: 8종 전체 OBB 비율 템플릿에서 가능한 크기·중심·yaw를 생성한다. 재관측 결과에 유의미한 잘림이 있어야 한다.
3. 두 후보군 모두 동일한 가림 계산과 관측 적합도, 사전 비중, 중요도 보정을 거친다. 완전 관측 후보를 계산 이후에 임의로 더하거나, 노란 관측 중심이 90% 타원 안에 들어오도록 공분산을 늘리지 않는다.

기본 초기 비중은 `Config(complete_prior=0.5)`다. **실험을 위한 설정이며 학습되거나 보정된 확률이 아니다.** 최종 비중은 후보군별 관측 적합도와 표집 적분에 따라 달라진다. 가까운 물체 때문에 실제로 가려지는 후보는 완전 관측으로 인정하지 않는다. 옆에 물체가 있다는 사실만으로 완전 관측을 금지하지 않는다.

완전 관측 여부는 예측 상자와 후보 전체 상자의 모서리 RMS 차이가 기존 관측 허용 척도 3 mm 이하인지로 판단한다. 양쪽 경계에 정확히 맞닿은 사례에서 엄격한 실수 좌표 동일성을 요구하면 유효한 pose 집합의 부피가 0이 되어 해당 후보군이 사라지기 때문이다. 따라서 이 구분은 **3 mm 이하의 미미한 잘림과 완전 관측을 구별하지 못하는 근사**다. 부분 관측 후보군은 같은 기준보다 큰 차이가 있어야 한다.

완전 관측 후보의 크기는 측정값으로 조건화되어 있다. 이는 별도 크기 생성모델의 주변우도를 정확히 적분한 일반적인 Bayesian model selection이 아니다. 초기 비중과 관측 허용 척도에 의존하는 **관측 조건부 복원 모델**로 해석해야 한다.

## 그다음 하나의 가우시안으로 압축한다

모든 후보 중심의 가중 평균을 최종 평균으로 사용한다. 공분산은 후보 중심과 그 평균의 차이에서 계산하고, 각 축에 기본 (2 mm)²의 분산 바닥을 더한다. 완전·부분 가설 사이의 중심 차이도 공분산에 포함한다. 후보군별 공분산만 평균내면 빠지는 이 불확실성을 보존한다.

최종 분포는 봉우리 하나와 타원형 등밀도선을 가진다. 평균과 MAP은 같다. 여러 가능한 중심 사이의 낮은 확률 구간에도 질량을 부여할 수 있고, 원래 기하 후보가 존재하지 않는 곳에도 Gaussian 꼬리가 생긴다. 이것은 단순화를 위해 수용한 손실이다. 90% 타원에 관측 중심이 반드시 들어간다는 제약은 없다.

**최초 추정 계산은 여전히 후보 평가를 수행한다.** 단순해지는 것은 최종 표현, 저장, 이후 확률 조회다. 최종 환경에는 아직 연결하지 않았다.

## 기하·템플릿 가정

주변 관측 OBB를 근사 가림 입력으로 사용한다. Z는 무한 가림으로 취급하고 앞뒤 관계를 판정한다. 노출된 각 연결 구간을 따로 자르며 **폭 2 cm 이상 AND 후보 원래 전체 폭의 20% 이상**인 경우만 관측으로 남긴다. 깊이와 높이를 보존해 최소 면적 OBB를 구한다. 같은 원본의 조각이라는 정답 연관 관계는 제공하지 않는다.

| 템플릿 | 기본 X × Y × Z (cm) |
| --- | --- |
| cylinder | 8 × 8 × 10 |
| cube | 6 × 8 × 10 |
| mug | 10.1 × 7 × 10 |
| bottle | 6 × 6 × 13.5 |
| 003_cracker_box | 3.792 × 10.112 × 13.272 |
| 004_sugar_box | 3.812 × 8.576 × 16.2 |
| 005_tomato_soup_can | 7.637 × 7.637 × 11.687 |
| 002_master_chef_can | 7.637 × 7.637 × 10.482 |

앞의 4개는 기존 기본 형상의 외접 치수다. YCB 4종은 실측 USD bounds가 아닌 선언한 근사 치수다. 정규화 전 크기 `(6,16,21)`, `(4,9,17)`, `(6.6,6.6,10.1)`, `(10.2,10.2,14)` cm에 배율 `1.35 * min(8 / hypot(X,Y), 12 / Z)`를 적용했다. 곡면·손잡이 메쉬 자체를 복원하지 않는다.

템플릿은 균등 종류 사전분포와 0.9–1.1 균일 스케일을 사용한다. 높이를 정확히 보존하는 관측에 맞춰 `관측 높이 / 기본 높이`로 스케일을 조건화하고, 형상별 높이 Jacobian을 반영해 정규화한 뒤 부분 관측 후보군의 비중을 배정한다. 높이 지원이 없으면 완전 관측 후보군만 사용하고 `no_template_height_support`를 알린다. 중심은 관측 주변의 제한된 사각형, yaw는 180도 구간에서 균등하다.

주변 OBB도 부분 관측의 외접 상자이므로 실제 물체의 가림 마스크와 다를 수 있다. 주변 원본 형상의 불확실성은 공동 적분하지 않는다. 학습된 센서 모델, 질량분포, 실제 충돌 판정은 사용하지 않는다.

## 실행과 검증

Workspace 루트에서 실행한다. 설치된 NumPy·SciPy 1.15 이상·Matplotlib이 필요하다. 이 workspace에서는 `/home/min/miniconda3/envs/env_isaaclab/bin/python`을 사용하며 Isaac Lab은 import하지 않는다.

```bash
OPENBLAS_NUM_THREADS=1 python src/centroid_belief/scripts/evaluate_gaussian.py
python src/centroid_belief/scripts/evaluate_gaussian.py --render-only
# 중단된 실행은 입력과 설정이 같은 저장 사례를 재사용한다.
python src/centroid_belief/scripts/evaluate_gaussian.py --resume
PYTHONPATH=src/centroid_belief:src/observed_obb PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q src/centroid_belief/tests src/observed_obb/tests
```

새 Python 환경에 로컬 설치하려면 다음을 사용한다.

```bash
python -m pip install -e src/observed_obb -e src/centroid_belief
```

기본 8종 × 4개 가림 조건, 회전·뒤쪽 물체·분리 관측 4개, 동일 관측이 작은 물체 전체에서 나온 대안 정답 1개로 총 37개를 확인한다. 마지막 사례는 원통 오른쪽 가림 사례와 입력·seed가 같아서 출력도 같아야 한다. 실제 중심은 다르며, 이 구별 불가능성을 점 추정 성공으로 포장하지 않는다.

`outputs/centroid_gaussian`에 사례별 5개 파라미터 JSON, 진단·평가 지표, CSV, 요약 JSON, 확률밀도·90% 타원 PNG/PDF를 저장한다. 진단·평가용 JSON에는 입력과 정답도 구분하여 기록하지만, 재사용에는 `parameters`의 5개 숫자만 필요하다. 그림도 이 값으로 복원한 가우시안만 사용한다.

## 검토용 입자 API와 이전 실험

`estimate_particles()`는 진단용 후보와 가중치를 반환한다. `shape_probabilities[COMPLETE_HYPOTHESIS]`에서 완전 관측 후보군의 최종 비중, `warnings`에서 낮은 ESS·적합 실패를 확인할 수 있다. 기본 `estimate()`는 후보를 버리고 Gaussian만 반환하며, 진단 경고가 있으면 Python `RuntimeWarning`으로 알린다.

이전 입자/KDE 실험은 `evaluate.py`가 `complete_prior=0`을 명시하여 재현한다. 기존 `outputs/centroid_belief` 결과는 보존한다. 이전의 `estimate(...).grid()` 호출은 현재 `estimate_particles(...).grid()`로 변경해야 한다. 기본 API를 단일 Gaussian 출력으로 바꾼 의도적인 변경이다.

## Isaac Sim에서 관측과 함께 표시

```bash
OPENBLAS_NUM_THREADS=1 python src/centroid_belief/scripts/show.py --seed 0
```

기존 `observed_obb`의 Simple 정지 장면·가림 배치·관측 스냅샷을 그대로 사용한다. 각 관측 ID마다 전체 관측 목록을 입력으로 추정하며 원본 물체 ID·종류·정답 중심은 전달하지 않는다. 분리된 조각도 각각 독립 추정한다.

- 초록 선: 관측 OBB. 빨간 점: Gaussian 평균. 청록색 띠: 90% XY 확률 경계. 파랑→노랑 색상 면: 각 Gaussian의 최대값으로 정규화한 상대 확률밀도(99% 영역까지 표시). 색은 서로 다른 물체의 절대 밀도를 비교하는 척도가 아니다.
- 타원과 평균은 가려지지 않도록 해당 관측 OBB의 윗면보다 25 mm 위에 표시한다. 이 높이는 표시 위치이며 Z 확률 추정이 아니다.
- `New placement`와 `Reobserve current placement`에서만 새 추정을 시작한다. 계산 중에도 두 버튼을 사용할 수 있다. 이전 추정을 취소하고 가장 최근 요청만 계산하며 이전 배치의 결과는 버린다. 카메라·참고 박스·Gaussian 토글도 계속 사용할 수 있다.
- CPU 백그라운드 작업이 ID별로 계산을 마칠 때마다 주 스레드에서 표시한다. 완료된 분포는 5개 값만 유지하며 프레임마다 다시 추론하지 않는다. 실패·진단 경고는 ID별 패널과 콘솔에 표시한다.
- 주황 원본 OBB 참고는 기존 토글로 표시하며 추정 입력에 사용하지 않는다. 밀도 면·경계·점에는 충돌·강체 스키마가 없고 원본 장면의 dynamics는 비활성 상태다.

뷰어 기본값은 `--estimator lookup`이다. 8종 형상을 미리 잘라 저장한 81,621개 OBB와 중심 오프셋을 조회하고, 최대 16개 후보만 실제 관측 목록으로 재검사한다. 약 776 KiB 데이터는 패키지에 포함되며 시작 시 한 번 읽는다. Reset마다 최적화·Monte Carlo 표집·학습을 수행하지 않는다. `--samples`는 비교용 `--estimator reference`에서만 사용한다.

```bash
# 사전 계산표 재생성: 시뮬레이션·학습 없이 CPU 약 13초
python src/centroid_belief/scripts/build_lookup.py
# CPU 속도·정확도 재현
OPENBLAS_NUM_THREADS=1 python src/centroid_belief/scripts/evaluate_fast.py
# 실제 GPU Reset 6회 측정 후 종료
python src/centroid_belief/scripts/show.py --benchmark-resets 6
# 기존 느린 추정기 비교
python src/centroid_belief/scripts/show.py --estimator reference --samples 128
```

빠른 경로의 프로그램 인터페이스:

```python
from centroid_belief import LookupEstimator, estimate_fast
model = LookupEstimator()  # 시작 시 한 번
result = model.estimate(observations, target_id)
belief = result.belief  # mean, covariance, parameters: 기존과 같은 5개 값
print(result.warnings, result.evaluated, result.seconds)
# 진단이 필요 없는 간단한 호출, 내부 인스턴스 재사용:
belief = estimate_fast(observations, target_id)
```

이 조회 모델은 기존 importance posterior의 수치적으로 동일한 가속판이 아니다. 높이로 크기를 정규화한 OBB 특징과 가림 방향을 기준으로 후보를 검색한다. 원본 ID·실제 중심·종류는 입력하지 않는다. 완전 관측 중심과 부분 관측 후보 중심을 가중 평균하고 공분산에 3 mm 바닥 오차 및 표의 격자 오차를 더해 단일 Gaussian으로 출력한다. 완전 관측 비중은 최소 50%이며, 후보 적합도에 따라 증가한다. 후보 가중치와 오차 바닥은 미보정 근사값으로 실제 빈도 90%를 보장하지 않는다.

관측된 가림 경계가 없으면 관측 중심을 그대로 쓴다. 따라서 가리는 물체 자체가 관측에서 누락된 분리 사례는 틀릴 수 있다. 높이가 8종의 0.9~1.1배 범위에 없으면 `no_template_height_support`, 조회 후보가 맞지 않으면 `lookup_low_support`를 표시한다. 후자는 넓은 대체 공분산을 사용한다. 조회 방식에는 ESS가 존재하지 않는다. 기본 `estimate()`와 `estimate_particles()` API는 비교용 기존 계산을 유지하며, 빠른 사용에는 `estimate_fast()` 또는 `LookupEstimator`를 명시한다.

확대 밀도 창은 메모리 RGBA 이미지를 직접 갱신한다. Reset마다 Matplotlib 실행·PNG 저장·파일 읽기를 하지 않는다. `Next completed observation`으로 관측을 전환한다. 3D Gaussian은 USD 재질·도형을 재사용하며 사용하지 않는 관측 도형은 숨긴다. Reset 로그의 `reset_to_rendered_frame`은 배치·관측·추정·표시 갱신과 마지막 `app.update()`까지의 시간이다. 앱 시작과 초기 자산 로딩은 별도다.

뷰어는 닫을 때까지 유지한다. GPU·디스플레이·기존 Simple 자산과 Isaac Lab 환경이 필요하다. 최종 학습 환경에는 아직 연결하지 않았다.
