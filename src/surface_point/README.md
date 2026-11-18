# 실제 가시 메쉬 기반 대표 표면점 벤치

현실의 **instance segmentation → 보이는 영역의 대표 픽셀 → RGB-D 역투영**에 대응하는 기하 기반 테스트 벤치다. 시뮬레이션에서는 RGB를 렌더링하거나 YOLO/SAM을 실행하지 않는다. 실제 시각 메쉬에 병렬 광선 질의를 수행해 가시 영역과 표면 좌표를 직접 얻는다. 기존 3D 관측 OBB를 함께 표시한다.

이전 기본값은 원본 OBB를 실제 형상처럼 사용했다. 현재 기본값은 USD 시각 메쉬다. 병목·어깨·손잡이의 빈 부분을 박스처럼 채우지 않는다. 기존 박스 알고리즘은 비교용 `geometry.py`와 `evaluate.py`에 보존했다.

## 실행

Workspace 루트와 Isaac Lab Python을 사용한다. 검증 실행은 `--validate`를 붙여 자동 종료한다. 일반 실행은 사용자가 닫을 때까지 유지된다.

```bash
cd /home/min/7cmdehdrb/grad
OPENBLAS_NUM_THREADS=1 /home/min/miniconda3/envs/env_isaaclab/bin/python src/surface_point/scripts/show.py --scene simple --seed 0

# 문제가 보였던 병 가림 배치부터 시작
OPENBLAS_NUM_THREADS=1 /home/min/miniconda3/envs/env_isaaclab/bin/python src/surface_point/scripts/show.py --scene simple --seed 0 --placement 3

# 가시 영역 bbox 중앙 방식과 비교
OPENBLAS_NUM_THREADS=1 /home/min/miniconda3/envs/env_isaaclab/bin/python src/surface_point/scripts/show.py --scene simple --placement 3 --representative bbox-center

# 간단한 박스 장면 또는 더 촘촘한 광선 간격
python src/surface_point/scripts/show.py --scene fixtures
python src/surface_point/scripts/show.py --spacing-mm 0.5

# 실제 장면 검증·메쉬/마스크 저장 후 자동 종료
python src/surface_point/scripts/show.py --scene simple --validate
python src/surface_point/scripts/show.py --scene simple --placement 3 --validate --capture-validation --output outputs/surface_point_mesh/review
```

`--ray-device auto`는 사용 가능한 경우 CUDA, 아니면 CPU를 선택한다. 명시적으로 `cpu` 또는 `cuda:0`를 지정할 수 있다. 샌드박스 CPU 테스트에서 Warp 캐시 경로를 지정해야 한다면 `WARP_CACHE_PATH=/tmp/surface_point_warp`를 사용한다.

## 계산 원리와 현실 대응

| 의미 | 현실 | 이 벤치 |
| --- | --- | --- |
| 입력 형상 | RGB 이미지의 물체 | USD 시각 메쉬 |
| 가시 영역 | instance segmentation mask | 각 광선의 첫 교점 소유자 배열 |
| 대표 픽셀 | mask 내부의 대표 위치 | 동일 규칙으로 선택한 광선 격자 위치 |
| 깊이 | 정렬된 depth 센서 값 | 해당 광선과 시각 메쉬의 첫 교점 |
| 3D 점 | 내·외부 파라미터로 역투영 | 월드 좌표 교점 자체 |

1. 실제 USD 시각 메쉬를 원본 물체의 로컬 좌표로 캐시한다. instance proxy와 하위 변환을 포함하고, 숨겨진 충돌 전용 메쉬·guide 형상을 제외한다. 원본 OBB는 새 가시 영역/대표점 추정에 사용하지 않는다.
2. 재배치 시 실제 루트 변환을 메쉬에 적용한다. 주변 물체 및 선반의 시각 메쉬도 가림 계산에 포함한다. 관측 표시는 광선 질의 대상이 아니다.
3. 고정 정면 +X → −X의 직교 광선을 기본 1 mm 간격으로 배치한다. 격자는 현재 물체들의 YZ 범위를 덮는다. 광선마다 가장 앞의 삼각형과 그 물체 ID를 구한다. 이는 해상도가 명시된 기하 샘플링이며 연속 형상의 해석적 가시성 계산은 아니다.
4. 같은 물체 ID의 4방향 연결 영역마다 독립 관측을 만든다. 분리된 가시 조각을 원본 ID로 다시 합치지 않는다. 물체 ID는 시뮬레이션에서 segmentation의 역할을 대신하며 최종 관측에는 독립 ID만 사용한다.
5. 각 가시 영역의 대표 픽셀을 정하고 해당 위치의 실제 메쉬 교점을 반환한다. 출력 점은 가우시안 추정이나 원래 물체 중심이 아니라 보이는 표면의 한 점이다.

노출 폭 2 cm 이상·원래 시각 메쉬의 Y 투영 폭 대비 20% 이상 조건은 유지한다. 별도의 높이/면적 필터는 추가하지 않았다. 출력에서 제외된 물체도 가림에는 참여한다. 선반은 가리기만 하고 관측점은 출력하지 않는다.

### 대표 픽셀 규칙

- 기본 `mask-centroid`: 해당 **가시 연결 영역의 픽셀 좌표를 면적 가중 없이 평균**한다. 모든 픽셀의 면적이 동일하므로 가시 면적의 중심이다. 이 연속 2D 위치에 가장 가까운 해당 영역 내부 픽셀을 선택한다. 얇은 병목이나 작은 돌출부가 bbox 범위를 크게 바꾸더라도 기여도는 실제 가시 면적만큼이다.
- 비교용 `bbox-center`: 해당 가시 영역의 2D bbox 중앙에 가장 가까운 영역 내부 픽셀을 선택한다. 원래 bbox 중심이 가림/구멍에 놓이면 경계 가까운 픽셀로 이동할 수 있다. 이는 명시적인 정의의 차이이며 고장으로 숨기지 않는다.

두 방식 모두 실물 segmentation mask에서도 동일하게 구현할 수 있다. 선택 규칙을 현실과 시뮬레이션에서 같게 맞춰야 한다. 실제로 보이는 병목도 포함하므로 점을 임의로 “오른쪽 끝”에 옮기지 않는다. 마스크 창의 빨간 점과 검은 bbox로 선택 근거를 확인한다.

`correction_m`은 연속 대표 위치에서 선택 픽셀 중심까지의 YZ 거리다. `corrected`는 반올림한 대표 픽셀이 해당 연결 영역 밖이었는지를 뜻한다. 정상 픽셀 양자화로 `corrected=False`여도 작은 이동 거리는 있을 수 있다.

## 카메라와 시각화

시작 화면은 **자유롭게 움직일 수 있는 사선 원근 카메라**다. 관측 계산의 고정 정면 카메라와 검토용 카메라는 별개다. 매 프레임/재관측/재배치에서 검토 카메라를 강제로 원위치로 돌리지 않는다.

- 뷰포트에서 Alt + 왼쪽 드래그: 회전, Alt + 가운데 드래그: 이동, 스크롤: 확대/축소.
- UI의 방위각·고도각·거리 입력으로도 사선 시점을 지정할 수 있다. 이 조절은 물체 중심을 기준으로 한 프리셋 방식이다.
- `Front orthographic inspection`: 관측 방향과 같은 정면 직교 시점.
- `Oblique perspective inspection`: 사선 원근 시점으로 복귀.
- 카메라 이동은 마스크/표면점에 영향을 주지 않는다. 마스크 창은 항상 고정 정면 관측이다.

| 표시 | 의미 |
| --- | --- |
| 빨간 구 | 실제 메쉬 교점. 마커 반지름은 4 mm이고 구의 중심이 계산점 |
| 초록 선 | 기존 `observed_obb` 부분 관측 3D OBB |
| 파란 구·연결선 | 대응 기존 OBB 중심과 새 점 사이 차이 |
| 주황 선 | 원본 전체 OBB 참고값 |
| 별도 마스크 창 | 광선 소유자 색상, 연결 영역 bbox, 선택 픽셀과 관측 ID |

마스크 창의 색은 형상/역할 분류가 아니라 디버깅용 물체 구분이다. 모든 관측 마커는 비충돌이다. 파란 OBB 중심은 물체 내부에 있어 가려질 수 있다.

## OBB 비교의 의미

기존 OBB는 기존 함수를 그대로 호출하므로 **박스 근사·높이 무한 가림**을 유지한다. 새 관측은 **실제 시각 메쉬·유한 높이 가림**이다. 두 결과는 반드시 일대일로 대응하거나 같은 개수를 가질 필요가 없다.

- `center delta`: 표면점과 대응 OBB 중심의 3D 거리. 표면과 내부 중심이므로 0을 목표로 하는 오차가 아니다.
- `front delta`: 같은 YZ 위치에서 기존 OBB 앞면 X와 새 메쉬 표면점 X의 차이. 박스와 실제 형상 사이의 오차를 포함한다.
- `unmatched`: 원본 물체와 노출 횡구간 기준으로 대응 가능한 기존 조각이 없다. 다른 물체와 강제 매칭하지 않는다.

source index는 벤치 평가용 메타데이터다. 두 관측 방법의 독립 ID를 동일하다고 가정하지 않는다. 대표점의 횡위치에 해당하는 기존 조각만 연결하며 전체 segmentation instance의 일대일 matching 성능을 평가하는 지표가 아니다.

## 성능·한계

USD 로컬 메쉬를 캐시하고, 삼각형 연결/소유자가 같으면 BVH를 refit한다. 활성 물체 구성이 바뀌면 BVH를 다시 만든다. 현재 테스트에서 초기 컴파일 이후 약 8만~14만 광선의 GPU 질의는 약 1 ms, 가시 영역 분리·대표점 선택·기존 OBB 비교까지 약 18~30 ms였다. 최초 Warp 컴파일·처음 쓰는 자산 추출·Isaac Sim 시작 시간은 별도다.

1 mm 격자는 1 mm 센서 정확도를 보장하지 않는다. 작은 영역의 연결 여부와 경계는 샘플링에 따라 변한다. 더 작은 간격은 광선 수를 면적에 비례해 늘린다. 최대 200만 광선 제한을 넘으면 간격을 키우라는 오류를 반환한다.

USD Mesh의 삼각형/볼록 polygon 면은 삼각분할하고, native Cube는 정확한 면을 사용한다. native Cylinder/Cone은 128분할 삼각형 근사다. 지원하지 않는 visible primitive는 박스로 조용히 대체하지 않고 오류를 낸다. 현재 자산의 숨겨진 collision proxy는 제외한다. 재질 투명도·alpha cutout·subdivision/displacement로 변형된 최종 렌더 표면, 실제 RGB-D noise와 segmentation 오류는 모델링하지 않는다. 원근 관측 센서는 아직 구현하지 않았으며 자유 원근 뷰는 검토용이다.

## API·재현

```python
from surface_point.mesh import MeshObserver
observer = MeshObserver(device="cuda:0", spacing=.001, representative="mask-centroid")
# 실제 월드 메쉬 vertices, triangle faces, triangle별 owner, 기존 비교용 boxes
frame = observer.observe(vertices, faces, face_owner, boxes)
for record in frame.records:
    observation = record.patch.observation
    print(observation.observation_id, observation.point)
# frame.owner: 기하 가시 mask; frame.hits: 광선별 월드 교점
```

triangle owner는 물체 목록 인덱스이며, -1은 선반 같은 가림 전용 형상이다. 배경 광선은 owner=-2다. 좌표 단위는 m이며 `bbox_yz` 순서는 `(ymin,zmin,ymax,zmax)`다. `frame.pixels`는 실제 선택된 격자 행/열과 대표 위치·픽셀 수를 기록한다. 최종 정책 관측에 원본 owner나 mesh를 전달하지 않는다.

`--validate`는 `outputs/surface_point_mesh/`에 배치별 `*_mesh_*.npz`, `*_mask_*.png`, `*_mesh_gpu.json`을 기록한다. `--capture-validation`은 첫 배치의 실제 검토 뷰포트 정면/사선 이미지를 추가한다. 저장한 메쉬는 Isaac Sim 없이 재생할 수 있다.

```bash
WARP_CACHE_PATH=/tmp/surface_point_warp OPENBLAS_NUM_THREADS=1 python src/surface_point/scripts/evaluate_mesh.py
WARP_CACHE_PATH=/tmp/surface_point_warp PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src/surface_point:src/observed_obb python -m pytest -q src/surface_point/tests src/observed_obb/tests
```

기존 `scripts/evaluate.py`와 `outputs/surface_point/`는 이전 박스 근사 비교 기록이며 새 메쉬 관측의 결과로 해석하지 않는다. 기존 baseline·관측 OBB·Gaussian 패키지 및 최종 학습 환경은 수정하지 않았다.
