# Shelf Object Gallery

Isaac Lab 전시 장면이다. 최신 Final 환경의 선반과 형상 생성 코드를 재사용하여 **절차적 물체 4종과 외부 YCB 물체 4종, 총 8종**을 기준 크기로 배치한다. 기본은 2×4이며, 더 큰 격자는 이 순서로 반복한다.

| 행 | 왼쪽부터 오른쪽 |
| --- | --- |
| 전면 행 | 원통 → 직육면체 → 머그컵 → 병 |
| 후면 행 | 크래커 상자 → 설탕 상자 → 수프 캔 → 커피 캔 |

## 실행

Isaac Lab과 Isaac Sim이 설치된 Python 환경에서 workspace root 기준으로 실행한다. checkout에서 직접 실행할 수 있으므로 별도 패키지 설치는 필요하지 않다.

```powershell
python src/shelf_object_gallery/scripts/show.py
python src/shelf_object_gallery/scripts/show.py --rows 2 --cols 4
```

설치하여 사용할 경우 먼저 `inspire-sweep-baseline`을 설치한다.

```powershell
python -m pip install -e src/inspire_sweep_baseline
python -m pip install -e src/shelf_object_gallery
```

| 옵션 | 기본값 | 의미 |
| --- | --- | --- |
| `--rows` | 2 | 깊이 방향 행 수, 전면부터 뒤쪽 |
| `--cols` | 4 | 좌우 방향 열 수, 정면에서 왼쪽부터 오른쪽 |
| `--gap` | 0.05 | 가장 큰 footprint 기준 셀 사이 간격, m |
| `--shelf-usd` | Final 환경의 경로 | 동일 선반의 로컬 복사본 또는 명시적 USD 경로 |
| `--steps` | 0 | 0이면 창을 닫을 때까지 유지, 양수이면 지정 step 후 종료 |
| `--validate` | 비활성 | 실제 장면의 치수·배치·고정 pose·로봇 부재 검사 |
| `--export-usd` | 없음 | 전시 장면 USD 내보내기 |
| `--screenshot` | 없음 | 기본 카메라 구도의 PNG 저장, 카메라 렌더링 자동 활성화 |

`--device`, `--headless` 등 Isaac Lab AppLauncher 옵션도 지원한다. 검증 실행 예시는 다음과 같다.

```powershell
python src/shelf_object_gallery/scripts/show.py --headless --steps 20 --validate
python src/shelf_object_gallery/scripts/show.py --headless --steps 20 --screenshot outputs/gallery.png
```

## 선반과 기준 크기

기본 선반은 활성 환경과 동일한 `SHELF_USD_PATH`를 사용한다. `INSPIRE_SWEEP_SHELF_USD_PATH`와 `SWEEP_POLICY_SHELF_USD_PATH` 환경변수도 그대로 적용된다. 원본 서버에 접근할 수 없다면 같은 자산의 로컬 복사본을 `--shelf-usd`로 지정한다. 임의 대체 선반을 자동으로 만들지는 않는다.

작업면은 400×1000 mm, 높이 1050 mm이며 선반 위치도 Final과 같은 `(-0.7, 0, 0)` m다. 물체는 1.0배이며 크기·배치 무작위화가 없다.

| 물체 | 기준 크기 |
| --- | --- |
| 원통 | 직경 80 mm × 높이 100 mm |
| 직육면체 | 60×80×100 mm |
| 머그컵 | 손잡이 포함 외곽 101×70×100 mm |
| 병 | 외곽 60×60×135 mm |
| 크래커 상자 | 약 109.93×48.12×143.04 mm |
| 설탕 상자 | 약 94.65×46.09×180.00 mm |
| 수프 캔 | 약 84.82×84.89×127.69 mm |
| 커피 캔 | 약 84.89×84.81×116.21 mm |

YCB는 Final과 동일한 외부 USD의 시각 mesh·텍스처와 별도 충돌 proxy를 사용한다. `size_fraction=1.0`으로 정규화한 뒤 추가 크기 변형은 적용하지 않는다. 위 YCB 수치는 현재 Isaac 5.1 자산을 측정한 값이며, 실행 시에는 실제 source spawner의 bounds를 다시 측정하므로 표의 반올림 값을 배치에 사용하지 않는다. 외부 자산 서버 또는 해당 USD·텍스처 캐시에 접근할 수 있어야 한다.

물체 외곽 중심을 균일한 셀 중심에 맞추고 바닥을 작업면에 맞춘다. 선반 가장자리 여유는 20 mm다. 격자가 들어가지 않으면 크기를 줄이지 않고 오류로 안내하므로 행·열 수 또는 간격을 조정한다.

선반과 물체는 kinematic으로 고정되어 전시 중 이동하지 않는다. 원본 시각·충돌 형상을 유지하며 로봇·정책·보상·학습 설정은 생성하지 않는다. 형상을 구분하기 위해 종류별 색상을 사용한다.

## 구조

- `shelf_object_gallery/layout.py`: 시뮬레이터와 독립적인 격자 배치·크기 검사.
- `shelf_object_gallery/scene.py`: 원본 spawner 재사용, Isaac Lab InteractiveScene 구성.
- `scripts/show.py`: AppLauncher, 카메라와 실행 루프.
- `tests/test_layout.py`: 외곽 경계·비대칭 중심 보정·간격·초과 크기 검사.
