# Initial observed OBB inspection

Simple 자산·배치 규칙을 재사용하는 단일 정지 장면 검토 패키지다. 학습 환경이나 정책을 생성하지 않는다. 기존 패키지의 코드는 변경하지 않는다.

## 실행

Isaac Lab과 기존 baseline 의존성이 설치된 Python 환경에서 workspace 루트 기준으로 실행한다.

```bash
python src/observed_obb/scripts/show.py --seed 0
```

뷰어는 직접 닫을 때까지 계속 실행한다. 프레임 수에 따른 자동 종료와 `--frames` 옵션은 제거했다. `--validate` 검사 후에도 화면을 유지한다.

뷰어의 `New placement`는 기존 배치 함수의 후보 중 정면에서 부분 가림이 가능한 장면을 선택해 관측을 생성한다. `Reobserve current placement`는 물체를 움직이지 않고 관측만 다시 계산한다. `Toggle full-box reference`는 주황색 원본 박스 참고 표시를 켜거나 끈다. 초록색 선과 ID·중심 좌표 목록은 관측 스냅샷이다. 화면을 회전해도 관측 방향은 선반 +X에서 -X를 향하도록 고정된다.

뷰어에는 Isaac Sim이 지원하는 GPU와 드라이버가 필요하다. 기하 계산은 NumPy 기반 CPU 함수이며 렌더링이나 센서 데이터에 의존하지 않는다. 선반 경로는 baseline과 같은 `INSPIRE_SWEEP_SHELF_USD_PATH` 설정을 사용한다. 자산을 임의 대체하지 않으며 원본 선반과 YCB 자산에 접근할 수 있어야 한다.

## 관측 계약

`observed_obb.observe`는 기하 중심·크기·yaw로 표현한 활성 물체들의 `Box`와 고정 관측 방향을 받아 불변 `Observation` 튜플을 반환한다. ID는 스냅샷 안에서만 유효하며, 역할 분류·원본 물체 ID·시간에 따른 추적을 제공하지 않는다. 계산용 입력에는 타겟과 활성 장애물을 모두 포함한다.

- 회전 박스의 경계와 경계 교차점을 기준으로 정면 가시 구간을 구한다. 높이에 상관없이 가장 앞 물체가 해당 구간을 가린다.
- 분리된 조각은 합치지 않는다. 각 조각의 노출 폭이 2cm 이상이고 원본 전체 정면 폭의 20% 이상이어야 한다. 필터 탈락 물체도 가림에는 참여한다.
- 살아남은 좌우 구간에서 전체 깊이를 유지해 바닥면을 절단하고, XY 최소 면적 직사각형과 실제 높이로 OBB를 만든다.
- 로컬 자산 기하와 배치 준비는 재사용한다. 관측은 초기 배치·새 배치·명시적 재관측에서만 계산한다.
- USD BasisCurves로 표시하며 관측·참고 표시에는 물리 스키마가 없다. 원본 장면도 rigid-body dynamics를 비활성화한다.

패키지 이름·버튼 이름은 구현 선택이다. 시각화 파일 내보내기·추가 관측 노이즈는 포함하지 않는다. 2026-11-18 후속 요청에 따라 검토용 배치만 정면 투영이 일부 겹치는 후보로 제한한다. 물체 사이 여유와 선반 경계는 기존 샘플러가 보장하며, 학습 배치 분포는 변경하지 않는다. 후보는 최대 256회 확인하고 조건을 만족하지 못하면 오류를 알린다. 다른 물체의 추가 가림까지 포함한 최종 조각 수는 장면마다 다를 수 있다.

## 검증

계산 테스트는 승인된 ‘배치된 물체 정보 → 독립 관측 ID와 OBB 스냅샷’ 경계에서 실행한다.

```bash
PYTHONPATH=src/observed_obb python -m pytest src/observed_obb/tests -q
```

라이브 장면 검증은 GPU와 원본 자산이 있는 환경에서 실행한다.

```bash
python src/observed_obb/scripts/show.py --validate --seed 0
```

이 검증은 실제 자산과 OBB의 기하, 선 표시, 비충돌 스키마, 정지 상태, 카메라 이동·토글 중 스냅샷 유지, 재관측과 새 배치를 확인한다. 성공 시 `[ObservedOBB] Live validation passed`가 출력된다. Isaac Sim 초기화 실패는 종료 코드가 0일 수도 있으므로 종료 코드만으로 통과 판정하지 않는다. 최종 화면 모양과 버튼 조작은 열린 뷰어에서도 확인한다.

실행기는 startup 기하 준비, 관측 계산, USD 표시 갱신 시간을 각각 출력한다. 표시 갱신 시간에는 GPU 프레임 렌더 시간이 포함되지 않는다. 수치 성능 보장은 추가하지 않는다.

2026-11-18 현재 계산 테스트·타입 검사·CPU USD 장면 검증을 수행했다. GPU 드라이버가 없어 실제 Simple 장면의 렌더링과 라이브 검증은 미완료다. 상세 결과는 [개발 로그](../../example/Contact-Rich-Manipulation-Context/02_implementation/logs/2026-11-18_observed_obb_implementation.md)에 기록한다.
