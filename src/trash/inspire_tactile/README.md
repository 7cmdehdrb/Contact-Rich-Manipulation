# UR5e + Inspire Hand 촉각 확인 데모

Isaac Lab에서 UR5e와 왼손 Inspire Hand를 한 articulation으로 조립하고, 작은 원통을
정해진 위치에서 손으로 잡았다 놓으며 17개 센서 영역의 이진 접촉 배열을 출력한다.
`example/Contact-Rich-Manipulation-Context`의 **촉각 센싱 feasibility 확인**을 위한 패키지다.

팔은 일정한 관절 자세를 유지한다. 원통의 초기 위치는 고정이며, 손만
`open → close → hold → release → released`를 11초 주기로 반복한다.
현재 원통은 지름 80 mm, 높이 90 mm, 질량 0.32 kg의 **동적 rigid body**이고 받침대 위에 놓인다.
반지름·질량은 사용자가 조정한 값을 유지한다. 값은 `scene.py`의 실린더 설정에서 변경한다.
닫는 도중에는 물체 위치/관절 상태를 강제로 덮어쓰지 않는다. 매 주기 시작 시 같은 초기 상태로 리셋한다.
물체를 들어 옮기는 pick-and-place나 학습 정책은 이 데모의 범위가 아니다.

## 실행

저장소 루트에서, Isaac Lab이 설치된 `env_isaaclab` 환경을 사용한다.
기본 실행에는 ROS 2가 필요 없다. ROS 2 시각화를 함께 쓰려면 아래 절을 참고한다.

```bash
conda activate env_isaaclab
python -m pip install --no-build-isolation -e src/inspire_tactile
python src/inspire_tactile/scripts/play.py
```

Isaac Sim을 먼저 수동으로 열 필요는 없다. 스크립트가 GUI와 환경을 함께 실행한다.
GUI를 닫거나 Ctrl+C를 누르면 종료한다. 최초 실행에는 URDF → USD 변환 시간이 추가된다.

## ROS 2 접촉 시각화

### 손가락 관절 수동 조작

자동 파지 `play.py`와 별도로 `scripts/manual.py`를 추가했다.
기존 로봇·원통·받침대·접촉 센서 설정을 공유하며 **자동 개폐나 주기적 리셋은 하지 않는다**.

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  /home/min/miniconda3/envs/env_isaaclab/bin/python src/inspire_tactile/scripts/manual.py
# 기존 ROS bridge / viewer에 촉각도 표시하려면 끝에 --ros2 추가
```

Isaac Sim 안의 **Inspire Hand - Manual Joints** 창에서 각 슬라이더를 움직이거나
오른쪽 숫자 칸에 각도를 입력한다. UI의 단위는 **도(deg)**이고 내부 명령/로그는 rad다.
엄지 4개, 검지·중지·약지·소지 각 2개, 총 12개 관절을 조작한다.
스크롤하면 아래 손가락이 보인다. 팔 자세는 고정되고 손가락만 바뀐다.

- 기본은 **12관절 독립 모드**다. 원래 mimic 종속 관절도 개별로 움직일 수 있는 시뮬레이션용 모드다.
  실제 Inspire Hand의 독립 구동 자유도가 12라는 뜻은 아니다.
- **Link original mimic joints**를 켜면 기존 6개 주 관절만 조작하고 종속 관절 입력은 비활성화된다.
  `--linked`로 이 모드를 켠 상태에서 시작할 수도 있다.
- **Open / Close**: 기존 열린 자세 / 관절 상한 기반 파지 목표를 지정한다.
  Close는 손의 모든 종속 관절을 각각 최대값으로 지정하는 기능이 아니라 기존 파지 preset이다.
- **Hold actual**: 최근 표시된 실제 관절각을 목표로 유지한다. 비상 정지나 토크 차단이 아니다.
- **Reset scene**: 열린 자세와 원통 초기 위치로 명시적으로 리셋한다.
- 슬라이더 값은 요청각, `command`는 속도 제한 후 명령각, `actual`은 물리 시뮬레이션의 실제 각도다.
  물체에 막히면 실제 각도는 목표에 도달하지 않을 수 있다.
- 요청각은 원본 URDF 관절 제한으로 제한한다. 목표의 변화 속도는 기본 `0.6 rad/s`이고
  `--speed`로 변경할 수 있다(각 관절의 원본 velocity 상한 이하로 제한).
  독립→연동 모드 전환 중에는 급변을 막기 위해 종속 목표가 새 mimic 비율에 서서히 수렴한다.
- self collision은 기존 장면처럼 꺼져 있다. 독립 모드에서 비현실적인 자세/자기 관통이 가능하므로
  이 프로그램은 관절·센서 점검용이며 실제 로봇의 충돌 안전성을 보장하지 않는다.

`--without_object`, `--fix_object`, `--object_lowering`, `--object_offset`, `--threshold`,
`--ros_port`, `--ros_hz`, `--log`도 사용할 수 있다. 수동 모드는 한 환경을 제어한다.
패널 또는 앱을 닫거나 Ctrl+C로 종료한다. Physics를 Pause하면 관절 이동도 멈춘다.

```bash
# 물체 없이 개별 관절 확인
python src/inspire_tactile/scripts/manual.py --without_object
# 초기 검지 두 번째 관절 목표를 30도로 지정
python src/inspire_tactile/scripts/manual.py --joint left_index_2_joint=30
# 창 없이 같은 관절 명령을 4초간 검증 (--duration은 시뮬레이션 시간)
python src/inspire_tactile/scripts/manual.py --headless --duration 4 \
  --without_object --joint left_index_2_joint=30 --log src/inspire_tactile/logs/manual.jsonl
```

`--joint`는 여러 번 지정할 수 있고, 이름의 `inspire_` 접두사는 생략할 수 있다.
headless에서는 조작 창이 없으므로 `--duration > 0`이 필수다.
ROS 메시지는 `phase="manual"`로 발행한다. 이전 실행 중인 bridge/viewer는
새 phase를 인식하도록 **재시작**해야 한다. ROS 노드 구조와 토픽 이름은 그대로다.

검증: 단위 테스트 16개(시뮬레이션/수동 제어 11개, ROS 5개) 통과.
물체 없는 독립 제어 실험에서 검지 2번 목표 30°에 실제 약 30.00°로 도달했고 검지 1번은 0°를 유지했다.
GUI의 연동 체크박스·관절 입력·Open/Close/Hold/Reset 콜백 및 35° 주 관절의 종속 관절 추종을 확인했다.
최종 실행에서 팔의 초기 자세 유지와 ROS `manual` 메시지 수신도 확인했다.
수동 실험 로그와 화면 캡처(`final_panel.png`)는 `logs/manual/`에 저장했다.

### ROS 발행 및 구독

별도 [inspire_tactile_ros 패키지](../inspire_tactile_ros/README.md)에 실행 순서를 정리했다.
Isaac Sim Python 3.11에서 ROS Humble의 Python 3.10 `rclpy`를 직접 import하지 않는다.

```text
Isaac Lab (--ros2) → localhost UDP → ROS 2 bridge → /inspire/tactile → ROS 2 viewer
```

viewer는 17개 센서 번호별 ON/OFF·힘 크기와 실제 센서 링크의 3D 위치를 표시한다.
초록색은 ON, 회색은 OFF, 주황색은 1초 이상 새 메시지가 없는 상태(판단 불가)다.
센서 ID의 2D 배치는 설명용이며, 오른쪽 3D 좌표는 시뮬레이션의 world 좌표다.
여기서 접촉 위치는 **센서 링크 원점**이며 충돌면의 정확한 접촉점/압력 중심은 아니다.

시뮬레이터는 기존 명령에 `--ros2`만 추가한다:

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  /home/min/miniconda3/envs/env_isaaclab/bin/python src/inspire_tactile/scripts/play.py --ros2
```

`--ros_hz 30`은 시뮬레이션 시간 기준 전송 빈도(기본 30 Hz), `--ros_port 9870`은
로컬 UDP 포트다. ROS 토픽이 필요 없으면 `--ros2`를 생략한다.
UDP는 최신 상태 모니터링용으로 일부 샘플이 누락될 수 있다. 240 Hz 전체 접촉 이벤트의 기록용이 아니다.

창 없이 정해진 횟수만 실행하고 신호 확인까지 수행하려면:

```bash
python src/inspire_tactile/scripts/play.py --headless --cycles 2 --check \
  --log src/inspire_tactile/logs/demo.jsonl
```

이 패키지 코드는 기본적으로 `IsaacLab/isaaclab.sh -p src/inspire_tactile/scripts/play.py`로도
실행할 수 있다. 다른 Python 환경에서는 먼저 같은 환경에 `xacro`, `numpy`, `scipy`, `PyYAML`을 설치해야 한다.
`torch`와 Isaac Lab/Sim은 기존 Isaac Lab 환경에서 제공한다.

## 출력과 코드에서 사용하기

### 물리를 보존하는 가상 센서 영역 관측 (기본)

`play.py`와 `manual.py`의 기본값은 `--tactile_mode projected`다.
기존 convex hull 손가락 충돌체가 센서 표면을 덮는 경우, 물체가 손가락에 계속 닿아도
센서 rigid body만 읽으면 0이 될 수 있다. 이를 **충돌 형상을 고치지 않고 관측 단계에서** 처리한다.

- URDF/STL/USD의 충돌 형상, 질량·관성, 마찰, contact/rest offset, 관절·drive·시간 간격은 그대로다.
- 원래 17개 센서 body의 힘은 유지한다. 추가 ContactSensor view로 센서의 부모 링크에
  실제로 발생한 **개별 접촉점과 법선 힘**을 읽는다(접촉점 평균 위치를 쓰지 않는다).
- 접촉점을 현재 센서 좌표계로 변환하여, 해당 부모에 붙은 센서 영역 안에 있을 때만
  그 채널에 힘 벡터를 더한다. 두 영역이 겹쳐도 한 점은 한 채널에만 배정한다.
- 영역은 기존 센서 STL의 convex hull이다. `--region_tolerance 0.002`는 각 경계 평면의
  관측 허용 거리 2 mm다. 부모 hull 안쪽에 묻힌 센서와의 간극을 허용하며,
  **메시 두께나 물리 접촉 여유를 2 mm로 바꾸는 옵션이 아니다**. 엄격한 영역은 `0`으로 설정한다.
- 영역 밖의 접촉은 가까운 센서로 강제 배정하지 않는다. `unmapped_parent_*`로 별도 기록한다.
  따라서 물체와 손이 닿아 있어도 해당 위치에 센서 영역이 없으면 OFF가 정상이다.
- 위치 기반 배정의 상대 필터는 **원통**이다. 현재 GPU backend는 정적 받침대·바닥의
  접촉점 필터를 지원하지 않으므로 이들의 물리 속성을 바꾸지 않고, 부모 링크의 나머지
  합력을 `unobserved_parent_force_w_N`으로 기록한다(미배정 합력에도 포함).
  원시 센서 body의 힘은 여전히 모든 환경 접촉을 포함한다. 새 동적 물체를 추가하면
  `scene.py`의 관측 필터에도 추가해야 한다. 자기 접촉은 기존대로 비활성화 상태다.
- 접촉 데이터 버퍼 초과/누락과 비정상 값은 조용히 0으로 처리하지 않고 실행 오류로 알린다.

이는 **가상 센서 영역별 법선 합력**이며, 실제 독립 센서에 전달되는 반력이나 피부/압력분포 모델은 아니다.
접촉을 만들어내거나 ON을 시간적으로 유지하는 필터는 적용하지 않는다. 실제 미끄러짐·분리,
센서 경계 통과, 임계값 근처의 변화에 따른 깜빡임은 여전히 가능하다.
Isaac Sim의 표시 점, 수동 패널, ROS viewer는 같은 최종 이진 배열을 사용한다.

```bash
# 새 기본 관측 + 기존 ROS 발행
python src/inspire_tactile/scripts/manual.py --ros2
# 비교용: 예전처럼 센서 rigid body 힘만 읽기 (물리 설정은 동일)
python src/inspire_tactile/scripts/manual.py --ros2 --tactile_mode raw
# 매 물리 스텝의 원시/배정 힘과 물체·관절 상태 기록
python src/inspire_tactile/scripts/manual.py --print_hz 240 --log /tmp/tactile.jsonl
```

로그에는 `raw_sensor_force_w_N`, `mapped_force_w_N`, `unmapped_parent_force_w_N`,
`unmapped_parent_contact_count`가 환경 차원을 포함해 별도 기록된다. 마지막 항목은 위치가
관측된 원통 접촉 중 미배정된 점의 수이며, 힘이 0인 manifold 점도 포함한다.
`unobserved_parent_force_w_N`은 위치 필터 밖의 부모 접촉 합력이다. 합력 벡터는 서로
상쇄될 수 있으므로 그 norm을 전체 접촉 강도의 합으로 해석하면 안 된다.
`force_w_N`은 원시 힘과 배정한 부모 접촉 힘의 합이다.
ROS JSON에는 `force_source="contact_region_projection"` 또는 `"sensor_body_raw"`와
`region_tolerance_m`이 실리며 viewer 상단에서 `virtual regions` / `raw bodies`로 구분한다.
실행 중인 bridge/viewer는 새 코드를 읽도록 재시작한다. 토픽 이름과 17채널 순서는 동일하다.

시작할 때 `TACTILE_CHANNELS`로 인덱스와 센서 링크의 대응표를 출력한다.
출력 배열은 `torch.uint8`, shape `(num_envs, 17)`이며 기본 출력 주기는 5 Hz다.
힘 읽기와 이진화는 물리 스텝마다 240 Hz로 수행한다. `--print_hz`는 콘솔/로그 출력 빈도만 바꾼다.

```python
from inspire_tactile.projected_tactile import ProjectedTactileReader

reader = ProjectedTactileReader(scene, urdf, model["sensor_names"], threshold=0.01,
                               tolerance=0.002, dt=1 / 240)
force_w, magnitude, binary = reader.read()
# force_w:   torch.float32, (num_envs, 17, 3), world frame, Newton
# magnitude: torch.float32, (num_envs, 17)
# binary:    torch.uint8,   (num_envs, 17)
array = binary[0].detach().cpu().numpy()  # NumPy (17,)
```

계산식은 다음과 같다. **0.01 N과 같아도 1**이다.

```python
magnitude = torch.linalg.vector_norm(force_w, ord=2, dim=-1)
binary = (magnitude >= 0.01).to(torch.uint8)
```

| 인덱스 | 센서 링크 (`inspire_` 접두사 생략) |
|---|---|
| 0 | `palm_force_sensor` |
| 1–4 | `thumb_force_sensor_1` … `thumb_force_sensor_4` |
| 5–7 | `index_force_sensor_1` … `index_force_sensor_3` |
| 8–10 | `middle_force_sensor_1` … `middle_force_sensor_3` |
| 11–13 | `ring_force_sensor_1` … `ring_force_sensor_3` |
| 14–16 | `little_force_sensor_1` … `little_force_sensor_3` |

인덱스는 PhysX의 링크 탐색 순서와 무관하게 이름으로 재정렬된다.
`--log`는 메타데이터, 출력 시점별 힘 벡터/크기/이진값/관절각/센서 위치/물체 위치,
마지막 실행 요약을 JSONL에 저장한다. 물체 위치는 디버그/검증용 기록이며 손 동작을 결정하는 입력이 아니다.

## 센서 모델과 로봇 조립

- Hand 원본: `src/inspire_robot/urdf_left_with_force_sensor/urdf/urdf_left_with_force_sensor.urdf`.
  원본을 읽어 생성물에서 메시 경로를 해석한다. 원본 URDF/STL은 수정하지 않는다.
- Arm 원본: `src/Universal_Robots_ROS2_Description`의 UR5e xacro, YAML, meshes.
  Nucleus 서버에 있는 로봇 자산에 의존하지 않는다.
- 손에 `inspire_` 접두사를 붙여 UR5e의 `base_link` 등과 이름 충돌을 피한다.
- `tool0`와 Hand `base_link`의 장착면 원점 및 +Z 축을 일치시킨다 (`xyz=0 0 0`, `rpy=0 0 0`).
  이전의 90° 꺾임과 모델링되지 않은 25 mm 간격을 제거했다. 실제 어댑터를 실측한 보정값은 아니다.
  손과 원통의 기존 world 위치·방향은 팔의 고정 관절 자세를 바꿔 유지한다.
  질량 없는 UR 좌표계 링크는 변환을 보존하여 생략한다.
- **17개 센서 링크와 고정 관절은 유지**한다. `merge_fixed_joints=False`,
  `activate_contact_sensors=True`를 사용하고, 초기화 시 17개 rigid body와 ContactReport API를 검사한다.
- ContactReport의 보고 임계값은 0으로 설정한다. `ContactSensorCfg.force_threshold`는
  접촉 시간 추적용 옵션이므로 사용자 요구 임계값은 `tactile.py`에서 직접 적용한다.
- 손의 6개 주 구동 관절과 6개 종속 관절에 대해 원본 mimic 비율로 목표각을 계산한다.
  생성 URDF에서는 `<mimic>`을 제거하여 PhysX mimic 제약과 개별 drive가 중복되지 않게 한다.
  목표각을 연동하는 간단한 시뮬레이션 제어이며 실제 기계적 링크/힘 전달을 재현한 것은 아니다.
- 충돌은 원본 STL의 convex hull을 사용한다. 손 충돌 여유는 0.5 mm,
  손 관절에는 안정화를 위한 armature `0.0001 kg·m²`를 적용한다.
  자가 충돌은 꺼져 있다. 힘은 환경과 손의 실제 접촉에서 나온다.

`net_forces_w`는 센서 rigid body에 작용하는 **접촉 법선 힘들의 합 벡터**다.
센서 영역별로 이를 축약하므로 17개 센서를 만들었다고 항상 17채널 모두 1이 되지는 않는다.
`raw` 모드에서는 비센서 손가락 링크의 힘이 촉각 배열에 들어오지 않는다.
기본 `projected` 모드는 위에 설명한 영역 판정으로 해당 부모 링크의 접촉도 반영한다.
이는 실물 촉각의 taxel grid, 마찰력 전체, 피부 변형, 노이즈나 실제 감도를 보정한 모델은 아니다.
ATI 손목 6축 F/T는 이 패키지에 포함하지 않는다.

센서 의미와 제약은 [Isaac Lab 2.3 ContactSensor 구현](https://isaac-sim.github.io/IsaacLab/v2.3.1/_modules/isaaclab/sensors/contact_sensor/contact_sensor.html)과
[PhysX contact tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)를 참고한다.
프로젝트 배경은 [인계 요약](../../example/Contact-Rich-Manipulation-Context/docs/00_HANDOFF_BRIEF.md)과
[결정·미정 목록](../../example/Contact-Rich-Manipulation-Context/docs/03_DECISIONS_AND_OPEN_QUESTIONS.md)에 있다.

## 대조 실험과 검증

관측 방식만 바뀌었는지 검사하는 재현 절차(Isaac Lab Python):

```bash
python src/inspire_tactile/tests/observation_probe.py --mode raw --output /tmp/raw.jsonl
python src/inspire_tactile/tests/observation_probe.py --mode projected --output /tmp/projected.jsonl
python src/inspire_tactile/tests/compare_observation.py /tmp/raw.jsonl /tmp/projected.jsonl
```

2026-09-18, 반지름 40 mm/질량 0.32 kg, 1초 뒤 Close를 누르는 8초 실험에서
1,920 물리 스텝의 손·팔 관절각, 물체 위치·회전·속도, 원시 센서 힘의 최대 차이는 모두 **0**이었다.
5–8초 유지 구간에서 원시 17채널은 전부 0인 반면, 가상 영역 ID **7, 10, 13, 16**은
모든 샘플에서 ON이었다. 다른 부위의 접촉은 영역 밖이면 미배정으로 남았다.
이는 해당 환경에서의 회귀 검증 결과이며 GPU/드라이버/PhysX 버전 간 bitwise 재현 보장은 아니다.
단위 테스트 24개(시뮬레이션 측 18 + ROS 측 6), 두 환경의 자동 접촉·해제 검사,
최종 코드의 원통 없는 두 환경에서 전 주기 17채널 힘 0 검사를 통과했다.
ROS viewer에 실제 파지 로그를 넣어 `virtual regions` 표시와 ON/OFF 색상도 확인했다.

원통 없이 똑같이 손을 움직이는 대조 실험:

```bash
python src/inspire_tactile/scripts/play.py --headless --cycles 1 --without_object --check
```

`--check`는 기본 모드에서 접촉 발생 후 해제를 검사하고, 원통 없는 모드에서는
전 구간에서 17채널 모두 임계값 미만인지 검사한다. `--cycles`를 반드시 함께 지정한다.
여러 환경/주기를 실행하면 각 환경의 각 주기를 따로 검사한다.
이 검사는 촉각 경로 확인이며, 물체를 공중에서 유지하는 파지 성공 판정은 아니다.

센서 위치만 확인할 때는 `--fix_object`로 원통을 kinematic 상태로 유지할 수 있다.
이 모드는 물리 파지 성공의 근거로 사용하지 않는다.

```bash
# 단위 테스트: 임계값 경계, 비정상 데이터, 채널 순서, 상태기계, 원본 보존/조립
python -m unittest discover -s src/inspire_tactile/tests -v

# 렌더링 이미지도 저장 (headless 카메라를 자동 활성화)
python src/inspire_tactile/scripts/play.py --headless --cycles 1 --check \
  --capture_dir src/inspire_tactile/logs/images
```

`--object_offset X Y Z`는 hand base 좌표계에서 원통 중심 위치를 m 단위로 조정한다.
그 다음 `--object_lowering`만큼 원통과 받침대를 함께 **world Z 방향으로** 내린다.
기본값은 `0.020`(20 mm)이며, 두 물체 사이의 높이 차이는 유지한다.
이 손 자세에서는 hand base의 Z가 수평이므로, 수직 높이는 `--object_lowering`으로 조절한다.

```bash
# 기본: 물체/받침대 20 mm 하강 + 강화된 닫힘 목표
python src/inspire_tactile/scripts/play.py --ros2
# 하강량을 10 mm로 변경 (0이면 이전 높이)
python src/inspire_tactile/scripts/play.py --ros2 --object_lowering 0.01
```

강화된 파지는 주 관절의 닫힘 목표를 원본 URDF 상한으로 설정한다:
엄지 1번 `1.1641 rad`, 엄지 2번 `0.5864 rad`, 나머지 네 손가락 1번 `1.4381 rad`.
종속 관절은 기존 mimic 비율을 유지하고 모든 목표각의 원본 제한을 검증한다.
손 drive는 stiffness `1.2`, damping `0.06`, 토크 제한 `0.20 N·m`로 설정했다
(이전 `0.4`, `0.02`, `0.08 N·m`). 실제 하드웨어의 구동력을 보정한 값은 아니다.
물체 접촉으로 실제 관절각은 목표보다 작을 수 있으며, 강제로 관절 위치를 덮어쓰지 않는다.
반지름·질량·마찰과 기존 팔 자세는 이 변경에서 수정하지 않았다.

메시를 수정한 경우 `--force_conversion`으로 USD를 다시 생성한다.
`generated/`와 `logs/`는 Git에서 제외되어 있으며, 전자는 원본 자산에서 재생성 가능하다.

### 초기 버전 실행 검증 기록 (2026-09-18)

Isaac Lab **2.3.2**, Isaac Sim **5.1.0**, Python **3.11**, RTX 3090 환경에서 확인했다.

| 검사 | 확인 결과 |
|---|---|
| 단위 테스트 5개 | 통과: 임계값 경계/배치 순서/원본 보존 포함 |
| 동적 원통, 1환경 × 2주기 | 닫기/유지 중 접촉 발생, 열기 후 17채널 모두 0 |
| 동적 원통, 2환경 × 2주기 | 두 환경 모두 접촉 후 해제, 출력 shape `(2,17)` |
| 원통 없음, 1환경 × 1주기 | 전체 물리 스텝에서 17채널 모두 0 |
| RGB 캡처 | 실제 UR5e+Hand 조립 및 원통을 감싼 손/열린 손 확인 |
| ROS 환경변수 제거 | `PYTHONPATH`, `ROS_PACKAGE_PATH`, `AMENT_PREFIX_PATH` 없이 실행 통과 |
| 자동 검증 종료 코드 | 통과 시 0, 의도적으로 불가능한 임계값을 지정한 실패 시 1 |

기본 배치에서는 주로 엄지 센서 2·3번이 반응했고 일부 순간에 검지 1번도 반응했다.
17개 모두의 감도를 검증한 결과는 아니다. 접촉 위치에 따라 활성 채널이 바뀐다.
측정값은 수치 해석, 렌더링과 초기화 조건에 따라 달라질 수 있으며 실물 보정값으로 사용하지 않는다.
실행 로그와 캡처는 로컬 `logs/verification/`에 있다.

### 손목 장착 / ROS 2 추가 후 검증 (2026-09-18)

- 단위 테스트 9개 통과: 장착 축·원점·손의 world pose 보존, 이진화, ROS 스키마,
  sender/receiver 호환, 다중 환경 순번, marker 위치·수명 포함.
- 최종 장착 모델, 동적 원통 1환경 × 2주기: 각 주기에서 접촉 후 해제, `--check` 통과.
- 최종 장착 모델, 원통 없는 2환경 × 1주기: 모든 물리 스텝의 17채널 힘이 0, `--check` 통과.
- 실제 시뮬레이션 → UDP → ROS 토픽 → 별도 구독 노드에서 힘·접촉·해제 확인.
- ROS 패키지 빌드, `ros2 run` 실행, matplotlib GUI 및 headless 이미지 저장 확인.

최종 동적 실험에서는 주로 닫는 과정의 엄지 3번(ID 3)이 반응했다.
잡고 있는 동안에도 비센서 표면이 물체를 지지하면 센서 배열은 0일 수 있다.
검증 결과는 파지 성공 지표가 아닌 센서 경로의 접촉/해제 확인이다.
수정 후 로그와 연결부 캡처는 `logs/mount_ros/flush_dynamic.jsonl`,
`logs/mount_ros/negative_batch.jsonl`, `logs/mount_ros/flush_images/`에 있다.

### 물체 하강 / 최대 닫힘 목표 추가 후 검증

- 반지름 32 mm, 질량 1.05 kg을 유지하고 원통/받침대를 함께 20 mm 내렸다.
- 단위 테스트 10개 통과(시뮬레이션 관련 6개 + ROS 관련 4개).
- 동적 원통 2주기에서 관절 발산 없이 접촉 후 해제를 확인했다.
- 원통 없는 1주기에서 전 채널 힘이 0이었다.
- 닫는 과정에 검지·중지·약지·소지 센서 접촉이 기록됐다. 물체를 유지하는 동안에는
  비센서 표면이 지지해 이진 배열이 0일 수도 있으며, 공중 파지 성공을 검사한 것은 아니다.
- 로그: `logs/firm_grasp/dynamic.jsonl`, `logs/firm_grasp/no_object.jsonl`.
  렌더링: `logs/firm_grasp/images/`.

## 파일 구성

```text
inspire_tactile/
  model.py       로컬 UR5e/Hand URDF 조립, mimic 목표각, 센서 순서
  scene.py       로봇, 원통, 받침대, ContactSensor 설정
  controller.py  룰베이스 손 개폐 주기
  placement.py   물체·받침대의 world 수직 하강량 및 배치
  manual_control.py  12관절 개별/6관절 연동 목표, 제한·속도 제한
  manual_ui.py   Isaac Sim 내부 관절 슬라이더·숫자 입력 UI
  tactile.py     접촉력 읽기, L2 norm, 이진 배열 인터페이스
  regions.py     기존 센서 메시에서 관측 영역만 읽기 (물리 형상 수정 없음)
  projected_tactile.py  실제 부모 접촉점의 가상 센서 영역 귀속, 원시 힘/미배정 진단
  transport.py   ROS 없는 Python에서 localhost로 보내는 선택적 전송 경로
scripts/play.py  GUI/headless 실행, 콘솔 출력, 로그, 검증
scripts/manual.py  자동 개폐 없는 수동 관절 제어 전용 실행
tests/test_*.py   시뮬레이터를 띄우지 않는 단위 테스트
tests/observation_probe.py  동일 Close 명령의 raw/projected 물리 궤적 비교용 GPU 실행
```
