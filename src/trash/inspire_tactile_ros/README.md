# Inspire 촉각 ROS 2 발행·시각화

Isaac Lab의 17개 접촉 센서를 ROS 2로 발행하는 `bridge` 노드와,
이를 구독하는 별도 matplotlib `viewer` 노드다.
원본 URDF는 변경하지 않는다. 시뮬레이션 패키지는 [inspire_tactile](../inspire_tactile/README.md).

## 실행 (저장소 루트에서)

ROS 노드는 **시스템 Python 3.10 / ROS Humble**, 시뮬레이션은 **env_isaaclab Python 3.11**을 쓴다.
따라서 Sim 내부에 Humble `rclpy`나 ROS Bridge 확장을 로딩하지 않고,
로컬 UDP로 작은 JSON 패킷을 전달해 별도 ROS 프로세스에서 발행한다.
현재 머신에는 rclpy, visualization_msgs, matplotlib, Tk가 설치되어 있다.
다른 머신에는 ROS Humble과 `python3-matplotlib`, `python3-tk`가 필요하다.

한 번 빌드:

```bash
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m colcon build --base-paths src/inspire_tactile_ros \
  --packages-select inspire_tactile_ros --symlink-install
```

터미널 1 — ROS 발행 노드:

```bash
source /opt/ros/humble/setup.bash
source install/inspire_tactile_ros/share/inspire_tactile_ros/package.bash
ros2 run inspire_tactile_ros bridge
```

터미널 2 — ROS 구독 시각화 노드:

```bash
source /opt/ros/humble/setup.bash
source install/inspire_tactile_ros/share/inspire_tactile_ros/package.bash
ros2 run inspire_tactile_ros viewer
```

터미널 3 — Isaac Lab:

```bash
env -u PYTHONPATH -u ROS_PACKAGE_PATH -u AMENT_PREFIX_PATH \
  /home/min/miniconda3/envs/env_isaaclab/bin/python src/inspire_tactile/scripts/play.py --ros2
```

각 터미널은 `~/7cmdehdrb/grad`에서 실행한다. ROS 노드끼리는 동일한 `ROS_DOMAIN_ID`를 사용해야 한다.
전체 `install/setup.bash` 대신 이 패키지의 `package.bash`만 읽어, 기존 다른 패키지의 손상된 overlay를 피한다.
`ros2` 명령은 `/opt/ros/humble/bin/ros2`의 시스템 Python을 사용한다.
`bridge`는 한 번만 실행한다(동일 UDP 포트 중복 바인딩 불가).

빌드 없이도 실행할 수 있다. ROS 설정을 읽은 별도 터미널에서:

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD/src/inspire_tactile_ros:$PYTHONPATH"
/usr/bin/python3 -m inspire_tactile_ros.bridge
# 다른 터미널에서 동일한 source/export 후:
/usr/bin/python3 -m inspire_tactile_ros.viewer
```

## 표시 의미

- 왼쪽: 손가락별 센서 ID 배치도(실측 형상이 아닌 설명용), 힘 L2 norm [N].
- 오른쪽: 실제로 움직이는 17개 **센서 링크 원점**의 world 좌표 [m]. 마우스로 회전할 수 있다.
- 초록: `norm >= threshold_N`, 기본 0.01 N. 회색: OFF.
- 주황: 마지막 수신 후 1초 경과. 이전 접촉 상태를 현재 상태로 오인하지 않도록 힘 표시를 숨긴다.
- 수신 전에는 Waiting, 메시지가 끊기면 STALE을 표시한다. 센서 배열은 팔 자세나 PhysX 탐색 순서와 무관하게 고정 ID를 쓴다.

센서 위치는 **충돌면의 정확한 접촉점 또는 압력 중심이 아니다**.
여러 접촉은 채널별 하나의 합력으로 축약된다. 힘은 Isaac Lab 2.3.2의 접촉 법선 힘이며,
마찰력 전체 또는 실물 taxel 데이터가 아니다.
기본 `virtual regions` 모드는 원시 센서 body 힘에, 실제 접촉 위치로 배정한 부모 링크의
접촉 힘을 더한다. 충돌체와 물리 설정은 바꾸지 않는다. `--tactile_mode raw`로 실행한
시뮬레이션은 예전 센서 body만 읽으며 viewer에 `raw bodies`로 표시된다.
가상 영역의 범위·제약은 [시뮬레이션 README](../inspire_tactile/README.md)를 참고한다.
새 표시를 적용하려면 실행 중인 bridge/viewer를 재시작한다.

## 토픽

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/inspire/tactile` | `std_msgs/msg/String` | 아래 version 1 JSON; viewer가 구독하는 정식 상태 메시지 |
| `/inspire/tactile/binary` | `std_msgs/msg/UInt8MultiArray` | 17개 ON/OFF, dimension label에 `env_<id>:sensors` |
| `/inspire/tactile/markers` | `visualization_msgs/msg/MarkerArray` | world 위치의 구/번호·이름 라벨, RViz용 |

RViz에서는 Fixed Frame을 `world`로 지정하고 MarkerArray display에 위 토픽을 선택한다.
표시용 marker는 새 데이터가 없으면 1초 뒤 사라진다. 별도 TF/로봇 모델 발행은 포함하지 않는다.
`binary`는 편의용이고 타임스탬프/이름은 없으므로 기록·동기화에는 정식 JSON 메시지를 사용한다.

각 메시지는 **한 환경의 전체 17채널**을 담는다:

```text
version: 1
session: 시뮬레이션 실행마다 바뀌는 UUID hex
sequence: 샘플 순번 (세션별 증가; 환경들은 같은 순번 공유)
env_id: 환경 번호
time_s: 물리 스텝 완료 후 시뮬레이션 시간 [s]
phase: open / close / hold / release / released / manual
ros_stamp: {sec, nanosec}, bridge의 ROS 수신/발행 시각
frame_id: world
position_kind: sensor_link_origin
names: [17개 센서 링크 이름]
positions_w_m: [17, 3]
forces_w_N: [17, 3]
norms_N: [17]
active: [17], 0 또는 1
threshold_N: 0.01 (실행 옵션에 따라 변경)
force_source: contact_region_projection / sensor_body_raw (옛 메시지는 raw로 취급)
region_tolerance_m: 0.002 (기본 projected 관측 허용 거리; 물리 접촉 여유 아님)
```

시뮬레이터는 `/clock`을 발행하지 않는다. 기본 ROS marker timestamp는 wall clock이며,
시뮬레이션 시간은 JSON의 `time_s`로 구분한다. 노드의 `use_sim_time`은 기본 false로 둔다.
기본적으로 reliable, volatile, queue depth 10이다. bridge는 중복/역순 패킷을 제거하고
한 타이머 호출에서 환경별 최신 샘플만 발행한다. 손실 없는 전수 수집용이 아니다.

## 옵션 및 확인

```bash
# 환경 1만 보기 (Sim은 --num_envs 2 이상)
ros2 run inspire_tactile_ros viewer --ros-args -p env_id:=1

# 포트 변경: Sim에서도 --ros_port 9871 사용
ros2 run inspire_tactile_ros bridge --ros-args -p port:=9871

# 콘솔에서 원시 메시지 확인
ros2 topic echo /inspire/tactile
ros2 topic echo /inspire/tactile/binary

# 디스플레이 없는 환경에서 15초 수신 후 PNG 저장
ros2 run inspire_tactile_ros viewer --headless --duration 15 --output /tmp/tactile.png

# protocol, malformed message, marker 좌표/수명 테스트
PYTHONPATH="$PWD/src/inspire_tactile_ros:$PYTHONPATH" \
  /usr/bin/python3 -m unittest discover -s src/inspire_tactile_ros/tests -v
```

`--headless`에서 데이터가 한 번도 들어오지 않으면 실패로 종료한다.
데이터가 끊긴 경우 최종 PNG에도 STALE 상태가 표시된다.
토픽 prefix는 두 노드 모두 `--ros-args -p topic:=/my/tactile`로 바꿀 수 있다.
UDP는 항상 `127.0.0.1`에만 바인딩하며, 원격 ROS 수신은 ROS의 DDS 설정으로 처리한다.
