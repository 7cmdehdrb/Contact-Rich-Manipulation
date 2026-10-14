# ROS 2와 Python 환경 분리

한 ROS 설치를 여러 Python 가상환경에서 참조할 수 있지만, 각 프로세스에서 Python ABI와
ROS 네이티브 라이브러리의 호환성이 성립해야 한다. `PYTHONPATH`에 ROS 경로가 보이는 것만으로
호환성이 보장되지는 않는다. 이 문서는 기존 일반 지식 노트에서 핵심 조건을 정리한 것이다.

## 실행 환경 선택

apt ROS 2 Humble의 CPython 3.10용 `rclpy` 바이너리는 해당 인터프리터와 호환되는 환경에서
사용한다. `rclpy`에는 컴파일된 확장 모듈이 있으므로 다른 Python minor 버전에
경로만 추가하는 방법으로 해결되지 않는다.
[ROS의 Python 패키지 안내](https://github.com/ros2/ros2_documentation/blob/humble/source/How-To-Guides/Using-Python-Packages.rst)를 따른다.

```bash
# 시스템 ROS Python과 호환되는 가상환경을 만들 때의 예시
/usr/bin/python3 -m venv --system-site-packages /path/to/ros-venv
/path/to/ros-venv/bin/python -m pip install -r /path/to/requirements.txt
```

Conda에서 Python 버전 숫자만 맞춘 것으로 apt ROS와의 직접 연동을 검증했다고 보지 않는다.
Isaac Lab Python과 시스템 ROS Python이 다른 런타임을 요구하면 각각의 프로세스로 실행하고
ROS 어댑터에서 통신하는 구성을 검토한다.

## 확인할 항목

| 항목 | 확인하는 이유 |
| --- | --- |
| `sys.executable`, Python minor 버전 | 실제 실행 인터프리터와 확장 모듈 ABI 확인 |
| `sys.path`, `PYTHONPATH` | 모듈 검색 경로 확인 |
| `AMENT_PREFIX_PATH`, workspace setup | 어떤 ROS 설치와 overlay를 참조하는지 확인 |
| 공유 라이브러리 경로 | C/C++ 런타임 충돌 확인 |
| 토픽·메시지 타입·QoS·discovery 설정 | 프로세스 간 통신 조건 확인 |

가상환경은 Python 패키지를 분리한다. GPU 드라이버, 커널, USB 권한, 네트워크와 장비 전원은
별도로 관리한다. 여러 환경을 한 Python 프로세스에서 동시에 import하는 것으로 환경 분리를
대체할 수는 없다.

Python의 [venv](https://docs.python.org/3.10/library/venv.html),
[ABI 설명](https://docs.python.org/3.10/c-api/stable.html),
colcon의 [환경 설정](https://colcon.readthedocs.io/en/released/developer/environment.html)을 참고한다.

[일반 지식 안내](README.md)로 돌아간다.
