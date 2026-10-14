# SweepPolicyCube 환경 원칙

적용 대상은 Gym task `Isaac-Sweep-Object-UR5e-SweepPolicyCube-v0`와 이를 구현하는
현재 디렉터리 전체다.

이 문서는 환경을 분석하거나 수정할 때 유지해야 하는 필수 원칙을 기록한다. 새로운
원칙은 사용자가 명시적으로 추가를 요청할 때만 반영한다.

## 1. Gripper는 관측하거나 제어하지 않는다

- 정책 observation에 gripper joint position, joint velocity, opening state 등
  gripper 자체의 상태를 포함하지 않는다.
- 정책 action에 gripper command를 포함하지 않는다. Action은 UR5e arm 6축만
  제어한다.
- Gripper는 모든 episode에서 자산이 정의한 최대 open 자세를 유지한다.
- 최대 open joint target은 reset 시에도 다시 적용하며, implicit actuator의
  stiffness와 damping으로 해당 자세를 유지한다.
- Robot/gripper 자산을 변경해 open joint 값의 의미가 달라지면, 새 자산의 실제
  maximum-open 값을 확인하여 고정값을 함께 갱신한다.
- End-effector pose처럼 arm task 수행에 필요한 기준 frame 정보는 gripper joint
  상태가 아니므로 사용할 수 있다. 다만 이를 통해 gripper 개폐 상태를 우회적으로
  관측하거나 제어해서는 안 된다.

