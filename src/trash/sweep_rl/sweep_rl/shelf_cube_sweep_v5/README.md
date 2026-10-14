# Standalone Shelf Cube Sweep v5

`Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v5`의 독립 구현이다. 다른 sweep task의
환경 설정이나 MDP 함수에 의존하지 않는다.

Policy observation은 39차원이다.

```text
joint_pos 6 + joint_vel 6 + pose_command 7
+ target_object_position 3 + cube_width 1
+ current_ee_pose 7 + sweep_goal_position 3 + actions 6 = 39
```

Action은 6D default-relative arm joint-position target이다. Runner action clip과
joint-target rate limiter는 사용하지 않으며, gripper 8개 joint는 매 control step
`0.0` open target으로 고정한다. Physics는 100 Hz, policy는 50 Hz이며 episode는
500 policy step, 즉 10초다.

Reach와 push zeta는 example random sweep과 동일한 하나의 contact point를 사용한다.
고정 `+Y` Sweep에서 contact point는 현재 Cube 중심 기준
`[-0.02, -0.08, +0.09] m`이다. staged latch, dwell, target interpolation은 없다.
TCP가 이 점에 가까워지면 Reach reward가 증가하고, TCP 거리와 wrist Y 오차가
각각 4 cm 미만이면 같은 step에 push zeta가 활성화된다. Push progress는 현재
goal 거리의 절대값을 반복 지급하지 않고, 방향이 적용된 step 간 Cube 이동량
`Δy / (0.18 × step_dt)`를 사용한다. 따라서 Cube가 정지하면 progress reward는
0이고 역방향으로 움직이면 음수가 된다. RewardManager의 `dt` 적용 후 episode
누적 progress는 실제 정규화 이동량 `ΣΔy / 0.18`과 일치한다.

Reward weight는 Reach `3.0`, EE local-Y/shelf local-Z alignment `2.0`, Push `6.0`,
Home `9.0`, action-rate와 arm joint-velocity 각각 `-0.03`이다. Contact-sensor 기반
shelf collision penalty는 `-10.0`을 유지한다. Home reward는 Sweep goal과 Cube의
YZ 거리에 대한 tanh gate와 앞 5개 arm joint의 default-pose error를 사용한다.

PPO는 example random sweep 설정을 현재 RSL-RL API로 옮긴다. Actor와 critic은
모두 `256-128-64`, rollout은 36 step/env, learning rate는 `1e-3 adaptive`,
`gamma=0.98`, `desired_kl=0.02`, entropy coefficient는 `0.005`다. Gaussian은
6D direct scalar std와 `init_std=1.0`을 사용하며 std floor나 log 변환은 없다.

성공 termination은 없다. Timeout 외에는 Cube 중심 높이 1.04 m 미만, Cube
local-up 기울기 0.9 rad 초과, Cube 선속도 0.3 m/s 초과, arm 관절 중 하나의
속도 1.0 rad/s 초과를 failure termination으로 사용한다.

```powershell
C:\Users\3cmde\miniconda3\envs\env_isaaclab\python.exe .\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v5 --num_envs 2048 --headless
```

Current contact and stability settings:

- The contact target is the center of the Cube's negative-Y face: `[0.0, -0.04, 0.0] m` relative to the Cube center.
- The Cube center of mass is offset by `[0.0, 0.0, -0.08] m`.
- Robot articulation self-collision is enabled explicitly.
