# Constant-velocity sweep training

`Isaac-Sweep-Object-UR5e-OSC-ConstantVelocity-v0` is derived directly from
`UR5eOscSweepEnvCfg`. It replaces the force command with:

```text
[direction_x, direction_y, distance_m, target_speed_mps]
```

The target speed is fixed at `0.08 m/s` for the first experiment. The desired
profile starts at 25% of cruise speed, ramps up over the first `0.025 m`,
cruises at the target speed, and ramps down over the last `0.04 m`. Success
requires endpoint error below `0.02 m`, object speed below `0.02 m/s`, and a
continuous `0.30 s` dwell.

Policy observations contain arm state, EEF pose, initial/current object pose,
object linear velocity, the four-dimensional motion command, and the last
action. F/T wrench, contact point, desired contact force, and force tolerance
are not exposed to the policy. The gripper target is held at the fully-open
joint position on every physics step.

Positive transit velocity reward is gated by actual forward object motion, so
a stationary object receives zero velocity reward. Endpoint distance is a
normalized negative running cost, and the pre-contact pose term is also a
penalty relative to the moving object. Early safety termination is charged for
the remaining episode horizon, so deliberately triggering the wrench limit
cannot avoid these running costs. Small target/central-contact bridge rewards
help discover contact, while the larger contact reward requires actual forward
object motion. Consequently, alignment or stationary contact without object
progress cannot form a positive-reward local optimum. TensorBoard reports
`endpoint_error`, `forward_speed`, and `progress_ratio` under
`Metrics/desired_motion`.

The complete Action, Observation, and Reward specification is in
[`constant_velocity_action_reward_observation.md`](constant_velocity_action_reward_observation.md).

Train with:

```bash
./IsaacLab/isaaclab.sh -p \
  src/sweep_rl/scripts/train_constant_velocity.py \
  --device cuda:0 \
  --num_envs 2048 \
  --headless
```
