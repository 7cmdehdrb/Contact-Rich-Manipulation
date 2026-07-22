# Sweep Shelve Force environment

## Scene and task

- UR5e + inline virtual F/T body + Robotiq 2F-85
- Speedrack shelf USD
- Six-object `RigidObjectCollection`: bottle, two cups, two mugs, can
- Original 2×3 placement at `x={-0.75,-0.60}`, `y={-0.20,0,0.20}`
- One target is selected at reset and swept by `±0.18 m` along world Y
- Objects blocking the selected sweep side are placed on the upper shelf at
  `z=1.8 m`, matching the original random-sweep event concept

## Action

The policy action is the same 12-D variable-stiffness OSC action used by
`sweep_jh`:

```text
[normalized_stiffness(6), relative_cartesian_pose_rpy(6)]
```

Normalized stiffness is mapped from `[-1,1]` to `[20,300]`. Translation and
orientation increments are scaled by `0.025 m/step` and `0.12 rad/step`.
The OSC uses gravity compensation, full inertial decoupling, and effort limit
clamping at `90%` of the UR5e limits. The gripper is not a policy action.

## Observation

The concatenated policy observation is 41-D.

| Term | Dimension | Description |
|---|---:|---|
| `eef_pose` | 6 | EEF `xyz + RPY` in robot-base frame |
| `eef_twist` | 6 | EEF linear + angular velocity in robot-base frame |
| `ft_sensor` | 6 | `[Fx,Fy,Fz,Tx,Ty,Tz]` in robot-base axes |
| `initial_target_pose` | 6 | Selected object's reset-time `xyz + RPY` in robot base |
| `desired_motion` | 5 | direction 2 + distance + desired force + tolerance |
| `last_action` | 12 | Previous OSC action |

Joint state, current target pose, object width, and Cartesian goal position are
not policy observations. The F/T reference point remains the virtual sensor
origin; only its expression axes are rotated to robot base.

The 5-D desired-motion command is:

```text
[direction_x_base, direction_y_base, 0.18, desired_force_N, tolerance_N]
```

Shelf sweep direction remains world `±Y` and is transformed into robot-base
axes. Desired force is sampled from `[8,25] N`, with tolerance `[3,6] N`.

## Reward formulation

Only six task-level terms are exposed to the reward manager. Related low-level
components are combined inside each term to keep weight tuning tractable.
Continuous values are bounded with `tanh`, Gaussian kernels, smoothstep gates,
or clamping.

| Group | Term | Range | Weight |
|---|---|---:|---:|
| approach | `approach_error` | `[0,1]` penalty | `-1.0` |
| force | `force_control_error` | `[0,1]` penalty | `-2.0` |
| motion | `sweep_task` | `[-0.7,1]` reward | `+6.0` |
| scene | `scene_safety` | `[0,1]` penalty | `-1.0` |
| action | `action_smoothness` | `[0,1]` penalty | `-0.5` |
| terminal | `failure` | `{0,1}` penalty | `-1000.0` |

The moving push pose retains the original shelf geometry: it is offset from
the selected object by its width opposite the sweep direction, `-0.02 m` in
world X, and `+0.09 m` in world Z. `approach_error` combines normalized push
position and orientation errors at a `0.7:0.3` ratio. It fades out after
proximity and planar force indicate contact.

Planar axial force is computed in robot-base axes. Base Z is excluded so the
tool gravity load is not interpreted as planar object contact. Force tracking
uses the command values as

```text
exp(-((measured_axial_force - desired_force) / tolerance)^2)
```

`force_control_error` is activated by push-pose proximity and combines
`1 - tracking_quality` with `0.3 * tanh(tangential_force / 25)`. It is an error
penalty rather than a positive contact reward, preventing the policy from
collecting reward by holding force without moving the object.

`sweep_task` uses the step-to-step change of a goal-distance potential, so
movement toward the full Cartesian goal is positive while lateral motion and
overshoot that increase goal distance are negative. This progress component
and the low-speed stopped-at-goal component are mixed at a `0.7:0.3` ratio.

`scene_safety` combines shelf interference and non-target-object motion.
`action_smoothness` combines pose and stiffness action rates, with stiffness
assigned half the relative coefficient of pose.

The following command diagnostics remain separate from reward terms and are
logged for TensorBoard analysis:

- `endpoint_error_m`, `progress_ratio`, `object_speed_mps`
- `push_pose_error_m`, `contact_gate`
- `axial_force_N`, `force_error_N`, `tangential_force_N`

The terminal failure term covers object drop/flip, excessive target speed,
shelf collision, arm overspeed, and excessive wrist wrench. With
`step_dt=0.02 s`, its configured weight contributes `-20` on the termination
step and prevents deliberate early termination from avoiding the task.

Isaac Lab multiplies every configured term by `step_dt=0.02 s` before adding
it to the per-step reward.

## Termination

- timeout: `10 s`
- any object below `1.04 m` or tilted over `0.9 rad`
- target speed over `0.3 m/s`
- shelf collision/proximity condition
- any arm joint speed over `1.0 rad/s`
- wrist force over `100 N` or wrist torque over `15 N m`

## PPO

The original random-task settings are retained with the current RSL-RL model
configuration API: 36 steps/environment, 90,000 iterations, `[256,128,64]`
actor and critic, `gamma=0.98`, `lambda=0.95`, and TensorBoard logging.
