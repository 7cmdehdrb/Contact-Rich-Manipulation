# Newton UR5e Reaching

Initial learning package for UR5e point-reaching experiments.

Current status:

- `ReachJointEnv`: policy action is 6D joint-space delta.
- `ReachCartesianEnv`: policy action is 3D EEF displacement, converted internally with differential IK.
- Observation is 21D: joint position, joint velocity, EEF position, target position, target minus EEF.
- The first backend is a UR5e kinematic smoke backend. The Newton URDF adapter should preserve this same env contract.
- UR5e description is expected at `src/Universal_Robots_ROS2_Description`.

Use the `newton` conda environment:

```bash
conda run -n newton python -m src.learning.scripts.train_joint --num-envs 4 --smoke-steps 2
conda run -n newton python -m src.learning.scripts.train_cartesian --num-envs 4 --smoke-steps 2
```

Run short PPO experiments by passing `--iterations N`.
