"""PPO configuration for the inherited shelf Cube sweep task."""

from copy import deepcopy

from isaaclab.utils import configclass
from sweep_rl.shelf_cube_pre_reach.agents.rsl_rl_ppo_cfg import (
    UR5eGripperShelfCubePreReachPPORunnerCfg,
    UR5eGripperShelfCubePreReachV1PPORunnerCfg,
)


@configclass
class UR5eGripperShelfCubeSweepPPORunnerCfg(UR5eGripperShelfCubePreReachPPORunnerCfg):
    experiment_name = "sweep_shelf_ur5e_gripper_cube"


@configclass
class UR5eGripperShelfCubeSweepV1PPORunnerCfg(
    UR5eGripperShelfCubePreReachV1PPORunnerCfg
):
    experiment_name = "sweep_shelf_ur5e_gripper_cube_v1"


@configclass
class UR5eGripperShelfCubeSweepV2PPORunnerCfg(UR5eGripperShelfCubeSweepV1PPORunnerCfg):
    experiment_name = "sweep_shelf_ur5e_gripper_cube_v2"


@configclass
class UR5eGripperShelfCubeSweepV3PPORunnerCfg(
    UR5eGripperShelfCubeSweepV2PPORunnerCfg
):
    """V2 PPO with only the learning-rate policy stabilized."""

    experiment_name = "sweep_shelf_ur5e_gripper_cube_v3"
    algorithm = deepcopy(UR5eGripperShelfCubeSweepV2PPORunnerCfg().algorithm)
    algorithm.learning_rate = 3.0e-4
    algorithm.schedule = "fixed"


@configclass
class UR5eGripperShelfCubeSweepV4PPORunnerCfg(
    UR5eGripperShelfCubeSweepV3PPORunnerCfg
):
    """V3 runner with the original Sweep-Policy actor capacity."""

    experiment_name = "sweep_shelf_ur5e_gripper_cube_v4"
    actor = deepcopy(UR5eGripperShelfCubeSweepV3PPORunnerCfg().actor)
    actor.hidden_dims = [256, 128, 64]
