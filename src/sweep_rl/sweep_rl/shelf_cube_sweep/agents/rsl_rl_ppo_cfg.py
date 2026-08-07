"""PPO configuration for the inherited shelf Cube sweep task."""

from isaaclab.utils import configclass

from sweep_rl.shelf_cube_pre_reach.agents.rsl_rl_ppo_cfg import (
    UR5eGripperShelfCubePreReachPPORunnerCfg,
)


@configclass
class UR5eGripperShelfCubeSweepPPORunnerCfg(
    UR5eGripperShelfCubePreReachPPORunnerCfg
):
    experiment_name = "sweep_shelf_ur5e_gripper_cube"
