"""PPO configuration for Cube-relative pre-reaching."""

from isaaclab.utils import configclass

from sweep_rl.shelf_reach.agents.rsl_rl_ppo_cfg import (
    UR5eGripperShelfReachPPORunnerCfg,
)


@configclass
class UR5eGripperShelfCubePreReachPPORunnerCfg(
    UR5eGripperShelfReachPPORunnerCfg
):
    experiment_name = "reach_shelf_ur5e_gripper_cube_pre_reach"
