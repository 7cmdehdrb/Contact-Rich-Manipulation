"""Shelf Cube Sweep task inherited from the collision-safe PreReach task."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from sweep_rl.shelf_cube_pre_reach.env_cfg import (
    BEHIND_WIDTH_SCALE,
    CUBE_WIDTH,
    GRIPPER_BASE_BODY_PATH,
    PRE_REACH_Z_OFFSET,
    ROBOT_CFG,
    TARGET_OBJECT_CFG,
    TARGET_PITCH,
    TARGET_POS_X,
    TARGET_POS_Y,
    TARGET_POS_Z,
    TARGET_ROLL,
    TARGET_YAW,
    CubePreReachCommandsCfg,
    CubePreReachRewardsCfg,
    ShelfCubePreReachSceneCfg,
    UR5eGripperShelfCubePreReachEnvCfg,
)
from sweep_rl.shelf_reach.env_cfg import ROBOT_BASE_BODY_PATH

from . import mdp

TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-v0"
PLAY_TASK_ID = "Isaac-Sweep-Shelf-UR5e-Gripper-Cube-Play-v0"

PRE_REACH_X_OFFSET = 0.0
PUSH_GOAL_OFFSET = (0.0, 0.18, 0.0)
PUSH_REWARD_WEIGHT = 6.0
WRIST_BACK_OFFSET = -0.14


@configclass
class ShelfCubeSweepSceneCfg(ShelfCubePreReachSceneCfg):
    """PreReach scene extended only with the wrist reference frame."""

    wrist_frame = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{ROBOT_BASE_BODY_PATH}",
        update_period=0.0,
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{GRIPPER_BASE_BODY_PATH}",
                name="wrist",
                offset=OffsetCfg(pos=(WRIST_BACK_OFFSET, 0.0, 0.0)),
            )
        ],
    )


@configclass
class CubeSweepCommandsCfg(CubePreReachCommandsCfg):
    """PreReach moving command extended with a fixed episode Sweep goal."""

    ee_pose = mdp.CubeSweepPoseCommandCfg(
        asset_name="robot",
        body_name=GRIPPER_BASE_BODY_PATH,
        frame_name="ee_frame",
        frame_index=0,
        object_name="target_object",
        cube_width=CUBE_WIDTH,
        behind_width_scale=BEHIND_WIDTH_SCALE,
        z_offset=PRE_REACH_Z_OFFSET,
        target_roll=TARGET_ROLL[0],
        target_pitch=TARGET_PITCH[0],
        target_yaw=TARGET_YAW[0],
        goal_offset=PUSH_GOAL_OFFSET,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.CubeSweepPoseCommandCfg.Ranges(
            pos_x=TARGET_POS_X,
            pos_y=TARGET_POS_Y,
            pos_z=TARGET_POS_Z,
            roll=TARGET_ROLL,
            pitch=TARGET_PITCH,
            yaw=TARGET_YAW,
        ),
    )


@configclass
class CubeSweepRewardsCfg(CubePreReachRewardsCfg):
    """Inherit Reach/orientation/collision terms and add Sweep shaping."""

    pushing_target = RewTerm(
        func=mdp.pushing_target,
        weight=PUSH_REWARD_WEIGHT,
        params={
            "command_name": "ee_pose",
            "cube_width": CUBE_WIDTH,
            "x_offset": PRE_REACH_X_OFFSET,
            "behind_width_scale": BEHIND_WIDTH_SCALE,
            "z_offset": PRE_REACH_Z_OFFSET,
            "object_cfg": TARGET_OBJECT_CFG,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "wrist_frame_cfg": SceneEntityCfg("wrist_frame"),
        },
    )


@configclass
class UR5eGripperShelfCubeSweepEnvCfg(UR5eGripperShelfCubePreReachEnvCfg):
    """PreReach task extended with implicit +Y Cube pushing."""

    scene: ShelfCubeSweepSceneCfg = ShelfCubeSweepSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    commands: CubeSweepCommandsCfg = CubeSweepCommandsCfg()
    rewards: CubeSweepRewardsCfg = CubeSweepRewardsCfg()


@configclass
class UR5eGripperShelfCubeSweepEnvCfg_PLAY(UR5eGripperShelfCubeSweepEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.ee_pose.debug_vis = True
