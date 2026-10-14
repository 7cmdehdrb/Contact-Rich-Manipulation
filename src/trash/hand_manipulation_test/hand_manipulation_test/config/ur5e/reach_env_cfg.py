"""UR5e–Axia80–Inspire specialization of the manager-based reaching task."""

from pathlib import Path

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from ...assets.robot import ARM_JOINT_NAMES, HAND_BASE_BODY_NAME, PALM_SENSOR_BODY_NAMES, make_robot_cfg
from ...constants import C_OFFSET_H_M, C_QUAT_H_WXYZ
from ...env_cfg import ActionsCfg, HandManipulationTestEnvCfg, ReachSceneCfg
from ...mdp.actions import CurrentFrameOscActionCfg, InspireHandSynergyActionCfg


def _eef_marker_cfg():
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.markers["frame"].usd_path = str(
        Path(__file__).resolve().parents[2] / "assets" / "data" / "frame_prim.usd"
    )
    marker_cfg.prim_path = "/Visuals/HandManipulationTest/EEF"
    return marker_cfg


@configclass
class UR5eInspireReachSceneCfg(ReachSceneCfg):
    robot = make_robot_cfg()
    palm_tactile = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(PALM_SENSOR_BODY_NAMES) + ")",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[],
        max_contact_data_count_per_prim=16,
        debug_vis=False,
    )
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=_eef_marker_cfg(),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + HAND_BASE_BODY_NAME,
                name="end_effector",
                offset=OffsetCfg(pos=C_OFFSET_H_M, rot=C_QUAT_H_WXYZ),
            ),
        ],
    )


@configclass
class UR5eInspireReachActionsCfg(ActionsCfg):
    arm_action = CurrentFrameOscActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=HAND_BASE_BODY_NAME,
        body_offset_pos=C_OFFSET_H_M,
        body_offset_quat=C_QUAT_H_WXYZ,
        translation_scale=(0.012, 0.012, 0.008),
        rotation_scale=(0.060, 0.060, 0.060),
        motion_stiffness=(100.0,) * 6,
        motion_damping_ratio=(1.0,) * 6,
        gravity_compensation=True,
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        effort_limit_scale=0.90,
    )
    hand_action = InspireHandSynergyActionCfg(
        asset_name="robot",
        max_synergy_rate=(1.5, 1.5),
        max_joint_target_rate=(1.0,) * 12,
    )


@configclass
class UR5eInspireReachEnvCfg(HandManipulationTestEnvCfg):
    scene: UR5eInspireReachSceneCfg = UR5eInspireReachSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, lazy_sensor_update=False
    )
    actions: UR5eInspireReachActionsCfg = UR5eInspireReachActionsCfg()
