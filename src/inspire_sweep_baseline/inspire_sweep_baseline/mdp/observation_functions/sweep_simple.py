"""Measured primary-drive joint positions; no projection or normalization."""


def hand_drive_positions(env, asset_cfg):
    """Return the resolved six joints in thumb-to-little, thumb-opposition order."""
    return env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
