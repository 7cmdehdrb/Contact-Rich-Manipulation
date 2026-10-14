"""Environment shell; all task behavior belongs to manager terms."""

from isaaclab.envs import ManagerBasedRLEnv


class HandManipulationTestEnv(ManagerBasedRLEnv):
    """Free-space reaching with standard Isaac Lab stepping and resets."""
