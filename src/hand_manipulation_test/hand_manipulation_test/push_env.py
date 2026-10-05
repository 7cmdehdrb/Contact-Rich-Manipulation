"""Manager-based planar pushing specialization of the Cube-contact task."""

from .contact_env import HandManipulationContactEnv


class HandManipulationPushEnv(HandManipulationContactEnv):
    """Push a physical Cube toward a fixed goal with standard manager terms."""
