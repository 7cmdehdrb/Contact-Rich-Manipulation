"""Simple Finger input width; shared obstacle encoder and MLP architecture."""
from .wrench_refactor_models import WrenchRefactorRelativeModel


class SweepSimpleFingerModel(WrenchRefactorRelativeModel):
    raw_observation_dim = 66
