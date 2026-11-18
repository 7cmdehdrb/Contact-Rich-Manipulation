"""Independent observation width with the shared obstacle encoder."""
from .wrench_refactor_models import WrenchRefactorRelativeModel


class StabilizeSweepModel(WrenchRefactorRelativeModel):
    raw_observation_dim = 67
