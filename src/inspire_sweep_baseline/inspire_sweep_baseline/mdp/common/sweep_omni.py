"""Baseline mechanics with realistic physics and goal-based omnidirectional rewards."""



# Measured middle-board dimensions of the same shelf asset used by Baseline.
# Shelf-local XY is horizontal and Z is up; the shelf root is (-0.7, 0, 0).
SHELF_HALF_EXTENTS = (0.20, 0.50)


SHELF_SURFACE_HEIGHT = 1.05


OBJECT_RADIUS = 0.04


FAILURE_TERMS = (
    "object_drop",
    "push_fast",
    "shelf_collision",
    "hand_velocity",
    "object_outside_shelf",
    "shelf_displacement",
)
