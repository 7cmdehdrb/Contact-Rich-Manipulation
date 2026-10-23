"""Dex Tactile v1 with goal occupancy rewards continuing until timeout.

Renewal variants with 17 tactile observations and shared exponential costs."""



# Only active failures inherited from v1; object_outside_shelf is disabled.
FAILURE_TERMS = (
    "object_drop", "push_fast", "shelf_collision", "hand_velocity", "shelf_displacement",
)


TACTILE_THRESHOLD_N = 0.01
