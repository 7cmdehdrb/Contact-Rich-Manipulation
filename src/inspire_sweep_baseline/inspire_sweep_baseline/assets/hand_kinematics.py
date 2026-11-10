"""Six right-hand drive channels and the original URDF mimic relationships.

The USD retains historical ``inspire_left`` names for the right-hand joints.
Mapping entries follow robot.HAND_JOINT_NAMES (12 physical joints).
"""

HAND_DRIVE_JOINT_NAMES = (
    'inspire_left_thumb_2_joint',
    'inspire_left_index_1_joint',
    'inspire_left_middle_1_joint',
    'inspire_left_ring_1_joint',
    'inspire_left_little_1_joint',
    'inspire_left_thumb_1_joint',
)
HAND_DRIVE_UPPER_LIMITS = (.5864, 1.4381, 1.4381, 1.4381, 1.4381, 1.1641)
HAND_MIMIC_DRIVE_INDICES = (5, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4)
HAND_MIMIC_MULTIPLIERS = (1., 1., .8024, .8024 * .9487, 1., 1.0843,
                          1., 1.0843, 1., 1.0843, 1., 1.0843)
