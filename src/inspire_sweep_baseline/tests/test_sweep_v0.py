"""Direction boundaries, target-only surface evidence and unchanged Sweep accounting."""

import importlib
import math
from types import SimpleNamespace as NS

import pytest
import torch


@pytest.fixture
def preference(load_mdp):
    return importlib.import_module('_baseline_cpu_pkg.mdp.reward_functions.sweep_contact_preference')


def test_pull_push_are_sixty_degrees_with_no_excluded_angles(preference):
    angles = torch.tensor([0, 29.999, 30.001, 90, 149.999, 150.001, 180,
                           -150.001, -149.999, -90, -30.001, -29.999], dtype=torch.float64)
    radians = angles * math.pi / 180
    direction = torch.stack((-radians.cos(), radians.sin(), torch.zeros_like(radians)), -1)
    assert preference.sweep_region(direction).tolist() == [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    # Exact boundaries belong to the narrower push/pull regions.
    a = torch.tensor([30., -30., 150., -150.], dtype=torch.float64) * math.pi / 180
    assert preference.sweep_region(torch.stack((-a.cos(), a.sin(), a * 0), -1)).tolist() == [1, 1, 3, 3]


def test_all_eight_side_direction_combinations_and_mixed_contacts(preference):
    regions = torch.tensor([1, 2, 3, 4])
    one, zero = torch.ones(4), torch.zeros(4)
    torch.testing.assert_close(preference.contact_side_multiplier(regions, one, zero),
                               torch.tensor([.7, .7, 1.5, 1.5]))
    torch.testing.assert_close(preference.contact_side_multiplier(regions, zero, one),
                               torch.tensor([1.5, 1.5, .7, .7]))
    torch.testing.assert_close(preference.contact_side_multiplier(regions, 3 * one, one),
                               torch.tensor([.9, .9, 1.3, 1.3]))
    torch.testing.assert_close(preference.contact_side_multiplier(regions, zero, zero), one)


@pytest.mark.parametrize('preferred,other', [(0., .7), (1.5, -1.), (float('nan'), .7), (1.5, float('inf'))])
def test_invalid_multipliers(preference, preferred, other):
    with pytest.raises(ValueError):
        preference.contact_side_multiplier(torch.tensor([1]), torch.ones(1), torch.zeros(1), preferred, other)


def test_reward_uses_episode_direction_in_rotated_shelf_and_preserves_retreat(preference, tensor_math):
    # Shelf yaw +90: local 12(-X) becomes world -Y; local 6(+X) becomes +Y.
    quaternion = torch.tensor([[math.sqrt(.5), 0., 0., math.sqrt(.5)]]).expand(2, -1)
    direction = torch.tensor([[-1., 0., 0.], [1., 0., 0.]])
    state = NS(positive=torch.tensor([.01, .01]), retreat=torch.tensor([.03, .04]))
    env = NS(scene={'shelf': NS(data=NS(root_quat_w=quaternion))},
             sweep_dir=tensor_math.quat_apply(quaternion, direction), _stabilize_state=state, step_dt=.02)
    term = object.__new__(preference.ContactSideSweepProgress)
    term._surface_loads = lambda _: (torch.ones(2), torch.zeros(2))
    torch.testing.assert_close(term(env), torch.tensor([.35, .75]))
    torch.testing.assert_close(state.positive, torch.tensor([.01, .01]))
    torch.testing.assert_close(state.retreat, torch.tensor([.03, .04]))


def test_real_surface_mapper_excludes_obstacles_and_masks_reset_rows(preference):
    # Actual convex-hull front/back mapper, with one target front and one back
    # contact, plus a much stronger obstacle contact which must be excluded.
    geometry = importlib.import_module('_baseline_cpu_pkg.mdp.observation_functions.wrench_refactor')
    metadata = {'footprint_halfspaces': [[1, 0, -.1], [-1, 0, -.1], [0, 1, -.1], [0, -1, -.1]],
                'parent_halfspaces': [[1, 0, 0, -.1], [-1, 0, 0, -.1], [0, 1, 0, -.1],
                                      [0, -1, 0, -.1], [0, 0, 1, -.01], [0, 0, -1, -.01]],
                'pad_bounds_m': {'min': [-.1, -.1, -.01], 'max': [.1, .1, .01]}}
    regions = geometry.PairedContactRegions([metadata], [0], 'cpu')
    forces = torch.tensor([[1.], [3.], [1000.], [0.]])
    points = torch.tensor([[0., 0., .01], [0., 0., -.01], [0., 0., -.01], [0., 0., 0.]])
    normals = torch.tensor([[0., 0., 1.], [0., 0., -1.], [0., 0., -1.], [0., 0., 0.]])
    counts, starts = torch.tensor([[2, 1]]), torch.tensor([[0, 2]])
    raw = (forces, points, normals, torch.zeros(4), counts, starts)
    term = object.__new__(preference.ContactSideSweepProgress)
    term._parents = [('parent', regions)]
    term._robot_ids = torch.tensor([0])
    term._pad_filters = torch.arange(17)
    matrix = torch.zeros(1, 1, len(preference.HAND_CONTACT_BODY_NAMES), 3)
    matrix[0, 0, 0, 0] = 5.
    term._target_sensor = NS(data=NS(force_matrix_w=matrix))
    env = NS(num_envs=1, device='cpu', physics_dt=.01, sensor_data_fresh=torch.ones(1, dtype=torch.bool),
             scene={'parent': NS(contact_physx_view=NS(get_contact_data=lambda **_: raw)),
                    'robot': NS(data=NS(body_link_pos_w=torch.zeros(1, 1, 3),
                                       body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]])))})
    palm, dorsal = term._surface_loads(env)
    torch.testing.assert_close(palm, torch.tensor([6.]))
    torch.testing.assert_close(dorsal, torch.tensor([3.]))
    # Reward reads must not trim/mutate shared PhysX contact ranges.
    assert counts.tolist() == [[2, 1]] and starts.tolist() == [[0, 2]]
    env.sensor_data_fresh.zero_()
    palm, dorsal = term._surface_loads(env)
    assert palm.item() == dorsal.item() == 0.
