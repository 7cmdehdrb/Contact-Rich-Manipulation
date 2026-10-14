"""Real table-frame EEF height and unweighted progressive rate boundaries."""

import math

import pytest
import torch

from hand_manipulation_test.action_math import quaternion_from_rotation_vector, quaternion_rotate
from hand_manipulation_test.height_math import eef_height_guard


def _guard(control, position=None, quaternion=None, **settings):
    if position is None:
        position = torch.zeros_like(control)
    if quaternion is None:
        quaternion = control.new_tensor([1., 0., 0., 0.]).expand(len(control), -1)
    return eef_height_guard(control, position, quaternion, table_thickness_m=settings.get('thickness', .04),
                            soft_height_m=settings.get('soft', .15), hard_height_m=settings.get('hard', .25))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_soft_is_zero_until_boundary_then_linear_and_hard_is_inclusive(dtype):
    expected_height = torch.tensor([.05455, .12955, .15, .20, .25, .30], dtype=dtype)
    control = torch.zeros(6, 3, dtype=dtype)
    control[:, 2] = expected_height+.02
    height, penalty, hard = _guard(control)
    torch.testing.assert_close(height, expected_height, atol=3e-8, rtol=0.)
    torch.testing.assert_close(penalty, torch.tensor([0., 0., 0., -.05, -.10, -.15], dtype=dtype), atol=3e-8, rtol=0.)
    assert hard.tolist() == [False, False, False, False, True, True]
    assert not penalty[:2].any(), "Both actual reset-C and contact-C heights must lie below the soft threshold"
    assert float(penalty[3]*5*.02) == pytest.approx(-.005, abs=1e-8)


def test_height_uses_actual_tilted_table_axes_and_is_invariant_to_environment_origins():
    position = torch.tensor([[0., 0., 1.03], [50., -48.75, 1.82505]], dtype=torch.float64)
    rotation = torch.tensor([[.3, -.2, .8], [.3, -.2, .8]], dtype=position.dtype)
    quaternion = quaternion_from_rotation_vector(rotation)
    relative = torch.tensor([[.3, -.1, .22], [.3, -.1, .22]], dtype=position.dtype)
    control = position+quaternion_rotate(quaternion, relative)
    height, penalty, hard = _guard(control, position, quaternion)
    torch.testing.assert_close(height, torch.full((2,), .20, dtype=position.dtype), atol=8e-15, rtol=0.)
    torch.testing.assert_close(penalty, torch.full((2,), -.05, dtype=position.dtype), atol=8e-15, rtol=0.)
    assert not hard.any()
    assert not torch.allclose(control[:, 2]-position[:, 2]-.02, height), "World Z alone must not define a tilted table's height"


def test_invalid_geometry_is_finite_and_fails_without_fabricating_a_height_penalty():
    control = torch.tensor([[0., 0., .15]]*5)
    position = torch.zeros_like(control)
    quaternion = torch.tensor([[1., 0., 0., 0.]]*5)
    control[0, 0] = float('nan')
    position[1, 1] = float('inf')
    quaternion[2] = 0
    quaternion[3, 2] = float('nan')
    quaternion[4] *= 2  # Nonunit finite quaternions normalize normally.
    height, penalty, hard = _guard(control, position, quaternion)
    assert hard.tolist() == [True, True, True, True, False]
    assert torch.isfinite(height).all() and torch.isfinite(penalty).all()
    assert not height[:4].any() and not penalty.any()
    assert float(height[4]) == pytest.approx(.13)


@pytest.mark.parametrize("settings", ({'soft':.25,'hard':.25}, {'soft':.3,'hard':.25},
                                      {'soft':float('nan')}, {'hard':float('inf')},
                                      {'thickness':0.}, {'thickness':float('nan')}))
def test_invalid_height_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        _guard(torch.zeros(1, 3), **settings)


def test_pose_shape_and_dtype_contracts_are_explicit():
    with pytest.raises(ValueError, match="control_pos_w"):
        _guard(torch.zeros(3))
    with pytest.raises(ValueError, match="Table pose"):
        _guard(torch.zeros(2, 3), torch.zeros(1, 3), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="floating-point"):
        _guard(torch.zeros(1, 3, dtype=torch.long))
