import math
import sys
import numpy as np
import pytest
from observed_obb import Box, Observation, observe
from centroid_belief import Config, Posterior, SHAPES, estimate_particles as estimate
from centroid_belief.forward import ForwardModel, corner_error
from centroid_belief.fixtures import make_case


def test_forward_matches_existing_observation_with_rotations_and_intersections():
    rng = np.random.default_rng(123)
    for _ in range(25):
        candidate = Box((*rng.uniform(-.03, .03, 2), .05), (.08, .10, .10), rng.uniform(-1, 1))
        others = tuple(Box((*rng.uniform(-.07, .07, 2), .02), (.04, .05, .04), rng.uniform(-1, 1))
                       for _ in range(2))
        expected = [o.box for o in observe((candidate, *others)) if o.box.size[2] == .10]
        actual = ForwardModel(others).predict(candidate)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert corner_error(a, b) < 1e-25


def test_rear_object_cannot_mask_and_height_does_not_limit_occlusion():
    target = Box((0, 0, .1), (.08, .08, .2))
    rear = Box((-.2, 0, .001), (.04, .08, .002))
    front = Box((.2, 0, .001), (.04, .08, .002))
    assert ForwardModel([rear]).predict(target) == (target,)
    assert ForwardModel([front]).predict(target) == ()


def test_candidate_own_full_width_controls_detection_threshold():
    mask = Box((.2, .1, .01), (.04, .2, .02))  # mask Y=[0,.2]
    narrow = Box((0, .025, .05), (.08, .10, .10))  # visible .025 / .10
    wide = Box((0, .075, .05), (.08, .20, .10))  # visible .025 / .20
    assert len(ForwardModel([mask]).predict(narrow)) == 1
    assert ForwardModel([mask]).predict(wide) == ()


def test_obb_axis_equivalence():
    a = Box((.1, .2, .05), (.06, .08, .10), .3)
    b = Box(a.center, (.08, .06, .10), .3 + math.pi/2)
    assert corner_error(a, b) < 1e-25


def test_normalized_multimodal_density_mean_and_hpd_are_distinct():
    posterior = Posterior(np.array(((-.03, 0), (.03, 0))), np.array((.5, .5)),
                          np.array((0, 0)), ("shape",), .002, 0., 0)
    grid = posterior.grid(step=.0005)
    assert np.allclose(posterior.mean, 0)
    assert abs(grid["map"][0]) > .025
    assert abs(grid["integral"]-1) < 1e-5
    assert posterior.density(posterior.mean)[0] < grid["threshold"]
    mass = grid["density"][grid["density"] >= grid["threshold"]].sum() * .0005**2
    assert .90 <= mass < .91
    assert posterior.covariance[0, 0] == pytest.approx(.03**2 + .002**2)
    assert posterior.probability_in_rectangle((-.05, -.02), (0, .02)) == pytest.approx(.5)
    assert posterior.probability_in_rectangle((0, 0), (0, 0)) == 0
    # Fast plotted grid and exact pointwise KDE agree within discretization error.
    points = np.array(((grid["map"][0], grid["map"][1]),))
    assert grid["density"].max() == pytest.approx(posterior.density(points)[0], rel=.02)


def test_observation_only_estimation_recovers_occluded_center():
    case = make_case(0, "right")
    # The estimator receives no Case, truth, physical identity or source label.
    posterior = estimate(case.observations, case.target_id, shapes=(SHAPES[0],),
                         config=Config(samples_per_shape=512, seed=12, complete_prior=0))
    assert np.isclose(posterior.weights.sum(), 1)
    assert np.linalg.norm(posterior.mean-case.truth.center[:2]) < .008
    assert posterior.mean[1] > case.observations[0].box.center[1] + .012
    assert posterior.effective_samples > 20
    assert "isaaclab" not in sys.modules


def test_id_selection_and_missing_prior_support():
    observed = Observation(9, Box((0, 0, .05), (.08, .08, .10)))
    with pytest.raises(ValueError, match="Unique"):
        estimate([observed, observed], 9)
    with pytest.raises(ValueError, match="Unique"):
        estimate([observed], 0)
    with pytest.raises(ValueError, match="height"):
        estimate([Observation(0, Box((0, 0, 1), (.08, .08, 2)))], 0, config=Config(complete_prior=0))


def test_translation_does_not_supply_hidden_origin_prior():
    first, second = make_case(3, "clear"), make_case(3, "clear", translation=(.7, -.4))
    config = Config(samples_per_shape=256, seed=5, complete_prior=0)
    a = estimate(first.observations, first.target_id, shapes=(SHAPES[3],), config=config)
    b = estimate(second.observations, second.target_id, shapes=(SHAPES[3],), config=config)
    assert np.linalg.norm((b.mean-a.mean) - (.7, -.4)) < .002


def test_mirrored_occlusion_preserves_multiple_completion_modes():
    means = []
    for side in ("left", "right"):
        case = make_case(6, side, scale=1.08)
        posterior = estimate(case.observations, case.target_id, shapes=(SHAPES[6],),
                             config=Config(samples_per_shape=2048, seed=17, complete_prior=0))
        means.append(posterior.mean)
        assert posterior.effective_samples > 50
    # A single proposal previously missed the displaced rotated completion mode
    # on one side and incorrectly put virtually all mass near the true center.
    assert abs(means[0][1] + means[1][1]) < .004
    assert abs(means[0][1]) > .003
    assert abs(means[1][1]) > .003
