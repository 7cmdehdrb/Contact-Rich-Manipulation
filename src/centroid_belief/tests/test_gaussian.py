import math
import numpy as np
import pytest
from observed_obb import Box, Observation, observe
from centroid_belief.forward import corner_error
from centroid_belief import (GaussianBelief, Posterior, Config, estimate,
                             estimate_particles, COMPLETE_HYPOTHESIS)
from centroid_belief.fixtures import make_case


def test_moment_matching_keeps_between_hypothesis_variance():
    p = Posterior(np.array(((0., 0.), (0., .04))), np.array((.25, .75)),
                  np.array((0, 1)), ("complete", "partial"), .002, 0., 0)
    g = GaussianBelief.from_particles(p)
    assert np.allclose(g.mean, (0., .03))
    assert g.covariance[1, 1] == pytest.approx(.25*.03**2 + .75*.01**2 + .002**2)
    assert np.array_equal(g.map, g.mean)
    assert len(g.parameters) == 5
    restored = GaussianBelief.from_parameters(g.parameters)
    assert np.array_equal(restored.covariance, g.covariance)
    assert np.array_equal(restored.density([[0., 0.], [0., .03]]), g.density([[0., 0.], [0., .03]]))
    assert not hasattr(restored, "centers")
    with pytest.raises(ValueError):
        restored.mean[0] = 1


@pytest.mark.parametrize("rho", [0., .7, -.7, .99999])
def test_correlated_rectangle_probability_and_normalization(rho):
    g = GaussianBelief(np.zeros(2), np.array(((1., rho), (rho, 1.))))
    assert g.probability_in_rectangle((-20, -20), (20, 20)) == pytest.approx(1, abs=1e-8)
    expected = .25 + math.asin(rho)/(2*math.pi)
    assert g.probability_in_rectangle((-20, -20), (0, 0)) == pytest.approx(expected, abs=1e-7)
    assert g.probability_in_rectangle((0, 0), (0, 1)) == 0


def test_ellipse_has_requested_mass():
    g = GaussianBelief(np.array((.02, -.01)), np.array(((.0004, .0001), (.0001, .0001))))
    points = g.ellipse(.9)
    assert np.allclose(g.mahalanobis_squared(points), -2*math.log(.1))
    samples = np.random.default_rng(72).multivariate_normal(g.mean, g.covariance, 100000)
    assert np.mean(g.contains(samples)) == pytest.approx(.9, abs=.004)
    assert g.credible_area(.9) == pytest.approx(math.pi*(-2*math.log(.1))*np.sqrt(np.linalg.det(g.covariance)))


def test_complete_candidate_is_evaluated_and_can_survive_adjacent_occluder():
    case = make_case(0, "right", scale=.94)
    p = estimate_particles(case.observations, case.target_id,
                           config=Config(samples_per_shape=768, seed=17))
    weight = p.shape_probabilities[COMPLETE_HYPOTHESIS]
    assert 0.01 < weight < .99
    g = GaussianBelief.from_particles(p)
    observed = np.array(case.observations[0].box.center[:2])
    assert g.probability_in_rectangle(observed-.005, observed+.005) > .01
    assert np.linalg.norm(g.mean-observed) > .002  # not simply return observation center


def test_default_api_is_a_single_gaussian_and_clear_scene_is_complete():
    case = make_case(1, "clear", translation=(.4, -.3))
    g = estimate(case.observations, case.target_id, config=Config(samples_per_shape=512, seed=4))
    assert isinstance(g, GaussianBelief)
    assert np.linalg.norm(g.mean-case.truth.center[:2]) < .003
    p = estimate_particles(case.observations, case.target_id,
                           config=Config(samples_per_shape=128, seed=4))
    assert p.shape_probabilities[COMPLETE_HYPOTHESIS] == pytest.approx(1.)


def test_inconsistent_complete_hypothesis_is_not_force_added():
    # A whole foreground box masks the selected observation. Giving a complete
    # hypothesis prior=1 must not manufacture a good fit at the observed center.
    snapshot = (Observation(0, Box((0, 0, .05), (.08, .08, .1))),
                Observation(1, Box((.2, 0, .03), (.04, .20, .06))))
    p = estimate_particles(snapshot, 0, config=Config(samples_per_shape=256, complete_prior=1, seed=9))
    assert "poor_observation_fit" in p.warnings
    assert p.best_rms > .009


def test_exact_two_sided_contact_keeps_complete_hypothesis_support():
    case = make_case(0, "both", scale=.94)
    p = estimate_particles(case.observations, case.target_id,
                           config=Config(samples_per_shape=512, complete_prior=1, seed=17))
    assert p.shape_probabilities[COMPLETE_HYPOTHESIS] == pytest.approx(1.)
    assert p.best_rms < .003


def test_same_observation_can_come_from_a_complete_smaller_object():
    partial = make_case(0, "right", scale=.94)
    complete_box = partial.observations[0].box
    complete_observation = observe((complete_box, *partial.scene[1:]))
    assert len(complete_observation) == len(partial.observations)
    assert np.linalg.norm(np.asarray(complete_box.center)-partial.truth.center) > .02
    for a, b in zip(complete_observation, partial.observations):
        assert corner_error(a.box, b.box) < 1e-24
        assert a.box.size[2] == b.box.size[2]


def test_invalid_gaussian_and_hypothesis_prior():
    with pytest.raises(ValueError, match="positive definite"):
        GaussianBelief(np.zeros(2), np.array(((1., 2.), (2., 1.))))
    with pytest.raises(ValueError, match="symmetric"):
        GaussianBelief(np.zeros(2), np.array(((1., .2), (.3, 1.))))
    with pytest.raises(ValueError):
        GaussianBelief.from_parameters((0, 0))
    with pytest.raises(ValueError):
        Config(complete_prior=1.01)
