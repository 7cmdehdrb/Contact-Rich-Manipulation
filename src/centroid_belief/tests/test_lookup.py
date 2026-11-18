import numpy as np
import pytest
from observed_obb import Box, Observation
from centroid_belief.lookup import LookupEstimator, feature
from centroid_belief.fixtures import make_case
from centroid_belief.preview import preview_rgba


@pytest.fixture(scope='module')
def estimator():
    return LookupEstimator()


def test_bounded_lookup_preserves_complete_and_partial_possibilities(estimator):
    case = make_case(0,'right')
    result = estimator.estimate(case.observations,case.target_id)
    assert 0 < result.evaluated <= 16
    assert .5 <= result.complete_weight < 1
    assert not result.warnings
    assert np.linalg.eigvalsh(result.belief.covariance).min() > 0
    assert result.belief.contains(case.observations[0].box.center[:2])[0]
    assert result.belief.contains(case.truth.center[:2])[0]
    image = preview_rgba(case.observations[0],result.belief.parameters)
    assert image.shape == (500,720,4) and image.dtype == np.uint8


def test_translation_and_equivalent_axes(estimator):
    case = make_case(2,'left',yaw=.43)
    shifted = make_case(2,'left',yaw=.43,translation=(1.2,-.9))
    a=estimator.estimate(case.observations,case.target_id).belief
    b=estimator.estimate(shifted.observations,shifted.target_id).belief
    np.testing.assert_allclose(b.mean-a.mean,[1.2,-.9],atol=1e-8)
    np.testing.assert_allclose(a.covariance,b.covariance,atol=1e-10)
    box=case.observations[0].box
    equivalent=Box(box.center,(box.size[1],box.size[0],box.size[2]),box.yaw+np.pi/2)
    np.testing.assert_allclose(feature(box),feature(equivalent),atol=1e-12)


def test_clear_and_unsupported_and_invalid_input(estimator):
    case=make_case(1,'clear')
    result=estimator.estimate(case.observations,case.target_id)
    assert result.evaluated==0 and result.complete_weight==1
    np.testing.assert_allclose(result.belief.mean,case.truth.center[:2])
    result=estimator.estimate([Observation(0,Box((0,0,1),(.1,.1,2)))],0)
    assert 'no_template_height_support' in result.warnings
    with pytest.raises(ValueError): estimator.estimate(case.observations,999)
    with pytest.raises(ValueError): estimator.estimate(case.observations*2,0)


def test_fast_worker_uses_actual_snapshot():
    from centroid_belief.live import SnapshotJob, lookup_inference
    case=make_case(0,'right')
    job=SnapshotJob(case.observations,infer=lookup_inference).start()
    assert job.done.wait(5)
    results=[job.results.get_nowait() for _ in case.observations]
    assert all(r.error is None and len(r.parameters)==5 for r in results)
