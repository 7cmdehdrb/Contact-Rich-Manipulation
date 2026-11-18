"""Snapshot boundary and asynchronous lifecycle, independent of the simulator."""
from threading import Event
from types import SimpleNamespace
import numpy as np
from observed_obb import Box, Observation, observe
from centroid_belief import Config
from centroid_belief.live import SnapshotJob


def test_observation_snapshot_infers_every_fragment_without_source_metadata():
    # Genuine observation output, not a source-object ID mapping.
    snapshot = observe((Box((0., 0., .05), (.08, .08, .10)),
                        Box((.2, .03, .05), (.04, .06, .10))))
    seen = []
    def infer(data, target_id, *, config, cancelled):
        assert data == snapshot
        assert all(isinstance(o, Observation) for o in data)
        seen.append(target_id)
        item = next(o for o in data if o.observation_id == target_id)
        return SimpleNamespace(mean=np.asarray(item.box.center[:2]),
                               covariance=np.eye(2)*1e-4, warnings=("diagnostic",))
    job = SnapshotJob(snapshot, infer=infer).start()
    assert job.done.wait(5)
    assert seen == [o.observation_id for o in snapshot]
    results = [job.results.get_nowait() for _ in snapshot]
    assert all(len(r.parameters) == 5 and r.warnings == ("diagnostic",) for r in results)


def test_worker_does_not_block_and_cancel_discards_late_results():
    entered, release = Event(), Event()
    def infer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return SimpleNamespace(mean=np.zeros(2), covariance=np.eye(2), warnings=())
    snapshot = (Observation(4, Box((0., 0., .05), (.08, .08, .1))),)
    job = SnapshotJob(snapshot, infer=infer).start()
    assert entered.wait(5)
    assert not job.done.is_set()
    job.close()
    release.set()
    assert job.done.wait(5)
    assert job.results.empty()


def test_worker_reports_error_and_continues_other_observations():
    snapshot = tuple(Observation(i, Box((0., i*.1, .05), (.08, .08, .1))) for i in range(2))
    def infer(data, target_id, *, config, cancelled):
        if target_id == 0:
            raise ValueError("no fit")
        return SimpleNamespace(mean=np.zeros(2), covariance=np.eye(2), warnings=())
    job = SnapshotJob(snapshot, infer=infer).start()
    assert job.done.wait(5)
    assert job.results.get_nowait().error == "ValueError: no fit"
    assert job.results.get_nowait().parameters is not None


def test_actual_observation_to_gaussian_worker():
    snapshot = observe((Box((-.7, .1, 1.10), (.08, .08, .10)),))
    job = SnapshotJob(snapshot, Config(samples_per_shape=128)).start()
    assert job.done.wait(30)
    result = job.results.get_nowait()
    assert result.error is None
    np.testing.assert_allclose(result.parameters[:2], (-.7, .1), atol=.015)


def test_latest_snapshot_replaces_busy_job_and_discards_old_result():
    from centroid_belief.live import SnapshotRunner
    jobs = []
    class Job:
        def __init__(self, snapshot, config):
            from queue import Queue
            self.snapshot = snapshot
            self.done = Event()
            self.results = Queue()
            self.cancelled = False
            jobs.append(self)
        def start(self): return self
        def close(self): self.cancelled = True
    runner = SnapshotRunner(Config(), job_factory=Job)
    runner.replace(('first',))
    runner.replace(('second',))
    runner.replace(('latest',))
    assert jobs[0].cancelled
    assert len(jobs) == 1  # no overlapping candidate searches
    jobs[0].results.put('stale result')
    jobs[0].done.set()
    assert runner.poll() == []
    assert len(jobs) == 2 and jobs[1].snapshot == ('latest',)
    jobs[1].results.put('latest result')
    jobs[1].done.set()
    assert runner.poll() == ['latest result']
    assert not runner.busy
    runner.close()


def test_estimator_cancels_during_candidate_search():
    from concurrent.futures import CancelledError
    from centroid_belief import estimate_particles
    import pytest
    checks = 0
    def cancelled():
        nonlocal checks
        checks += 1
        return checks >= 8
    snapshot = observe((Box((0., 0., .05), (.08, .08, .1)),))
    with pytest.raises(CancelledError):
        estimate_particles(snapshot, snapshot[0].observation_id, cancelled=cancelled)
    assert checks == 8


def test_finished_worker_is_busy_until_results_are_drained():
    from centroid_belief.live import SnapshotRunner, lookup_inference
    from functools import partial
    snapshot = (Observation(0,Box((0,0,.05),(.08,.08,.1))),)
    runner = SnapshotRunner(Config(),job_factory=partial(SnapshotJob,infer=lookup_inference))
    runner.replace(snapshot)
    assert runner.job.done.wait(5)
    assert runner.busy
    assert len(runner.poll()) == 1
    assert not runner.busy
    runner.close()
