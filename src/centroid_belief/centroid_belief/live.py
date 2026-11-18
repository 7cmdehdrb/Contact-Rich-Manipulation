"""CPU snapshot inference; no simulator, source objects, or USD access."""
from dataclasses import dataclass, replace
from queue import Empty, Queue
from concurrent.futures import CancelledError
from threading import Event, Thread
from time import perf_counter

from .estimator import Config, estimate_particles
from .gaussian import GaussianBelief


@dataclass(frozen=True)
class Result:
    observation_id: int
    parameters: tuple | None
    seconds: float
    warnings: tuple[str, ...] = ()
    error: str | None = None


class SnapshotJob:
    """One immutable snapshot, one daemon worker; display consumes queue on main thread."""

    def __init__(self, snapshot, config=Config(), *, infer=estimate_particles):
        self.snapshot = tuple(snapshot)
        self.results: Queue[Result] = Queue()
        self.done = Event()
        self.cancelled = Event()
        self._config = config
        self._infer = infer
        self._thread = Thread(target=self._run, daemon=True, name="centroid-snapshot")

    def start(self):
        self._thread.start()
        return self

    def close(self):
        # Candidate evaluation checks this flag, including optimizer evaluations.
        self.cancelled.set()

    def _run(self):
        try:
            for item in self.snapshot:
                if self.cancelled.is_set():
                    break
                start = perf_counter()
                try:
                    posterior = self._infer(self.snapshot, item.observation_id,
                        config=replace(self._config, seed=self._config.seed + item.observation_id),
                        cancelled=self.cancelled.is_set)
                    belief = GaussianBelief.from_particles(posterior)
                    result = Result(item.observation_id, belief.parameters, perf_counter()-start,
                                    tuple(posterior.warnings))
                except CancelledError:
                    break
                except Exception as error:
                    result = Result(item.observation_id, None, perf_counter()-start,
                                    error=f"{type(error).__name__}: {error}")
                if not self.cancelled.is_set():
                    self.results.put(result)
        finally:
            self.done.set()


class SnapshotRunner:
    """Main-thread scheduler: only latest request survives; at most one worker."""

    def __init__(self, config=Config(), *, job_factory=SnapshotJob):
        self.config = config
        self.job_factory = job_factory
        self.job = None
        self.pending = None

    def replace(self, snapshot):
        self.pending = tuple(snapshot)
        if self.job is not None:
            self.job.close()
        self._start_pending()

    def _start_pending(self):
        if self.pending is not None and (self.job is None or self.job.done.is_set()):
            self.job = self.job_factory(self.pending, self.config).start()
            self.pending = None

    @property
    def busy(self):
        return self.pending is not None or (self.job is not None and
                (not self.job.done.is_set() or not self.job.results.empty()))

    def poll(self):
        if self.pending is not None:
            self._start_pending()
            return []  # discard every result from the superseded snapshot
        results = []
        if self.job is not None:
            while True:
                try:
                    results.append(self.job.results.get_nowait())
                except Empty:
                    break
        return results

    def close(self):
        self.pending = None
        if self.job is not None:
            self.job.close()


def lookup_inference(snapshot, target_id, *, config, cancelled):
    """Adapter preserving the worker result contract; no Monte Carlo or optimizer."""
    from types import SimpleNamespace
    from .lookup import default_estimator
    if cancelled():
        raise CancelledError()
    result = default_estimator().estimate(snapshot, target_id)
    if cancelled():
        raise CancelledError()
    return SimpleNamespace(mean=result.belief.mean, covariance=result.belief.covariance,
                           warnings=result.warnings)
