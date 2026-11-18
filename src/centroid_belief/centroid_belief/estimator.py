"""Internal importance-corrected completion hypotheses from an OBB snapshot.

Exact observed height conditions the uniform-scale shape prior analytically.
Continuous latent variables are center X/Y and yaw; discrete shape is unknown.
The complete-box hypothesis conditions its dimensions on the observed OBB.
This is a conditional approximation, not a calibrated joint scene posterior.
"""
from dataclasses import dataclass
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError
import math
import numpy as np
from scipy.optimize import Bounds, differential_evolution
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp, ndtr
from scipy.stats import multivariate_normal
from observed_obb import Box, Observation
from .forward import ForwardModel, corner_error
from .shapes import SHAPES, Shape

COMPLETE_HYPOTHESIS = "__complete_observation__"


@dataclass(frozen=True)
class Config:
    samples_per_shape: int = 2048
    observation_sigma: float = .003
    bandwidth: float = .002
    seed: int = 17
    minimum_width: float = .02
    minimum_ratio: float = .20
    complete_prior: float = .5

    def __post_init__(self):
        if self.samples_per_shape < 32:
            raise ValueError("At least 32 samples per shape are required")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.observation_sigma, self.bandwidth, self.minimum_width)):
            raise ValueError("Noise, bandwidth and width must be positive finite values")
        if not math.isfinite(self.minimum_ratio) or not 0 <= self.minimum_ratio <= 1:
            raise ValueError("minimum_ratio must lie in [0,1]")
        if not math.isfinite(self.complete_prior) or not 0 <= self.complete_prior <= 1:
            raise ValueError("complete_prior must lie in [0,1]")


@dataclass
class Posterior:
    centers: np.ndarray
    weights: np.ndarray
    shape_indices: np.ndarray
    shape_names: tuple[str, ...]
    bandwidth: float
    best_rms: float
    evaluations: int
    warnings: tuple[str, ...] = ()

    @property
    def mean(self):
        return self.weights @ self.centers

    @property
    def covariance(self):
        delta = self.centers - self.mean
        return (delta * self.weights[:, None]).T @ delta + np.eye(2) * self.bandwidth**2

    @property
    def effective_samples(self):
        return float(1 / (self.weights @ self.weights))

    @property
    def shape_probabilities(self):
        return {name: float(self.weights[self.shape_indices == i].sum())
                for i, name in enumerate(self.shape_names)}

    def density(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        result = np.empty(len(points))
        # Chunk to bound memory on ordinary CPU laptops.
        for start in range(0, len(points), 256):
            delta = points[start:start+256, None, :] - self.centers[None, :, :]
            kernel = np.exp(-np.sum(delta**2, axis=2) / (2 * self.bandwidth**2))
            result[start:start+256] = kernel @ self.weights / (2 * math.pi * self.bandwidth**2)
        return result

    def probability_in_rectangle(self, lower, upper) -> float:
        """Integrated probability mass, unlike the density at a single point."""
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lower.shape != (2,) or upper.shape != (2,) or
                not np.isfinite([lower, upper]).all() or np.any(upper < lower)):
            raise ValueError("Finite ordered XY bounds required")
        mass = ndtr((upper-self.centers)/self.bandwidth) - ndtr((lower-self.centers)/self.bandwidth)
        return float(self.weights @ np.prod(mass, axis=1))

    def grid(self, step=.001, mass=.90):
        if not math.isfinite(step) or step <= 0 or not 0 < mass < 1:
            raise ValueError("Positive finite grid step and mass in (0,1) required")
        if step > self.bandwidth / 2:
            raise ValueError("Grid step must be at most half the KDE bandwidth")
        # Include every particle and six kernel standard deviations.
        lower = self.centers.min(axis=0) - 6 * self.bandwidth
        upper = self.centers.max(axis=0) + 6 * self.bandwidth
        x = np.arange(lower[0], upper[0] + step, step)
        y = np.arange(lower[1], upper[1] + step, step)
        # Bilinear mass deposition + Gaussian convolution approximates the KDE
        # without O(grid_cells * particles) cost. Pointwise density remains exact.
        coordinates = (self.centers - lower) / step
        base = np.floor(coordinates).astype(int)
        fraction = coordinates - base
        histogram = np.zeros((len(y), len(x)))
        for dx in (0, 1):
            for dy in (0, 1):
                weight = self.weights * (fraction[:, 0] if dx else 1-fraction[:, 0])
                weight = weight * (fraction[:, 1] if dy else 1-fraction[:, 1])
                np.add.at(histogram, (base[:, 1]+dy, base[:, 0]+dx), weight)
        density = gaussian_filter(histogram, self.bandwidth/step,
                                  mode="constant", truncate=6.) / step**2
        cells = density.ravel() * step**2
        order = np.argsort(cells)[::-1]
        cumulative = np.cumsum(cells[order])
        if cumulative[-1] < mass:
            raise RuntimeError("Grid does not capture requested probability mass")
        threshold = density.ravel()[order[np.searchsorted(cumulative, mass)]]
        index = np.unravel_index(np.argmax(density), density.shape)
        return {"x": x, "y": y, "density": density, "threshold": float(threshold),
                "map": np.array((x[index[1]], y[index[0]])),
                "integral": float(cells.sum()),
                "credible_area": float(np.sum(density >= threshold) * step**2)}


def _wrap(yaw):
    return (yaw + math.pi / 2) % math.pi - math.pi / 2


def _proposal_logpdf(points, means, covariances, mixture, lower, upper):
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    uniform = np.where(inside, -np.log(np.prod(upper - lower)), -np.inf)
    # Wrapped normal on yaw modulo pi. +-2 periods safely cover our capped yaw SD.
    normal = []
    for mean, covariance, probability in zip(means, covariances, mixture):
        periods = []
        for turn in range(-2, 3):
            shifted = points.copy()
            shifted[:, 2] += turn * math.pi
            periods.append(multivariate_normal.logpdf(shifted, mean=mean, cov=covariance))
        normal.append(math.log(probability) + logsumexp(periods, axis=0))
    components = np.array(normal).T
    logq = np.logaddexp(np.log(.2) + uniform, np.log(.8) + logsumexp(components, axis=1))
    return logq, components


def estimate_particles(observations: Sequence[Observation], target_id: int, *,
             shapes: Sequence[Shape] = SHAPES, config: Config = Config(),
             cancelled: Callable[[], bool] = lambda: False) -> Posterior:
    """Estimate original OBB center. No kind, source IDs or true sizes are inputs.

    IDs must be unique within this snapshot. target_id selects one observed
    fragment, not a known physical-object identity. Template completions use a
    uniform population/scale prior. The separate complete hypothesis uses the
    observed dimensions. complete_prior=0 reproduces the old template-only model.
    """
    if cancelled():
        raise CancelledError()
    observations, shapes = tuple(observations), tuple(shapes)
    ids = [o.observation_id for o in observations]
    if len(ids) != len(set(ids)) or target_id not in ids:
        raise ValueError("Unique snapshot IDs and an existing target_id are required")
    if not shapes or len({s.name for s in shapes}) != len(shapes):
        raise ValueError("A nonempty population with unique shape names is required")
    if any(s.name == COMPLETE_HYPOTHESIS for s in shapes):
        raise ValueError("Reserved complete-observation hypothesis name")
    target = observations[ids.index(target_id)].box
    others = [o.box for o in observations if o.observation_id != target_id]
    forward = ForwardModel(others, config.minimum_width, config.minimum_ratio)
    compatible = [(i, target.size[2] / s.size[2]) for i, s in enumerate(shapes)
                  if .9 - 1e-12 <= target.size[2] / s.size[2] <= 1.1 + 1e-12]
    if not compatible and config.complete_prior == 0:
        raise ValueError("Observed height lies outside the declared shape/scale prior")
    radius = max([math.hypot(*s.size[:2]) * 1.1 for s in shapes] +
                 [math.hypot(*target.size[:2])])
    lower = np.array((target.center[0]-radius, target.center[1]-radius, -math.pi/2))
    upper = np.array((target.center[0]+radius, target.center[1]+radius, math.pi/2))
    rng = np.random.default_rng(config.seed)
    all_points, all_logweights, all_shapes = [], [], []
    best_error, evaluations = math.inf, 0
    hypotheses = []
    height_normalizer = sum(1 / shapes[i].size[2] for i, _ in compatible)
    for shape_index, scale in compatible:
        size = tuple(scale * v for v in shapes[shape_index].size)
        probability = (1-config.complete_prior) / (shapes[shape_index].size[2] * height_normalizer)
        if probability > 0 and (others or config.complete_prior == 0):
            hypotheses.append((shape_index, size, probability, False))
    names = tuple(s.name for s in shapes)
    if config.complete_prior > 0:
        probability = config.complete_prior if compatible else 1.
        hypotheses.append((len(shapes), target.size, probability, True))
        names += (COMPLETE_HYPOTHESIS,)
    for shape_index, size, probability, complete in hypotheses:

        def error(point):
            nonlocal best_error, evaluations
            if cancelled():
                raise CancelledError()
            evaluations += 1
            candidate = Box((point[0], point[1], target.center[2]), size, point[2])
            predicted = forward.predict(candidate)
            if not predicted:
                return math.inf
            # Complete within the same measurement tolerance as the likelihood.
            # Strict equality gives a zero-volume pose set for an object exactly
            # touching two masks, incorrectly eliminating that valid hypothesis.
            fully_visible = len(predicted) == 1 and (
                predicted[0] == candidate or
                (config.complete_prior > 0 and
                 corner_error(predicted[0], candidate) <= config.observation_sigma**2))
            if complete and not fully_visible:
                return math.inf
            if not complete and config.complete_prior > 0 and fully_visible:
                return math.inf
            errors = [corner_error(box, target) for box in predicted]
            chosen = int(np.argmin(errors))
            total = errors[chosen]
            # Extra candidate fragments need support somewhere in the snapshot.
            # This permits possible matches without merging observation IDs.
            for j, box in enumerate(predicted):
                if j != chosen:
                    total += min((corner_error(box, other) +
                                  (box.size[2]-other.size[2])**2 for other in others),
                                 default=radius**2)
            best_error = min(best_error, total)
            return total

        def objective(point):
            value = error(point)
            return min(value / (2 * config.observation_sigma**2), 1e6)

        # Optimization builds an efficient proposal only; it is not a posterior
        # estimate. Final samples always receive the full prior/proposal ratio.
        # Separate yaw sectors find modes whose centers ALSO differ. Merely
        # rotating one fitted proposal misses valid displaced completion modes.
        mode_centers, scores = [], []
        yaw_edges = np.linspace(-math.pi/2, math.pi/2, 9)
        for yaw_lo, yaw_hi in zip(yaw_edges[:-1], yaw_edges[1:]):
            sector_lo, sector_hi = lower.copy(), upper.copy()
            sector_lo[2], sector_hi[2] = yaw_lo, yaw_hi
            fitted = differential_evolution(objective, Bounds(sector_lo, sector_hi),
                rng=rng, popsize=5, maxiter=22, polish=True,
                x0=np.array((*target.center[:2], (yaw_lo+yaw_hi)/2)))
            mode_centers.append(fitted.x)
            scores.append(fitted.fun)
        means = np.asarray(mode_centers)
        means[:, 2] = _wrap(means[:, 2])
        covariances = np.tile(np.diag((.008**2, .02**2, math.radians(9)**2)), (8, 1, 1))
        mixture = np.exp(-np.asarray(scores) - logsumexp(-np.asarray(scores)))
        mixture = .95 * mixture + .05/8
        for stage in range(2):
            count = config.samples_per_shape
            assignment = rng.choice(8, count, p=mixture)
            points = np.empty((count, 3))
            for component in range(8):
                select = assignment == component
                points[select] = rng.multivariate_normal(means[component], covariances[component],
                                                         int(select.sum()))
            global_draw = rng.random(count) < .2
            points[global_draw] = rng.uniform(lower, upper, (int(global_draw.sum()), 3))
            points[:, 2] = _wrap(points[:, 2])
            logq, components = _proposal_logpdf(points, means, covariances, mixture, lower, upper)
            inside = np.all((points >= lower) & (points <= upper), axis=1)
            errors = np.full(count, math.inf)
            for i in np.flatnonzero(inside):
                errors[i] = error(points[i])
            # Template weights are conditioned on height BEFORE mixing with the
            # observation-conditioned complete-box hypothesis. Its fitted size
            # is not claimed to be an independent population shape prior.
            logprior = math.log(probability)
            logprior -= np.log(np.prod(upper-lower))
            logweights = -errors / (2 * config.observation_sigma**2) + logprior - logq
            if stage == 0:
                if not np.isfinite(logweights).any():
                    continue
                weights = np.exp(logweights - logsumexp(logweights))
                normalizer = np.asarray(logsumexp(components, axis=1))
                responsibility = np.exp(components-normalizer[:, None])
                masses = weights @ responsibility
                for component in range(8):
                    if masses[component] < 1e-10:
                        continue
                    local_weights = weights * responsibility[:, component] / masses[component]
                    unwrapped = points.copy()
                    unwrapped[:, 2] = means[component, 2] + _wrap(points[:, 2]-means[component, 2])
                    mean = local_weights @ unwrapped
                    delta = unwrapped - mean
                    covariance = 2 * (delta * local_weights[:, None]).T @ delta
                    covariance += np.diag((.002**2, .003**2, math.radians(3)**2))
                    if covariance[2, 2] > .5**2:
                        factor = .5 / math.sqrt(covariance[2, 2])
                        covariance[2, :] *= factor
                        covariance[:, 2] *= factor
                    mean[2] = _wrap(mean[2])
                    means[component], covariances[component] = mean, covariance
                mixture = .95 * masses / masses.sum() + .05/8
            else:
                all_points.append(points[:, :2])
                all_logweights.append(logweights)
                all_shapes.append(np.full(count, shape_index))
    logweights = np.concatenate(all_logweights)
    if not np.isfinite(logweights).any():
        raise ValueError("No sampled candidate explains a detectable target; inspect prior and scene")
    weights = np.exp(logweights - logsumexp(logweights))
    centers, shape_indices = np.concatenate(all_points), np.concatenate(all_shapes)
    # Remove exactly zero weights only (no posterior truncation for convenience).
    keep = weights > 0
    warnings = []
    if not compatible:
        warnings.append("no_template_height_support")
    if 1 / (weights @ weights) < 50:
        warnings.append("low_effective_sample_size")
    if math.sqrt(best_error) > 3 * config.observation_sigma:
        warnings.append("poor_observation_fit")
    return Posterior(centers[keep], weights[keep], shape_indices[keep],
                     names, config.bandwidth,
                     math.sqrt(best_error), evaluations, tuple(warnings))
