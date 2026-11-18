"""Five-number Gaussian output; using a saved belief needs no particles or OBBs."""
from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from .estimator import Config, estimate_particles
from .shapes import SHAPES


@dataclass(frozen=True)
class GaussianBelief:
    mean: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        mean = np.array(self.mean, dtype=float, copy=True)
        covariance = np.array(self.covariance, dtype=float, copy=True)
        if mean.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("Expected XY mean and 2x2 covariance")
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("Gaussian parameters must be finite")
        if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-15):
            raise ValueError("Covariance must be symmetric")
        covariance = (covariance + covariance.T) / 2
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("Covariance must be positive definite")
        mean.flags.writeable = covariance.flags.writeable = False
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    @classmethod
    def from_particles(cls, posterior):
        # Total covariance includes within-candidate (kernel) uncertainty AND
        # between-hypothesis displacement. No fitting of density peaks is used.
        return cls(posterior.mean, posterior.covariance)

    @property
    def parameters(self):
        """(mu_x, mu_y, cov_xx, cov_xy, cov_yy), metres and metres squared."""
        return (float(self.mean[0]), float(self.mean[1]), float(self.covariance[0, 0]),
                float(self.covariance[0, 1]), float(self.covariance[1, 1]))

    @classmethod
    def from_parameters(cls, values):
        if len(values) != 5:
            raise ValueError("Exactly five Gaussian parameters required")
        x, y, xx, xy, yy = values
        return cls(np.array((x, y)), np.array(((xx, xy), (xy, yy))))

    @property
    def map(self):
        return self.mean

    def mahalanobis_squared(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Finite XY points required")
        delta = points - self.mean
        return np.sum(delta * np.linalg.solve(self.covariance, delta.T).T, axis=1)

    def density(self, points):
        return np.exp(-self.mahalanobis_squared(points)/2) / (2*math.pi*np.sqrt(np.linalg.det(self.covariance)))

    @staticmethod
    def _radius_squared(mass):
        if not math.isfinite(mass) or not 0 < mass < 1:
            raise ValueError("Probability mass must lie in (0,1)")
        return -2*math.log1p(-mass)  # exact chi-square quantile, df=2

    def contains(self, points, mass=.9):
        return self.mahalanobis_squared(points) <= self._radius_squared(mass)

    def credible_area(self, mass=.9):
        return math.pi*self._radius_squared(mass)*np.sqrt(np.linalg.det(self.covariance))

    def ellipse(self, mass=.9, count=181):
        if count < 4:
            raise ValueError("At least four ellipse points required")
        values, vectors = np.linalg.eigh(self.covariance)
        angle = np.linspace(0, 2*math.pi, count)
        circle = np.column_stack((np.cos(angle), np.sin(angle)))
        return self.mean + (circle*np.sqrt(values*self._radius_squared(mass))) @ vectors.T

    def probability_in_rectangle(self, lower, upper):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (lower.shape != (2,) or upper.shape != (2,) or
                not np.isfinite([lower, upper]).all() or np.any(upper < lower)):
            raise ValueError("Finite ordered XY bounds required")
        if np.any(lower == upper):
            return 0.
        sx, sy = np.sqrt(np.diag(self.covariance))
        rho = self.covariance[0, 1] / (sx*sy)
        lo, hi = (lower-self.mean)/(sx, sy), (upper-self.mean)/(sx, sy)
        conditional_sd = math.sqrt(max(1-rho*rho, np.finfo(float).eps))

        def integrand(x):
            mass = ndtr((hi[1]-rho*x)/conditional_sd)-ndtr((lo[1]-rho*x)/conditional_sd)
            return math.exp(-x*x/2)/math.sqrt(2*math.pi)*mass

        # Truncation below 8e-24 Gaussian tail mass; avoid missing a narrow peak
        # when callers provide an extremely wide rectangle.
        a, b = max(float(lo[0]), -10.), min(float(hi[0]), 10.)
        if a >= b:
            return 0.
        points = [float(v) for v in (0., lo[1]/rho, hi[1]/rho)
                  if a < v < b] if abs(rho) > 1e-12 else [0.] if a < 0 < b else []
        value = quad(integrand, a, b, points=points, epsabs=1e-10, limit=150)[0]
        return float(np.clip(value, 0., 1.))


def estimate(observations, target_id, *, shapes=SHAPES, config=Config()) -> GaussianBelief:
    """Infer once, discard particles, return only a 2D Gaussian."""
    posterior = estimate_particles(observations, target_id, shapes=shapes, config=config)
    if posterior.warnings:
        warnings.warn("; ".join(posterior.warnings), RuntimeWarning, stacklevel=2)
    return GaussianBelief.from_particles(posterior)
