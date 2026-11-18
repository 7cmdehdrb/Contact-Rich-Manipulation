"""Offline clipping dictionary and bounded OBB-only retrieval at reset time.

This is a discrete geometric approximation, not importance-sampled Bayes.
No evaluation truth, source identity or simulator state enters inference.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
from observed_obb import Box
from observed_obb.observation import _footprint
from .forward import ForwardModel, corner_error
from .gaussian import GaussianBelief
from .shapes import SHAPES

BANK_PATH = Path(__file__).parent / 'data' / 'completion_bank.npz'
BANK_VERSION = 1


def feature(box):
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    a, b = np.asarray(box.size[:2])/box.size[2]
    # Covariance signature is invariant to equivalent OBB axis representations.
    return np.array([abs(c)*a+abs(s)*b, abs(s)*a+abs(c)*b,
                     c*c*a*a+s*s*b*b, c*s*(a*a-b*b), s*s*a*a+c*c*b*b])


@dataclass(frozen=True)
class FastResult:
    belief: GaussianBelief
    warnings: tuple[str, ...]
    evaluated: int
    seconds: float
    complete_weight: float


class LookupEstimator:
    def __init__(self, path=BANK_PATH, *, candidates=16):
        if not 1 <= candidates <= 128:
            raise ValueError('candidates must be in [1,128]')
        self.candidates = candidates
        with np.load(path, allow_pickle=False) as bank:
            metadata = json.loads(str(bank['metadata']))
            if metadata['version'] != BANK_VERSION or metadata['shapes'] != [list(s.size) for s in SHAPES]:
                raise ValueError('Completion bank is stale; rebuild it')
            self.features = bank['features'].astype(float)
            self.records = bank['records'].astype(float)
        self.trees = {}
        self.rows = {}
        for shape_index in range(len(SHAPES)):
            for mask in (1,2,3):
                rows = np.flatnonzero((self.records[:,0]==shape_index) & (self.records[:,1]==mask))
                self.rows[shape_index,mask] = rows
                self.trees[shape_index,mask] = cKDTree(self.features[rows])

    def estimate(self, observations, target_id):
        start = perf_counter()
        snapshot = tuple(observations)
        ids = [o.observation_id for o in snapshot]
        if len(set(ids)) != len(ids) or target_id not in ids:
            raise ValueError('Unique snapshot IDs and existing target_id required')
        target = snapshot[ids.index(target_id)].box
        others = [o.box for o in snapshot if o.observation_id != target_id]
        target_points = _footprint(target)
        lo, hi = target_points[:,1].min(), target_points[:,1].max()
        front = target_points[:,0].min()
        mask = 0
        for other in others:
            points = _footprint(other)
            if points[:,0].max() <= front:
                continue
            a,b = points[:,1].min(),points[:,1].max()
            if a < lo and b >= lo-.006: mask |= 1
            if b > hi and a <= hi+.006: mask |= 2
        mean0 = np.array(target.center[:2])
        floor = np.eye(2)*.003**2
        compatible = [i for i,s in enumerate(SHAPES) if .9-1e-10 <= target.size[2]/s.size[2] <= 1.1+1e-10]
        warnings = []
        if not compatible:
            warnings.append('no_template_height_support')
        # No observed occlusion boundary => complete-box approximation. As before,
        # a missing/filtered occluder is unidentifiable from these inputs alone.
        if not mask or not compatible:
            return FastResult(GaussianBelief(mean0,floor),tuple(warnings),0,perf_counter()-start,1.)
        query = feature(target)
        options = []
        # Also permit one-sided hypotheses when evidence exists on both sides.
        masks = (1,2,3) if mask == 3 else (mask,)
        for shape_index in compatible:
            for candidate_mask in masks:
                key = shape_index,candidate_mask
                distances, indices = self.trees[key].query(query,k=min(8,len(self.rows[key])))
                for distance,index in zip(np.atleast_1d(distances),np.atleast_1d(indices)):
                    options.append((float(distance),int(self.rows[key][index])))
        options.sort()
        model = ForwardModel(others)
        centers, likelihoods = [], []
        evaluated = 0
        seen = set()
        for distance, index in options:
            if evaluated >= self.candidates: break
            shape_index, _, yaw, ox, oy, width_ratio = self.records[index]
            size = np.asarray(SHAPES[int(shape_index)].size)*target.size[2]/SHAPES[int(shape_index)].size[2]
            center = mean0 + np.array([ox,oy])*target.size[2]
            dedup = tuple(np.round(np.r_[center,yaw],5))
            if dedup in seen: continue
            seen.add(dedup)
            candidate = Box((*center,target.center[2]),tuple(size),float(yaw))
            predictions = model.predict(candidate)
            evaluated += 1
            if not predictions: continue
            errors = [corner_error(p,target) for p in predictions]
            j = int(np.argmin(errors))
            error = errors[j]
            for n,p in enumerate(predictions):
                if n != j:
                    error += min((corner_error(p,o)+(p.size[2]-o.size[2])**2 for o in others),default=.01)
            if corner_error(predictions[j],candidate) <= .003**2: continue
            likelihoods.append(np.exp(-error/(2*.004**2)))
            centers.append(center)
        if not likelihoods or max(likelihoods) < .01:
            warnings.append('lookup_low_support')
            # Do not return a falsely tight complete-only result at a suspicious boundary.
            return FastResult(GaussianBelief(mean0,floor+np.diag([.005**2,.02**2])),
                              tuple(warnings),evaluated,perf_counter()-start,1.)
        likelihood_array = np.array(likelihoods)
        # Complete-box hypothesis always remains; relative family support is
        # bounded by the best retrieved fit. This is explicit heuristic weighting.
        partial_support = float(likelihood_array.max())
        complete_weight = 1/(1+partial_support)
        weights = likelihood_array/likelihood_array.sum()*(1-complete_weight)
        weights = np.r_[complete_weight,weights]
        centers = np.vstack((mean0,centers))
        mean = weights @ centers
        delta = centers-mean
        # Bank discretization uncertainty (yaw/cut grid) in addition to sensor floor.
        covariance = (delta*weights[:,None]).T @ delta+floor+np.eye(2)*(.025*target.size[2])**2
        return FastResult(GaussianBelief(mean,covariance),tuple(warnings),evaluated,
                          perf_counter()-start,complete_weight)


@lru_cache(maxsize=1)
def default_estimator():
    return LookupEstimator()


def estimate_fast(observations, target_id):
    """Five-value Gaussian API; table loaded once per process."""
    return default_estimator().estimate(observations,target_id).belief
