"""Exact orthographic visibility of upright box proxies, without image sampling.

Finite-height YZ cells partition visibility. Source identity is evaluation-only
metadata in Patch; it is not part of SurfaceObservation.
"""
from dataclasses import dataclass
import numpy as np
from observed_obb import Box, Observation, observe
from observed_obb.observation import _footprint, _visibility_edges, _front_depth

FRAME = np.array(((0., 1.), (-1., 0.)))


@dataclass(frozen=True)
class SurfaceObservation:
    observation_id: int
    point: tuple[float, float, float]
    bbox_yz: tuple[float, float, float, float]  # ymin,zmin,ymax,zmax
    corrected: bool
    correction_m: float


@dataclass(frozen=True)
class Patch:
    observation: SurfaceObservation
    source_index: int  # evaluation ONLY, not perception output
    cells: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True)
class Comparison:
    patch: Patch
    legacy: Observation | None
    center_distance_m: float | None
    front_depth_error_m: float | None


def _components(owner, source):
    remaining = set(map(tuple, np.argwhere(owner == source)))
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, component = [first], []
        while todo:
            i,j = todo.pop()
            component.append((i,j))
            for neighbor in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    todo.append(neighbor)
        yield sorted(component)


def benchmark(boxes, *, minimum_width=.02, minimum_ratio=.20):
    """Return surface/legacy comparison records and the unmodified legacy snapshot.

    New finite-height components retain legacy lateral-width filtering. No
    area/height threshold is silently introduced. Objects filtered from output
    still occlude. Equal depths use input order, as in the legacy kernel.
    """
    boxes = tuple(boxes)
    legacy = observe(boxes, minimum_width=minimum_width, minimum_ratio=minimum_ratio)
    if not boxes:
        return (), legacy
    polygons = [_footprint(b) @ FRAME.T for b in boxes]
    ys = _visibility_edges(polygons)
    zs = np.unique([z for b in boxes for z in (b.center[2]-b.size[2]/2,b.center[2]+b.size[2]/2)])
    mids_y, mids_z = (ys[:-1]+ys[1:])/2, (zs[:-1]+zs[1:])/2
    depths = np.array([[_front_depth(p,y) for y in mids_y] for p in polygons])
    heights = np.array([[b.center[2]-b.size[2]/2 < z < b.center[2]+b.size[2]/2
                         for z in mids_z] for b in boxes])
    values = np.where(heights[:,None,:],depths[:,:,None],np.inf)
    owners = values.argmin(axis=0)
    owners[~np.isfinite(values.min(axis=0))] = -1

    # Match old output ordering to physical sources ONLY for benchmark comparison.
    # Independent IDs from the two methods are never assumed equal.
    flat_owner = depths.argmin(axis=0)
    flat_owner[~np.isfinite(depths.min(axis=0))] = -1
    associations = []
    for source,p in enumerate(polygons):
        threshold = max(minimum_width,minimum_ratio*np.ptp(p[:,0]))
        segments: list[tuple[float, float]] = []
        for k in np.flatnonzero(flat_owner==source):
            lo,hi = ys[k:k+2]
            if segments and segments[-1][1] == lo:
                segments[-1] = (segments[-1][0],hi)
            else:
                segments.append((lo,hi))
        for lo,hi in segments:
            tolerance = 4*np.finfo(float).eps*max(abs(lo),abs(hi),threshold)
            if hi-lo >= threshold-tolerance:
                associations.append((source,lo,hi))
    if len(associations) != len(legacy):
        raise RuntimeError('Legacy comparison association disagrees with observation kernel')

    records: list[Comparison] = []
    for source,(box,polygon) in enumerate(zip(boxes,polygons)):
        threshold = max(minimum_width,minimum_ratio*np.ptp(polygon[:,0]))
        for component in _components(owners,source):
            cells = np.array([(ys[i],zs[j],ys[i+1],zs[j+1]) for i,j in component])
            low,high = cells[:,:2].min(0),cells[:,2:].max(0)
            tolerance = 4*np.finfo(float).eps*max(abs(low[0]),abs(high[0]),threshold)
            if high[0]-low[0] < threshold-tolerance:
                continue
            center = (low+high)/2
            # Verify actual ray ownership, including cell seams. If necessary
            # choose the nearest visible cell,
            # inset by <=1 micrometre; deterministic ties use cell order.
            ray_depths = [(_front_depth(p,float(center[0]))
                           if b.center[2]-b.size[2]/2 <= center[1] <= b.center[2]+b.size[2]/2
                           else np.inf) for b,p in zip(boxes,polygons)]
            inside = (int(np.argmin(ray_depths)) == source and
                      np.any(np.all((center>=cells[:,:2]) & (center<=cells[:,2:]),axis=1)))
            if inside:
                selected = center
            else:
                margin = np.minimum(1e-6,(cells[:,2:]-cells[:,:2])*.25)
                projected = np.clip(center,cells[:,:2]+margin,cells[:,2:]-margin)
                selected = projected[np.argmin(np.sum((projected-center)**2,axis=1))]
            point = (-_front_depth(polygon,float(selected[0])),float(selected[0]),float(selected[1]))
            shift = float(np.linalg.norm(selected-center))
            item = SurfaceObservation(len(records),point,tuple(np.r_[low,high]),not inside,shift)
            patch = Patch(item,source,tuple(map(tuple,cells)))
            matches = [old for old,(s,lo,hi) in zip(legacy,associations)
                       if s==source and lo-1e-12 <= selected[0] <= hi+1e-12]
            old = matches[0] if matches else None
            distance,depth_error = None,None
            if old is not None:
                distance = float(np.linalg.norm(np.array(point)-old.box.center))
                fitted_front = -_front_depth(_footprint(old.box) @ FRAME.T,float(selected[0]))
                depth_error = abs(point[0]-fitted_front)
            records.append(Comparison(patch,old,distance,depth_error))
    return tuple(records), legacy
