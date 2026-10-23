"""Uniform scaling into the parent's yaw-invariant placement envelope."""

import math as math

def fit_scale(lower, upper, size_fraction=1.0):
    if not 0.8 <= size_fraction <= 1.0:
        raise ValueError('size_fraction must be in [0.8, 1.0]')
    size = [b-a for a,b in zip(lower,upper)]
    if len(size)!=3 or not all(math.isfinite(x) and x>0 for x in size):
        raise ValueError('Expected a finite, nonempty 3D visual bound')
    scale = 1.5 * size_fraction * min(0.04 / (math.hypot(size[0],size[1])/2), 0.12/size[2])
    offset = (-scale*(lower[0]+upper[0])/2, -scale*(lower[1]+upper[1])/2, -scale*lower[2])
    return scale, offset


def flat_collision_mesh(dimensions, kind, segments=32):
    """Closed convex proxy in the bottom-origin body frame, inside the visual OBB.

    Scanned bottoms are not planar. A box or elliptical cylinder provides a
    horizontal support polygon without altering the rendered asset or its scale.
    """
    if len(dimensions) != 3 or not all(math.isfinite(v) and v > 0 for v in dimensions):
        raise ValueError('Expected three finite positive dimensions')
    x, y, height = dimensions
    if kind == 'box':
        ring = [(-x / 2, -y / 2), (x / 2, -y / 2), (x / 2, y / 2), (-x / 2, y / 2)]
    elif kind == 'can':
        if not isinstance(segments, int) or segments < 8 or segments % 4:
            raise ValueError('Cylinder segments must be a multiple of four, at least eight')
        ring = [(x / 2 * math.cos(2 * math.pi * i / segments),
                 y / 2 * math.sin(2 * math.pi * i / segments)) for i in range(segments)]
    else:
        raise ValueError(f'Unsupported collision proxy: {kind}')
    count = len(ring)
    points = [(a, b, z) for z in (0., height) for a, b in ring]
    faces = [list(reversed(range(count))), list(range(count, 2 * count))]
    faces.extend([[i, (i + 1) % count, (i + 1) % count + count, i + count] for i in range(count)])
    return points, [len(face) for face in faces], [index for face in faces for index in face]
