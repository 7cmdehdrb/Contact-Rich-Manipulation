"""Initial visible mesh snapshot; the surface_point ray/mask algorithm without its UI.

Only returned XYZ points enter observations. Mesh ownership is used locally for
exposure filtering and selecting the largest target component, never exported.
"""
from time import perf_counter
import numpy as np
from scipy import ndimage
import warp as wp


# Host-typed constants avoid Warp 1.8 narrowing Python literals to int32 or
# float32 before an in-kernel cast. Width filtering must match NumPy float64.
_COMPONENT_DISTANCE_LIMIT = wp.constant(wp.int64(9223372036854775807))
_COMPONENT_LEGACY_LIMIT = wp.constant(wp.float64(1.0e300))
_COMPONENT_SPACING = wp.constant(wp.float64(.001))
_COMPONENT_WIDTH_EPSILON = wp.constant(wp.float64(1.0e-9))


@wp.kernel(enable_backward=False)
def _cast(mesh: wp.uint64, owners: wp.array(dtype=int), width: int,  # type: ignore[valid-type]
          x: float, y: float, z: float, spacing: float,
          labels: wp.array(dtype=int), hits: wp.array(dtype=wp.vec3)):  # type: ignore[valid-type]
    i=wp.tid()
    origin=wp.vec3(x,y+(float(i%width)+0.5)*spacing,z+(float(i//width)+0.5)*spacing)
    direction=wp.vec3(-1.0,0.0,0.0)
    query=wp.mesh_query_ray(mesh,origin,direction,100.0)
    labels[i]=-2
    hits[i]=wp.vec3(0.0)
    if query.result:
        labels[i]=owners[query.face]
        hits[i]=origin+query.t*direction


def box_mesh(center, size):
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    vertices=np.asarray(center)+signs*np.asarray(size)/2
    quads=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
    faces=np.array([(q[0],q[i],q[i+1]) for q in quads for i in (1,2)],dtype=np.int32)
    return vertices,faces


def observe_mesh(vertices, faces, owners, *, device, spacing=.001):
    """Front +X rays, 4-connected masks, >=2cm and >=20% exposed width."""
    start=perf_counter()
    vertices=np.asarray(vertices,dtype=np.float32)
    faces=np.asarray(faces,dtype=np.int32).reshape(-1,3)
    owners=np.asarray(owners,dtype=np.int32)
    if not len(faces) or not np.isfinite(vertices).all() or len(faces)!=len(owners):
        raise ValueError('Finite visible mesh triangles and ownership are required')
    visible=vertices[np.unique(faces[owners>=0])]
    low=np.floor(visible[:,1:].min(0)/spacing)*spacing-spacing
    high=np.ceil(visible[:,1:].max(0)/spacing)*spacing+spacing
    width,height=np.ceil((high-low)/spacing).astype(int)
    count=int(width*height)
    cuda=wp.get_device(device).is_cuda
    if cuda:wp.synchronize_device(device)
    transfer_start=perf_counter()
    points=wp.array(vertices,dtype=wp.vec3,device=device)
    indices=wp.array(faces.ravel(),dtype=int,device=device)
    source_ids=wp.array(owners,dtype=int,device=device)
    if cuda:wp.synchronize_device(device)
    upload_ms=1000*(perf_counter()-transfer_start)
    mesh=wp.Mesh(points=points,indices=indices)
    labels=wp.empty(count,dtype=int,device=device);hits=wp.empty(count,dtype=wp.vec3,device=device)
    if cuda:wp.synchronize_device(device)
    build=perf_counter()
    wp.launch(_cast,dim=count,inputs=[mesh.id,source_ids,int(width),
              float(vertices[:,0].max()+.05),float(low[0]),float(low[1]),spacing,labels,hits],device=device)
    if cuda:wp.synchronize_device(device)
    ray_end=perf_counter()
    mask=labels.numpy().reshape(height,width);points=hits.numpy().reshape(height,width,3)
    readback=perf_counter()
    target=None;target_area=0;obstacles=[]
    for source in np.unique(owners[owners>=0]):
        full_width=np.ptp(vertices[np.unique(faces[owners==source]),1])
        components,_=ndimage.label(mask==source)
        for label,slices in enumerate(ndimage.find_objects(components),1):
            if slices is None:continue
            rows,cols=slices
            if (cols.stop-cols.start)*spacing+1e-9<max(.02,.2*full_width):continue
            rr,cc=np.nonzero(components[rows,cols]==label);rr+=rows.start;cc+=cols.start
            nearest=np.argmin((rr-rr.mean())**2+(cc-cc.mean())**2)
            point=points[rr[nearest],cc[nearest]].copy()
            if source==0:
                if len(rr)>target_area:target,target_area=point,len(rr)
            else:obstacles.append(point)
    return target,np.asarray(obstacles,dtype=np.float32).reshape(-1,3),dict(
        total_ms=1000*(perf_counter()-start),build_ms=1000*(build-start),
        ray_ms=1000*(ray_end-build),upload_ms=upload_ms,readback_ms=1000*(readback-ray_end),
        components_ms=1000*(perf_counter()-readback),rays=count)


def _triangles(prim):
    from pxr import UsdGeom
    if prim.IsA(UsdGeom.Mesh):
        mesh=UsdGeom.Mesh(prim)
        vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32)
        holes=set(mesh.GetHoleIndicesAttr().Get() or [])
        faces=[];offset=0
        for index,count in enumerate(mesh.GetFaceVertexCountsAttr().Get()):
            polygon=indices[offset:offset+count];offset+=count
            if index not in holes:faces.extend((polygon[0],polygon[k],polygon[k+1]) for k in range(1,count-1))
        return vertices,np.asarray(faces,dtype=np.int32).reshape(-1,3)
    if prim.IsA(UsdGeom.Cube):
        size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        return box_mesh((0,0,0),(size,size,size))
    if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Cone):
        geom=UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Cone(prim)
        radius,height=float(geom.GetRadiusAttr().Get()),float(geom.GetHeightAttr().Get())
        angle=np.arange(128)*2*np.pi/128
        ring=np.column_stack((radius*np.cos(angle),radius*np.sin(angle)))
        top=1. if prim.IsA(UsdGeom.Cylinder) else 0.
        vertices=np.vstack((np.column_stack((ring,np.full(128,-height/2))),
                            np.column_stack((ring*top,np.full(128,height/2))),
                            (0,0,-height/2),(0,0,height/2)))
        faces=[]
        for i in range(128):
            j=(i+1)%128
            faces.extend(((i,j,j+128),(i,j+128,i+128),(256,j,i),(257,i+128,j+128)))
        axis=str(geom.GetAxisAttr().Get()).upper()
        if axis=='X':vertices=vertices[:,[2,0,1]]
        elif axis=='Y':vertices=vertices[:,[1,2,0]]
        return vertices,np.asarray(faces,dtype=np.int32)
    if prim.IsA(UsdGeom.Gprim):raise ValueError(f'Unsupported visible geometry {prim.GetPath()}')
    return None


class SurfaceGeometry:
    """Cache scaled geometry in each rigid-root frame; plan poses without USD writes."""
    def __init__(self, stage):
        self.stage=stage;self.cache={}

    def local(self,path):
        if path in self.cache:return self.cache[path]
        from pxr import Usd,UsdGeom
        transforms=UsdGeom.XformCache();root=self.stage.GetPrimAtPath(path)
        matrix=np.asarray(transforms.GetLocalToWorldTransform(root))
        rotation=matrix[:3,:3]/np.linalg.norm(matrix[:3,:3],axis=1)[:,None]
        vertices=[];faces=[];offset=0
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Imageable):continue
            image=UsdGeom.Imageable(prim)
            if image.ComputeVisibility()=='invisible' or image.ComputePurpose()=='guide':continue
            result=_triangles(prim)
            if result is None:continue
            v,f=result
            if not len(f):continue
            transform=np.asarray(transforms.GetLocalToWorldTransform(prim))
            v=(v@transform[:3,:3]+transform[3,:3]-matrix[3,:3])@rotation.T
            vertices.append(v);faces.append(f+offset);offset+=len(v)
        if not faces:raise RuntimeError(f'No visible triangles below {path}')
        result=np.concatenate(vertices),np.concatenate(faces)
        self.cache[path]=result
        return result

    def observe(self, env_path, local, yaw, active, names, *, device):
        vertices=[];faces=[];owners=[];offset=0
        for source in [*np.flatnonzero(active),-1]:
            path=env_path+('/Shelf' if source==-1 else '/'+names[source])
            v,f=self.local(path)
            if source>=0:
                c,s=np.cos(yaw[source]),np.sin(yaw[source])
                rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                v=v@rotation.T+local[source]
            vertices.append(v);faces.append(f+offset);owners.append(np.full(len(f),source,dtype=np.int32));offset+=len(v)
        return observe_mesh(np.concatenate(vertices),np.concatenate(faces),np.concatenate(owners),device=device)


@wp.kernel(enable_backward=False)
def _cast_batch(mesh_ids: wp.array(dtype=wp.uint64), shelf_mesh: wp.uint64,  # type: ignore[valid-type]
                offsets: wp.array(dtype=int), widths: wp.array(dtype=int),  # type: ignore[valid-type]
                grid: wp.array(dtype=wp.vec3), spacing: float,  # type: ignore[valid-type]
                kinds: wp.array2d(dtype=int), scales: wp.array2d(dtype=float),  # type: ignore[valid-type]
                poses: wp.array2d(dtype=wp.vec3), rotations: wp.array2d(dtype=wp.vec2),  # type: ignore[valid-type]
                sources: wp.array2d(dtype=int), source_counts: wp.array(dtype=int),  # type: ignore[valid-type]
                lower: wp.array2d(dtype=wp.vec3), upper: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
                output: wp.array(dtype=wp.vec2)):  # type: ignore[valid-type]
    """Query immutable local meshes; return only owner and surface X per ray."""
    index = wp.tid()
    # Ragged images avoid padding every row to the largest scene in the chunk.
    lo = int(0)
    hi = offsets.shape[0] - 1
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if offsets[middle] <= index:
            lo = middle
        else:
            hi = middle
    row = lo
    pixel = index - offsets[row]
    origin = wp.vec3(grid[row][0],
                     grid[row][1] + (float(pixel % widths[row]) + 0.5) * spacing,
                     grid[row][2] + (float(pixel // widths[row]) + 0.5) * spacing)
    nearest = float(100.0)
    owner = int(-2)
    for slot in range(source_counts[row]):
        source = sources[row, slot]
        low = lower[row, source]
        high = upper[row, source]
        # Bounds are the actual projected visible vertices, not an OBB.
        if (origin[1] >= low[1] - 1.0e-6 and origin[1] <= high[1] + 1.0e-6
                and origin[2] >= low[2] - 1.0e-6 and origin[2] <= high[2] + 1.0e-6
                and origin[0] - high[0] <= nearest + 1.0e-6):
            c = rotations[row, source][0]
            s = rotations[row, source][1]
            scale = scales[row, source]
            norm_squared = c * c + s * s
            norm = wp.sqrt(norm_squared)
            delta = origin - poses[row, source]
            local_origin = wp.vec3((c * delta[0] + s * delta[1]) / (scale * norm_squared),
                                   (-s * delta[0] + c * delta[1]) / (scale * norm_squared),
                                   delta[2] / scale)
            direction = wp.vec3(-c / norm, s / norm, 0.0)
            query = wp.mesh_query_ray(mesh_ids[kinds[row, source]], local_origin,
                                      direction, nearest / (scale * norm))
            if query.result:
                distance = query.t * scale * norm
                if distance < nearest:
                    nearest = distance
                    owner = source
    # A shelf hit only affects observations when it occludes an object hit.
    # Background pixels never enter any object component or representative.
    if owner >= 0:
        shelf_hit = wp.mesh_query_ray(shelf_mesh, origin, wp.vec3(-1.0, 0.0, 0.0), nearest)
        if shelf_hit.result and shelf_hit.t < nearest:
            nearest = shelf_hit.t
            owner = -1
    x = float(0.0)
    if owner != -2:
        x = origin[0] - nearest
    # Owners are small exact integers. Packing them with X needs one readback;
    # Y/Z are reconstructed from the unchanged float32 pixel-center lattice.
    output[index] = wp.vec2(float(owner), x)


@wp.func
def _image_row(offsets: wp.array(dtype=int), index: int):  # type: ignore[valid-type]
    lo = int(0)
    hi = offsets.shape[0] - 1
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if offsets[middle] <= index:
            lo = middle
        else:
            hi = middle
    return lo


@wp.func
def _component_root(parent: wp.array(dtype=int), index: int):  # type: ignore[valid-type]
    root = index
    while parent[root] != root:
        root = parent[root]
    return root


@wp.func
def _join_components(parent: wp.array(dtype=int), first: int, second: int):  # type: ignore[valid-type]
    # Every successful CAS strictly decreases a root pointer. Concurrent joins
    # therefore cannot form a cycle; retry only if another edge changed it.
    joined = bool(False)
    while not joined:
        a = _component_root(parent, first)
        b = _component_root(parent, second)
        if a == b:
            joined = True
        else:
            low = wp.min(a, b)
            high = wp.max(a, b)
            previous = wp.atomic_cas(parent, high, high, low)
            if previous == high:
                joined = True


@wp.kernel(enable_backward=False)
def _initialize_components(pixels: wp.array(dtype=wp.vec2), parent: wp.array(dtype=int),  # type: ignore[valid-type]
                           area: wp.array(dtype=int),  # type: ignore[valid-type]
                           sum_row: wp.array(dtype=wp.uint64), sum_col: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
                           minimum_col: wp.array(dtype=int), maximum_col: wp.array(dtype=int),  # type: ignore[valid-type]
                           distance: wp.array(dtype=wp.int64), legacy_distance: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
                           winner: wp.array(dtype=int)):  # type: ignore[valid-type]
    index = wp.tid()
    parent[index] = -1
    area[index] = 0
    # Only object pixels can become roots. Every later statistics access is
    # guarded by parent/area, so background needs no other buffer writes.
    if pixels[index][0] >= 0.0:
        parent[index] = index
        sum_row[index] = wp.uint64(0)
        sum_col[index] = wp.uint64(0)
        minimum_col[index] = 2147483647
        maximum_col[index] = -1
        distance[index] = _COMPONENT_DISTANCE_LIMIT
        legacy_distance[index] = _COMPONENT_LEGACY_LIMIT
        winner[index] = 2147483647


@wp.kernel(enable_backward=False)
def _union_components(pixels: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=int),  # type: ignore[valid-type]
                      widths: wp.array(dtype=int), parent: wp.array(dtype=int)):  # type: ignore[valid-type]
    index = wp.tid()
    owner = pixels[index][0]
    if owner >= 0.0:
        row = _image_row(offsets, index)
        pixel = index - offsets[row]
        width = widths[row]
        if pixel % width != 0:
            if pixels[index - 1][0] == owner:
                _join_components(parent, index, index - 1)
        if pixel >= width:
            if pixels[index - width][0] == owner:
                # All horizontal edges remain. A same-owner 2x2 square's
                # vertical connection is already represented farther left;
                # each overlapping row-run keeps its first vertical edge.
                # Components and their minimum row-major root stay unchanged.
                redundant = bool(False)
                if pixel % width != 0:
                    if pixels[index - 1][0] == owner:
                        redundant = pixels[index - width - 1][0] == owner
                if not redundant:
                    _join_components(parent, index, index - width)


@wp.func_native(r'''
// Group only equal component roots. Integer sums and extrema are exact,
// so scheduling and warp membership cannot affect representative points.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// Native reductions use uint32 lanes. Current images are much smaller;
// retain scalar atomics if an unusually large image could overflow a group.
if (row <= 134217727 && col <= 134217727) {
const unsigned mask = __activemask();
const unsigned group = __match_any_sync(mask, root);
const unsigned n = __popc(group);
const unsigned r = __reduce_add_sync(group, static_cast<unsigned>(row));
const unsigned c = __reduce_add_sync(group, static_cast<unsigned>(col));
const int lower = __reduce_min_sync(group, col);
const int upper = __reduce_max_sync(group, col);
const unsigned lane = static_cast<unsigned>(threadIdx.x) & 31u;
if (lane == static_cast<unsigned>(__ffs(group) - 1)) {
    wp::atomic_add(&area[root], static_cast<int>(n));
    wp::atomic_add(&sum_row[root], wp::uint64(r));
    wp::atomic_add(&sum_col[root], wp::uint64(c));
    wp::atomic_min(&minimum_col[root], lower);
    wp::atomic_max(&maximum_col[root], upper);
}
return;
}
#endif
wp::atomic_add(&area[root], 1);
wp::atomic_add(&sum_row[root], wp::uint64(row));
wp::atomic_add(&sum_col[root], wp::uint64(col));
wp::atomic_min(&minimum_col[root], col);
wp::atomic_max(&maximum_col[root], col);
''')
def _accumulate_component(root: int, row: int, col: int,
                          area: wp.array(dtype=int),  # type: ignore[valid-type]
                          sum_row: wp.array(dtype=wp.uint64), sum_col: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
                          minimum_col: wp.array(dtype=int), maximum_col: wp.array(dtype=int)):  # type: ignore[valid-type]
    ...


@wp.kernel(enable_backward=False)
def _reduce_components(offsets: wp.array(dtype=int), widths: wp.array(dtype=int),  # type: ignore[valid-type]
                       parent: wp.array(dtype=int), area: wp.array(dtype=int),  # type: ignore[valid-type]
                       sum_row: wp.array(dtype=wp.uint64), sum_col: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
                       minimum_col: wp.array(dtype=int), maximum_col: wp.array(dtype=int)):  # type: ignore[valid-type]
    index = wp.tid()
    if parent[index] >= 0:
        root = _component_root(parent, index)
        parent[index] = root
        image = _image_row(offsets, index)
        pixel = index - offsets[image]
        row = pixel // widths[image]
        col = pixel % widths[image]
        # Warp 1.8's signed int64 atomic_add is broken; retain uint64 sums.
        # Ampere+ combines equal-root lanes before the same integer atomics.
        _accumulate_component(root, row, col, area, sum_row, sum_col, minimum_col, maximum_col)


@wp.kernel(enable_backward=False)
def _filter_components(pixels: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=int),  # type: ignore[valid-type]
                       thresholds: wp.array2d(dtype=wp.float64), area: wp.array(dtype=int),  # type: ignore[valid-type]
                       minimum_col: wp.array(dtype=int), maximum_col: wp.array(dtype=int)):  # type: ignore[valid-type]
    index = wp.tid()
    if area[index] > 0:
        image = _image_row(offsets, index)
        source = int(pixels[index][0])
        width = wp.float64(maximum_col[index] - minimum_col[index] + 1) * _COMPONENT_SPACING
        if width + _COMPONENT_WIDTH_EPSILON < thresholds[image, source]:
            area[index] = 0


@wp.func
def _centroid_distance(pixel: int, width: int, size: int, sum_row: wp.uint64, sum_col: wp.uint64):
    # Common area^2 denominator cancels within a component, leaving an exact
    # integer comparison and deterministic ties independent of reduction order.
    dr = wp.int64(pixel // width) * wp.int64(size) - wp.int64(sum_row)
    dc = wp.int64(pixel % width) * wp.int64(size) - wp.int64(sum_col)
    return dr * dr + dc * dc


@wp.func
def _legacy_centroid_distance(pixel: int, width: int, size: int, sum_row: wp.uint64, sum_col: wp.uint64):
    dr = wp.float64(pixel // width) - wp.float64(sum_row) / wp.float64(size)
    dc = wp.float64(pixel % width) - wp.float64(sum_col) / wp.float64(size)
    return dr * dr + dc * dc


@wp.kernel(enable_backward=False)
def _minimum_component_distance(offsets: wp.array(dtype=int), widths: wp.array(dtype=int),  # type: ignore[valid-type]
                                parent: wp.array(dtype=int), area: wp.array(dtype=int),  # type: ignore[valid-type]
                                sum_row: wp.array(dtype=wp.uint64), sum_col: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
                                distance: wp.array(dtype=wp.int64)):  # type: ignore[valid-type]
    index = wp.tid()
    root = parent[index]
    if root >= 0:
        if area[root] > 0:
            image = _image_row(offsets, index)
            value = _centroid_distance(index - offsets[image], widths[image], area[root], sum_row[root], sum_col[root])
            wp.atomic_min(distance, root, value)


@wp.kernel(enable_backward=False, module='unique')
def _minimum_legacy_distance(offsets: wp.array(dtype=int), widths: wp.array(dtype=int),  # type: ignore[valid-type]
                             parent: wp.array(dtype=int), area: wp.array(dtype=int),  # type: ignore[valid-type]
                             sum_row: wp.array(dtype=wp.uint64), sum_col: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
                             distance: wp.array(dtype=wp.int64), legacy_distance: wp.array(dtype=wp.float64)):  # type: ignore[valid-type]
    index = wp.tid()
    root = parent[index]
    if root >= 0:
        if area[root] > 0:
            image = _image_row(offsets, index)
            pixel = index - offsets[image]
            value = _centroid_distance(pixel, widths[image], area[root], sum_row[root], sum_col[root])
            if value == distance[root]:
                # Mathematical ties can differ by one floating-point rounding bit
                # in NumPy's old expression. Reproduce that expression without FMA
                # before applying its row-major argmin tie rule.
                legacy = _legacy_centroid_distance(pixel, widths[image], area[root], sum_row[root], sum_col[root])
                wp.atomic_min(legacy_distance, root, legacy)


@wp.kernel(enable_backward=False, module='unique')
def _pick_component_pixel(offsets: wp.array(dtype=int), widths: wp.array(dtype=int),  # type: ignore[valid-type]
                          parent: wp.array(dtype=int), area: wp.array(dtype=int),  # type: ignore[valid-type]
                          sum_row: wp.array(dtype=wp.uint64), sum_col: wp.array(dtype=wp.uint64),  # type: ignore[valid-type]
                          distance: wp.array(dtype=wp.int64), legacy_distance: wp.array(dtype=wp.float64),  # type: ignore[valid-type]
                          winner: wp.array(dtype=int)):  # type: ignore[valid-type]
    index = wp.tid()
    root = parent[index]
    if root >= 0:
        if area[root] > 0:
            image = _image_row(offsets, index)
            value = _centroid_distance(index - offsets[image], widths[image], area[root], sum_row[root], sum_col[root])
            if value == distance[root]:
                legacy = _legacy_centroid_distance(index - offsets[image], widths[image], area[root], sum_row[root], sum_col[root])
                if legacy == legacy_distance[root]:
                    wp.atomic_min(winner, root, index)


# Isaac Sim bundles Warp 1.8, whose decorator has no module_options keyword.
# Configure these isolated modules before compilation, without changing the
# reference raycaster or any other kernel's floating-point compilation flags.
_minimum_legacy_distance.module.options['fuse_fp'] = False
_minimum_legacy_distance.module.mark_modified()
_pick_component_pixel.module.options['fuse_fp'] = False
_pick_component_pixel.module.mark_modified()


@wp.kernel(enable_backward=False)
def _collect_components(pixels: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=int),  # type: ignore[valid-type]
                        area: wp.array(dtype=int), target_key: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
                        obstacle_count: wp.array(dtype=int), obstacle_roots: wp.array2d(dtype=int)):  # type: ignore[valid-type]
    index = wp.tid()
    if area[index] > 0:
        image = _image_row(offsets, index)
        source = int(pixels[index][0])
        if source == 0:
            key = wp.int64(area[index]) * wp.int64(pixels.shape[0] + 1) + wp.int64(pixels.shape[0] - index)
            wp.atomic_max(target_key, image, key)
        else:
            slot = wp.atomic_add(obstacle_count, image, 1)
            if slot < 16:
                obstacle_roots[image, slot] = index


@wp.func
def _pixel_point(pixels: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=int),  # type: ignore[valid-type]
                 widths: wp.array(dtype=int), grid: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
                 image: int, index: int):
    pixel = index - offsets[image]
    return wp.vec3(pixels[index][1],
                   grid[image][1] + (float(pixel % widths[image]) + 0.5) * 0.001,
                   grid[image][2] + (float(pixel // widths[image]) + 0.5) * 0.001)


@wp.kernel(enable_backward=False)
def _pack_components(pixels: wp.array(dtype=wp.vec2), offsets: wp.array(dtype=int),  # type: ignore[valid-type]
                     widths: wp.array(dtype=int), grid: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
                     winner: wp.array(dtype=int), target_key: wp.array(dtype=wp.int64),  # type: ignore[valid-type]
                     obstacle_count: wp.array(dtype=int), obstacle_roots: wp.array2d(dtype=int),  # type: ignore[valid-type]
                     output: wp.array2d(dtype=wp.vec4)):  # type: ignore[valid-type]
    image = wp.tid()
    output[image, 0] = wp.vec4(0.0)
    key = target_key[image]
    if key > wp.int64(0):
        root = pixels.shape[0] - int(key % wp.int64(pixels.shape[0] + 1))
        point = _pixel_point(pixels, offsets, widths, grid, image, winner[root])
        output[image, 0] = wp.vec4(1.0, point[0], point[1], point[2])
    count = obstacle_count[image]
    output[image, 1] = wp.vec4(float(count), 0.0, 0.0, 0.0)
    # At most 16 accepted components: sort by source, then canonical row-major
    # root. Atomic append order is deliberately never exposed to the policy.
    kept = wp.min(count, 16)
    for i in range(1, kept):
        root = obstacle_roots[image, i]
        source = int(pixels[root][0])
        j = i - 1
        moving = bool(True)
        while j >= 0 and moving:
            previous = obstacle_roots[image, j]
            previous_source = int(pixels[previous][0])
            if previous_source > source or (previous_source == source and previous > root):
                obstacle_roots[image, j + 1] = previous
                j -= 1
            else:
                moving = False
        obstacle_roots[image, j + 1] = root
    for i in range(kept):
        root = obstacle_roots[image, i]
        point = _pixel_point(pixels, offsets, widths, grid, image, winner[root])
        output[image, i + 2] = wp.vec4(1.0, point[0], point[1], point[2])


@wp.func_native(r'''
const float c = rotation[0], s = rotation[1];
const float norm_squared = c * c + s * s;
const float norm = sqrtf(norm_squared);
const wp::vec3 direction(-c / norm, s / norm, 0.0f);
const float ay = scale * (a[0] * s + a[1] * c) + pose[1];
const float by = scale * (b[0] * s + b[1] * c) + pose[1];
const float cy = scale * (c_vertex[0] * s + c_vertex[1] * c) + pose[1];
const float az = scale * a[2] + pose[2];
const float bz = scale * b[2] + pose[2];
const float cz = scale * c_vertex[2] + pose[2];
const int left = wp::max(0, static_cast<int>(floorf((wp::min(ay, wp::min(by, cy)) - grid[1]) / spacing)) - 1);
const int right = wp::min(width - 1, static_cast<int>(ceilf((wp::max(ay, wp::max(by, cy)) - grid[1]) / spacing)) + 1);
const int bottom = wp::max(0, static_cast<int>(floorf((wp::min(az, wp::min(bz, cz)) - grid[2]) / spacing)) - 1);
const int top = wp::min(height - 1, static_cast<int>(ceilf((wp::max(az, wp::max(bz, cz)) - grid[2]) / spacing)) + 1);
const int rect_width = right - left + 1;
const int total = rect_width * (top - bottom + 1);
for (int pixel = lane; pixel < total; pixel += lanes) {
    const int row = bottom + pixel / rect_width;
    const int col = left + pixel % rect_width;
    const float z = grid[2] + (static_cast<float>(row) + 0.5f) * spacing;
    {
        const float y = grid[1] + (static_cast<float>(col) + 0.5f) * spacing;
        const wp::vec3 delta = wp::vec3(grid[0], y, z) - pose;
        const wp::vec3 origin((c * delta[0] + s * delta[1]) / (scale * norm_squared),
                              (-s * delta[0] + c * delta[1]) / (scale * norm_squared), delta[2] / scale);
        float t, u, v, sign;
        if (wp::intersect_ray_tri_woop(origin, direction, a, b, c_vertex, t, u, v, sign, nullptr)) {
            const float distance = t * scale * norm;
            if (distance >= 0.0f && distance < 100.0f) {
                union { float f; unsigned int u; } bits;
                bits.f = distance;
                const wp::uint64 key = (wp::uint64(bits.u) << 32) | wp::uint64(source);
                wp::atomic_min(&keys[offset + row * width + col], key);
            }
        }
    }
}
''')
def _raster_triangle(a: wp.vec3, b: wp.vec3, c_vertex: wp.vec3,
                    pose: wp.vec3, rotation: wp.vec2, scale: float,
                    grid: wp.vec3, spacing: float, width: int, height: int,
                    offset: int, source: int, lane: int, lanes: int, keys: wp.array(dtype=wp.uint64)):  # type: ignore[valid-type]
    ...


@wp.func_native(r'''
union { unsigned int u; float f; } bits;
bits.u = static_cast<unsigned int>(key >> 32);
return bits.f;
''')
def _raster_distance(key: wp.uint64) -> float:
    return 0.  # Native body above is used by Warp; never executed in Python.


_RASTER_EMPTY = wp.constant(wp.uint64(0xFFFFFFFFFFFFFFFF))
_RASTER_SOURCE_MASK = wp.constant(wp.uint64(0xFFFFFFFF))


@wp.kernel(enable_backward=False, module='unique')
def _initialize_raster_keys(keys: wp.array(dtype=wp.uint64)):  # type: ignore[valid-type]
    keys[wp.tid()] = _RASTER_EMPTY


@wp.kernel(enable_backward=False, module='unique')
def _rasterize_triangles(prefix: wp.array(dtype=int), rows: wp.array(dtype=int),  # type: ignore[valid-type]
              sources: wp.array(dtype=int), face_starts: wp.array(dtype=int),  # type: ignore[valid-type]
              faces: wp.array(dtype=wp.vec3i), points: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
              poses: wp.array2d(dtype=wp.vec3), rotations: wp.array2d(dtype=wp.vec2),  # type: ignore[valid-type]
              scales: wp.array2d(dtype=float), grids: wp.array(dtype=wp.vec3),  # type: ignore[valid-type]
              offsets: wp.array(dtype=int), dimensions: wp.array(dtype=wp.vec2i),  # type: ignore[valid-type]
              spacing: float, keys: wp.array(dtype=wp.uint64)):  # type: ignore[valid-type]
    triangle = wp.tid()
    lo = int(0)
    hi = prefix.shape[0] - 1
    while lo + 1 < hi:
        middle = (lo + hi) // 2
        if prefix[middle] <= triangle:
            lo = middle
        else:
            hi = middle
    row, source = rows[lo], sources[lo]
    # Four lanes share each triangle's original pixel rectangle.
    lanes = int(4)
    lane = (triangle - prefix[lo]) % lanes
    face = faces[face_starts[lo] + (triangle - prefix[lo]) // lanes]
    _raster_triangle(points[face[0]], points[face[1]], points[face[2]],
                    poses[row, source], rotations[row, source], scales[row, source],
                    grids[row], spacing, dimensions[row][0], dimensions[row][1],
                    offsets[row], source, lane, lanes, keys)


@wp.kernel(enable_backward=False, module='unique')
def _decode_raster(keys: wp.array(dtype=wp.uint64), offsets: wp.array(dtype=int),  # type: ignore[valid-type]
           widths: wp.array(dtype=int), grids: wp.array(dtype=wp.vec3), spacing: float,  # type: ignore[valid-type]
           shelf: wp.uint64, output: wp.array(dtype=wp.vec2)):  # type: ignore[valid-type]
    index = wp.tid()
    key = keys[index]
    owner = int(-2)
    x = float(0.0)
    if key != _RASTER_EMPTY:
        row = _image_row(offsets, index)
        pixel = index - offsets[row]
        origin = wp.vec3(grids[row][0],
                         grids[row][1] + (float(pixel % widths[row]) + 0.5) * spacing,
                         grids[row][2] + (float(pixel // widths[row]) + 0.5) * spacing)
        nearest = _raster_distance(key)
        owner = int(key & _RASTER_SOURCE_MASK)
        hit = wp.mesh_query_ray(shelf, origin, wp.vec3(-1.0, 0.0, 0.0), nearest)
        if hit.result:
            if hit.t < nearest:
                nearest = hit.t
                owner = -1
        x = origin[0] - nearest
    output[index] = wp.vec2(float(owner), x)



def _support_vertices(vertices):
    """Yaw extrema need only the XY hull plus the two original Z extrema."""
    from scipy.spatial import ConvexHull, QhullError
    vertices = np.asarray(vertices, dtype=np.float64)
    try:
        # Reset rotates only about Z. X/Y are linear functions of the original
        # XY coordinates, while positive uniform scaling preserves Z extrema.
        # This selects existing vertices; mesh triangles and the float64 pose
        # arithmetic in _bounds remain unchanged.
        indices = ConvexHull(vertices[:, :2]).vertices
        indices = np.unique(np.concatenate((indices, [vertices[:, 2].argmin(), vertices[:, 2].argmax()])))
        return vertices[indices]
    except QhullError:
        # Collinear/degenerate XY projections retain every original vertex.
        return vertices


@wp.kernel(enable_backward=False, module='unique')
def _bounds_cpu_kernel(kinds: wp.array2d(dtype=int), scales: wp.array2d(dtype=wp.float64),  # type: ignore[valid-type]
                       local: wp.array2d(dtype=wp.vec3), rotation: wp.array2d(dtype=wp.vec2),  # type: ignore[valid-type]
                       active: wp.array2d(dtype=bool), support: wp.array(dtype=wp.vec3d),  # type: ignore[valid-type]
                       starts: wp.array(dtype=int), all_support: wp.array(dtype=wp.vec3d),  # type: ignore[valid-type]
                       all_starts: wp.array(dtype=int), lower: wp.array2d(dtype=wp.vec3),  # type: ignore[valid-type]
                       upper: wp.array2d(dtype=wp.vec3), origin_x: wp.array(dtype=float)):  # type: ignore[valid-type]
    row = wp.tid()
    origin = origin_x[row]
    for source in range(kinds.shape[1]):
        lower[row, source] = wp.vec3(0.0)
        upper[row, source] = wp.vec3(0.0)
        if active[row, source]:
            kind = kinds[row, source]
            scale = scales[row, source]
            c = wp.float64(rotation[row, source][0])
            s = wp.float64(rotation[row, source][1])
            px = wp.float64(local[row, source][0])
            py = wp.float64(local[row, source][1])
            pz = wp.float64(local[row, source][2])
            lx = wp.float64(wp.inf); ly = wp.float64(wp.inf); lz = wp.float64(wp.inf)
            hx = wp.float64(-wp.inf); hy = wp.float64(-wp.inf); hz = wp.float64(-wp.inf)
            for index in range(starts[kind], starts[kind + 1]):
                vertex = support[index]
                x = scale * (vertex[0] * c - vertex[1] * s) + px
                y = scale * (vertex[0] * s + vertex[1] * c) + py
                z = scale * vertex[2] + pz
                lx = wp.min(lx, x); ly = wp.min(ly, y); lz = wp.min(lz, z)
                hx = wp.max(hx, x); hy = wp.max(hy, y); hz = wp.max(hz, z)
            lower[row, source] = wp.vec3(float(lx), float(ly), float(lz))
            upper[row, source] = wp.vec3(float(hx), float(hy), float(hz))
            all_x = wp.float64(-wp.inf)
            for index in range(all_starts[kind], all_starts[kind + 1]):
                vertex = all_support[index]
                x = scale * (vertex[0] * c - vertex[1] * s) + px
                all_x = wp.max(all_x, x)
            origin = wp.max(origin, float(all_x))
    origin_x[row] = origin


# Match NumPy's individual float64 operations, followed by its float32 cast.
# This isolated module leaves the original observer's compilation untouched.
_bounds_cpu_kernel.module.options['fuse_fp'] = False
_bounds_cpu_kernel.module.mark_modified()


class _CPUSurfaceBounds:
    """Immutable CPU support cache; fuse bounds without per-kind temporaries."""

    def __init__(self, support, all_support, shelf_max_x):
        self.shelf_max_x = shelf_max_x
        wp.get_device('cpu')

        def flatten(values):
            starts = np.zeros(max(values) + 2, dtype=np.int32)
            parts = []
            for kind in range(len(starts) - 1):
                vertices = values.get(kind, np.empty((0, 3)))
                parts.append(vertices)
                starts[kind + 1] = starts[kind] + len(vertices)
            return (wp.array(np.concatenate(parts), dtype=wp.vec3d, device='cpu'),
                    wp.array(starts, dtype=int, device='cpu'))

        self.support, self.starts = flatten(support)
        self.all_support, self.all_starts = flatten(all_support)

    def bounds(self, kinds, scales, local, rotation, active):
        lower = np.empty((*kinds.shape, 3), dtype=np.float32)
        upper = np.empty_like(lower)
        origin_x = np.full(len(kinds), self.shelf_max_x, dtype=np.float32)
        array = lambda value, dtype: wp.array(value, dtype=dtype, device='cpu', copy=False)
        arguments = [array(kinds, int), array(scales, wp.float64), array(local, wp.vec3),
                     array(rotation, wp.vec2), array(active, bool), self.support, self.starts,
                     self.all_support, self.all_starts, array(lower, wp.vec3),
                     array(upper, wp.vec3), array(origin_x, float)]
        if len(kinds):
            # CPU launch completes before the caller creates the ray image.
            wp.launch(_bounds_cpu_kernel, dim=len(kinds), inputs=arguments, device='cpu')
        return lower, upper, origin_x


class BatchedSurfaceGeometry:
    """Exact visible-mesh semantics with shared BVHs and chunked readbacks.

    Revision objects share eight immutable shapes; only their uniform root
    scale differs between environments. Read each shape once from the real USD
    scene, remove that reference root's scale, and retain its child transforms.
    CPU connected components deliberately use the reference SciPy algorithm.
    """

    def __init__(self, env, chunk_size=64, components_backend='cpu', ray_backend='bvh', bounds_backend='numpy'):
        from isaaclab.sim.utils import get_current_stage
        if chunk_size < 1:
            raise ValueError('Surface observation chunk_size must be positive')
        if components_backend not in ('cpu', 'gpu'):
            raise ValueError('Surface components_backend must be cpu or gpu')
        if ray_backend not in ('bvh', 'raster'):
            raise ValueError('Surface ray_backend must be bvh or raster')
        if bounds_backend not in ('numpy', 'cpu_kernel'):
            raise ValueError('Surface bounds_backend must be numpy or cpu_kernel')
        self.ray_backend = ray_backend
        self.device = env.device
        self.chunk_size = int(chunk_size)
        self.components_backend = components_backend
        self._workspace_arrays: dict = {}
        self.spacing = .001
        self.kinds = np.asarray(env._revision_object_kinds, dtype=np.int32).copy()
        self.scales = np.asarray(env._final_object_scales, dtype=np.float64).copy()
        if self.kinds.shape != self.scales.shape or not np.isfinite(self.scales).all() or (self.scales <= 0).any():
            raise ValueError('Surface observation requires fixed positive per-object scale metadata')
        names = tuple(env.cfg.scene.object_collection.rigid_objects)
        if self.kinds.shape != (env.num_envs, len(names)):
            raise ValueError('Surface geometry metadata does not match the object roster')
        geometry = SurfaceGeometry(get_current_stage())
        self.prototypes = {}
        self.support = {}
        self.all_support = {}
        self.meshes = []
        mesh_ids = np.zeros(int(self.kinds.max()) + 1, dtype=np.uint64)
        for kind in np.unique(self.kinds):
            row, source = np.argwhere(self.kinds == kind)[0]
            path = env.scene.env_prim_paths[row] + '/' + names[source]
            vertices, faces = geometry.local(path)
            vertices = vertices / self.scales[row, source]
            self.prototypes[int(kind)] = vertices, faces
            self.support[int(kind)] = _support_vertices(vertices[np.unique(faces)])
            self.all_support[int(kind)] = _support_vertices(vertices)
            mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                           indices=wp.array(faces.ravel(), dtype=int, device=self.device))
            self.meshes.append(mesh)
            mesh_ids[kind] = mesh.id
        self.mesh_ids = wp.array(mesh_ids, dtype=wp.uint64, device=self.device)
        shelf_vertices, shelf_faces = geometry.local(env.scene.env_prim_paths[0] + '/Shelf')
        self.shelf_mesh = wp.Mesh(points=wp.array(shelf_vertices, dtype=wp.vec3, device=self.device),
                                 indices=wp.array(shelf_faces.ravel(), dtype=int, device=self.device))
        self.shelf_max_x = np.float32(np.asarray(shelf_vertices, dtype=np.float32)[:, 0].max())
        self._cpu_bounds = (_CPUSurfaceBounds(self.support, self.all_support, self.shelf_max_x)
                            if bounds_backend == 'cpu_kernel' else None)
        if ray_backend == 'raster':
            points, faces = [], []
            self.face_start = np.zeros(max(self.prototypes) + 1, dtype=np.int32)
            self.face_count = np.zeros_like(self.face_start)
            vertex_offset = 0
            face_offset = 0
            for kind, (vertices, indices) in self.prototypes.items():
                self.face_start[kind] = face_offset
                self.face_count[kind] = len(indices)
                faces.append(indices + vertex_offset)
                points.append(vertices)
                vertex_offset += len(vertices)
                face_offset += len(indices)
            self.raster_points = wp.array(np.concatenate(points), dtype=wp.vec3, device=self.device)
            self.raster_faces = wp.array(np.concatenate(faces), dtype=wp.vec3i, device=self.device)

    def _workspace(self, name, count, dtype, *, zero=False):
        """Reuse large chunk buffers; CUDA pool synchronization must not free them."""
        buffer = self._workspace_arrays.get(name)
        if buffer is None or buffer.shape[0] < count:
            buffer = wp.empty(count, dtype=dtype, device=self.device)
            self._workspace_arrays[name] = buffer
        result = buffer if buffer.shape[0] == count else buffer[:count]
        if zero:
            result.zero_()
        return result

    def _bounds(self, kinds, scales, local, rotation, active):
        lower = np.zeros((*kinds.shape, 3), dtype=np.float32)
        upper = np.zeros_like(lower)
        origin_x = np.full(len(kinds), self.shelf_max_x, dtype=np.float32)
        for kind, vertices in self.support.items():
            rows, sources = np.nonzero((kinds == kind) & active)
            if not len(rows):
                continue
            scale = scales[rows, sources, None]
            c = rotation[rows, sources, 0, None].astype(np.float64)
            s = rotation[rows, sources, 1, None].astype(np.float64)
            # The reference composes yaw/translation in NumPy float64 before
            # casting scene vertices to float32. Keep that boundary here too.
            x = scale * (vertices[None, :, 0] * c - vertices[None, :, 1] * s) + local[rows, sources, 0, None]
            y = scale * (vertices[None, :, 0] * s + vertices[None, :, 1] * c) + local[rows, sources, 1, None]
            z = scale * vertices[None, :, 2] + local[rows, sources, 2, None]
            lower[rows, sources] = np.column_stack((x.min(1), y.min(1), z.min(1)))
            upper[rows, sources] = np.column_stack((x.max(1), y.max(1), z.max(1)))
            # Unreferenced vertices affect the old ray's starting X, although
            # only referenced visible vertices define its YZ image bounds.
            all_vertices = self.all_support[kind]
            x_all = scale * (all_vertices[None, :, 0] * c - all_vertices[None, :, 1] * s) + local[rows, sources, 0, None]
            np.maximum.at(origin_x, rows, x_all.max(1).astype(np.float32))
        return lower, upper, origin_x

    def observe_batch(self, ids, local, yaw, active):
        """Return reference-shaped observations in the requested row order."""
        ids = np.asarray(ids, dtype=np.int64)
        local = np.asarray(local, dtype=np.float32)
        yaw = np.asarray(yaw, dtype=np.float32)
        active = np.asarray(active, dtype=bool)
        expected = (len(ids), self.kinds.shape[1])
        if local.shape != (*expected, 3) or yaw.shape != expected or active.shape != expected:
            raise ValueError('Surface observation batch shapes do not match selected environments')
        if len(ids) and (not np.isfinite(local).all() or not np.isfinite(yaw).all() or not active.any(1).all()):
            raise ValueError('Surface observation requires finite poses and an active object per row')
        result = []
        for start in range(0, len(ids), self.chunk_size):
            stop = min(start + self.chunk_size, len(ids))
            result.extend(self._observe_chunk(ids[start:stop], local[start:stop], yaw[start:stop], active[start:stop]))
        return result

    def _observe_chunk(self, ids, local, yaw, active):
        start = perf_counter()
        kinds, scales = self.kinds[ids], self.scales[ids]
        rotation = np.stack((np.cos(yaw), np.sin(yaw)), -1)
        bounds = self._bounds if self._cpu_bounds is None else self._cpu_bounds.bounds
        lower, upper, origin_x = bounds(kinds, scales, local, rotation, active)
        low = np.floor(np.where(active[..., None], lower[..., 1:], np.inf).min(1) / self.spacing) * self.spacing - self.spacing
        high = np.ceil(np.where(active[..., None], upper[..., 1:], -np.inf).max(1) / self.spacing) * self.spacing + self.spacing
        dimensions = np.ceil((high - low) / self.spacing).astype(np.int32)
        counts = dimensions.prod(1, dtype=np.int64)
        offsets = np.concatenate(([0], counts.cumsum())).astype(np.int32)
        grid = np.column_stack((origin_x + .05, low)).astype(np.float32)
        if self.ray_backend == 'raster':
            rows, source_ids = np.nonzero(active)
            instance_kinds = kinds[rows, source_ids]
            # Prefix entries count raster work lanes, four per original face.
            prefix = np.concatenate(([0], (self.face_count[instance_kinds] * 4).cumsum())).astype(np.int32)
        else:
            source_counts = active.sum(1).astype(np.int32)
            sources = np.sort(np.where(active, np.arange(active.shape[1]), active.shape[1]), axis=1)
            sources = sources[:, :int(source_counts.max())].astype(np.int32)
        prepared = perf_counter()
        cuda = wp.get_device(self.device).is_cuda
        events = None
        if cuda and self.components_backend == 'gpu':
            if not hasattr(self, '_timing_events'):
                self._timing_events = tuple(wp.Event(device=self.device, enable_timing=True) for _ in range(4))
            events = self._timing_events
            wp.get_stream(self.device).record_event(events[0])
        array = lambda values, dtype: wp.array(values, dtype=dtype, device=self.device)
        image_arrays = [array(offsets, int), array(dimensions[:, 0], int), array(grid, wp.vec3)]
        if self.ray_backend == 'raster':
            arguments = [array(prefix, int), array(rows, int), array(source_ids, int),
                         array(self.face_start[instance_kinds], int), self.raster_faces, self.raster_points,
                         array(local, wp.vec3), array(rotation, wp.vec2), array(scales, float),
                         image_arrays[2], image_arrays[0], array(dimensions, wp.vec2i), self.spacing]
            keys = self._workspace('raster_keys', int(offsets[-1]), wp.uint64)
        else:
            arguments = [self.mesh_ids, self.shelf_mesh.id, *image_arrays, self.spacing,
                         array(kinds, int), array(scales, float), array(local, wp.vec3), array(rotation, wp.vec2),
                         array(sources, int), array(source_counts, int), array(lower, wp.vec3), array(upper, wp.vec3)]
        packed = (self._workspace('pixels', int(offsets[-1]), wp.vec2) if self.components_backend == 'gpu'
                  else wp.empty(int(offsets[-1]), dtype=wp.vec2, device=self.device))
        if events is not None:
            wp.get_stream(self.device).record_event(events[1])
        elif cuda:
            wp.synchronize_stream(self.device)
        uploaded = perf_counter()
        if self.ray_backend == 'raster':
            # Test every original visual triangle with Warp's same Woop ray
            # intersection; raster bounds only avoid rays outside its projection.
            wp.launch(_initialize_raster_keys, dim=len(keys), inputs=[keys], device=self.device)
            wp.launch(_rasterize_triangles, dim=int(prefix[-1]), inputs=[*arguments, keys], device=self.device)
            wp.launch(_decode_raster, dim=len(keys),
                      inputs=[keys, *image_arrays, self.spacing, self.shelf_mesh.id, packed], device=self.device)
        else:
            wp.launch(_cast_batch, dim=int(offsets[-1]), inputs=[*arguments, packed], device=self.device)
        if events is not None:
            wp.get_stream(self.device).record_event(events[2])
        elif cuda:
            wp.synchronize_stream(self.device)
        cast = perf_counter()
        if self.components_backend == 'gpu':
            return self._components_gpu(packed, offsets, dimensions, grid, lower, upper, active,
                                        image_arrays, start, prepared, uploaded, cast, events)
        pixels = packed.numpy()
        readback = perf_counter()
        # Empty pixels outside a source's projected geometry cannot participate
        # in its connected components. Crop those empty borders before SciPy,
        # then restore global pixel indices before centroid/tie selection.
        crop_low = np.floor((lower[..., 1:] - grid[:, None, 1:]) / self.spacing).astype(np.int32) - 1
        crop_high = np.ceil((upper[..., 1:] - grid[:, None, 1:]) / self.spacing).astype(np.int32) + 2
        crop_low = np.maximum(crop_low, 0)
        crop_high = np.minimum(crop_high, dimensions[:, None])
        shared = dict(build_ms=1000 * (prepared - start) / len(ids),
                      upload_ms=1000 * (uploaded - prepared) / len(ids),
                      ray_ms=1000 * (cast - uploaded) / len(ids),
                      readback_ms=1000 * (readback - cast) / len(ids))
        result = []
        for row, (width, height) in enumerate(dimensions):
            components_start = perf_counter()
            image = pixels[offsets[row]:offsets[row + 1]].reshape(height, width, 2)
            mask = image[..., 0]
            target = None
            target_area = 0
            obstacles = []
            for source in np.flatnonzero(active[row]):
                full_width = upper[row, source, 1] - lower[row, source, 1]
                col_start, row_start = crop_low[row, source]
                col_stop, row_stop = crop_high[row, source]
                components, _ = ndimage.label(mask[row_start:row_stop, col_start:col_stop] == source)
                for label, slices in enumerate(ndimage.find_objects(components), 1):
                    if slices is None:
                        continue
                    rows, cols = slices
                    if (cols.stop - cols.start) * self.spacing + 1e-9 < max(.02, .2 * full_width):
                        continue
                    rr, cc = np.nonzero(components[rows, cols] == label)
                    rr += rows.start + row_start
                    cc += cols.start + col_start
                    nearest = np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)
                    r, c = rr[nearest], cc[nearest]
                    point = np.array([image[r, c, 1],
                                      grid[row, 1] + (np.float32(c) + np.float32(.5)) * np.float32(self.spacing),
                                      grid[row, 2] + (np.float32(r) + np.float32(.5)) * np.float32(self.spacing)], dtype=np.float32)
                    if source == 0:
                        if len(rr) > target_area:
                            target, target_area = point, len(rr)
                    else:
                        obstacles.append(point)
            component_ms = 1000 * (perf_counter() - components_start)
            timing = dict(shared, components_ms=component_ms,
                          total_ms=sum(shared.values()) + component_ms, rays=int(counts[row]))
            result.append((target, np.asarray(obstacles, dtype=np.float32).reshape(-1, 3), timing))
        return result

    def _components_gpu(self, pixels, offsets, dimensions, grid, lower, upper, active,
                        image_arrays, start, prepared, uploaded, cast, events=None):
        """Optional device CCL; the normal CPU-crop backend remains the default."""
        count = int(offsets[-1])
        batch = len(dimensions)
        device = self.device
        device_offsets, widths, device_grid = image_arrays
        parent = self._workspace('parent', count, int)
        area = self._workspace('area', count, int)
        sum_row = self._workspace('sum_row', count, wp.uint64)
        sum_col = self._workspace('sum_col', count, wp.uint64)
        minimum_col = self._workspace('minimum_col', count, int)
        maximum_col = self._workspace('maximum_col', count, int)
        distance = self._workspace('distance', count, wp.int64)
        legacy_distance = self._workspace('legacy_distance', count, wp.float64)
        winner = self._workspace('winner', count, int)
        target_key = wp.zeros(batch, dtype=wp.int64, device=device)
        obstacle_count = wp.zeros(batch, dtype=int, device=device)
        obstacle_roots = wp.empty((batch, 16), dtype=int, device=device)
        compact = wp.empty((batch, 18), dtype=wp.vec4, device=device)
        threshold_values = np.maximum(.02, .2 * (upper[..., 1] - lower[..., 1]).astype(np.float64))
        thresholds = wp.array(threshold_values, dtype=wp.float64, device=device)
        wp.launch(_initialize_components, dim=count,
                  inputs=[pixels, parent, area, sum_row, sum_col, minimum_col, maximum_col, distance, legacy_distance, winner], device=device)
        wp.launch(_union_components, dim=count, inputs=[pixels, device_offsets, widths, parent], device=device)
        wp.launch(_reduce_components, dim=count,
                  inputs=[device_offsets, widths, parent, area, sum_row, sum_col, minimum_col, maximum_col], device=device)
        wp.launch(_filter_components, dim=count,
                  inputs=[pixels, device_offsets, thresholds, area, minimum_col, maximum_col], device=device)
        wp.launch(_minimum_component_distance, dim=count,
                  inputs=[device_offsets, widths, parent, area, sum_row, sum_col, distance], device=device)
        wp.launch(_minimum_legacy_distance, dim=count,
                  inputs=[device_offsets, widths, parent, area, sum_row, sum_col, distance, legacy_distance], device=device)
        wp.launch(_pick_component_pixel, dim=count,
                  inputs=[device_offsets, widths, parent, area, sum_row, sum_col, distance, legacy_distance, winner], device=device)
        wp.launch(_collect_components, dim=count,
                  inputs=[pixels, device_offsets, area, target_key, obstacle_count, obstacle_roots], device=device)
        wp.launch(_pack_components, dim=batch,
                  inputs=[pixels, device_offsets, widths, device_grid, winner, target_key, obstacle_count,
                          obstacle_roots, compact], device=device)
        if events is not None:
            wp.get_stream(device).record_event(events[3])
        if wp.get_device(device).is_cuda:
            # Only this observer's stream owns these arrays. One final wait is
            # enough for compact readback; timing events avoid earlier waits.
            wp.synchronize_stream(device)
        reduced = perf_counter()
        observations = compact.numpy()
        readback = perf_counter()
        shared = dict(build_ms=1000 * (prepared - start) / batch,
                      upload_ms=1000 * (uploaded - prepared) / batch,
                      ray_ms=1000 * (cast - uploaded) / batch,
                      readback_ms=1000 * (readback - reduced) / batch,
                      components_ms=1000 * (reduced - cast) / batch)
        if events is not None:
            shared.update(upload_ms=wp.get_event_elapsed_time(events[0], events[1], synchronize=False) / batch,
                          ray_ms=wp.get_event_elapsed_time(events[1], events[2], synchronize=False) / batch,
                          components_ms=wp.get_event_elapsed_time(events[2], events[3], synchronize=False) / batch)
        result = []
        for row, observation in enumerate(observations):
            target = observation[0, 1:].copy() if observation[0, 0] else None
            number = int(observation[1, 0])
            obstacles = observation[2:2 + min(number, 16), 1:].copy()
            extra_ms = 0.
            if number > 16:
                # The caller removes these obstacles and retries target-only.
                # Still return the complete reference-shaped result instead of
                # silently truncating or manufacturing overflow coordinates.
                extra_start = perf_counter()
                raw = pixels[int(offsets[row]):int(offsets[row + 1])].numpy()
                target, obstacles = self._overflow_components_numpy(raw, dimensions[row], grid[row],
                                                                    lower[row], upper[row], active[row])
                extra_ms = 1000 * (perf_counter() - extra_start)
            timing = dict(shared, total_ms=1000 * (readback - start) / batch + extra_ms,
                          rays=int(offsets[row + 1] - offsets[row]))
            timing['components_ms'] += extra_ms
            result.append((target, obstacles, timing))
        return result

    def _overflow_components_numpy(self, pixels, dimensions, grid, lower, upper, active):
        width, height = dimensions
        image = pixels.reshape(height, width, 2)
        target = None
        target_area = 0
        obstacles = []
        for source in np.flatnonzero(active):
            full_width = upper[source, 1] - lower[source, 1]
            components, _ = ndimage.label(image[..., 0] == source)
            for label, slices in enumerate(ndimage.find_objects(components), 1):
                if slices is None:
                    continue
                rows, cols = slices
                if (cols.stop - cols.start) * self.spacing + 1e-9 < max(.02, .2 * full_width):
                    continue
                rr, cc = np.nonzero(components[rows, cols] == label)
                rr += rows.start
                cc += cols.start
                nearest = np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)
                r, c = rr[nearest], cc[nearest]
                point = np.array([image[r, c, 1],
                                  grid[1] + (np.float32(c) + np.float32(.5)) * np.float32(self.spacing),
                                  grid[2] + (np.float32(r) + np.float32(.5)) * np.float32(self.spacing)], dtype=np.float32)
                if source == 0:
                    if len(rr) > target_area:
                        target, target_area = point, len(rr)
                else:
                    obstacles.append(point)
        return target, np.asarray(obstacles, dtype=np.float32).reshape(-1, 3)
