"""Conservative collision boxes at candidate joint poses, without physics steps."""
import itertools
import numpy as np
import torch
import warp as wp
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, matrix_from_quat


_FACE_MARGIN=wp.constant(wp.float64(1e-8))


@wp.func
def _face_proves_separation(plane: wp.vec4d, relative: wp.mat33d, delta: wp.vec3d,
                           vertices: wp.array(dtype=wp.vec3d), start: int, count: int):
    normal=wp.vec3d(plane[0],plane[1],plane[2])
    projected=relative*normal
    constant=wp.dot(normal,delta)+plane[3]
    separated=bool(True)
    for vertex in range(start,start+count):
        if wp.dot(projected,vertices[vertex])+constant<=_FACE_MARGIN:
            separated=False
            break
    return separated


@wp.func
def _hull_faces_separated(planes: wp.array(dtype=wp.vec4d), start: int, count: int,
                         relative: wp.mat33d, delta: wp.vec3d,
                         vertices: wp.array(dtype=wp.vec3d), vertex_start: int, vertex_count: int,
                         centroid: wp.vec3d):
    first=int(-1);second=int(-1)
    best=wp.float64(-1.0e20);next_best=wp.float64(-1.0e20)
    for index in range(start,start+count):
        plane=planes[index];normal=wp.vec3d(plane[0],plane[1],plane[2])
        score=wp.dot(relative*normal,centroid)+wp.dot(normal,delta)+plane[3]
        if score>best:
            second=first;next_best=best;first=index;best=score
        elif score>next_best:
            second=index;next_best=score
    separated=bool(False)
    if first>=0:
        separated=_face_proves_separation(planes[first],relative,delta,vertices,vertex_start,vertex_count)
    if not separated and second>=0:
        separated=_face_proves_separation(planes[second],relative,delta,vertices,vertex_start,vertex_count)
    if not separated:
        for index in range(start,start+count):
            if index!=first and index!=second:
                if _face_proves_separation(planes[index],relative,delta,vertices,vertex_start,vertex_count):
                    separated=True
                    break
    return separated


@wp.kernel(enable_backward=False,module='unique')
def _convex_face_kernel(candidates: wp.array2d(dtype=wp.int64),
                       centers: wp.array2d(dtype=wp.vec3), rotations: wp.array2d(dtype=wp.mat33),
                       planes: wp.array(dtype=wp.vec4d), face_starts: wp.array(dtype=int), face_counts: wp.array(dtype=int),
                       vertices: wp.array(dtype=wp.vec3d), vertex_starts: wp.array(dtype=int), vertex_counts: wp.array(dtype=int),
                       centroids: wp.array(dtype=wp.vec3d), separated: wp.array(dtype=int)):
    index=wp.tid();row=int(candidates[index,0]);a=int(candidates[index,1]);b=int(candidates[index,2])
    ar=rotations[row,a];br=rotations[row,b]
    ra=wp.mat33d(wp.float64(ar[0,0]),wp.float64(ar[0,1]),wp.float64(ar[0,2]),
                wp.float64(ar[1,0]),wp.float64(ar[1,1]),wp.float64(ar[1,2]),
                wp.float64(ar[2,0]),wp.float64(ar[2,1]),wp.float64(ar[2,2]))
    rb=wp.mat33d(wp.float64(br[0,0]),wp.float64(br[0,1]),wp.float64(br[0,2]),
                wp.float64(br[1,0]),wp.float64(br[1,1]),wp.float64(br[1,2]),
                wp.float64(br[2,0]),wp.float64(br[2,1]),wp.float64(br[2,2]))
    ac=centers[row,a];bc=centers[row,b]
    difference=wp.vec3d(wp.float64(bc[0])-wp.float64(ac[0]),wp.float64(bc[1])-wp.float64(ac[1]),
                       wp.float64(bc[2])-wp.float64(ac[2]))
    # R^-1 moves the other hull's transformed plane back into this hull's
    # vertex coordinates. This matches the LP even for nonorthogonal R.
    safe=_hull_faces_separated(planes,face_starts[a],face_counts[a],wp.inverse(rb)*ra,wp.transpose(ra)*difference,
                              vertices,vertex_starts[b],vertex_counts[b],centroids[b])
    if not safe:
        safe=_hull_faces_separated(planes,face_starts[b],face_counts[b],wp.inverse(ra)*rb,wp.transpose(rb)*(-difference),
                                  vertices,vertex_starts[a],vertex_counts[a],centroids[a])
    separated[index]=int(safe)


_convex_face_kernel.module.options['fuse_fp']=False


@wp.func
def _fk_mul(a: wp.vec4, b: wp.vec4):
    w1=a[0];x1=a[1];y1=a[2];z1=a[3]
    w2=b[0];x2=b[1];y2=b[2];z2=b[3]
    ww=(z1+x1)*(x2+y2)
    yy=(w1-y1)*(w2+z2)
    zz=(w1+y1)*(w2-z2)
    xx=ww+yy+zz
    qq=0.5*(xx+(z1-x1)*(x2-y2))
    return wp.vec4(qq-ww+(z1-y1)*(y2-z2),qq-xx+(x1+w1)*(x2+w2),
                   qq-yy+(w1-x1)*(y2+z2),qq-zz+(z1+y1)*(w2-x2))


@wp.func
def _fk_apply(q: wp.vec4, v: wp.vec3):
    xyz=wp.vec3(q[1],q[2],q[3])
    t=wp.cross(xyz,v)*2.0
    return v+q[0]*t+wp.cross(xyz,t)


@wp.func
def _fk_matrix(q: wp.vec4):
    r=q[0];i=q[1];j=q[2];k=q[3]
    two_s=2.0/(((r*r+i*i)+j*j)+k*k)
    return wp.mat33(1.0-two_s*(j*j+k*k),two_s*(i*j-k*r),two_s*(i*k+j*r),
                    two_s*(i*j+k*r),1.0-two_s*(i*i+k*k),two_s*(j*k-i*r),
                    two_s*(i*k-j*r),two_s*(j*k+i*r),1.0-two_s*(i*i+j*j))


@wp.func
def _fk_child(row: int, step: int, pos: wp.vec3, quat: wp.vec4,
              joints: wp.array2d(dtype=float),
              p0: wp.array(dtype=wp.vec3),q0: wp.array(dtype=wp.vec4),
              p1: wp.array(dtype=wp.vec3),q1: wp.array(dtype=wp.vec4),
              axes: wp.array(dtype=int),columns: wp.array(dtype=int)):
    p=pos+_fk_apply(quat,p0[step])
    q=_fk_mul(quat,q0[step])
    col=columns[step]
    if col>=0:
        angle=joints[row,col]*0.5
        rot=wp.vec4(wp.cos(angle),0.0,0.0,0.0)
        rot[axes[step]+1]=wp.sin(angle)
        q=_fk_mul(q,rot)
    inv=q1[step]
    q=_fk_mul(q,wp.vec4(inv[0],-inv[1],-inv[2],-inv[3]))
    p=p-_fk_apply(q,p1[step])
    return p,q


@wp.kernel(enable_backward=False,module='unique')
def _fk_owner_kernel(joints: wp.array2d(dtype=float),
    base_pos: wp.array(dtype=wp.vec3),base_quat: wp.array(dtype=wp.vec4),
    chain_starts: wp.array(dtype=int),chain_steps: wp.array(dtype=int),
    p0: wp.array(dtype=wp.vec3),q0: wp.array(dtype=wp.vec4),
    p1: wp.array(dtype=wp.vec3),q1: wp.array(dtype=wp.vec4),axes: wp.array(dtype=int),columns: wp.array(dtype=int),
    local_centers: wp.array(dtype=wp.vec3),centers: wp.array2d(dtype=wp.vec3),rotations: wp.array2d(dtype=wp.mat33)):
    row,owner=wp.tid()
    p=base_pos[row];q=base_quat[row]
    for offset in range(chain_starts[owner],chain_starts[owner+1]):
        step=chain_steps[offset]
        p,q=_fk_child(row,step,p,q,joints,p0,q0,p1,q1,axes,columns)
    centers[row,owner]=p+_fk_apply(q,local_centers[owner])
    rotations[row,owner]=_fk_matrix(q)


_fk_owner_kernel.module.options['fuse_fp']=False


@wp.func
def _sat_column(matrix: wp.mat33, col: int):
    return wp.vec3(matrix[0, col], matrix[1, col], matrix[2, col])


@wp.func
def _sat_sum3(value: wp.vec3):
    return (value[0] + value[1]) + value[2]


@wp.func
def _sat_dot3(a: wp.vec3, b: wp.vec3):
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]


@wp.func
def _sat_pair(center_a: wp.vec3, half_a: wp.vec3, ar: wp.mat33,
              center_b: wp.vec3, half_b: wp.vec3, br: wp.mat33, clearance: float):
    ah = half_a + wp.vec3(clearance, clearance, clearance)
    bh = half_b; delta = center_b - center_a
    intersects = bool(True); uncertain = bool(False)
    for index in range(15):
        axis = wp.vec3(0.0)
        if index < 3:
            axis = _sat_column(ar, index)
        elif index < 6:
            axis = _sat_column(br, index - 3)
        else:
            offset = index - 6
            axis = wp.cross(_sat_column(ar, offset // 3), _sat_column(br, offset % 3))
        length = wp.sqrt(_sat_dot3(axis, axis))
        unit = axis / wp.max(length, 1.0e-9)
        da = wp.vec3(wp.abs(_sat_dot3(unit, _sat_column(ar, 0))),
                     wp.abs(_sat_dot3(unit, _sat_column(ar, 1))),
                     wp.abs(_sat_dot3(unit, _sat_column(ar, 2))))
        db = wp.vec3(wp.abs(_sat_dot3(unit, _sat_column(br, 0))),
                     wp.abs(_sat_dot3(unit, _sat_column(br, 1))),
                     wp.abs(_sat_dot3(unit, _sat_column(br, 2))))
        reach = _sat_sum3(wp.cw_mul(da, ah)) + _sat_sum3(wp.cw_mul(db, bh))
        distance = wp.abs(_sat_dot3(delta, unit))
        margin = distance - (reach - 1.0e-6)
        # Preserve the original SAT comparisons. A wider numerical guard
        # sends near-contact and nearly parallel axes to the Torch reference.
        # Dividing by length accounts for cross-axis normalization error.
        bound = 1.0e-5 * (distance + reach + 1.0) / wp.max(length, 1.0e-7)
        if length > 1.0e-7 and wp.abs(margin) < bound:
            uncertain = True
        axis_unreliable = length <= 2.0e-6
        if axis_unreliable:
            uncertain = True
        if margin >= 0.0 and length > 1.0e-7:
            intersects = False
            # A well separated axis resolves any earlier ambiguous axes.
            if margin >= bound and not axis_unreliable:
                uncertain = False
            break
    return int(intersects), int(uncertain)


@wp.kernel(enable_backward=False, module='unique')
def _broad_sat(ca: wp.array2d(dtype=wp.vec3), ha: wp.array2d(dtype=wp.vec3),
               ra: wp.array2d(dtype=wp.mat33), ct: wp.array2d(dtype=wp.vec3),
               ht: wp.array2d(dtype=wp.vec3), rt: wp.array2d(dtype=wp.mat33),
               cs: wp.array2d(dtype=wp.vec3), hs: wp.array2d(dtype=wp.vec3),
               rs: wp.array2d(dtype=wp.mat33), pairs: wp.array2d(dtype=int),
               result: wp.array2d(dtype=int), ambiguous: wp.array2d(dtype=int)):
    row, index = wp.tid()
    kind = pairs[index,0]; a = pairs[index,1]; b = pairs[index,2]
    cb=wp.vec3(0.0); hb=wp.vec3(0.0); rb=wp.mat33(0.0)
    if kind == 0:
        cb=ct[row,b]; hb=ht[row,b]; rb=rt[row,b]
    elif kind == 1:
        cb=cs[row,b]; hb=hs[row,b]; rb=rs[row,b]
    else:
        cb=ca[row,b]; hb=ha[row,b]; rb=ra[row,b]
    hit, unsure = _sat_pair(ca[row,a], ha[row,a], ra[row,a], cb, hb, rb, 0.0)
    result[row,index] = hit
    ambiguous[row,index] = unsure


_broad_sat.module.options['fuse_fp'] = False



def overlap(a, b, clearance=0.):
    """OBB separating-axis test, returning [batch, A, B]. Touching is permitted."""
    ca, ha, ra = a
    cb, hb, rb = b
    axes_a = ra.transpose(-1, -2)[:, :, None]
    axes_b = rb.transpose(-1, -2)[:, None]
    shape = (len(ca), ca.shape[1], cb.shape[1], 3, 3)
    cross = torch.cross(axes_a[..., :, None, :], axes_b[..., None, :, :], dim=-1).flatten(-3, -2)
    axes = torch.cat((axes_a.expand(shape), axes_b.expand(shape), cross), -2)
    length = axes.norm(dim=-1)
    unit = axes / length.clamp_min(1e-9)[..., None]
    da = torch.einsum('nabki,naij->nabkj', unit, ra).abs()
    db = torch.einsum('nabki,nbij->nabkj', unit, rb).abs()
    reach = (da * (ha[:, :, None, None] + clearance)).sum(-1) + (db * hb[:, None, :, None]).sum(-1)
    distance = ((cb[:, None] - ca[:, :, None])[..., None, :] * unit).sum(-1).abs()
    separated = (distance >= reach - 1e-6) & (length > 1e-7)
    return ~separated.any(-1)


def corners(boxes):
    center, half, rotation = boxes
    signs = center.new_tensor(list(itertools.product((-1., 1.), repeat=3)))
    return center[..., None, :] + torch.einsum('...ij,...kj->...ki', rotation, half[..., None, :] * signs)


def collision_bounds(stage, root_path, relative_path):
    """USD collision shapes, not visual geometry, enclosed in the rigid-body frame."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy', 'guide'], useExtentsHint=False)
    root, relative = stage.GetPrimAtPath(root_path), stage.GetPrimAtPath(relative_path)
    result = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        bounds = cache.ComputeRelativeBound(prim, relative).ComputeAlignedRange()
        if bounds.IsEmpty():
            continue
        result.append((tuple(bounds.GetMin()), tuple(bounds.GetMax())))
    return result


def convex_overlap(a,b):
    """Positive interior intersection of two convex half-space systems."""
    from scipy.optimize import linprog
    planes=np.concatenate((a,b))
    result=linprog([0.,0.,0.,-1.],A_ub=np.column_stack((planes[:,:3],np.ones(len(planes)))),
                   b_ub=-planes[:,3],bounds=[(None,None)]*4,method='highs')
    if result.status==2:return False
    if not result.success:raise RuntimeError(f'Collision narrow phase failed: {result.message}')
    return result.x[3]>1e-6


def convex_face_separated(planes, vertices):
    """Prove separation using existing hull faces; unresolved pairs need the LP.

    Each row contains one hull's outward planes and the other hull's vertices.
    Checking a subset of faces can only miss a separating axis. It cannot make
    an intersecting pair pass. Try the two faces nearest the other hull first,
    then the remaining faces only for unresolved rows. All arithmetic is double
    precision and a positive margin protects nearly touching configurations.
    """
    planes=np.asarray(planes,dtype=np.float64)
    vertices=np.asarray(vertices,dtype=np.float64)
    center=vertices.mean(axis=1)
    score=np.einsum('npi,ni->np',planes[...,:3],center)+planes[...,3]
    nearest=np.argsort(score,axis=1)[:,-2:]
    first=np.take_along_axis(planes,nearest[...,None],axis=1)
    margin=np.einsum('npi,nvi->npv',first[...,:3],vertices)+first[...,3,None]
    separated=(margin.min(axis=-1)>1e-8).any(axis=-1)
    remaining=np.flatnonzero(~separated)
    if len(remaining):
        p=planes[remaining];v=vertices[remaining]
        margin=np.einsum('npi,nvi->npv',p[...,:3],v)+p[...,3,None]
        separated[remaining]=(margin.min(axis=-1)>1e-8).any(axis=-1)
    return separated


def collision_hulls(stage,path):
    from pxr import Usd,UsdGeom,UsdPhysics
    from scipy.spatial import ConvexHull
    from .stabilize_sweep_surface import _triangles
    cache=UsdGeom.XformCache();root=stage.GetPrimAtPath(path)
    inverse=cache.GetLocalToWorldTransform(root).GetInverse()
    result=[]
    for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
        vertices=[]
        for shape in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
            data=_triangles(shape)
            if data is None:continue
            v,_=data;matrix=np.asarray(cache.GetLocalToWorldTransform(shape)*inverse)
            vertices.append(v@matrix[:3,:3]+matrix[3,:3])
        if not vertices:raise RuntimeError(f'Unsupported collision shape: {prim.GetPath()}')
        cloud=np.concatenate(vertices);hull=ConvexHull(cloud)
        result.append((hull.equations,cloud[hull.vertices]))
    return result


class RobotResetGeometry:
    """USD joint-frame FK for collision checks at the fixed initial hand posture."""
    def __init__(self, stage, robot_path, joint_names, device):
        from pxr import Usd, UsdPhysics
        self.device = device
        incoming = {}
        adjacency = set()
        fixed = []
        def q(value):
            return (float(value.GetReal()), *map(float, value.GetImaginary()))
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(parents) != 1 or len(children) != 1:
                continue
            parent, child = str(parents[0]), str(children[0])
            column = joint_names.index(prim.GetName()) if prim.GetName() in joint_names else None
            axis = 'XYZ'.index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()) if column is not None else None
            if column is None and not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f'Unmapped reset joint: {prim.GetPath()}')
            values = (joint.GetLocalPos0Attr().Get(), q(joint.GetLocalRot0Attr().Get()),
                      joint.GetLocalPos1Attr().Get(), q(joint.GetLocalRot1Attr().Get()))
            incoming[child] = (parent, *(torch.tensor(tuple(v),device=device,dtype=torch.float32) for v in values),axis,column)
            adjacency.add(frozenset((parent,child)))
            if column is None: fixed.append((parent,child))
        base = robot_path + '/base_link'
        self.base = base
        self.steps = []
        available = {base}
        pending = dict(incoming)
        while pending:
            ready = [child for child,value in pending.items() if value[0] in available]
            if not ready: break  # root/world attachment is outside the base subtree
            for child in ready:
                self.steps.append((child,pending.pop(child)))
                available.add(child)
        self.paths = sorted(available)
        groups = {path:path for path in self.paths}
        for parent,child in fixed:
            if parent in groups and child in groups:
                old,new=groups[child],groups[parent]
                groups={key:(new if value==old else value) for key,value in groups.items()}
        linked={frozenset(groups[p] for p in pair) for pair in adjacency if all(p in groups for p in pair)}
        centers,half,owners=[],[],[]
        self.hulls=[]
        self.vertex_clouds=[]
        for path in self.paths:
            for equations,vertices in collision_hulls(stage,path):
                self.hulls.append(equations);self.vertex_clouds.append(vertices)
            # Stop traversal at nested bodies: spawned bodies are siblings here.
            for low,high in collision_bounds(stage,path,path):
                lo,hi=torch.tensor(low),torch.tensor(high)
                centers.append((lo+hi)/2);half.append((hi-lo)/2);owners.append(path)
        if not centers: raise RuntimeError('No robot collision shapes')
        self.centers=torch.stack(centers).to(device);self.half=torch.stack(half).to(device)
        self.owners=owners
        # A small deterministic selection of authored hull faces is sufficient
        # to reject most OBB false positives. The exact LP remains the fallback.
        directions=np.array([v for v in itertools.product((-1.,0.,1.),repeat=3) if any(v)])
        directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        self.face_subsets=[h[np.unique(np.argmax(h[:,:3]@directions.T,axis=0))] for h in self.hulls]
        self.self_pairs=torch.tensor([[i<j and groups[a]!=groups[b] and frozenset((groups[a],groups[b])) not in linked
                                       for j,b in enumerate(owners)] for i,a in enumerate(owners)],device=device)

    def boxes(self, joints, base_pos, base_quat):
        if joints.is_cuda and all(v.dtype==torch.float32 for v in (joints,base_pos,base_quat)):
            return self.boxes_kernel(joints,base_pos,base_quat)
        return self.boxes_reference(joints,base_pos,base_quat)

    def boxes_reference(self, joints, base_pos, base_quat):
        poses={self.base:(base_pos,base_quat)}
        for child,(parent,p0,q0,p1,q1,axis,column) in self.steps:
            pos,quat=poses[parent]
            pos=pos+quat_apply(quat,p0.expand_as(pos))
            quat=quat_mul(quat,q0.expand_as(quat))
            if column is not None:
                rot=torch.zeros_like(quat);rot[:,0]=(joints[:,column]/2).cos();rot[:,axis+1]=(joints[:,column]/2).sin()
                quat=quat_mul(quat,rot)
            quat=quat_mul(quat,quat_conjugate(q1.expand_as(quat)))
            poses[child]=(pos-quat_apply(quat,p1.expand_as(pos)),quat)
        positions=torch.stack([poses[path][0] for path in self.owners],1)
        rotations=torch.stack([poses[path][1] for path in self.owners],1)
        centers=positions+quat_apply(rotations,self.centers.expand(len(joints),-1,-1))
        return centers,self.half.expand(len(joints),-1,-1),matrix_from_quat(rotations)

    def boxes_kernel(self, joints, base_pos, base_quat):
        """Fuse each collision body's complete USD joint chain into one kernel.

        The reference quaternion equations and matrix normalization are retained;
        the fixed initial hand geometry is still authored through boxes_reference.
        """
        wp.init();device=joints.device
        stream=wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type=='cuda' else None
        if not hasattr(self,'_fk_kernel_geometry'):
            chains={self.base:[]}
            p0=[];q0=[];p1=[];q1=[];axes=[];columns=[]
            for step,(child,(parent,pos0,rot0,pos1,rot1,axis,column)) in enumerate(self.steps):
                chains[child]=[*chains[parent],step]
                p0.append(pos0);q0.append(rot0);p1.append(pos1);q1.append(rot1)
                axes.append(axis if axis is not None else -1)
                columns.append(column if column is not None else -1)
            owners=[chains[owner] for owner in self.owners]
            starts=np.r_[0,np.cumsum([len(chain) for chain in owners])]
            steps=[step for chain in owners for step in chain]
            def integers(values):
                return wp.from_torch(torch.tensor(values,dtype=torch.int32,device=device),dtype=wp.int32)
            def vectors(values,dtype):
                return wp.from_torch(torch.stack(values).to(device=device),dtype=dtype)
            # Every upload uses Torch's current stream, as does the launch.
            self._fk_kernel_geometry=(integers(starts),integers(steps),vectors(p0,wp.vec3),vectors(q0,wp.vec4),
                vectors(p1,wp.vec3),vectors(q1,wp.vec4),integers(axes),integers(columns),
                wp.from_torch(self.centers.contiguous(),dtype=wp.vec3))
        centers=torch.empty((len(joints),len(self.owners),3),dtype=joints.dtype,device=device)
        rotations=torch.empty((len(joints),len(self.owners),3,3),dtype=joints.dtype,device=device)
        wp.launch(_fk_owner_kernel,dim=(len(joints),len(self.owners)),inputs=[
            wp.from_torch(joints.contiguous(),dtype=wp.float32),wp.from_torch(base_pos.contiguous(),dtype=wp.vec3),
            wp.from_torch(base_quat.contiguous(),dtype=wp.vec4),*self._fk_kernel_geometry,
            wp.from_torch(centers,dtype=wp.vec3),wp.from_torch(rotations,dtype=wp.mat33)],
            device=str(device),stream=stream,block_dim=64)
        return centers,self.half.expand(len(joints),-1,-1),rotations

    def hand_vertices(self, joints, base_pos, base_quat, eef_pos, eef_quat, hand_names):
        boxes=self.boxes_reference(joints,base_pos,base_quat)
        keep=[i for i,path in enumerate(self.owners) if path.rsplit('/',1)[-1] in hand_names]
        if not keep: raise RuntimeError('No hand collision shapes')
        owner_ids=torch.cat([torch.full((len(self.vertex_clouds[i]),),i,device=self.device,dtype=torch.long) for i in keep])
        local=torch.tensor(np.concatenate([self.vertex_clouds[i] for i in keep]),device=self.device,dtype=joints.dtype)
        local=local-self.centers[owner_ids]
        points=boxes[0][:,owner_ids]+torch.einsum('nvij,vj->nvi',boxes[2][:,owner_ids],local)
        return quat_apply(quat_conjugate(eef_quat[:,None].expand(-1,points.shape[1],-1)),points-eef_pos[:,None])[0]

    def convex_separated_kernel(self, boxes, candidates):
        """Same conservative face proof on the tensor device, without readback."""
        device=boxes[0].device
        wp.init()
        stream=wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type=='cuda' else None
        if not hasattr(self,'_face_kernel_geometry'):
            offsets=self.centers.cpu().numpy()
            faces=[];vertices=[]
            for index,hull in enumerate(self.face_subsets):
                plane=hull.copy();plane[:,3]+=hull[:,:3]@offsets[index]
                faces.append(plane);vertices.append(self.vertex_clouds[index]-offsets[index])
            face_counts=np.array([len(v) for v in faces],dtype=np.int32)
            vertex_counts=np.array([len(v) for v in vertices],dtype=np.int32)
            self._face_kernel_geometry=(
                wp.array(np.concatenate(faces),dtype=wp.vec4d,device=str(device)),
                wp.array(np.r_[0,np.cumsum(face_counts[:-1])],dtype=int,device=str(device)),
                wp.array(face_counts,dtype=int,device=str(device)),
                wp.array(np.concatenate(vertices),dtype=wp.vec3d,device=str(device)),
                wp.array(np.r_[0,np.cumsum(vertex_counts[:-1])],dtype=int,device=str(device)),
                wp.array(vertex_counts,dtype=int,device=str(device)),
                wp.array(np.stack([v.mean(axis=0) for v in vertices]),dtype=wp.vec3d,device=str(device)))
            if stream is not None:
                # Geometry uploads use Warp's stream; the proof kernel shares
                # Torch's current stream with the input/output tensors.
                stream.wait_stream(wp.get_stream(str(device)))
        result=torch.empty(len(candidates),dtype=torch.int32,device=device)
        wp.launch(_convex_face_kernel,dim=len(candidates),inputs=[
            wp.from_torch(candidates.contiguous(),dtype=wp.int64),
            wp.from_torch(boxes[0].contiguous(),dtype=wp.vec3),wp.from_torch(boxes[2].contiguous(),dtype=wp.mat33),
            *self._face_kernel_geometry,wp.from_torch(result,dtype=wp.int32)],device=str(device),stream=stream,block_dim=64)
        return result.bool()

    def broadphase_reference(self, boxes, target, shelf):
        """Original Torch OBB checks and exact relevant-pair ordering."""
        bad=overlap(boxes,target).any(dim=(1,2)) | overlap(boxes,shelf).any(dim=(1,2))
        candidates=(overlap(boxes,boxes)&self.self_pairs&~bad[:,None,None]).nonzero()
        return bad,candidates

    def broadphase(self, boxes, target, shelf):
        if boxes[0].is_cuda and all(v.dtype==torch.float32 for group in (boxes,target,shelf) for v in group):
            return self.broadphase_kernel(boxes,target,shelf)
        return self.broadphase_reference(boxes,target,shelf)

    def broadphase_kernel(self, boxes, target, shelf):
        """Fuse relevant SAT pairs; preserve Torch decisions near boundaries."""
        device=boxes[0].device;wp.init()
        mask_version=None if torch.is_inference(self.self_pairs) else self.self_pairs._version
        key=(str(device),boxes[0].shape[1],target[0].shape[1],shelf[0].shape[1],
             id(self.self_pairs),mask_version)
        if getattr(self,'_sat_kernel_key',None)!=key:
            self_pairs=self.self_pairs.cpu().nonzero().tolist()
            other_pairs=[(kind,a,b) for kind,other in enumerate((target,shelf))
                         for a,b in itertools.product(range(boxes[0].shape[1]),range(other[0].shape[1]))]
            table=torch.tensor([*other_pairs,*[(2,a,b) for a,b in self_pairs]],device=device,dtype=torch.int32)
            self._sat_kernel_geometry=(table,len(other_pairs))
            self._sat_kernel_key=key
        table,n_other=self._sat_kernel_geometry
        outputs=torch.empty((len(boxes[0]),len(table)),dtype=torch.int32,device=device)
        ambiguity=torch.empty_like(outputs)
        inputs=[wp.from_torch(v,dtype=dtype) for values in (boxes,target,shelf)
                for v,dtype in zip(values,(wp.vec3,wp.vec3,wp.mat33))]
        stream=wp.stream_from_torch(torch.cuda.current_stream(device)) if boxes[0].is_cuda else None
        wp.launch(_broad_sat,dim=outputs.shape,inputs=[*inputs,wp.from_torch(table,dtype=wp.int32),
                  wp.from_torch(outputs,dtype=wp.int32),wp.from_torch(ambiguity,dtype=wp.int32)],
                  device=str(device),stream=stream,block_dim=64)
        ambiguous=ambiguity.nonzero()
        if len(ambiguous):
            for kind,other in enumerate((target,shelf,boxes)):
                selected=ambiguous[table[ambiguous[:,1],0]==kind]
                if len(selected):
                    rows=selected[:,0];slots=selected[:,1]
                    a=table[slots,1].long();b=table[slots,2].long()
                    left=tuple(v[rows,a].unsqueeze(1) for v in boxes)
                    right=tuple(v[rows,b].unsqueeze(1) for v in other)
                    outputs[rows,slots]=overlap(left,right).flatten().to(torch.int32)
        flags=outputs.bool()
        bad=flags[:,:n_other].any(-1)
        eligible=(flags[:,n_other:]&~bad[:,None]).nonzero()
        candidates=torch.column_stack((eligible[:,0],table[eligible[:,1]+n_other,1:].long()))
        return bad,candidates

    def valid(self, boxes, target, shelf):
        bad,candidates_t=self.broadphase(boxes,target,shelf)
        if boxes[0].is_cuda and len(candidates_t):
            candidates_t=candidates_t[~self.convex_separated_kernel(boxes,candidates_t)]
        candidates=candidates_t.cpu().numpy()
        if len(candidates):
            centers=boxes[0].detach().cpu().numpy()
            rotations=boxes[2].detach().cpu().numpy()
            offsets=self.centers.cpu().numpy()
            bad_cpu=bad.cpu().numpy().copy()
            unresolved=np.ones(len(candidates),dtype=bool)
            for a,b in np.unique(candidates[:,1:],axis=0):
                indices=np.flatnonzero((candidates[:,1]==a)&(candidates[:,2]==b))
                for start in range(0,len(indices),128):
                    selected=indices[start:start+128];rows=candidates[selected,0]
                    planes=[];vertices=[]
                    for index in (a,b):
                        hull=self.face_subsets[index]
                        rotation=rotations[rows,index].astype(np.float64)
                        normal=np.einsum('nij,pj->npi',rotation,hull[:,:3])
                        constant=hull[:,3]+hull[:,:3]@offsets[index]-np.einsum('npi,ni->np',normal,centers[rows,index])
                        planes.append(np.concatenate((normal,constant[...,None]),axis=-1))
                        # Match the LP's rotated planes even when the float32
                        # rotation is not perfectly orthogonal.
                        local=self.vertex_clouds[index]-offsets[index]
                        vertices.append(np.einsum('vj,nji->nvi',local,np.linalg.inv(rotation))+centers[rows,index,None])
                    separated=convex_face_separated(planes[0],vertices[1])|convex_face_separated(planes[1],vertices[0])
                    unresolved[selected]=~separated
            for row,a,b in candidates[unresolved]:
                if bad_cpu[row]:continue
                transformed=[]
                for index in (a,b):
                    hull=self.hulls[index];normal=hull[:,:3]@rotations[row,index].T
                    constant=hull[:,3]+hull[:,:3]@offsets[index]-normal@centers[row,index]
                    transformed.append(np.column_stack((normal,constant)))
                if convex_overlap(*transformed):bad_cpu[row]=True
            bad=torch.from_numpy(bad_cpu).to(device=bad.device)
        return ~bad

    def exclude_existing_hand_overlaps(self, joints, base_pos, base_quat, hand_names):
        # User-approved exception: preserve the source hand's zero-radian pose.
        # This cannot exclude an arm/hand or arm/arm pair.
        hand=torch.tensor([p.rsplit('/',1)[-1] in hand_names for p in self.owners],device=self.device)
        boxes=self.boxes_reference(joints,base_pos,base_quat)
        existing=overlap(boxes,boxes)[0] & hand[:,None] & hand[None,:]
        self.self_pairs &= ~existing

    def diagnostics(self, boxes, target, shelf):
        return dict(target=overlap(boxes,target).nonzero().tolist(),
                    shelf=overlap(boxes,shelf).nonzero().tolist(),
                    self_collision=(overlap(boxes,boxes)&self.self_pairs).nonzero().tolist(),
                    bodies=self.owners)
