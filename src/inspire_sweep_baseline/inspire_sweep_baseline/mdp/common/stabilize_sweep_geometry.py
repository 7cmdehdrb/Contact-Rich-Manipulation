"""Conservative collision boxes at candidate joint poses, without physics steps."""
import itertools
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul, matrix_from_quat


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
        self.self_pairs=torch.tensor([[i<j and groups[a]!=groups[b] and frozenset((groups[a],groups[b])) not in linked
                                       for j,b in enumerate(owners)] for i,a in enumerate(owners)],device=device)

    def boxes(self, joints, base_pos, base_quat):
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

    def hand_vertices(self, joints, base_pos, base_quat, eef_pos, eef_quat, hand_names):
        boxes=self.boxes(joints,base_pos,base_quat)
        keep=[i for i,path in enumerate(self.owners) if path.rsplit('/',1)[-1] in hand_names]
        if not keep: raise RuntimeError('No hand collision shapes')
        owner_ids=torch.cat([torch.full((len(self.vertex_clouds[i]),),i,device=self.device,dtype=torch.long) for i in keep])
        local=torch.tensor(np.concatenate([self.vertex_clouds[i] for i in keep]),device=self.device,dtype=joints.dtype)
        local=local-self.centers[owner_ids]
        points=boxes[0][:,owner_ids]+torch.einsum('nvij,vj->nvi',boxes[2][:,owner_ids],local)
        return quat_apply(quat_conjugate(eef_quat[:,None].expand(-1,points.shape[1],-1)),points-eef_pos[:,None])[0]

    def valid(self, boxes, target, shelf):
        bad=overlap(boxes,target).any(dim=(1,2)) | overlap(boxes,shelf).any(dim=(1,2))
        candidates=(overlap(boxes,boxes)&self.self_pairs).nonzero().tolist()
        if candidates:
            centers,_,rotations=(v.detach().cpu().numpy() for v in boxes)
            offsets=self.centers.cpu().numpy()
            for row,a,b in candidates:
                if bool(bad[row]):continue
                transformed=[]
                for index in (a,b):
                    hull=self.hulls[index];normal=hull[:,:3]@rotations[row,index].T
                    constant=hull[:,3]+hull[:,:3]@offsets[index]-normal@centers[row,index]
                    transformed.append(np.column_stack((normal,constant)))
                if convex_overlap(*transformed):bad[row]=True
        return ~bad

    def exclude_existing_hand_overlaps(self, joints, base_pos, base_quat, hand_names):
        # User-approved exception: preserve the source hand's zero-radian pose.
        # This cannot exclude an arm/hand or arm/arm pair.
        hand=torch.tensor([p.rsplit('/',1)[-1] in hand_names for p in self.owners],device=self.device)
        boxes=self.boxes(joints,base_pos,base_quat)
        existing=overlap(boxes,boxes)[0] & hand[:,None] & hand[None,:]
        self.self_pairs &= ~existing

    def diagnostics(self, boxes, target, shelf):
        return dict(target=overlap(boxes,target).nonzero().tolist(),
                    shelf=overlap(boxes,shelf).nonzero().tolist(),
                    self_collision=(overlap(boxes,boxes)&self.self_pairs).nonzero().tolist(),
                    bodies=self.owners)
