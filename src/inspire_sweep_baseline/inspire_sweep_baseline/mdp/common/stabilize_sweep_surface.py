"""Initial visible mesh snapshot; the surface_point ray/mask algorithm without its UI.

Only returned XYZ points enter observations. Mesh ownership is used locally for
exposure filtering and selecting the largest target component, never exported.
"""
from time import perf_counter
import numpy as np
from scipy import ndimage
import warp as wp


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
    points=wp.array(vertices,dtype=wp.vec3,device=device)
    mesh=wp.Mesh(points=points,indices=wp.array(faces.ravel(),dtype=int,device=device))
    labels=wp.empty(count,dtype=int,device=device);hits=wp.empty(count,dtype=wp.vec3,device=device)
    build=perf_counter()
    wp.launch(_cast,dim=count,inputs=[mesh.id,wp.array(owners,dtype=int,device=device),int(width),
              float(vertices[:,0].max()+.05),float(low[0]),float(low[1]),spacing,labels,hits],device=device)
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
        ray_readback_ms=1000*(readback-build),rays=count)


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
