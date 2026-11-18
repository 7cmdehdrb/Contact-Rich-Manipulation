"""Cache visible USD geometry in object-local space; transform only on reset."""
import numpy as np
from pxr import Usd, UsdGeom
from observed_obb import Box
from .mesh import box_triangles


def triangles(prim):
    if prim.IsA(UsdGeom.Mesh):
        mesh=UsdGeom.Mesh(prim)
        vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32)
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int)
        holes=set(mesh.GetHoleIndicesAttr().Get() or [])
        faces: list[tuple[int,int,int]] = []
        offset=0
        for index,count in enumerate(counts):
            polygon=indices[offset:offset+count];offset+=count
            if index not in holes:
                faces.extend((polygon[0],polygon[k],polygon[k+1]) for k in range(1,count-1))
        return vertices,np.asarray(faces,dtype=np.int32).reshape(-1,3)
    if prim.IsA(UsdGeom.Cube):
        size=float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        return box_triangles(Box((0,0,0),(size,size,size)))
    if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Cone):
        geom=UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Cone(prim)
        radius,height=float(geom.GetRadiusAttr().Get()),float(geom.GetHeightAttr().Get())
        angles=np.arange(128)*2*np.pi/128
        ring=np.column_stack((radius*np.cos(angles),radius*np.sin(angles)))
        top_radius=1. if prim.IsA(UsdGeom.Cylinder) else 0.
        vertices=np.vstack((np.column_stack((ring,np.full(128,-height/2))),
                            np.column_stack((ring*top_radius,np.full(128,height/2))),
                            (0,0,-height/2),(0,0,height/2)))
        faces=[]
        for i in range(128):
            j=(i+1)%128
            faces.extend(((i,j,j+128),(i,j+128,i+128),(256,j,i),(257,i+128,j+128)))
        axis=str(geom.GetAxisAttr().Get()).upper()
        if axis=='X':vertices=vertices[:,[2,0,1]]
        elif axis=='Y':vertices=vertices[:,[1,2,0]]
        return vertices,np.asarray(faces,dtype=np.int32)
    if prim.IsA(UsdGeom.Gprim):
        raise ValueError(f'Unsupported visible geometry {prim.GetTypeName()} at {prim.GetPath()}')
    return None


class StageGeometry:
    def __init__(self,stage):
        self.stage=stage
        self.cache={}

    def _local(self,path):
        if path in self.cache:return self.cache[path]
        root=self.stage.GetPrimAtPath(path)
        transforms=UsdGeom.XformCache()
        root_inverse=transforms.GetLocalToWorldTransform(root).GetInverse()
        vertices=[];faces=[];offset=0
        for prim in Usd.PrimRange(root,Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Imageable):continue
            imageable=UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility()=='invisible' or imageable.ComputePurpose()=='guide':continue
            data=triangles(prim)
            if data is None:continue
            v,f=data
            if not len(f):continue
            transform=np.asarray(transforms.GetLocalToWorldTransform(prim)*root_inverse)
            v=v @ transform[:3,:3]+transform[3,:3]
            vertices.append(v);faces.append(f+offset);offset+=len(v)
        if not faces:raise ValueError(f'No visible triangle geometry below {path}')
        value=np.concatenate(vertices),np.concatenate(faces)
        self.cache[path]=value
        return value

    def collect(self,paths,occluder_paths=()):
        transforms=UsdGeom.XformCache()
        vertices=[];faces=[];owners=[];offset=0
        for owner,path in [*enumerate(paths),*((-1,p) for p in occluder_paths)]:
            v,f=self._local(path)
            transform=np.asarray(transforms.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path)))
            world=v @ transform[:3,:3]+transform[3,:3]
            vertices.append(world);faces.append(f+offset);owners.append(np.full(len(f),owner,dtype=np.int32))
            offset+=len(v)
        return np.concatenate(vertices),np.concatenate(faces),np.concatenate(owners)
