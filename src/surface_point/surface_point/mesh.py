"""Visible triangle masks from batched geometric rays, no RGB render or model.

A metric orthographic ray lattice is the simulation equivalent of a depth/
instance sensor. Resolution is explicit; it is not analytic continuous visibility.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy import ndimage
import warp as wp
from observed_obb import observe
from observed_obb.observation import _footprint, _front_depth, _visibility_edges
from .geometry import SurfaceObservation, Patch, Comparison, FRAME


@wp.kernel
def cast(mesh: wp.uint64, face_owner: wp.array(dtype=int), width: int,  # type: ignore[valid-type]
         x: float, y0: float, z0: float, spacing: float,
         owner: wp.array(dtype=int), hits: wp.array(dtype=wp.vec3)):  # type: ignore[valid-type]
    i=wp.tid()
    origin=wp.vec3(x,y0+(float(i%width)+0.5)*spacing,z0+(float(i//width)+0.5)*spacing)
    direction=wp.vec3(-1.0,0.0,0.0)
    query=wp.mesh_query_ray(mesh,origin,direction,100.0)
    owner[i]=-2
    hits[i]=wp.vec3(0.0,0.0,0.0)
    if query.result:
        owner[i]=face_owner[query.face]
        hits[i]=origin+query.t*direction


@dataclass
class MeshFrame:
    records: tuple
    legacy: tuple
    owner: np.ndarray
    hits: np.ndarray
    y0: float
    z0: float
    spacing: float
    pixels: list
    timings: dict


def box_triangles(box):
    xy=_footprint(box)
    vertices=np.concatenate([np.column_stack((xy,np.full(4,box.center[2]+sign*box.size[2]/2))) for sign in (-1,1)])
    quads=((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
    faces=np.array([(q[0],q[i],q[i+1]) for q in quads for i in (1,2)],dtype=np.int32)
    return vertices,faces


def legacy_associations(boxes,legacy):
    polygons=[_footprint(b) @ FRAME.T for b in boxes]
    edges=_visibility_edges(polygons)
    parts: list[list[tuple[float,float]]] = [[] for _ in boxes]
    for lo,hi in zip(edges[:-1],edges[1:]):
        depths=[_front_depth(p,(lo+hi)/2) for p in polygons]
        owner=int(np.argmin(depths))
        if not np.isfinite(depths[owner]):continue
        if parts[owner] and parts[owner][-1][1]==lo:
            parts[owner][-1]=(parts[owner][-1][0],hi)
        else:parts[owner].append((lo,hi))
    association=[]
    for source,intervals in enumerate(parts):
        threshold=max(.02,.20*np.ptp(polygons[source][:,0]))
        for lo,hi in intervals:
            tolerance=4*np.finfo(float).eps*max(abs(lo),abs(hi),threshold)
            if hi-lo>=threshold-tolerance:association.append((source,lo,hi))
    if len(association)!=len(legacy):raise RuntimeError('Legacy association count mismatch')
    return list(zip(legacy,association))


class MeshObserver:
    def __init__(self,*,device='cpu',spacing=.001,representative='mask-centroid',max_rays=2_000_000):
        if not np.isfinite(spacing) or spacing<=0:raise ValueError('spacing must be finite and positive')
        if representative not in ('mask-centroid','bbox-center'):raise ValueError('Unknown representative rule')
        self.device=wp.get_device(device)
        self.spacing=float(spacing)
        self.representative=representative
        self.max_rays=max_rays
        self.mesh=None
        self._topology=None
        self._owners=None

    def observe(self,vertices,faces,face_owner,boxes):
        start=perf_counter()
        vertices=np.asarray(vertices,dtype=np.float32)
        faces=np.asarray(faces,dtype=np.int32).reshape(-1,3)
        face_owner=np.asarray(face_owner,dtype=np.int32)
        if not boxes:raise ValueError('At least one observed object required')
        if not len(faces) or len(face_owner)!=len(faces):raise ValueError('Triangle ownership required')
        if not np.isfinite(vertices).all():raise ValueError('Finite vertices required')
        if faces.min()<0 or faces.max()>=len(vertices):raise ValueError('Invalid triangle indices')
        if face_owner.min() < -1 or face_owner.max() >= len(boxes):raise ValueError('Invalid source owner')
        target_vertices=vertices[np.unique(faces[face_owner>=0])]
        low=np.floor(target_vertices[:,1:].min(0)/self.spacing)*self.spacing-self.spacing
        high=np.ceil(target_vertices[:,1:].max(0)/self.spacing)*self.spacing+self.spacing
        width,height=np.ceil((high-low)/self.spacing).astype(int)
        count=int(width*height)
        if count>self.max_rays:raise ValueError(f'{count} rays exceed limit; increase --spacing-mm')
        points=wp.array(vertices,dtype=wp.vec3,device=self.device)
        same=(self.mesh is not None and len(vertices)==len(self.mesh.points) and
              np.array_equal(faces,self._topology) and np.array_equal(face_owner,self._owners))
        if same:
            assert self.mesh is not None
            self.mesh.points.assign(points);self.mesh.refit()
        else:
            self.mesh=wp.Mesh(points=points,indices=wp.array(faces.ravel(),dtype=int,device=self.device))
            self._face_owner=wp.array(face_owner,dtype=int,device=self.device)
            self._topology=faces.copy();self._owners=face_owner.copy()
        assert self.mesh is not None
        build_ms=(perf_counter()-start)*1000
        ray_start=perf_counter()
        owner=wp.empty(count,dtype=int,device=self.device)
        hits=wp.empty(count,dtype=wp.vec3,device=self.device)
        wp.launch(cast,dim=count,inputs=[self.mesh.id,self._face_owner,int(width),
                  float(vertices[:,0].max()+.05),float(low[0]),float(low[1]),self.spacing,owner,hits],device=self.device)
        owners=owner.numpy().reshape(height,width)
        hit_points=hits.numpy().reshape(height,width,3)
        ray_ms=(perf_counter()-ray_start)*1000
        reduce_start=perf_counter()
        legacy=observe(boxes);association=legacy_associations(boxes,legacy)
        records: list[Comparison] = []
        pixels=[]
        for source in range(len(boxes)):
            source_faces=faces[face_owner==source]
            if not len(source_faces):continue
            full_width=float(np.ptp(vertices[np.unique(source_faces),1]))
            labels,num=ndimage.label(owners==source)  # type: ignore[misc]  # 4-neighbour visible components
            for label_index,slices in enumerate(ndimage.find_objects(labels),start=1):
                if slices is None:continue
                rs,cs=slices
                if (cs.stop-cs.start)*self.spacing+1e-9 < max(.02,.2*full_width):continue
                rr,cc=np.nonzero(labels[rs,cs]==label_index)
                rr,cc=rr+rs.start,cc+cs.start
                bbox=(float(low[0]+cs.start*self.spacing),float(low[1]+rs.start*self.spacing),
                      float(low[0]+cs.stop*self.spacing),float(low[1]+rs.stop*self.spacing))
                reference=(np.array([rr.mean(),cc.mean()]) if self.representative=='mask-centroid'
                           else np.array([(rs.start+rs.stop-1)/2,(cs.start+cs.stop-1)/2]))
                index=np.argmin((rr-reference[0])**2+(cc-reference[1])**2)
                row,col=int(rr[index]),int(cc[index])
                rounded=np.rint(reference).astype(int)
                corrected=labels[tuple(rounded)]!=label_index
                shift=float(np.linalg.norm(np.array([row,col])-reference)*self.spacing)
                hit=hit_points[row,col]
                point=(float(hit[0]),float(hit[1]),float(hit[2]))
                observation=SurfaceObservation(len(records),point,bbox,bool(corrected),shift)
                # Dense cells are not copied into each record; frame.owner is the mask.
                patch=Patch(observation,source,())
                matches=[o for o,(s,lo,hi) in association if s==source and lo-1e-6<=point[1]<=hi+1e-6
                         and abs(point[2]-o.box.center[2])<=o.box.size[2]/2+1e-6]
                old=matches[0] if matches else None
                distance,depth_error=None,None
                if old:
                    distance=float(np.linalg.norm(np.array(point)-old.box.center))
                    front=-_front_depth(_footprint(old.box) @ FRAME.T,point[1])
                    depth_error=abs(point[0]-front)
                records.append(Comparison(patch,old,distance,depth_error))
                pixels.append(dict(observation_id=observation.observation_id,source_index=source,
                                   pixel_rc=(row,col),reference_rc=reference.tolist(),pixel_count=len(rr),
                                   bbox_rc=(rs.start,cs.start,rs.stop,cs.stop),rule=self.representative))
        timings=dict(bvh_ms=build_ms,ray_ms=ray_ms,reduce_ms=(perf_counter()-reduce_start)*1000,
                     total_ms=(perf_counter()-start)*1000,rays=count,triangles=len(faces))
        return MeshFrame(tuple(records),legacy,owners,hit_points,float(low[0]),float(low[1]),self.spacing,pixels,timings)
