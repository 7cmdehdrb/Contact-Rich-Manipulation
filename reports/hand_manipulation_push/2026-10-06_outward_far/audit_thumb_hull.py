"""CPU USD-joint FK and exact convex-hull silhouette, no Simulator/GPU."""
from pathlib import Path
import sys,json,itertools,math,time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
sys.path.insert(0,'/tmp')
import push_cpu_ik_audit as h
from pxr import Usd,UsdGeom,UsdPhysics,Gf
HERE=Path(__file__).resolve().parent
raw=[]
for body in h.st.Traverse():
 if not body.HasAPI(UsdPhysics.RigidBodyAPI) or not body.GetName().startswith('inspire_'):continue
 bi=h.ca.GetLocalToWorldTransform(body).GetInverse()
 seen=set()
 for col in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
  if not col.HasAPI(UsdPhysics.CollisionAPI):continue
  for shape in Usd.PrimRange(col,Usd.TraverseInstanceProxies()):
   if not shape.IsA(UsdGeom.Mesh) or str(shape.GetPath()) in seen:continue
   seen.add(str(shape.GetPath()));mesh=UsdGeom.Mesh(shape)
   tf=h.ca.GetLocalToWorldTransform(shape)*bi
   p=np.array([list(tf.Transform(Gf.Vec3d(v))) for v in mesh.GetPointsAttr().Get()],dtype=float)
   collider=shape
   while collider!=body and not collider.HasAPI(UsdPhysics.MeshCollisionAPI):collider=collider.GetParent()
   approximation=UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() if collider.HasAPI(UsdPhysics.MeshCollisionAPI) else None
   attrs={a.GetName():str(a.Get()) for a in collider.GetAttributes() if ('physx' in a.GetName().lower() or 'physics:' in a.GetName())}
   raw.append(dict(name=body.GetName(),path=str(shape.GetPath()),points=p,approximation=approximation,attrs=attrs))
print('GEOMETRY',len(raw),'mesh colliders',flush=True)
# Contact variant changes only central-palm collision position, in rest H frame.
for r in raw:
 if r['name']=='inspire_palm_force_sensor':
  body=h.st.GetPrimAtPath(h.rt+'/'+r['name']);inv=h.ca.GetLocalToWorldTransform(body).GetInverse()
  shift=np.array((h.H0*inv).TransformDir(Gf.Vec3d(0,.035,0)))
  r['points']=r['points']+shift

def at(common,thumb,exposure=0):
 mats=h.all_fk(h.q0,common,thumb);hi=mats['inspire_base_link'].GetInverse();out=[]
 for r in raw:
  tf=mats[r['name']]*hi;p=np.array([list(tf.Transform(Gf.Vec3d(*v))) for v in r['points']])
  if exposure and r['name'] in ('inspire_thumb_force_sensor_3','inspire_thumb_force_sensor_4'):
   restbody=h.ca.GetLocalToWorldTransform(h.st.GetPrimAtPath(h.rt+'/'+r['name'])).GetInverse()
   dv=(h.H0*restbody).TransformDir(Gf.Vec3d(0,exposure,0))
   p+=np.array(tf.TransformDir(dv))
  hull=ConvexHull(p)
  out.append({**r,'points_h':p,'equations':hull.equations})
 return out

def support(g,xlo,xhi,zlo,zhi):
 eq=g['equations']
 result=linprog([0,-1,0],A_ub=eq[:,:3],b_ub=-eq[:,3]+1e-10,bounds=[(xlo,xhi),(None,None),(zlo,zhi)],method='highs')
 if result.success:return float(result.x[1])
 if result.status==2:return None
 raise RuntimeError(result.message)

def face(gs,name):
 p=np.concatenate([g['points_h'] for g in gs if g['name']==name]);mask=p[:,1]>=p[:,1].max()-.0005
 return p[mask].mean(0),p.min(0),p.max(0)

def query(gs,which,height):
 f,lo,hi=face(gs,which);xc=f[0]+height;zc=f[2];sup=[]
 for g in gs:
  v=support(g,xc-.03,xc+.03,zc-.03,zc+.03)
  if v is not None:sup.append((v,g['name'],g['approximation']))
 sup.sort(reverse=True)
 carriers=[x for x in sup if 'force_sensor' not in x[1]]
 pads=[x for x in sup if 'force_sensor' in x[1]]
 own=[x for x in pads if x[1]==which]
 ownfront=max(x[0] for x in own)
 return dict(pad=which,height=height,face_h=f.tolist(),min_h=lo.tolist(),max_h=hi.tolist(),
   cube_patch_h={'x':[xc-.03,xc+.03],'z':[zc-.03,zc+.03]},
   top_carriers=[dict(y=x[0],name=x[1],approximation=x[2]) for x in carriers[:5]],
   top_pads=[dict(y=x[0],name=x[1],approximation=x[2]) for x in pads[:5]],
   own_max_y=ownfront,carrier_to_own_gap_mm=1000*(carriers[0][0]-ownfront),
   required_extra_plus2mm=1000*max(0,carriers[0][0]+.002-ownfront))
report={'geometry':[{k:v for k,v in r.items() if k!='points'} for r in raw], 'queries':[], 'low_central_table':[]}
for common,thumb in itertools.product((.8,.9,.94,.98),repeat=2):
 gs=at(common,thumb)
 for pad,height in itertools.product((3,4),(0.,.01)):
  row=query(gs,f'inspire_thumb_force_sensor_{pad}',height);row.update(common=common,thumb=thumb);report['queries'].append(row)
 # Full strict4mm body OBB gate on physically fixed outward-flat left P at low center.
 mats=h.all_fk(h.q0,common,thumb);Hinv=mats['inspire_base_link'].GetInverse()
 R=np.stack([np.array([0,0,-1]),np.array([0,-1,0]),np.array([-1,0,0])],axis=1)
 for height in (.025,.04,.075,.10):
  P=np.array([-.70,.038,1.08+height]);H=P-R@h.ph;hits=[];mins=[]
  for name,c,half in h.bounds:
   if not name.startswith('inspire_'):continue
   tf=mats[name]*Hinv;Rb=np.array(tf.ExtractRotationMatrix()).T;pb=np.array(tf.ExtractTranslation());W=R@Rb;cc=H+R@pb+W@c
   if h.overlap(cc,W,half+.004,np.array([-.75,0,1.03]),np.array([.18,.5,.02])):hits.append(name)
   mins.append((float(cc[2]-(np.abs(W[2])@half)),name))
  report['low_central_table'].append(dict(common=common,thumb=thumb,height=height,table_hits=hits,lowest=sorted(mins)[:6]))
 print('POSE',common,thumb,'done',flush=True)
# Include actual under-load sag found in live fixture, plus compare +4/+8 rest-frame shifts.
for exposure in (0.,.004,.008):
 gs=at(.942,.985,exposure)
 for pad in (3,4):
  row=query(gs,f'inspire_thumb_force_sensor_{pad}',0.);row.update(common=.942,thumb=.985,exposure=exposure);report['queries'].append(row)
path=HERE/'thumb_hull_audit.json';path.write_text(json.dumps(report,indent=2));print('SAVED',path,flush=True)
for r in report['queries']:
 if r['common']==r['thumb'] or 'exposure' in r:
  print(json.dumps({k:r[k] for k in ('common','thumb','pad','height','carrier_to_own_gap_mm','required_extra_plus2mm','top_carriers','top_pads')}),flush=True)
print('LOW_CENTRAL',json.dumps([r for r in report['low_central_table'] if r['common']==r['thumb']]),flush=True)
