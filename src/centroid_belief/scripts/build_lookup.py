"""Generate a deterministic, simulation-free clipping dictionary once."""
from pathlib import Path
import sys,json,time
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/centroid_belief'),str(ROOT/'src/observed_obb')]
from observed_obb import Box
from observed_obb.observation import _footprint,_clip,_fit
from centroid_belief.lookup import BANK_PATH,BANK_VERSION,feature
from centroid_belief.shapes import SHAPES


def main():
    start=time.perf_counter();features=[];records=[]
    for i,shape in enumerate(SHAPES):
        for yaw in np.linspace(-np.pi/2,np.pi/2,72,endpoint=False):
            source=Box((0.,0.,0.),shape.size,float(yaw))
            points=_footprint(source)
            # clipping helper expects lateral coordinate in column 0
            polygon=points[:,::-1]
            low,high=polygon[:,0].min(),polygon[:,0].max()
            for left in range(17):
                for right in range(left+4,21):
                    if left==0 and right==20: continue
                    a,b=low+(high-low)*left/20,low+(high-low)*right/20
                    # Height normalization allows continuous runtime scaling.
                    if b-a < .02/1.1: continue
                    clipped=_clip(_clip(polygon,a,True),b,False)[:,::-1]
                    observed=_fit(clipped,source)
                    mask=(1 if left>0 else 0)|(2 if right<20 else 0)
                    features.append(feature(observed))
                    records.append((i,mask,yaw,-observed.center[0]/shape.size[2],
                                    -observed.center[1]/shape.size[2],(right-left)/20))
        print(shape.name,len(records),flush=True)
    BANK_PATH.parent.mkdir(parents=True,exist_ok=True)
    metadata=json.dumps(dict(version=BANK_VERSION,shapes=[s.size for s in SHAPES],yaw_steps=72,cut_steps=20))
    np.savez_compressed(BANK_PATH,features=np.array(features,dtype=np.float32),
                        records=np.array(records,dtype=np.float32),metadata=metadata)
    print(BANK_PATH,len(records),'rows',time.perf_counter()-start,'seconds',flush=True)

if __name__=='__main__': main()
