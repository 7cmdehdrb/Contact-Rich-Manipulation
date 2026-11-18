"""CPU geometry comparison, CSV/JSON and frontal/top-view plots; no simulator."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'/p) for p in ('surface_point','observed_obb')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon
import numpy as np
from observed_obb.observation import _footprint
from surface_point import benchmark
from surface_point.fixtures import scenes


def main():
    output=ROOT/'outputs/surface_point';output.mkdir(parents=True,exist_ok=True)
    rows=[];reports=[];fig,axes=plt.subplots(4,2,figsize=(11,13),constrained_layout=True)
    selected=['clear','split','hole_center','rotated']
    for name,boxes in scenes().items():
        start=perf_counter();records,legacy=benchmark(boxes);ms=(perf_counter()-start)*1000
        reports.append(dict(scene=name,geometry_ms=ms,sources=[asdict(b) for b in boxes],
                            comparisons=[asdict(r) for r in records],legacy=[asdict(o) for o in legacy]))
        for r in records:
            p=r.patch.observation
            rows.append(dict(scene=name,point_id=p.observation_id,evaluation_source=r.patch.source_index,
                             x=p.point[0],y=p.point[1],z=p.point[2],corrected=p.corrected,
                             correction_mm=p.correction_m*1000,legacy_id=r.legacy.observation_id if r.legacy else None,
                             center_distance_mm=r.center_distance_m*1000 if r.center_distance_m is not None else None,
                             front_depth_error_mm=r.front_depth_error_m*1000 if r.front_depth_error_m is not None else None))
        if name not in selected:continue
        front,top=axes[selected.index(name)]
        for b in boxes:
            lo=np.array(b.center)-np.array(b.size)/2
            lateral=_footprint(b)[:,1]
            front.add_patch(Rectangle((lateral.min()*100,lo[2]*100),np.ptp(lateral)*100,b.size[2]*100,
                                      fill=False,ls='--',ec='#df8b17',lw=1))
            top.add_patch(Polygon(_footprint(b)[:,::-1]*100,fill=False,ls='--',ec='#df8b17',lw=1))
        for r in records:
            for y0,z0,y1,z1 in r.patch.cells:
                front.add_patch(Rectangle((y0*100,z0*100),(y1-y0)*100,(z1-z0)*100,
                                         facecolor=('#b6cce4' if r.patch.source_index==0 else '#d5b29c'),edgecolor='none'))
            p=r.patch.observation
            bounds=p.bbox_yz
            front.add_patch(Rectangle((bounds[0]*100,bounds[1]*100),(bounds[2]-bounds[0])*100,
                                     (bounds[3]-bounds[1])*100,fill=False,ec='#9945bb',lw=1))
            for ax,xy in ((front,(p.point[1]*100,p.point[2]*100)),(top,(p.point[1]*100,p.point[0]*100))):
                ax.plot(*xy,'o',color='red',ms=5,zorder=5)
                ax.annotate(f'P{p.observation_id}',xy,xytext=(4,4),textcoords='offset points',fontsize=8)
            if r.legacy:
                center=r.legacy.box.center
                top.plot([p.point[1]*100,center[1]*100],[p.point[0]*100,center[0]*100],color='#1688e8',lw=1)
        for o in legacy:
            b=o.box;poly=_footprint(b)
            front.add_patch(Rectangle((poly[:,1].min()*100,(b.center[2]-b.size[2]/2)*100),
                                     np.ptp(poly[:,1])*100,b.size[2]*100,fill=False,ec='#16883c',lw=1.5))
            top.add_patch(Polygon(poly[:,::-1]*100,fill=False,ec='#16883c',lw=1.5))
            top.plot(b.center[1]*100,b.center[0]*100,'x',color='#1688e8',ms=6)
        front.set_title(f'{name}: frontal YZ (visible regions)')
        top.set_title(f'{name}: top YX (observer above plot)')
        front.set_ylabel('Z (cm)');top.set_ylabel('X (cm)')
        for ax in (front,top):
            ax.set_xlabel('Y (cm)');ax.autoscale_view();ax.margins(.15);ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle('Box-proxy test bench: visible surface points vs legacy observed OBBs\n'
                 'Red: surface point | Blue X/link: OBB center | Green: observed OBB\n'
                 'Orange dashed: source OBB | Purple: visible-region YZ bbox',fontsize=11)
    fig.savefig(output/'comparison.png',dpi=160);fig.savefig(output/'comparison.pdf');plt.close(fig)
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'fixtures.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps([dict(scene=r['scene'],points=len(r['comparisons']),legacy=len(r['legacy']),ms=r['geometry_ms']) for r in reports],indent=2))


if __name__=='__main__':main()
