"""Replay a saved real-mesh snapshot without Isaac Sim; compare point definitions."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src'/p) for p in ('surface_point','observed_obb')]
import numpy as np
from PIL import Image
from observed_obb import Box
from surface_point import benchmark
from surface_point.mesh import MeshObserver
from surface_point.mask_preview import preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'outputs/surface_point_mesh/simple_mesh_3.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/surface_point_mesh/replay')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.snapshot,allow_pickle=False) as data:
        vertices,faces,owners=data['vertices'],data['faces'],data['owners']
        boxes=tuple(Box(**b) for b in json.loads(str(data['boxes'])))
    old,_=benchmark(boxes)
    rows=[]
    for spacing,rule in ((.001,'mask-centroid'),(.0005,'mask-centroid'),(.001,'bbox-center')):
        model=MeshObserver(device='cpu',spacing=spacing,representative=rule)
        frame=model.observe(vertices,faces,owners,boxes)
        Image.fromarray(preview(frame)).save(args.output/f'{rule}_{spacing*1000:g}mm.png')
        for r in frame.records:
            p=r.patch.observation
            rows.append(dict(spacing_mm=spacing*1000,rule=rule,source=r.patch.source_index,
                             id=p.observation_id,point=p.point,corrected=p.corrected,
                             projection_shift_mm=p.correction_m*1000))
    (args.output/'comparison.json').write_text(json.dumps(dict(
        source=str(args.snapshot),mesh_points=rows,
        old_box_proxy_points=[dict(source=r.patch.source_index,point=r.patch.observation.point) for r in old]),indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
