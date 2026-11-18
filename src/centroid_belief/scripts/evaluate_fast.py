"""Reproducible CPU speed and off-grid accuracy checks, without simulation."""
import json
import csv
import os
from pathlib import Path
import sys
from time import perf_counter
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src/centroid_belief'), str(ROOT/'src/observed_obb')]
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from PIL import Image
from centroid_belief.lookup import LookupEstimator
from centroid_belief.fixtures import standard_cases, make_case
from centroid_belief.preview import preview_rgba


def main():
    output = ROOT/'outputs/centroid_fast'
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    estimator = LookupEstimator()
    load_ms = (perf_counter()-start)*1000
    rng = np.random.default_rng(31)
    cases = standard_cases()
    # Unseen continuous yaw/scale: not table grid points. Truth is evaluation only.
    holdout = []
    filtered = 0
    for i in range(8):
        for pattern in ('clear','left','right','both'):
            for _ in range(4):
                try:
                    holdout.append(make_case(i,pattern,scale=rng.uniform(.9,1.1),yaw=rng.uniform(-np.pi/2,np.pi/2)))
                except ValueError:
                    filtered += 1
    rows = []
    for group, fixtures in [('standard',cases),('holdout',holdout)]:
        for case in fixtures:
            durations = []
            for _ in range(3):
                start = perf_counter()
                result = estimator.estimate(case.observations,case.target_id)
                durations.append((perf_counter()-start)*1000)
            observed = next(o.box for o in case.observations if o.observation_id == case.target_id)
            truth = np.array(case.truth.center[:2])
            rows.append(dict(group=group,case=case.name,ms=float(np.median(durations)),
                             evaluated=result.evaluated, warnings=';'.join(result.warnings),
                             mean_error_cm=float(np.linalg.norm(result.belief.mean-truth)*100),
                             observed_error_cm=float(np.linalg.norm(np.array(observed.center[:2])-truth)*100),
                             truth90=bool(result.belief.contains(truth)[0]),
                             observed90=bool(result.belief.contains(observed.center[:2])[0])))
    case=cases[2]; result=estimator.estimate(case.observations,case.target_id)
    start=perf_counter()
    rgba=preview_rgba(case.observations[0],result.belief.parameters)
    preview_ms=(perf_counter()-start)*1000
    Image.fromarray(rgba).save(output/'fast_preview.png')
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary: dict = {'filtered_holdout':filtered,'load_ms':load_ms,'preview_ms':preview_ms,'bank_rows':len(estimator.records)}
    for group in ('standard','holdout'):
        selected=[r for r in rows if r['group']==group]
        summary[group]={'count':len(selected),'p50_ms':float(np.median([r['ms'] for r in selected])),
                        'p95_ms':float(np.percentile([r['ms'] for r in selected],95)),
                        'max_ms':max(r['ms'] for r in selected),
                        **{k:float(np.mean([r[k] for r in selected])) for k in ('mean_error_cm','observed_error_cm','truth90','observed90')},
                        'warning_count':sum(bool(r['warnings']) for r in selected)}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
