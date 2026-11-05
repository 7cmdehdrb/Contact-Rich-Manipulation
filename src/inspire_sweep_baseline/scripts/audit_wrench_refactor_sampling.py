"""Seeded audit of the production goal sampler without starting Isaac Sim."""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import types
import numpy as np
from scipy.stats import chisquare, binomtest, kstest
import torch


def direction_report(delta, degrees):
    theta = np.arctan2(delta[:,1],delta[:,0]) % (2*np.pi)
    alpha = math.radians(degrees)
    folded = theta % np.pi
    unit = ((folded-alpha)/(np.pi-2*alpha) + (theta>=np.pi)) / 2
    counts,_ = np.histogram(unit,bins=24,range=(0,1))
    left = int((delta[:,1]<0).sum())
    return dict(degrees=degrees,n=len(delta),negative_y_count=left,
        negative_y_fraction=left/len(delta),bins24=counts.tolist(),
        chi_square_p=float(chisquare(counts).pvalue),ks_p=float(kstest(unit,'uniform').pvalue),
        side_binomial_p=float(binomtest(left,len(delta),.5).pvalue),
        forbidden_count=int(((folded<alpha-1e-6)|(folded>np.pi-alpha+1e-6)).sum()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=200000)
    parser.add_argument('--output',type=Path,default=Path('outputs/wrench_refactor_validation/sampling'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    package=types.ModuleType('_refactor_geometry');package.__path__=[str(Path(__file__).resolve().parents[1]/'inspire_sweep_baseline/mdp/common')]
    sys.modules[package.__name__]=package
    layout=importlib.import_module('_refactor_geometry.wrench_refactor_layout')
    torch.set_num_threads(4);rows=[];deltas=[]
    for degrees in (30,25,20,15,10,5,0):
        rng=torch.Generator().manual_seed(42000+degrees)
        scale=.9+.2*torch.rand((args.samples,1),generator=rng)
        bounds=torch.zeros(args.samples,2,3);bounds[:,0,:2]=-.04*scale;bounds[:,1,:2]=.04*scale;bounds[:,1,2]=.1*scale[:,0]
        start,goal=layout.sample_commands(bounds,rng,excluded_x_half_angle=math.radians(degrees))
        delta=(goal-start).numpy();row=direction_report(delta,degrees);rows.append(row);deltas.append(delta)
        print(json.dumps(row),flush=True)
    # Familywise 0.001 over 7 stages x 3 tests. Passing is evidence, not proof
    # that every finite random draw has equal per-bin counts.
    cutoff=.001/21
    passed=all(r['forbidden_count']==0 and min(r['chi_square_p'],r['ks_p'],r['side_binomial_p'])>cutoff for r in rows)
    (args.output/'sampling.json').write_text(json.dumps(dict(passed=passed,cutoff=cutoff,scope='production sampler, no PhysX',rows=rows),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(14,7),subplot_kw={'projection':'polar'})
    for ax,row,delta in zip(axes.flat,rows,deltas):
        a=np.arctan2(delta[:,1],delta[:,0]); ax.hist(a,bins=72,range=(-np.pi,np.pi));ax.set_title(f"deadzone +/-{row['degrees']} deg; n={row['n']}")
    axes.flat[-1].set_visible(False);fig.tight_layout();fig.savefig(args.output/'azimuth_histograms.png',dpi=160)
    if not passed:raise SystemExit('Directional uniformity audit rejected')

if __name__=='__main__':main()
