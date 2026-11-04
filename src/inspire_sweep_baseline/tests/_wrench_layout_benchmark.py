"""CPU/GPU pure-layout benchmark; this does NOT measure Isaac collection time."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import types
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--num-envs', type=int, default=2048)
parser.add_argument('--bank-size', type=int, default=2)
parser.add_argument('--device', default='cpu')
parser.add_argument('--all-stages', action='store_true')
parser.add_argument('--output', type=Path, default=Path('outputs/wrench_validation/layout_benchmark.json'))
args = parser.parse_args()
torch.set_num_threads(1)
package = types.ModuleType('_wrench_benchmark')
package.__path__ = [str(Path(__file__).resolve().parents[1]/'inspire_sweep_baseline/mdp/common')]
sys.modules[package.__name__] = package
old = importlib.import_module('_wrench_benchmark.revision_layout')
new = importlib.import_module('_wrench_benchmark.wrench_layout')
bounds = torch.zeros(args.num_envs,10,2,3,device=args.device)
half = torch.linspace(.025,.055,20,device=args.device).reshape(10,2)
bounds[:,:,0,:2], bounds[:,:,1,:2] = -half, half
bounds[:,:,1,2] = .12
rng = torch.Generator(device=args.device).manual_seed(42)
def timed(fn):
    if args.device.startswith('cuda'): torch.cuda.synchronize()
    started = time.perf_counter()
    value = fn()
    if args.device.startswith('cuda'): torch.cuda.synchronize()
    return time.perf_counter()-started,value
old_s,_ = timed(lambda: old.revision_layout(bounds,rng,3,3))
new_s,_ = timed(lambda: new.wrench_layout(bounds,rng,3,3))
stages = new.LAYOUT_STAGES if args.all_stages else ((30,3,3),)
init_s,bank = timed(lambda: new.WrenchLayoutBank(bounds,rng,size=args.bank_size,stages=stages))
ids = torch.arange(args.num_envs,device=args.device)
samples = [timed(lambda: bank.sample(ids,rng,3,3,torch.pi/6))[0] for _ in range(20)]
report = dict(device=args.device,num_envs=args.num_envs,bank_size=args.bank_size,stages=stages,
              old_layout_s=old_s,new_layout_s=new_s,bank_initialize_s=init_s,
              bank_sample_mean_s=sum(samples)/len(samples),bank_sample_max_s=max(samples),
              bank_bytes=sum(x.numel()*x.element_size() for values in bank.banks.values() for x in values),
              complete_fraction={str(k):v[-1].float().mean().item() for k,v in bank.banks.items()},
              scope='Pure tensor layout only; excludes PhysX, observations, policy and PPO collection')
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
