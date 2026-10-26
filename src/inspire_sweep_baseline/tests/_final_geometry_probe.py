"""CPU stress of Final layout fallback; this does not certify robot dynamics."""
import json
from pathlib import Path
import sys
import time
from collections import Counter

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Skip only the asset package initializer, which imports simulator USD APIs.
# The geometry modules below execute unmodified.
from types import ModuleType
assets = ModuleType('inspire_sweep_baseline.assets')
assets.__path__ = [str(Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline/assets')]
sys.modules[assets.__name__] = assets
from inspire_sweep_baseline.mdp.common.final_geometry import sample_target_goal, sample_population, valid_layout

started=time.perf_counter()
report={}
for wide in (False, True):
    for requested in ((0,0),(1,1),(8,8),(10,10)):
        actual=Counter()
        for seed in range(8):
            rng=np.random.default_rng(seed)
            # Includes the widest new mug at +10% and enclosing scanned boxes.
            half=rng.uniform(.025,.047,(21,2))*rng.uniform(.9,1.1,(21,1))
            half[0]=np.array([.054,.035])*1.1 if seed%2 else [.044,.044]
            target,goal,config=sample_target_goal(rng,half[0],wide,.02,.15,.02,np.pi/6)
            centers,yaw,gray,white=sample_population(rng,target,goal,half,10,*requested,config,.15)
            indices=[0,*range(1,1+gray),*range(11,11+white)]
            assert valid_layout(np.concatenate((target[None],centers)),yaw[indices],half[indices])
            assert np.all(np.abs(goal)+half[0]+.02 <= [.20,.50])
            assert .05-1e-8 <= np.linalg.norm(goal-target) <= .15+1e-8
            actual[f'{gray}/{white}']+=1
        report[f'v{2 if wide else 1}:requested={requested[0]}/{requested[1]}']=dict(actual)
report['seconds']=time.perf_counter()-started
report['layouts']=64
out=Path('outputs/final_versions_validation')
out.mkdir(parents=True,exist_ok=True)
(out/'geometry_stress.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
