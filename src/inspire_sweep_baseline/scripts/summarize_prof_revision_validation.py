"""Audit actual train.py collection logs, TensorBoard scalars and checkpoints."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(log, runs, expected):
    text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text())
    iterations = [int(x) for x in re.findall(r'Learning iteration\s+(\d+)/', text)]
    times = [float(x) for x in re.findall(r'Collection time:\s+([\d.]+)s', text)]
    assert iterations == list(range(expected)), iterations
    assert len(times) == expected and 'Training time:' in text
    assert 'Traceback' not in text and 'NumericalTrainingError' not in text
    events = list(runs.rglob('events.out.tfevents.*'))
    assert events, f'No TensorBoard event file below {runs}'
    scalar_ranges = {}
    for event in events:
        accumulator = EventAccumulator(str(event), size_guidance={'scalars':0})
        accumulator.Reload()
        for tag in accumulator.Tags()['scalars']:
            values = [e.value for e in accumulator.Scalars(tag)]
            assert all(math.isfinite(v) for v in values),tag
            if tag.startswith(('Loss/','Policy/','ProfRevision/','Reset/','Metrics/')):
                previous = scalar_ranges.get(tag)
                lo, hi = min(values), max(values)
                scalar_ranges[tag] = [min(previous[0],lo),max(previous[1],hi)] if previous else [lo,hi]
    assert any(tag.startswith('Loss/') for tag in scalar_ranges)
    for tag,(_,hi) in scalar_ranges.items():
        if tag in ('Loss/value','Loss/value_function'):
            assert hi <= 1e6,(tag,hi)
    checkpoints = list(runs.rglob(f'model_{expected-1}.pt'))
    assert len(checkpoints) == 1, checkpoints
    checkpoint = torch.load(checkpoints[0],map_location='cpu',weights_only=False)
    def finite(value):
        if isinstance(value,torch.Tensor):
            assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for v in value.values():finite(v)
        elif isinstance(value,(tuple,list)):
            for v in value:finite(v)
        elif isinstance(value,float):
            assert math.isfinite(value)
    finite(checkpoint)
    assert not list(runs.rglob('numerical_failure.*'))
    # Preserve upper rounding error for a strict boundary verdict.
    worst = max(times)+.0005
    result = dict(iterations=iterations,collection_seconds=times,
        collection_mean=statistics.mean(times),collection_median=statistics.median(times),
        collection_p95=sorted(times)[math.ceil(.95*len(times))-1],collection_max=max(times),
        all_below_10_seconds=worst<10.,all_at_most_5_seconds=worst<=5.,
        scalar_ranges=scalar_ranges,checkpoint=str(checkpoints[0]),checkpoint_finite=True)
    first = list(runs.rglob('model_0.pt'))
    if first:
        initial = torch.load(first[0],map_location='cpu',weights_only=False)
        changes = {}
        for kind in ('actor_state_dict','critic_state_dict'):
            model = checkpoint[kind]
            names = [name for name in model if name.startswith('obstacle_encoder.') and name.endswith('weight')]
            assert names
            changes[kind] = any(not torch.equal(model[name],initial[kind][name]) for name in names)
            assert changes[kind]
        result['encoder_weights_changed'] = changes
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--baseline-log',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=32)
    parser.add_argument('--require-performance',action='store_true')
    args = parser.parse_args()
    result = summarize(args.log,args.runs,args.iterations)
    if args.baseline_log:
        values = [float(x) for x in re.findall(r'Collection time:\s+([\d.]+)s',args.baseline_log.read_text())]
        assert len(values)==args.iterations
        result['baseline'] = dict(collection_mean=statistics.mean(values),collection_max=max(values),
            mean_difference_seconds=result['collection_mean']-statistics.mean(values))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('scalar_ranges','collection_seconds','iterations')},indent=2))
    if args.require_performance:
        assert result['all_below_10_seconds'],'Collection performance gate not met'
