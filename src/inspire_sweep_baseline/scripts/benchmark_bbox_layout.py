"""Standalone CPU/CUDA reset-layout benchmark and gallery. No training or Isaac Sim."""
import argparse
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import torch

MODULE = Path(__file__).resolve().parents[1] / 'inspire_sweep_baseline/mdp/common/bbox_scene_layout.py'
spec = importlib.util.spec_from_file_location('bbox_scene_layout_benchmark', MODULE)
layout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = layout
spec.loader.exec_module(layout)


def make_inputs(batch, cfg, device):
    rng = torch.Generator(device=device).manual_seed(118)
    target = torch.tensor([0., .23], device=device).expand(batch, -1).clone()
    angle = torch.arange(batch, device=device) * 2.399963
    goal = target + .09 * torch.stack((angle.cos(), angle.sin()), -1)
    th = torch.tensor([.025, .04], device=device).expand(batch, -1).clone()
    ty = angle
    half = .012 + .025 * torch.rand(batch, cfg.max_gray+cfg.max_white, 2, device=device, generator=rng)
    return target, goal, th, ty, half


def run_case(batch, device, cfg, repeats):
    args = make_inputs(batch, cfg, device)
    rng = torch.Generator(device=device).manual_seed(310)
    for _ in range(2):
        result = layout.generate_layout(*args, cfg=cfg, generator=rng, check_inputs=False)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = layout.generate_layout(*args, cfg=cfg, generator=rng, check_inputs=False)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter()-start)*1000)
    counts = result.actual.float().mean(0).cpu().tolist()
    return dict(device=device, batch=batch, config=asdict(cfg), repeats=repeats,
                median_ms=sorted(elapsed)[len(elapsed)//2], min_ms=min(elapsed), max_ms=max(elapsed),
                mean_gray=counts[0], mean_white=counts[1],
                full_request_fraction=result.complete.float().mean().item(),
                peak_allocated_mb=(torch.cuda.max_memory_allocated()/1024**2 if device.startswith('cuda') else None))


def gallery(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon
    import numpy as np
    cfg = layout.LayoutConfig(max_gray=4, max_white=5)
    args = make_inputs(6, cfg, 'cpu')
    result = layout.generate_layout(*args, cfg=cfg, generator=torch.Generator().manual_seed(319))
    x, y = torch.linspace(-.5, .5, 260), torch.linspace(0., .4, 120)
    gy, gx = torch.meshgrid(y, x, indexing='ij')
    points = torch.stack((gx.flatten(), gy.flatten()), -1)[None].expand(6, -1, -1)
    red, gray, _ = layout.zone_masks(points, torch.zeros_like(points), torch.zeros(points.shape[:2]), *args[:4], cfg)
    colors = torch.where(red, 0, torch.where(gray, 1, 2)).reshape(6, len(y), len(x))
    def box(center, half, yaw):
        corners = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]) * half
        return (corners @ layout.axes(yaw) + center).numpy()
    fig, plots = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    for i, ax in enumerate(plots.flat):
        ax.pcolormesh(x, y, colors[i], cmap=ListedColormap(['#f6c5c5', '#ceced3', '#f6f8fa']), shading='auto', vmin=0, vmax=2)
        ax.add_patch(Polygon(box(args[0][i], args[2][i], args[3][i]), facecolor='#ca3b43', edgecolor='#79151a'))
        ax.add_patch(Polygon(box(args[1][i], args[2][i], args[3][i]), fill=False, edgecolor='#ca3b43', linestyle='--'))
        ax.annotate('', xy=args[1][i], xytext=args[0][i], arrowprops=dict(arrowstyle='->', color='#9b1721'))
        for j in torch.nonzero(result.active[i]).flatten().tolist():
            ax.add_patch(Polygon(box(result.centers[i, j], args[4][i, j], result.yaw[i, j]),
                                 facecolor='#566780' if j < cfg.max_gray else '#e6ad4b', edgecolor='#263244'))
        counts = result.actual[i].tolist()
        ax.set(title=f'Scene {i+1}: Gray {counts[0]}/4, White {counts[1]}/5', xlabel='x (m)', ylabel='y (m)')
        ax.set_aspect('equal')
        ax.set_xlim(-.5, .5)
        ax.set_ylim(0, .4)
    fig.suptitle('OBB placement: 20 mm surface gap, 60 mm Gray padding\nBackground zones show point footprints; whole OBBs are tested for placement', fontsize=13)
    fig.savefig(output/'gallery.png', dpi=160)
    plt.close(fig)
    np.savez(output/'gallery.npz', target=args[0].numpy(), goal=args[1].numpy(), target_half=args[2].numpy(),
             target_yaw=args[3].numpy(), obstacle_half=args[4].numpy(), centers=result.centers.numpy(),
             yaw=result.yaw.numpy(), active=result.active.numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/bbox_scene_layout'))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = layout.LayoutConfig()
    cases = [('cpu', 1, cfg), ('cpu', 64, cfg), ('cpu', 512, cfg)]
    if torch.cuda.is_available():
        cases += [('cuda', n, cfg) for n in (1, 64, 512, 2048)]
        cases += [('cuda', 512, replace(cfg, restarts=1, candidates=16, passes=1)),
                  ('cuda', 512, replace(cfg, restarts=8, candidates=64, passes=2)),
                  ('cuda', 512, replace(cfg, max_gray=10, max_white=10))]
    report = dict(platform=platform.platform(), torch=torch.__version__, cpu_threads=args.threads,
                  gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                  scope='layout search only; CUDA synchronized; excludes physics, assets and validation', cases=[])
    for device, batch, config in cases:
        case = run_case(batch, device, config, args.repeats)
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
        (args.output/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    gallery(args.output)


if __name__ == '__main__':
    main()
