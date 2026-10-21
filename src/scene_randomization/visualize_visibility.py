"""Generate the original scenes, classify visibility, benchmark and visualize.

From repository root: PYTHONPATH=src python -m scene_randomization.visualize_visibility
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import csv
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .generator import Config, generate_parallel
from .validation import validate_scenes
from .visibility import VisibilityResult, classify_visibility, classify_visibility_torch


def benchmark(call, synchronize, repeats):
    durations = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        call()
        synchronize()
        durations.append(time.perf_counter() - started)
    return {"repeats":repeats, "median_seconds":float(np.median(durations)),
            "p95_seconds":float(np.percentile(durations,95)), "min_seconds":min(durations)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20261013)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--config",type=Path,help="Existing generator Config JSON overrides")
    parser.add_argument("--output",type=Path,default=Path("outputs/scene_visibility"))
    parser.add_argument("--threshold",type=float,default=.35)
    parser.add_argument("--backend",choices=("numpy","torch"),default="numpy")
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--dtype",choices=("float32","float64"),default="float64")
    parser.add_argument("--torch-threads",type=int,default=None)
    parser.add_argument("--repeats",type=int,default=30)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--plots",type=int,default=9)
    parser.add_argument("--no-plots",action="store_true")
    args = parser.parse_args()
    if min(args.count,args.workers,args.repeats,args.plots)<1 or args.warmup<0:
        parser.error("count/workers/repeats/plots must be positive and warmup nonnegative")
    if not np.isfinite(args.threshold) or not 0<args.threshold<=1:
        parser.error("threshold must be in (0,1]")
    if args.torch_threads is not None and args.torch_threads<1:
        parser.error("torch-threads must be positive")
    if args.backend=="numpy" and args.device!="cpu":
        parser.error("NumPy backend runs on cpu; select --backend torch for CUDA")
    c = Config(**(json.loads(args.config.read_text()) if args.config else {}))
    c.check()
    args.output.mkdir(parents=True,exist_ok=False)
    report = {"status":"running", "seed":args.seed, "count":args.count, "workers":args.workers,
              "config":asdict(c), "backend":args.backend, "device":args.device, "dtype":args.dtype,
              "threshold":args.threshold, "python":platform.python_version(), "numpy":np.__version__,
              "model":"orthographic_front_equal_height_disjoint_cylinders",
              "view_direction":[0,1], "object_order":"T, O1, O2, ... (target then generator obstacles)",
              "timing_scope":"generation excludes geometry audit, classification, plotting and I/O; "
                             "hot decision excludes input audit, transfer and I/O; CUDA calls synchronize",
              "warmup_calls":args.warmup}
    report_path = args.output/"report.json"
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    try:
        started=time.perf_counter()
        batch=generate_parallel(args.count,c,args.seed,args.workers)
        report["generation_seconds"]=time.perf_counter()-started
        started=time.perf_counter()
        report["geometry"]=validate_scenes(batch)
        report["geometry_validation_seconds"]=time.perf_counter()-started
        if not report["geometry"]["valid"]:
            raise RuntimeError("Generated scene geometry did not pass independent validation")

        started=time.perf_counter()
        centers=np.concatenate((batch.target[:,None,:],batch.obstacles),axis=1).astype(args.dtype)
        radii=c.radius
        synchronize=lambda:None
        context=nullcontext()
        function=classify_visibility
        if args.backend=="torch":
            import torch
            if args.torch_threads is not None:torch.set_num_threads(args.torch_threads)
            device=torch.device(args.device)
            if device.type=="cuda":
                if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; use cpu or expose the NVIDIA device")
                synchronize=lambda:torch.cuda.synchronize(device)
                report["device_name"]=torch.cuda.get_device_name(device)
            report.update(torch=torch.__version__,torch_threads=torch.get_num_threads())
            centers=torch.as_tensor(centers,device=device)
            radii=torch.full(centers.shape[:2],c.radius,dtype=centers.dtype,device=device)
            synchronize()
            function=classify_visibility_torch
            context=torch.inference_mode()
        report["input_preparation_seconds"]=time.perf_counter()-started
        with context:
            synchronize();started=time.perf_counter()
            result=function(centers,radii,threshold=args.threshold,check_inputs=True)
            synchronize()
            report["first_validated_decision_seconds"]=time.perf_counter()-started
            call=lambda:function(centers,radii,threshold=args.threshold,check_inputs=False)
            for _ in range(args.warmup):call()
            report["batch_decision"]=benchmark(call,synchronize,args.repeats)
            single_radii=radii[:1] if args.backend=="torch" else radii
            single_call=lambda:function(centers[:1],single_radii,threshold=args.threshold,check_inputs=False)
            for _ in range(args.warmup):single_call()
            report["single_scene_decision"]=benchmark(single_call,synchronize,args.repeats)

        # Device-to-host conversion for plotting is explicitly outside timing.
        def host(value):
            return value.detach().cpu().numpy() if hasattr(value,"detach") else value
        result=VisibilityResult(**{f.name:host(getattr(result,f.name)) for f in fields(result)})
        report["generation_amortized_ms_per_scene"]=1000*report["generation_seconds"]/args.count
        median=report["batch_decision"]["median_seconds"]
        report["decision_amortized_ms_per_scene"]=1000*median/args.count
        report["visible_objects"]=int(result.visible.sum())
        report["not_visible_objects"]=int(result.not_visible.sum())
        report["objects_per_scene"]=centers.shape[1]
        report["detail_scene_index"]=int(np.argmax(result.not_visible.sum(axis=1)))
        started=time.perf_counter()
        batch.save(str(args.output/"scenes.npz"))
        np.savez_compressed(args.output/"visibility.npz",ratio=result.ratio,visible=result.visible,
                            not_visible=result.not_visible,visible_width=result.visible_width,
                            edges=result.edges,owners=result.owners,threshold=result.threshold)
        with (args.output/"visibility.csv").open("w",newline="") as file:
            writer=csv.writer(file)
            writer.writerow(["scene_id","object_id","name","x_cm","y_cm","visible_ratio","recognized"])
            all_centers=np.concatenate((batch.target[:,None,:],batch.obstacles),axis=1)
            for scene in range(args.count):
                for obj in range(all_centers.shape[1]):
                    writer.writerow([scene,obj,"T" if obj==0 else f"O{obj}",*all_centers[scene,obj],
                                     result.ratio[scene,obj],bool(result.visible[scene,obj])])
        report["serialization_seconds"]=time.perf_counter()-started
        report["plot_seconds"]=None
        if not args.no_plots:
            started=time.perf_counter()
            os.environ.setdefault("MPLCONFIGDIR",str(args.output.resolve()/".matplotlib"))
            import matplotlib
            matplotlib.use("Agg")
            from .visibility_plotting import save_visibility_detail,save_visibility_gallery
            timing=(f"Generation {report['generation_seconds']:.4f} s / {args.count:,} scenes "
                    f"({report['generation_amortized_ms_per_scene']:.4f} ms/scene amortized) | "
                    f"Decision [{args.backend}:{args.device}] batch median {1000*median:.3f} ms | "
                    f"Single scene median {1e6*report['single_scene_decision']['median_seconds']:.1f} us")
            report["gallery_scene_indices"]=save_visibility_gallery(batch,result,args.output/"visibility_gallery.png",args.plots,timing)
            save_visibility_detail(batch,result,args.output/"visibility_scene.png",report["detail_scene_index"],timing)
            report["plot_seconds"]=time.perf_counter()-started
        report["status"]="complete"
    except BaseException as exc:
        report.update(status="failed",error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Generation: {report['generation_seconds']:.6f} s / {args.count:,} scenes; "
          f"{report['generation_amortized_ms_per_scene']:.6f} ms/scene amortized")
    print(f"First validated decision: {1000*report['first_validated_decision_seconds']:.3f} ms")
    print(f"Decision batch median: {1000*median:.3f} ms; "
          f"{report['decision_amortized_ms_per_scene']:.6f} ms/scene amortized")
    print(f"Single-scene decision median: {1e6*report['single_scene_decision']['median_seconds']:.3f} us")
    print(f"Seen/not seen: {report['visible_objects']}/{report['not_visible_objects']}; output: {args.output.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
