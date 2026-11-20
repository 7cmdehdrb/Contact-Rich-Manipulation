"""Render randomized eight-object shelf scenes from a level, exact front view.

No robot, policy, extension or interactive UI. Default: 24 distinct 1920x800
orthographic PNGs, a contact sheet, metadata, and a reopenable first-scene USD.
Add --occlusion for mask-verified, physically separated front/back pairs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from isaaclab.app import AppLauncher
    from cli_args import apply_renderer_compatibility

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--occlusion", action="store_true",
                        help="Place physically separated front/back objects with partial image occlusion.")
    parser.add_argument("--heavy-occlusion", action="store_true",
                        help="Require a rear object with only 0.1--10 percent of its rendered area visible.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--warmup-frames", type=int, default=24)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.heavy_occlusion:
        args.occlusion = True
    if args.count < 1 or min(args.width, args.height) < 256 or args.warmup_frames < 8:
        parser.error("Positive count, dimensions >=256, warmup >=8 required")
    args.enable_cameras = True
    apply_renderer_compatibility(args)
    app = AppLauncher(args).app
    try:
        import math
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        import isaaclab.sim as sim
        from isaaclab.scene import InteractiveScene
        from isaaclab.sensors import Camera, CameraCfg
        from inspire_sweep_baseline.assets.scenes.shelf_photography import ShelfPhotographySceneCfg
        from inspire_sweep_baseline.assets.revision_spawners import (
            REVISION_OBJECT_KINDS, RevisionObjectSpawnerCfg, spawn_revision_objects,
        )
        from inspire_sweep_baseline.assets.paths import SHELF_USD_PATH
        from inspire_sweep_baseline.mdp.common.contact_sweep_clutter_fusion import (
            SHELF_HALF_EXTENTS, SHELF_SURFACE_HEIGHT,
        )

        default_output = ("outputs/shelf_front_heavy_24" if args.heavy_occlusion else
                          "outputs/shelf_front_occluded_24" if args.occlusion else "outputs/shelf_front_24")
        output = (args.output_dir or Path(default_output)).resolve()
        output.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        simulation = sim.SimulationContext(sim.SimulationCfg(
            device=args.device or "cuda:0", gravity=(0., 0., 0.), use_fabric=False,
        ))
        scene = InteractiveScene(ShelfPhotographySceneCfg())
        stage = sim.get_current_stage()
        segmentation = "instance_id_segmentation_fast"
        camera = Camera(CameraCfg(prim_path="/World/FrontCamera", width=args.width, height=args.height,
            data_types=["rgb", segmentation] if args.occlusion else ["rgb"],
            colorize_instance_id_segmentation=False, spawn=sim.PinholeCameraCfg(lock_camera=False,
                clipping_range=(.01, 100.), focal_length=24., horizontal_aperture=36.)))
        # USD orthographic aperture units are tenths of a stage unit (metres).
        width_m = 1.16
        usd_camera = UsdGeom.Camera(stage.GetPrimAtPath("/World/FrontCamera"))
        usd_camera.CreateProjectionAttr("orthographic")
        usd_camera.CreateHorizontalApertureAttr(width_m * 10)
        usd_camera.CreateVerticalApertureAttr(width_m * args.height / args.width * 10)
        eye, target = (1.4, 0., 1.22), (-.7, 0., 1.22)
        simulation.reset()
        scene.reset()
        camera.set_world_poses_from_view(torch.tensor([eye], device=simulation.device),
                                         torch.tensor([target], device=simulation.device))
        assert eye[1:] == target[1:], "Front camera must have zero yaw/pitch"
        assert not any(p.GetName() == "Robot" for p in stage.Traverse())

        def disable_dynamics(root):
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)

        def pixel_counts(paths):
            ids = camera.data.output[segmentation][0, ..., 0].cpu().numpy()
            labels = camera.data.info[0][segmentation]["idToLabels"]
            counts = {}
            for slot, path in enumerate(paths):
                matching = [int(key) for key, value in labels.items()
                            if str(value) == path or path + "/" in str(value)]
                counts[slot] = int(np.isin(ids, matching).sum()) if matching else 0
            if not any(counts.values()):
                raise RuntimeError(f"Cannot resolve object instance IDs: {str(labels)[:1500]}")
            return counts

        records, hashes, base_bounds = [], set(), []
        for index in range(args.count):
            seed = args.seed + index
            random.seed(seed)
            rng = np.random.default_rng(seed)
            paths, local_bounds, scales = [], [], []
            for slot, kind in enumerate(REVISION_OBJECT_KINDS):
                path = f"/World/envs/env_0/PhotoObject_{slot}"
                if index == 0:
                    cfg = RevisionObjectSpawnerCfg(kind_override=kind)
                    # Preserve source shape generation, YCB fitting/textures and
                    # source random scale; keep geometry alive between images.
                    prim = spawn_revision_objects(path, cfg, translation=(0., 0., 10.))
                    disable_dynamics(prim)
                    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
                    bound = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
                    base_bounds.append(np.array([bound.GetMin(), bound.GetMax()]))
                    scale = float(prim.GetCustomDataByKey("revision_scale"))
                else:
                    prim = stage.GetPrimAtPath(path)
                    scale = float(rng.uniform(.9, 1.1))
                    prim.GetAttribute("xformOp:scale").Set(Gf.Vec3d(scale))
                    prim.SetCustomDataByKey("revision_scale", scale)
                local_bounds.append(base_bounds[slot] * scale)
                scales.append(scale)
                paths.append(path)

            # Draw source-range yaw and reject layouts outside the board.
            # The optional front/back grouping allows partial image occlusion.
            for attempt in range(1000):
                yaws = rng.uniform(-math.pi, math.pi, 8)
                rotated = []
                for bounds, yaw in zip(local_bounds, yaws):
                    points = np.array([[x, y, z] for x in bounds[:, 0]
                                       for y in bounds[:, 1] for z in bounds[:, 2]])
                    c, s = math.cos(yaw), math.sin(yaw)
                    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                    points = points @ rotation.T
                    rotated.append(np.array([points.min(0), points.max(0)]))
                widths = np.array([b[1, 1] - b[0, 1] for b in rotated])
                available = 2 * (SHELF_HALF_EXTENTS[1] - .065)
                if args.occlusion:
                    # Two, three or four occluding pairs. Larger world X is
                    # closer to the camera. Pair AABBs are separated in depth;
                    # groups are separated horizontally, so meshes never cross.
                    pair_count = 2 + index % 3
                    if args.heavy_occlusion:
                        # Broad, tall boxes can nearly cover the smaller source
                        # shapes without scaling them outside their source range.
                        fronts = list(map(int, rng.permutation([4, 5])))
                        rears = list(map(int, rng.choice([0, 1, 2], size=2, replace=False)))
                        permutation = [fronts[0], rears[0], fronts[1], rears[1]]
                        permutation += list(map(int, rng.permutation(
                            [s for s in range(8) if s not in permutation])))
                        if not all(widths[f] > widths[r] * 1.08 and
                                   rotated[f][1, 2] - rotated[f][0, 2] >
                                   (rotated[r][1, 2] - rotated[r][0, 2]) * 1.05
                                   for f, r in zip(fronts, rears)):
                            continue
                    else:
                        permutation = list(map(int, rng.permutation(8)))
                    groups, pairs = [], []
                    for pair_index in range(pair_count):
                        front, rear = permutation[2 * pair_index:2 * pair_index + 2]
                        heavy_pair = args.heavy_occlusion and pair_index < 2
                        if heavy_pair:
                            ranges = ((.006, .018), (.018, .035), (.035, .065))
                            target_visible = float(rng.uniform(*ranges[index % 3]))
                            overlap = float((1. - target_visible) * widths[rear])
                        else:
                            overlap = float(rng.uniform(.3, .72) * min(widths[front], widths[rear]))
                        separation = float((widths[front] + widths[rear]) / 2 - overlap)
                        side = float(rng.choice([-1., 1.]))
                        offsets = {front: -side * separation / 2, rear: side * separation / 2}
                        low = min(offsets[s] - widths[s] / 2 for s in offsets)
                        high = max(offsets[s] + widths[s] / 2 for s in offsets)
                        offsets = {s: y - (low + high) / 2 for s, y in offsets.items()}
                        # Reserve lateral room for mask-based fine adjustment.
                        groups.append((offsets, float(high - low) + (.04 if heavy_pair else 0.)))
                        pairs.append(dict(front_slot=front, rear_slot=rear,
                                          projected_overlap_m=overlap, heavy_target=heavy_pair))
                    groups.extend(({s: 0.}, float(widths[s])) for s in permutation[2 * pair_count:])
                    slack = available - sum(group[1] for group in groups) - .015 * (len(groups) - 1)
                    depth_ok = all(
                        sum(rotated[s][1, 0] - rotated[s][0, 0]
                            for s in (pair["front_slot"], pair["rear_slot"])) < .32
                        for pair in pairs)
                    if slack >= 0 and depth_ok:
                        break
                    continue
                slack = available - widths.sum() - 7 * .008
                if slack >= 0 and all(b[1, 0] - b[0, 0] < .34 for b in rotated):
                    break
            else:
                raise RuntimeError("Eight-object photography layout does not fit the source shelf")
            proposals = []
            if args.occlusion:
                rng.shuffle(groups)
                gaps = rng.dirichlet(np.ones(len(groups) + 1)) * slack
                cursor = -SHELF_HALF_EXTENTS[1] + .065 + gaps[0]
                front_slots = {p["front_slot"] for p in pairs}
                rear_slots = {p["rear_slot"] for p in pairs}
                for rank, (offsets, group_width) in enumerate(groups):
                    group_center = cursor + group_width / 2
                    for slot, offset in offsets.items():
                        bounds = rotated[slot]
                        depth = bounds[1, 0] - bounds[0, 0]
                        if slot in front_slots:
                            center_x = .175 - depth / 2 - float(rng.uniform(0., .008))
                        elif slot in rear_slots:
                            center_x = -.175 + depth / 2 + float(rng.uniform(0., .008))
                        else:
                            center_x = float(rng.uniform(-.175 + depth / 2, .175 - depth / 2))
                        root_x = center_x - (bounds[0, 0] + bounds[1, 0]) / 2
                        root_y = group_center + offset - (bounds[0, 1] + bounds[1, 1]) / 2
                        proposals.append((slot, (-.7 + root_x, root_y,
                                                  SHELF_SURFACE_HEIGHT - bounds[0, 2])))
                    cursor += group_width + .015 + gaps[rank + 1]
            else:
                gaps = rng.dirichlet(np.ones(9)) * slack
                order = rng.permutation(8)
                cursor = -SHELF_HALF_EXTENTS[1] + .065 + gaps[0]
                for rank, slot in enumerate(order):
                    bounds = rotated[slot]
                    center_y = cursor + widths[slot] / 2
                    x_min = -SHELF_HALF_EXTENTS[0] + .025 - bounds[0, 0]
                    x_max = SHELF_HALF_EXTENTS[0] - .025 - bounds[1, 0]
                    root_x = float(rng.uniform(x_min, x_max))
                    root_y = float(center_y - (bounds[0, 1] + bounds[1, 1]) / 2)
                    proposals.append((slot, (-.7 + root_x, root_y, SHELF_SURFACE_HEIGHT - bounds[0, 2])))
                    cursor += widths[slot] + .008 + gaps[rank + 1]
            objects = []
            for slot, position in proposals:
                bounds = rotated[slot]
                prim = stage.GetPrimAtPath(paths[slot])
                prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
                yaw = float(yaws[slot])
                prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(math.cos(yaw / 2), Gf.Vec3f(0., 0., math.sin(yaw / 2))))
                objects.append(dict(slot=int(slot), kind=REVISION_OBJECT_KINDS[slot], scale=scales[slot],
                    yaw_rad=yaw, position_m=position, bounds_world_m=(bounds + position).tolist()))
            if args.occlusion:
                for first in range(8):
                    a = np.array(objects[first]["bounds_world_m"])
                    for second in range(first):
                        b = np.array(objects[second]["bounds_world_m"])
                        if np.all(np.minimum(a[1, :2], b[1, :2]) - np.maximum(a[0, :2], b[0, :2]) > 0):
                            raise RuntimeError("Occlusion layout contains intersecting object bounds")
            # No policy/reset/physics step: render randomized static source assets.
            for _ in range(args.warmup_frames if index else max(48, args.warmup_frames)):
                simulation.render()
            camera.update(simulation.get_physics_dt(), force_recompute=True)
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
            if rgb.shape != (args.height, args.width, 3) or rgb.std() < 2:
                raise RuntimeError("Invalid/empty front photograph")
            occlusion_checks = []
            if args.occlusion:
                visible = pixel_counts(paths)
                for slot in front_slots:
                    UsdGeom.Imageable(stage.GetPrimAtPath(paths[slot])).MakeInvisible()
                for _ in range(8):
                    simulation.render()
                camera.update(simulation.get_physics_dt(), force_recompute=True)
                unobstructed = pixel_counts(paths)
                for pair in pairs:
                    rear = pair["rear_slot"]
                    total = unobstructed[rear]
                    ratio = visible[rear] / total if total else 0.
                    occlusion_checks.append(dict(**pair, visible_pixels=visible[rear],
                        unobstructed_pixels=total, visible_fraction=ratio))
                for slot in front_slots:
                    UsdGeom.Imageable(stage.GetPrimAtPath(paths[slot])).MakeVisible()
                for _ in range(8):
                    simulation.render()
                if args.heavy_occlusion:
                    by_slot = {o["slot"]: o for o in objects}
                    initial_y = {p["rear_slot"]: by_slot[p["rear_slot"]]["position_m"][1]
                                 for p in pairs if p["heavy_target"]}
                    # Scan meshes are not exact rectangles. Close the loop on
                    # actual rendered masks, keeping a thin visible sliver.
                    for adjustment in range(12):
                        bad = [c for c in occlusion_checks if c["heavy_target"] and
                               not .005 <= c["visible_fraction"] <= .08]
                        if not bad:
                            break
                        for check in bad:
                            rear, front = check["rear_slot"], check["front_slot"]
                            obj = by_slot[rear]
                            rear_box = np.array(obj["bounds_world_m"])
                            front_box = np.array(by_slot[front]["bounds_world_m"])
                            side = 1. if rear_box[:, 1].mean() > front_box[:, 1].mean() else -1.
                            desired = (.015, .025, .045)[index % 3]
                            delta = side * float(np.clip((desired - check["visible_fraction"]) *
                                                       widths[rear], -.006, .006))
                            old_y = obj["position_m"][1]
                            new_y = float(np.clip(old_y + delta, initial_y[rear] - .018,
                                                  initial_y[rear] + .018))
                            delta = new_y - old_y
                            obj["position_m"] = (obj["position_m"][0], new_y, obj["position_m"][2])
                            obj["bounds_world_m"] = (rear_box + np.array([0., delta, 0.])).tolist()
                            stage.GetPrimAtPath(paths[rear]).GetAttribute("xformOp:translate").Set(
                                Gf.Vec3d(*obj["position_m"]))
                        for _ in range(8):
                            simulation.render()
                        camera.update(simulation.get_physics_dt(), force_recompute=True)
                        visible = pixel_counts(paths)
                        for slot in front_slots:
                            UsdGeom.Imageable(stage.GetPrimAtPath(paths[slot])).MakeInvisible()
                        for _ in range(8):
                            simulation.render()
                        camera.update(simulation.get_physics_dt(), force_recompute=True)
                        unobstructed = pixel_counts(paths)
                        occlusion_checks = []
                        for pair in pairs:
                            rear = pair["rear_slot"]
                            total = unobstructed[rear]
                            occlusion_checks.append(dict(**pair, visible_pixels=visible[rear],
                                unobstructed_pixels=total, visible_fraction=visible[rear] / total if total else 0.))
                        for slot in front_slots:
                            UsdGeom.Imageable(stage.GetPrimAtPath(paths[slot])).MakeVisible()
                        for _ in range(8):
                            simulation.render()
                    camera.update(simulation.get_physics_dt(), force_recompute=True)
                    rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
                    for check in occlusion_checks:
                        rear_box = np.array(by_slot[check["rear_slot"]]["bounds_world_m"])
                        front_box = np.array(by_slot[check["front_slot"]]["bounds_world_m"])
                        check["projected_overlap_m"] = float(max(0., min(rear_box[1, 1], front_box[1, 1]) -
                                                                    max(rear_box[0, 1], front_box[0, 1])))
                    for first, obj in enumerate(objects):
                        a = np.array(obj["bounds_world_m"])
                        for other in objects[:first]:
                            b = np.array(other["bounds_world_m"])
                            if np.all(np.minimum(a[1, :2], b[1, :2]) - np.maximum(a[0, :2], b[0, :2]) > 0):
                                raise RuntimeError("Heavy-occlusion adjustment intersects object bounds")
                partial_minimum = .001 if args.heavy_occlusion else .03
                if not any(partial_minimum < check["visible_fraction"] < .97 for check in occlusion_checks):
                    raise RuntimeError(f"Rendered image has no confirmed partial occlusion: {occlusion_checks}")
                if args.heavy_occlusion and not any(
                        .001 < check["visible_fraction"] <= .10 for check in occlusion_checks
                        if check["heavy_target"]):
                    raise RuntimeError(f"Rendered image has no nearly hidden rear object: {occlusion_checks}")
                if any(count < (20 if args.heavy_occlusion else 100) for count in visible.values()):
                    raise RuntimeError(f"An object is completely hidden: {visible}")
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in hashes:
                raise RuntimeError("Repeated image detected")
            hashes.add(digest)
            prefix = "front_heavy" if args.heavy_occlusion else "front_occluded" if args.occlusion else "front"
            filename = f"{prefix}_{index + 1:03d}.png"
            Image.fromarray(rgb).save(output / filename)
            records.append(dict(file=filename, seed=seed, rgb_sha256=digest, objects=objects,
                                occlusion_checks=occlusion_checks))
            if index == 0:
                snapshot = Usd.Stage.Open(stage.Flatten())
                snapshot.SetDefaultPrim(snapshot.GetPrimAtPath("/World"))
                disable_dynamics(snapshot.GetPrimAtPath("/World"))
                snapshot.GetRootLayer().Export(str(output / "front_scene_001.usdc"))
            print(f"[Front] {index + 1}/{args.count}: {filename}, objects=8, seed={seed}", flush=True)

        metadata = dict(count=len(records), objects_per_image=8, kinds=REVISION_OBJECT_KINDS,
            shelf_usd=SHELF_USD_PATH, resolution=[args.width, args.height],
            camera=dict(projection="orthographic", eye_m=eye, target_m=target,
                        horizontal_width_m=width_m, pitch_deg=0., yaw_deg=0., roll_deg=0.),
            layout="heavy_occlusion" if args.heavy_occlusion else "partial_occlusion" if args.occlusion else "separated",
            randomization="source 8 kinds, uniform scale .9--1.1 and yaw; randomized positions; " +
                ("2--4 physically separated front/back pairs; mask-verified rear visible area <=10%" if args.heavy_occlusion else
                 "2--4 physically separated front/back pairs with mask-verified partial occlusion" if args.occlusion
                 else "randomized horizontal order/gaps conditioned on eight visible silhouettes"),
            robot=False, policy=False, physics_steps=0, images=records)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        columns, thumb_w = 4, 480
        thumb_h = round(thumb_w * args.height / args.width)
        rows = math.ceil(len(records) / columns)
        sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + 24)), "#202328")
        draw = ImageDraw.Draw(sheet)
        for index, record in enumerate(records):
            thumb = Image.open(output / record["file"]).convert("RGB")
            thumb.thumbnail((thumb_w, thumb_h))
            x, y = (index % columns) * thumb_w, (index // columns) * (thumb_h + 24)
            sheet.paste(thumb, (x, y))
            draw.text((x + 8, y + thumb_h + 3), record["file"], fill="white")
        sheet.save(output / "contact_sheet.jpg", quality=92)
        print(f"[Front] Complete: {output}; unique RGB hashes={len(hashes)}", flush=True)
    except BaseException:
        # Immediate Kit shutdown may exit before Python prints an uncaught
        # exception, so preserve the error in the render log first.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
