"""Identical cylinders with a distinct obstacle color and shared tactile filters."""

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ..assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name
from ..contact_sweep.scene_cfg import ContactSweepSceneCfg
from .layout import DEFAULT_LAYOUT_CONFIG


@configclass
class ContactSweepClutterSceneCfg(ContactSweepSceneCfg):
    def __post_init__(self):
        super().__post_init__()
        target = self.object_collection.rigid_objects["mass_cylinder"]
        # Three actual obstacles; seven observation slots are padding,
        # never physical scene objects.
        for index in range(DEFAULT_LAYOUT_CONFIG.obstacles):
            obstacle = target.copy()
            obstacle.prim_path = f"{{ENV_REGEX_NS}}/obstacle_{index}"
            obstacle.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.35, 0.10),
            )
            # Non-overlapping construction poses, replaced before any rollout.
            obstacle.init_state.pos = (-0.82 + 0.12 * (index % 3), -0.30 if index < 3 else 0.30, 1.05)
            self.object_collection.rigid_objects[f"obstacle_{index}"] = obstacle
        paths = [obj.prim_path for obj in self.object_collection.rigid_objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            sensor = getattr(self, parent_sensor_name(parent))
            sensor.filter_prim_paths_expr = paths.copy()
            # Capacity is per hand body, covering contact with all real objects.
            sensor.max_contact_data_count_per_prim = 16 * len(paths)
