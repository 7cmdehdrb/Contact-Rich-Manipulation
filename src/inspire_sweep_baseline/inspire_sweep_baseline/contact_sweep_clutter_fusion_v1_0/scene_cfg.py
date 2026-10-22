"""Add whole-hand and object-specific contacts while preserving tactile input."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ..assets.robot import HAND_CONTACT_BODY_NAMES
from ..assets.tactile_regions import TACTILE_PARENT_BODY_NAMES, parent_sensor_name
from ..contact_sweep_clutter_fusion.scene_cfg import ContactSweepClutterFusionSceneCfg
from .contacts import object_hand_sensor_name
from .population import PARKING_HEIGHT, validate_capacity


@configclass
class ContactSweepClutterFusionV1_0SceneCfg(ContactSweepClutterFusionSceneCfg):
    whole_hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(" + "|".join(HAND_CONTACT_BODY_NAMES) + ")",
        update_period=0.0, history_length=0, force_threshold=0.01, debug_vis=False,
    )

    def __post_init__(self):
        super().__post_init__()
        self.configure_population_capacity(max_gray_obstacles=3, max_white_obstacles=3)

    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles):
        """Build fixed gray/white slots before simulation starts.

        Counts live on EnvCfg because InteractiveScene treats scalar scene
        fields as assets. The runtime calls this again after Hydra overrides.
        """
        validate_capacity(max_gray_obstacles, max_white_obstacles)
        for name in tuple(vars(self)):
            if name.startswith("hand_contact_"):
                delattr(self, name)
        target = self.object_collection.rigid_objects["mass_cylinder"].copy()
        objects = {"mass_cylinder": target}
        for index in range(max_gray_obstacles + max_white_obstacles):
            obstacle = target.copy()
            obstacle.prim_path = f"{{ENV_REGEX_NS}}/obstacle_{index}"
            # Slots remain stable for the entire run: gray capacity first,
            # then white capacity, irrespective of the current active counts.
            color = (0.45, 0.45, 0.45) if index < max_gray_obstacles else (0.90, 0.90, 0.90)
            obstacle.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            obstacle.init_state.pos = (-0.7, 0.2 * index, PARKING_HEIGHT)
            objects[f"obstacle_{index}"] = obstacle
        self.object_collection.rigid_objects = objects
        paths = [obj.prim_path for obj in objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            tactile = getattr(self, parent_sensor_name(parent))
            tactile.filter_prim_paths_expr = paths.copy()
            tactile.max_contact_data_count_per_prim = 16 * len(paths)
        # Reverse the contact query: each cylinder is ONE reporting rigid body
        # filtered against all hand bodies. One view per object suffices, rather than a
        # separate filtered sensor for every hand link/pad. Explicit paths select
        # the 13 hand links plus all 17 tactile pads, excluding the arm and Axia80
        # housing. Existing tactile sensors retain their own filters.
        for name, obj in self.object_collection.rigid_objects.items():
            obj.spawn.activate_contact_sensors = True
            setattr(self, object_hand_sensor_name(name), ContactSensorCfg(
                prim_path=obj.prim_path,
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + body for body in HAND_CONTACT_BODY_NAMES],
                update_period=0.0, history_length=0, debug_vis=False,
                max_contact_data_count_per_prim=16 * len(HAND_CONTACT_BODY_NAMES),
            ))
