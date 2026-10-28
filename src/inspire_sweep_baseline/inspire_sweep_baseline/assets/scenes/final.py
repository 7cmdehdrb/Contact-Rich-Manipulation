"""Independent Final v0 scene with the ObjectRandom v3 asset and sensor layout."""

from ...mdp.common.contact_sweep_clutter_fusion import OBJECT_MASS
from ...mdp.common.contact_sweep_clutter_fusion_contacts import object_hand_sensor_name
from ...mdp.common.contact_sweep_clutter_fusion_population import PARKING_HEIGHT
from ...mdp.common.contact_sweep_clutter_fusion_population import validate_capacity
from ..paths import SHELF_USD_PATH
from ..realistic_spawners import RealisticCylinderSpawnerCfg
from ..realistic_spawners import RealisticShelfSpawnerCfg
from ..robot import HAND_CONTACT_BODY_NAMES
from ..robot import PALM_SENSOR_BODY_NAMES
from ..robot import make_robot_cfg
from ..tactile_regions import TACTILE_PARENT_BODY_NAMES
from ..tactile_regions import parent_sensor_name
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from ..contact_sweep_clutter_fusion_holding_osc_object_random_spawners import prop_variants
from .contact_sweep_clutter_fusion import _tool_frame


def _target_spawn():
    """Fresh target physics template, also used for randomized prop variants."""
    return RealisticCylinderSpawnerCfg(
        mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0, max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0, disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.001, rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 0.8)),
    )


@configclass
class FinalSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path='/World/light',
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    shelf = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/Shelf',
        spawn=RealisticShelfSpawnerCfg(
            usd_path=SHELF_USD_PATH,
            mass_props=sim_utils.MassPropertiesCfg(mass=100),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 0.0)),
    )
    robot = make_robot_cfg().replace(prim_path='{ENV_REGEX_NS}/Robot')
    object_collection = RigidObjectCollectionCfg(
        rigid_objects={
            'mass_cylinder': RigidObjectCfg(
                prim_path='{ENV_REGEX_NS}/mass_cylinder',
                init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.7, 0.0, 1.05)),
                spawn=_target_spawn(),
            ),
        },
    )
    ee_frame = _tool_frame(
        (('end_effector', (0.0, 0.0, 0.13 + robot.spawn.sensor_height / 2)),),
        body='axia80_link',
        debug_vis=True,
    )
    finger_frame = _tool_frame((('l_finger', (0.07, 0.0, 0.13)), ('r_finger', (-0.07, 0.0, 0.13))))
    wrist_frame = _tool_frame((('wrist', (0.0, 0.0, -0.17)),))
    palm_tactile = None
    hand_tactile = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/(' + '|'.join(PALM_SENSOR_BODY_NAMES) + ')',
        update_period=0.0,
        history_length=0,
        force_threshold=0.01,
        debug_vis=False,
    )
    whole_hand_contact = ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/(' + '|'.join(HAND_CONTACT_BODY_NAMES) + ')',
        update_period=0.0,
        history_length=0,
        force_threshold=0.01,
        debug_vis=False,
    )
    replicate_physics: bool = False

    def __post_init__(self):
        super().__post_init__()
        paths = [obj.prim_path for obj in self.object_collection.rigid_objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            setattr(
                self,
                parent_sensor_name(parent),
                ContactSensorCfg(
                    prim_path='{ENV_REGEX_NS}/Robot/' + parent,
                    filter_prim_paths_expr=paths.copy(),
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    max_contact_data_count_per_prim=16 * len(paths),
                ),
            )
        self.configure_population_capacity(8, 8, 16)

    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=16):
        """Build fixed gray/white slots before simulation starts.

        Counts live on EnvCfg because InteractiveScene treats scalar scene
        fields as assets. The runtime calls this again after Hydra overrides.
        """
        validate_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)
        for name in tuple(vars(self)):
            if name.startswith('hand_contact_'):
                delattr(self, name)
        target = self.object_collection.rigid_objects['mass_cylinder'].copy()
        objects = {'mass_cylinder': target}
        for index in range(max_gray_obstacles + max_white_obstacles):
            obstacle = target.copy()
            obstacle.prim_path = f'{{ENV_REGEX_NS}}/obstacle_{index}'
            color = (0.45, 0.45, 0.45) if index < max_gray_obstacles else (0.9, 0.9, 0.9)
            obstacle.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            obstacle.init_state.pos = (-0.7, 0.2 * index, PARKING_HEIGHT)
            objects[f'obstacle_{index}'] = obstacle
        self.object_collection.rigid_objects = objects
        paths = [obj.prim_path for obj in objects.values()]
        for parent in TACTILE_PARENT_BODY_NAMES:
            tactile = getattr(self, parent_sensor_name(parent))
            tactile.filter_prim_paths_expr = paths.copy()
            tactile.max_contact_data_count_per_prim = 16 * len(paths)
        for name, obj in self.object_collection.rigid_objects.items():
            obj.spawn.activate_contact_sensors = True
            setattr(
                self,
                object_hand_sensor_name(name),
                ContactSensorCfg(
                    prim_path=obj.prim_path,
                    filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/' + body for body in HAND_CONTACT_BODY_NAMES],
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    max_contact_data_count_per_prim=16 * len(HAND_CONTACT_BODY_NAMES),
                ),
            )
        original = _target_spawn()
        for name, obj in self.object_collection.rigid_objects.items():
            if name != 'mass_cylinder':
                obj.spawn = sim_utils.MultiAssetSpawnerCfg(
                    assets_cfg=prop_variants(original),
                    random_choice=True,
                    activate_contact_sensors=True,
                )


from ..final_spawners import final_prop_variants


@configclass
class FinalV1SceneCfg(FinalSceneCfg):
    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=20):
        if max_gray_obstacles > 10 or max_white_obstacles > 10:
            raise ValueError('Final v1/v2 support at most 10 Gray and 10 White slots')
        # Reconstruct from a cylinder template, never from a previous MultiAsset.
        self.object_collection.rigid_objects['mass_cylinder'].spawn = _target_spawn()
        super().configure_population_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)
        original = _target_spawn()
        for name, obj in self.object_collection.rigid_objects.items():
            obj.spawn = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=final_prop_variants(original, target=name == 'mass_cylinder'),
                random_choice=True, activate_contact_sensors=True,
            )


@configclass
class FinalGoalAccuracyV0_1SceneCfg(FinalV1SceneCfg):
    def configure_population_capacity(self, max_gray_obstacles, max_white_obstacles, observation_slots=20):
        super().configure_population_capacity(max_gray_obstacles, max_white_obstacles, observation_slots)
        for variant in self.object_collection.rigid_objects['mass_cylinder'].spawn.assets_cfg:
            if getattr(variant, 'kind', None) == 'cube':
                variant.cube_size = (0.06, 0.08, 0.10)
