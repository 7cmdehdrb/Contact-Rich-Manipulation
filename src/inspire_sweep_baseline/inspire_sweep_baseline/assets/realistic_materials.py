"""Private, reusable physics materials for continuous episode randomization.

Updating existing USD material attributes follows Isaac Sim PhysicsMaterial's
set_static_friction/set_dynamic_friction behavior. Unlike allocating a fresh
material for each continuous TensorAPI sample, this keeps the material count
bounded across long training runs. Original restitution/compliance are kept.
"""

from pxr import Sdf, Usd, UsdPhysics, UsdShade


def bind_role_materials(stage, body_paths, material_root, static_friction, defaults):
    """Bind a role's friction to its colliders, preserving other material data.

    One private material is reused for colliders with the same non-friction
    properties. Materials live outside the cloned environment hierarchy so
    writing env_0's properties cannot propagate to the other environments.
    """
    groups = {}
    for body_path in body_paths:
        body = stage.GetPrimAtPath(body_path)
        if not body.IsValid():
            raise RuntimeError(f"Missing friction target: {body_path}")
        # The copied Hand uses instanceable collision meshes. Traverse their
        # proxies to find colliders, then bind on the nearest editable ancestor.
        colliders = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if prim.HasAPI(UsdPhysics.CollisionAPI)]
        if not colliders:
            raise RuntimeError(f"No editable collision prims below {body_path}")
        for collider in colliders:
            source, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            properties = dict(defaults)
            if source:
                for attr in source.GetPrim().GetAttributes():
                    if attr.GetName().startswith(("physics:", "physxMaterial:")) and attr.Get() is not None:
                        properties[attr.GetName()] = (attr.GetTypeName(), attr.Get())
            # These three fields are the only existing material values changed.
            properties.pop("physics:staticFriction", None)
            properties.pop("physics:dynamicFriction", None)
            properties.pop("physxMaterial:frictionCombineMode", None)
            signature = tuple(sorted((name, str(kind), repr(value)) for name, (kind, value) in properties.items()))
            if signature not in groups:
                material = UsdShade.Material.Define(stage, f"{material_root}_{len(groups)}")
                prim = material.GetPrim()
                physics = UsdPhysics.MaterialAPI.Apply(prim)
                prim.AddAppliedSchema("PhysxMaterialAPI")
                for name, (kind, value) in properties.items():
                    prim.CreateAttribute(name, kind, custom=False).Set(value)
                prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, custom=False).Set("average")
                static_attr = physics.CreateStaticFrictionAttr(float(static_friction))
                dynamic_attr = physics.CreateDynamicFrictionAttr(float(static_friction * 0.8))
                groups[signature] = (material, static_attr, dynamic_attr)
            material = groups[signature][0]
            binding_prim = collider
            while binding_prim.IsInstanceProxy():
                binding_prim = binding_prim.GetParent()
            UsdShade.MaterialBindingAPI.Apply(binding_prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics",
            )
    return [(static_attr, dynamic_attr) for _, static_attr, dynamic_attr in groups.values()]


def material_defaults(cfg):
    """Use the unchanged simulation defaults when a collider has no material."""
    return {
        "physics:restitution": (Sdf.ValueTypeNames.Float, cfg.restitution),
        "physxMaterial:restitutionCombineMode": (Sdf.ValueTypeNames.Token, cfg.restitution_combine_mode),
        "physxMaterial:compliantContactStiffness": (Sdf.ValueTypeNames.Float, cfg.compliant_contact_stiffness),
        "physxMaterial:compliantContactDamping": (Sdf.ValueTypeNames.Float, cfg.compliant_contact_damping),
    }
