"""
Largely adopted from Elevate3D 
"""

import argparse
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

try:
    import bpy
    from bpy_extras.io_utils import axis_conversion
    from mathutils import Matrix, Vector
except ImportError as exc:
    raise SystemExit("This script must be run from Blender.") from exc


_CONTEXT = bpy.context
_SCENE = _CONTEXT.scene
_RENDER = _SCENE.render

IMPORT_FUNCTIONS = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "ply": bpy.ops.import_mesh.ply,
}

AXIS_MAP = {
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "-X": "NEGATIVE_X",
    "-Y": "NEGATIVE_Y",
    "-Z": "NEGATIVE_Z",
}


@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Process one mesh with Blender.")

    axis_choices = ["X", "Y", "Z", "-X", "-Y", "-Z"]
    parser.add_argument("--input_path", type=Path, required=True, help="Input mesh path")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--axis_forward", type=str, default="-Z", choices=axis_choices)
    parser.add_argument("--axis_up", type=str, default="Y", choices=axis_choices)
    parser.add_argument("--env_map_path", type=Path, default=None)
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--num_renders", type=int, default=24)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--random_camera", action="store_true")
    parser.add_argument("--use_emission_shader", action="store_true")
    parser.add_argument("--baked", action="store_true")
    return parser.parse_args(argv)


def reset_scene() -> None:
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            obj.select_set(True)
    if _CONTEXT.selected_objects:
        bpy.ops.object.delete()

    for collection in [
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.meshes,
        bpy.data.actions,
        bpy.data.armatures,
        bpy.data.node_groups,
    ]:
        while collection:
            collection.remove(collection[0])


def load_object(mesh_path: Path, axis_forward: str, axis_up: str) -> None:
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    extension = mesh_path.suffix.lstrip(".").lower()
    import_func = IMPORT_FUNCTIONS.get(extension)
    if import_func is None:
        raise ValueError(f"Unsupported mesh type: {extension}")

    kwargs = {}
    correction = Matrix.Identity(4)

    if extension in {"obj", "fbx", "stl"}:
        kwargs["forward_axis"] = AXIS_MAP[axis_forward]
        kwargs["up_axis"] = AXIS_MAP[axis_up]
    elif extension in {"glb", "gltf"}:
        kwargs["merge_vertices"] = True
        target = axis_conversion(
            from_forward=axis_forward,
            from_up=axis_up,
            to_forward="Y",
            to_up="Z",
        ).to_4x4()
        correction = target @ Matrix.Rotation(math.radians(-90.0), 4, "X")
    else:
        correction = axis_conversion(
            from_forward=axis_forward,
            from_up=axis_up,
            to_forward="Y",
            to_up="Z",
        ).to_4x4()

    before = set(_SCENE.objects)
    import_func(filepath=str(mesh_path), **kwargs)

    if correction.is_identity:
        return

    imported = set(_SCENE.objects) - before
    roots = [obj for obj in imported if obj.parent is None or obj.parent not in imported]
    for obj in roots:
        obj.matrix_world = correction @ obj.matrix_world


def get_scene_aabb() -> tuple[Vector, Vector]:
    bbox_min = Vector((math.inf, math.inf, math.inf))
    bbox_max = Vector((-math.inf, -math.inf, -math.inf))
    found_mesh = False

    for obj in _SCENE.objects:
        if obj.type != "MESH":
            continue
        found_mesh = True
        for vertex in obj.data.vertices:
            point = obj.matrix_world @ vertex.co
            for index in range(3):
                bbox_min[index] = min(bbox_min[index], point[index])
                bbox_max[index] = max(bbox_max[index], point[index])

    if not found_mesh:
        raise RuntimeError("No mesh objects found in scene.")

    return bbox_min, bbox_max


def normalize_scene() -> None:
    root_objects = [obj for obj in _SCENE.objects if obj.parent is None and obj.type != "CAMERA"]
    if not root_objects:
        raise RuntimeError("No objects available to normalize.")

    target = root_objects[0]
    if len(root_objects) > 1:
        parent = bpy.data.objects.new("NormalizationParent", None)
        _SCENE.collection.objects.link(parent)
        for obj in root_objects:
            obj.parent = parent
        target = parent

    _CONTEXT.view_layer.update()
    bbox_min, bbox_max = get_scene_aabb()
    scale = 1.0 / max((bbox_max - bbox_min).length, 1e-6)
    offset = -(bbox_min + bbox_max) / 2.0

    target.scale = Vector((scale, scale, scale))
    target.location = offset * scale
    _CONTEXT.view_layer.update()


def replace_material_with_emission(obj, setup_nodes) -> None:
    material = bpy.data.materials.new(name=f"{obj.name}_Emission")
    material.use_nodes = True
    setup_nodes(material.node_tree)

    if obj.material_slots:
        for slot in obj.material_slots:
            slot.material = material
    else:
        obj.data.materials.append(material)


def set_emission_shader_from_texture() -> None:
    for obj in _SCENE.objects:
        if obj.type != "MESH":
            continue

        source_image = None
        for slot in obj.material_slots:
            material = slot.material
            if material and material.use_nodes:
                image_node = next((node for node in material.node_tree.nodes if node.type == "TEX_IMAGE"), None)
                if image_node and image_node.image:
                    source_image = image_node.image
                    break
        if source_image is None:
            continue

        def setup_nodes(node_tree):
            nodes = node_tree.nodes
            links = node_tree.links
            nodes.clear()
            texture = nodes.new(type="ShaderNodeTexImage")
            texture.image = source_image
            emission = nodes.new(type="ShaderNodeEmission")
            output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(texture.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        replace_material_with_emission(obj, setup_nodes)


def set_emission_shader_from_vertex_color() -> None:
    for obj in _SCENE.objects:
        if obj.type != "MESH" or not obj.data.color_attributes or not obj.data.color_attributes.active:
            continue

        color_layer = obj.data.color_attributes.active

        def setup_nodes(node_tree):
            nodes = node_tree.nodes
            links = node_tree.links
            nodes.clear()
            attribute = nodes.new(type="ShaderNodeAttribute")
            attribute.attribute_name = color_layer.name
            emission = nodes.new(type="ShaderNodeEmission")
            output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(attribute.outputs["Color"], emission.inputs["Color"])
            links.new(emission.outputs["Emission"], output.inputs["Surface"])

        replace_material_with_emission(obj, setup_nodes)


def load_environment_map(env_map_path: Path) -> None:
    if not env_map_path.exists():
        raise FileNotFoundError(f"Environment map not found: {env_map_path}")

    world = _SCENE.world or bpy.data.worlds.new("World")
    _SCENE.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    background = nodes.new(type="ShaderNodeBackground")
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    output = nodes.new(type="ShaderNodeOutputWorld")

    env_tex.image = bpy.data.images.load(str(env_map_path))
    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def set_lighting(args: argparse.Namespace) -> None:
    if args.baked:
        set_emission_shader_from_vertex_color()
    elif args.use_emission_shader:
        set_emission_shader_from_texture()
    elif args.env_map_path is not None:
        load_environment_map(args.env_map_path)


def setup_render_settings(args: argparse.Namespace) -> None:
    _RENDER.engine = args.engine
    _RENDER.image_settings.file_format = "PNG"
    _RENDER.image_settings.color_mode = "RGBA"
    _RENDER.resolution_x = args.resolution
    _RENDER.resolution_y = args.resolution
    _RENDER.film_transparent = True

    if args.engine == "CYCLES":
        _SCENE.cycles.device = "GPU"
        _SCENE.cycles.samples = args.samples
        _SCENE.cycles.use_denoising = True


def get_camera_positions(num_renders: int, radius: float, random_camera: bool):
    if random_camera:
        for _ in range(num_renders):
            azimuth = random.uniform(0.0, 2.0 * math.pi)
            elevation = math.asin(random.uniform(-1.0, 1.0))
            x = radius * math.cos(elevation) * math.sin(azimuth)
            y = -radius * math.cos(elevation) * math.cos(azimuth)
            z = radius * math.sin(elevation)
            yield Vector((x, y, z)), f"train_{math.degrees(azimuth):.1f}_{math.degrees(elevation):.1f}.png"
    else:
        for index in range(num_renders):
            azimuth = math.radians((360.0 / max(num_renders, 1)) * index)
            x = radius * math.sin(azimuth)
            y = -radius * math.cos(azimuth)
            yield Vector((x, y, 0.0)), f"orbit_{index:03d}.png"


def setup_camera() -> tuple[object, object]:
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    camera = _CONTEXT.active_object
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 1.0
    _SCENE.camera = camera

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    target = _CONTEXT.active_object
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    return camera, target


def render_views(output_dir: Path, num_renders: int, radius: float, random_camera: bool) -> None:
    if num_renders <= 0:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    camera, target = setup_camera()
    for location, filename in get_camera_positions(num_renders, radius, random_camera):
        camera.location = location
        _RENDER.filepath = str(output_dir / filename)
        with suppress_output():
            bpy.ops.render.render(write_still=True)

    bpy.ops.object.select_all(action="DESELECT")
    camera.select_set(True)
    target.select_set(True)
    bpy.ops.object.delete()


def apply_mesh_transforms() -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in _SCENE.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if _CONTEXT.selected_objects:
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def export_normalized_mesh(output_dir: Path, axis_forward: str, axis_up: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in _SCENE.objects:
        if obj.type == "MESH":
            obj.select_set(True)
    if not _CONTEXT.selected_objects:
        return

    bpy.ops.wm.obj_export(
        filepath=str(output_dir / "mesh_processed.obj"),
        forward_axis=axis_forward,
        up_axis=axis_up,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with suppress_output():
        reset_scene()
        load_object(args.input_path, args.axis_forward, args.axis_up)

    set_lighting(args)
    normalize_scene()
    apply_mesh_transforms()
    setup_render_settings(args)
    render_views(args.output_dir / "textures", args.num_renders, args.radius, args.random_camera)

    with suppress_output():
        export_normalized_mesh(args.output_dir, args.axis_forward, args.axis_up)


if __name__ == "__main__":
    main()