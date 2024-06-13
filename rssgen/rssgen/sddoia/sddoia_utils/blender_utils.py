import bpy
import bpy_extras
import os
import random


class SuppressOutput:
    """Suppress output in case the generation wants to be quiet"""

    def __enter__(self):
        self.stdout_orig = os.dup(1)
        self.stderr_orig = os.dup(2)
        self.null_path = os.devnull
        os.close(1)
        os.close(2)
        os.open(self.null_path, os.O_RDWR)
        os.dup2(1, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.stdout_orig, 1)
        os.dup2(self.stderr_orig, 2)
        os.close(self.stdout_orig)
        os.close(self.stderr_orig)


def add_material(obj, name, **properties):
    """
    Create a new material and assign it to the active object. "name" should be the
    name of a material that has been previously loaded using load_materials.
    """
    # Figure out how many materials are already in the scene
    mat_count = len(bpy.data.materials)

    # Create a new material; it is not attached to anything and
    # it will be called "Material"
    bpy.ops.material.new()

    # Get a reference to the material we just created and rename it;
    # then the next time we make a new material it will still be called
    # "Material" and we will still be able to look it up by name

    mat = bpy.data.materials["Material"]
    mat.name = "Material_%d" % mat_count

    # Attach the new material to the active object
    # Make sure it doesn't already have materials
    obj.data.materials.clear()

    assert len(obj.data.materials) == 0
    obj.data.materials.append(mat)

    # Find the output node of the new material
    output_node = None
    for n in mat.node_tree.nodes:
        if n.name == "Material Output":
            output_node = n
            break

    # Add a new GroupNode to the node tree of the active material,
    # and copy the node tree from the preloaded node group to the
    # new group node. This copying seems to happen by-value, so
    # we can create multiple materials of the same type without them
    # clobbering each other
    group_node = mat.node_tree.nodes.new("ShaderNodeGroup")
    group_node.node_tree = bpy.data.node_groups[name]

    # Find and set the "Color" input of the new group node
    for inp in group_node.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Wire the output of the new group node to the input of
    # the MaterialOutput node
    mat.node_tree.links.new(
        group_node.outputs["Shader"],
        output_node.inputs["Surface"],
    )


def delete_object(obj, delete=False):
    """Delete a specified blender object"""

    if delete:
        for o in bpy.data.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.ops.object.delete()
    else:
        obj.hide_set(True)
        obj.hide_render = True
        obj.name = "deleted"


def load_materials(material_dir):
    """
    Load materials from a directory. We assume that the directory contains .blend
    files with one material each. The file X.blend has a single NodeTree item named
    X; this NodeTree item must have a "Color" input that accepts an RGBA value.
    """
    reset_context()  # I don't know why honestly, this context thing is a mess
    # debug_context()
    for fn in os.listdir(material_dir):
        if not fn.endswith(".blend"):
            continue
        name = os.path.splitext(fn)[0]
        filepath = os.path.join(material_dir, fn, "NodeTree", name)
        bpy.ops.wm.append(filename=filepath)


def set_up_scene_eevee(
    output_image,
    width=469,
    height=387,
    render_num_samples=128,
    camera_jitter=0.5,
    key_light_jitter=1.0,
    back_light_jitter=1.0,
    fill_light_jitter=1.0,
    *args,
    **kwargs,
):
    """Set up eevee scene"""
    render_args = bpy.context.scene.render
    render_args.engine = "BLENDER_EEVEE"  # Use EEVEE rendering engine
    render_args.filepath = output_image
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100

    # Set up rendering options
    render_args.use_motion_blur = False  # Disable motion blur for EEVEE
    render_args.use_simplify = False  # Disable simplify options for EEVEE

    # Set render samples (EEVEE uses samples for anti-aliasing)
    bpy.context.scene.eevee.taa_render_samples = render_num_samples

    # Set light bounces (Note: EEVEE does not use bounces like Cycles)
    bpy.context.scene.eevee.use_gtao = (
        True  # Enable Ambient Occlusion for better lighting
    )
    bpy.context.scene.eevee.gtao_distance = 10  # Set the distance for AO
    bpy.context.scene.eevee.gtao_factor = 1.0  # Set the strength of AO

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if camera_jitter > 0:
        for i in range(3):
            bpy.data.objects["Camera"].location[i] += rand(camera_jitter)

    # Add random jitter to lamp positions
    if key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Key"].location[i] += rand(key_light_jitter)
    if back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Back"].location[i] += rand(back_light_jitter)
    if fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Fill"].location[i] += rand(fill_light_jitter)


def render_scene(args, output_path):
    """
    Render the scene and save it as a PNG file.
    """
    # set_up_scene(output_image=output_path, **vars(args))
    set_up_scene_eevee(output_image=output_path, **vars(args))

    # Set output format to PNG
    bpy.context.scene.render.image_settings.file_format = "PNG"
    # Set the output path
    bpy.context.scene.render.filepath = output_path

    # Render the scene
    with SuppressOutput():
        bpy.ops.render.render(write_still=True)

    print("Rendered", output_path)


def load_blend_scene(args, blend_file_path: str):
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blend_file_path))

    scene = bpy.context.scene

    load_materials(args.material_dir)


def debug_context():
    print("Context", bpy.context)
    for attr in dir(bpy.context):
        if not attr.startswith("__"):
            print(attr, ":", getattr(bpy.context, attr))


def reset_context():
    bpy.context.view_layer.objects.active = None


def debug_plane():
    plane_object = bpy.data.objects.get("Plane")
    print(plane_object.material_slots)

    # Loop through material slots
    for slot in plane_object.material_slots:
        material = slot.material  # Get the material assigned to the slot
        if material is not None:
            print("Material Name:", material.name)  # Print the name of the material


def count_shape_occurrences():
    """
    Count the occurrences of each shape in the scene.
    Returns a dictionary containing the shape name and its count.
    """
    shape_counts = {}

    # Iterate through all objects in the scene
    for obj in bpy.data.objects:
        # Check if the object is a mesh (geometry)
        if obj.type == "MESH":
            # Get the mesh data
            mesh = obj.data
            # Get the name of the mesh (shape)
            shape_name = mesh.name
            # Remove the suffix (if any) to agglomerate same shapes
            shape_name = shape_name.split(".")[0]
            # Skip mesh or plane
            if shape_name == "Mesh" or shape_name == "Plane":
                continue
            # Increment the count for the shape in the dictionary
            shape_counts[shape_name] = shape_counts.get(shape_name, 0) + 1

    return shape_counts


def get_camera_coords(cam, pos):
    """
    For a specified point, get both the 3D coordinates and 2D pixel-space
    coordinates of the point from the perspective of the camera.

    Inputs:
    - cam: Camera object
    - pos: Vector giving 3D world-space position

    Returns a tuple of:
    - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
      in the range [-1, 1]
    """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def cleanup_scene(to_cleanup, delete=False):
    # Get the current scene
    scene = bpy.context.scene

    for remove_me in to_cleanup:
        for obj in scene.objects:
            if obj.name.startswith(remove_me):
                if delete:
                    bpy.data.objects.remove(obj)
                else:
                    obj.hide_set(True)
                    obj.hide_render = True
                    obj.name = "deleted"


def change_material_color(obj_name, material_name, material_colors):
    obj = bpy.data.objects.get(obj_name)
    if obj is not None:
        # Iterate over the object's materials
        for slot in obj.material_slots:
            # Check if the slot has a material assigned
            if slot.material is not None and slot.material.name == material_name:
                # Modify the diffuse color of the material
                nodes = slot.material.node_tree.nodes
                principled_node = nodes.get("Principled BSDF")

                if principled_node is not None:
                    # Set the base color
                    color_name, rgb = random.choice(list(material_colors.items()))
                    # Add jittering to the color
                    jittered_rgb = [
                        min(1.0, max(0.0, c + random.uniform(-0.1, 0.1))) for c in rgb
                    ]
                    # Set the base color
                    principled_node.inputs["Base Color"].default_value = jittered_rgb


def set_up_scene(
    output_image,
    width=469,
    height=387,
    use_gpu=0,
    render_num_samples=128,
    render_min_bounces=8,
    render_max_bounces=8,
    render_tile_size=256,
    camera_jitter=0.5,
    key_light_jitter=1.0,
    back_light_jitter=1.0,
    fill_light_jitter=1.0,
    *args,
    **kwargs,
):
    """Set up scene with CYCLES"""
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    if use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = "CUDA"
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
            cycles_prefs.compute_device_type = "CUDA"

    # Some CYCLES-specific stuff
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.tile_size = render_tile_size

    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces

    if use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if camera_jitter > 0:
        for i in range(3):
            bpy.data.objects["Camera"].location[i] += rand(camera_jitter)

    # Add random jitter to lamp positions
    if key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Key"].location[i] += rand(key_light_jitter)
    if back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Back"].location[i] += rand(back_light_jitter)
    if fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Fill"].location[i] += rand(fill_light_jitter)
