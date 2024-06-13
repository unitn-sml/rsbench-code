# This script is used to generate the first person dataset for the SDDOIA project
# The dataset is generated using the Blender API and the CLEVR dataset
# The script generates the dataset in parallel using the multiprocessing module
# The script also generates the json file containing the scene information
# The script also generates the images and stores them in the output directory

import copy
import argparse
import json
import os
import random
import sys
import bpy
from mathutils import Vector
from collections import OrderedDict

sys.path.append("/home/x/rss/rss-dataset-gen/rssgen/sddoia/")
sys.path.append("/home/x/rss/rss-dataset-gen/rssgen/clevr/")
sys.path.insert(0, "/home/x/rss/rss-dataset-gen/venv/rss/lib/python3.7/site-packages")

from sddoia_utils import *
from clevr_renderer import camera_view_bounds_2d, compute_all_relationships

"""SEMANTIC MATRIX CONSTANTS"""

CURRENT_CAR_POSITION_MATRIX = None

TURN_EVERY_WHERE_POSITION = [(1, 3)]
TURN_IMM_RIGHT_POSITION = [(2, 3)]
TURN_IMM_LEFT_POSITION = [(3, 3)]


def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--". Blender ignores command-line flags
    after --, so this lets us forward command line arguments from the blender
    invocation to our own script.
    """
    if input_argv is None:
        input_argv = sys.argv
    output_argv = []
    if "--" in input_argv:
        idx = input_argv.index("--")
        output_argv = input_argv[(idx + 1) :]
    return output_argv


def assign_random_counts(shape_counts):
    """
    Assign a random count between 1 and the actual number for each shape.
    """
    random_counts = {}

    for shape, count in shape_counts.items():
        # Generate a random count between 1 and the actual number
        random_count = random.randint(1, count)
        random_counts[shape] = random_count

    return random_counts


def generate_random_config():
    """Generate random configuration"""
    shape_counts = count_shape_occurrences()
    shape_counts = assign_random_counts(shape_counts)

    sem_model = random.choice(list(SEM_MODELS.values()))
    stop_sign_model = random.choice(list(STOP_SIGN_MODELS.values()))
    tree_model = TREE_MODEL if random.randint(0, 1) else None
    car_model = CAR_MODEL if random.randint(0, 1) else None
    pedestrian_model = PEDESTRIAN_MODEL if random.randint(0, 1) else None
    rider_model = RIDER_MODEL if random.randint(0, 1) else None
    cross_walk_model = CROSS_WALK_MODEL if random.randint(0, 1) else None
    line_model = LINE
    dashedline_model = DASHED_LINE

    # Update shape counts based on constraints
    update_shape_counts(
        shape_counts,
        sem_model,
        stop_sign_model,
        pedestrian_model,
        rider_model,
        tree_model,
        cross_walk_model,
    )

    # Randomly choose counts for each shape
    cil_count = shape_counts.get("Cylinder", 0)
    sphere_count = shape_counts.get("Sphere", 0)
    torus_count = shape_counts.get("Torus", 0)

    pedestrian_cil_count = random.randint(0, cil_count)
    rider_cil_count = cil_count - pedestrian_cil_count

    sem_sphere_count = random.randint(0, sphere_count)
    stop_sign_sphere_count = sphere_count - sem_sphere_count

    line_count = random.randint(0, torus_count)
    dashedline_count = torus_count - line_count

    # define the random world
    shape_models = {
        "Sphere": [
            [sem_model, SEM_NAME.get(sem_model), sem_sphere_count],
            [stop_sign_model, "Stop", stop_sign_sphere_count],
        ],
        "Cube": [[car_model, "Car", shape_counts.get("Cube", 0)]],
        "Cylinder": [
            [pedestrian_model, "Stickman", pedestrian_cil_count],
            [rider_model, "Rider", rider_cil_count],
        ],
        "Cone": [[tree_model, "Tree", shape_counts.get("Cone", 0)]],
        "Torus": [
            [line_model, "Line", line_count],
            [dashedline_model, "DashedLine", dashedline_count],
        ],
        "Icosphere": [
            [cross_walk_model, "CrossWalk", shape_counts.get("Icosphere", 0)]
        ],
    }

    return shape_models


def update_shape_counts(
    shape_counts,
    sem_model,
    stop_sign_model,
    pedestrian_model,
    rider_model,
    tree_model,
    cross_walk_model,
):
    """Update the shape count"""
    if sem_model is not None and stop_sign_model is not None:
        if shape_counts["Sphere"] < 2:
            shape_counts["Sphere"] = 2
    elif sem_model is None and stop_sign_model is None:
        shape_counts["Sphere"] = 0

    if pedestrian_model is not None and rider_model is not None:
        if shape_counts["Cylinder"] < 2:
            shape_counts["Cylinder"] = 2
    elif pedestrian_model is None and rider_model is None:
        shape_counts["Cylinder"] = 0

    if tree_model is None:
        shape_counts["Cone"] = 0

    if cross_walk_model is None:
        shape_counts["Icosphere"] = 0

    return shape_counts


def replace_objects(
    args,
    scene_struct,
    object_to_replace,
    blend_file_path,
    replacement_object_name,
    count=None,
    additional_constraints=False,
):
    """Replace objects"""

    def get_suffix(name):
        start_index = name.find("new") + len("new")
        end_index = name.find(".", start_index)
        content = name[start_index:end_index]
        return content

    def load_cars():
        for car in CARS_COLORS:
            content = get_suffix(car)
            load_blend_file(car, f"Car{content}")

    def get_random_car():
        # randomize the cars
        load_cars()
        car = random.choice(CARS_COLORS)
        replacement_object_name = f"Car{get_suffix(car)}"
        return replacement_object_name

    def load_blend_file(blend_file_path, replacement_object_name):
        blend_file_path = os.path.abspath(blend_file_path)
        with bpy.data.libraries.load(blend_file_path) as (data_from, data_to):
            data_to.objects = [
                name
                for name in data_from.objects
                if name.startswith(replacement_object_name)
            ]

    """
    Replace all instances of object_to_replace with replacement object from blend_file_path.
    """
    # Get the current scene
    scene = bpy.context.scene
    # Get the camera
    camera = bpy.data.objects["Camera"]

    load_blend_file(blend_file_path, replacement_object_name)

    all_objects = [obj for obj in scene.objects]

    # shuffle objects
    random.shuffle(all_objects)

    # Iterate through all objects in the scene
    for obj in all_objects:
        # end placing all the objects
        if count is not None and count == 0:
            continue
        # Check if the object is the one we want to replace
        if (additional_constraints and obj.name == object_to_replace) or (
            not additional_constraints and obj.name.startswith(object_to_replace)
        ):
            # random car
            if blend_file_path == CAR_MODEL:
                replacement_object_name = get_random_car()

            # Check if replacement object was loaded
            if replacement_object_name in bpy.data.objects:
                replacement_object = bpy.data.objects[replacement_object_name]

                # Duplicate the replacement object and place it at a slightly jittered location
                new_obj = replacement_object.copy()
                new_obj.data = replacement_object.data.copy()
                bpy.context.collection.objects.link(new_obj)
                new_obj.location = obj.location

                # SET BASE DIMENSION
                color_conf, material_conf = None, None
                let_color = False

                # Apply the random configuration
                if obj.name in CONFIG:
                    rotation = CONFIG[obj.name].get(
                        "rotation", 0
                    )  # Get rotation value from config, default to 0
                    new_obj.rotation_euler.z = rotation * (
                        3.14159 / 180.0
                    )  # Convert degrees to radians

                    color_conf = CONFIG[obj.name].get(
                        "color", None
                    )  # Get color from config, default to "white"
                    material_conf = CONFIG[obj.name].get(
                        "material", "Rubber"
                    )  # Get material from config, default to "Rubber"
                    let_color = CONFIG[obj.name].get(
                        "let_color", False
                    )  # Get let_color from config, default to False

                    if color_conf is not None:
                        color_conf = COLORS_TO_RGBA[color_conf]

                # Apply jitter to position
                new_obj.location.x += random.uniform(
                    -args.location_jitter, args.location_jitter
                )
                new_obj.location.y += random.uniform(
                    -args.location_jitter, args.location_jitter
                )

                # Apply jitter to rotation
                new_obj.rotation_euler.z += random.uniform(
                    -args.rotation_jitter, args.rotation_jitter
                ) * (
                    3.14159 / 180.0
                )  # Convert degrees to radians
                new_obj.rotation_euler.y += random.uniform(
                    -args.rotation_jitter, args.rotation_jitter
                ) * (
                    3.14159 / 180.0
                )  # Convert degrees to radians

                # Set dimensions proportionally to depth (using object bouding box dimensions as depth factor)
                depth_factor = obj.dimensions.y  # Using y dimension as depth factor
                # Scale the new object's dimensions proportionally to depth
                new_obj.dimensions *= depth_factor
                # Place it at the same location as the original object
                new_obj.location = obj.location

                # Apply jitter to scale
                scale_factor = random.uniform(
                    1 - args.scale_jitter, 1 + args.scale_jitter
                )
                new_obj.scale *= scale_factor

                # Jitter sunlight position
                sun_light = bpy.data.lights.get("Sun")
                if sun_light:
                    sun_light.energy *= random.uniform(0.8, 1.2)

                material_name, mat_file = random.choice(list(MATERIALS.items()))
                color_name, rgba = random.choice(list(COLORS_TO_RGBA.items()))

                # set object as active
                bpy.context.view_layer.objects.active = new_obj

                if not let_color:
                    if color_conf is None:
                        add_material(new_obj, mat_file, Color=rgba)
                    else:
                        try:
                            add_material(new_obj, material_conf, Color=color_conf)
                        except Exception as e:
                            print(e, new_obj.name)

                # Delete the original object
                # bpy.data.objects.remove(obj)
                obj.hide_set(True)
                obj.hide_render = True
                obj.name = "deleted"

                # Add the object to struct
                pixel_coords = get_camera_coords(camera, new_obj.location)
                # Get 2D pixel coordinates for all 8 points in the bounding box
                bound_box = camera_view_bounds_2d(bpy.context.scene, camera, new_obj)

                scene_struct["objects"].append(
                    {
                        "shape": replacement_object_name,
                        "material": material_name,
                        "3d_coords": tuple(new_obj.location),
                        "pixel_coords": pixel_coords,
                        "color": color_name,
                        "x": bound_box.x,
                        "y": bound_box.y,
                        "width": bound_box.width,
                        "height": bound_box.height,
                    }
                )
            else:
                print(
                    "Replacement object not found in blend file:",
                    replacement_object_name,
                )
                exit(1)

            # reduce the count
            if count is not None:
                count = count - 1


def replace_camera(position, name_to_replace, delete=False):
    """
    Replace the name_to_replace with the camera in the scene.
    """
    # Original camera
    camera = bpy.data.objects.get("Camera")

    # Get the object to replace
    obj_to_replace = bpy.data.objects.get(name_to_replace)
    if obj_to_replace is None:
        print(f"Object {name_to_replace} not found in scene.")
        return

    # Remove parenting from camera if it exists
    if camera.parent:
        camera.parent = None

    # Set camera location and rotation to match the object to replace
    camera.location = obj_to_replace.location
    camera.location.z = obj_to_replace.location.z
    camera.rotation_euler = obj_to_replace.rotation_euler

    # Set the camera as the active object
    bpy.context.view_layer.objects.active = camera

    # Add a constraint to the camera to copy the transformation of the empty
    copy_transform_constraint = camera.constraints.new(type="COPY_TRANSFORMS")
    copy_transform_constraint.target = obj_to_replace

    # Turn the car slightly right if it is in TURN_IMM_RIGHT_POSITION
    if position == TURN_IMM_RIGHT_POSITION:
        camera.rotation_euler.z += 0.1

    # Turn the car slightly left if it is in TURN_IMM_LEFT_POSITION
    if position == TURN_IMM_LEFT_POSITION:
        camera.rotation_euler.z -= 0.1

    if delete:
        # Remove the object to replace from the scene
        bpy.data.objects.remove(obj_to_replace)
    else:
        obj_to_replace.hide_set(True)
        obj_to_replace.hide_render = True
        obj_to_replace.name = "deleted"

    # Copy transformations from object to camera
    camera.matrix_world = obj_to_replace.matrix_world


def replace_objects_given_config(
    args,
    split,
    output_idx,
    output_img,
    models_count,
    additional_constraints,
    elements_to_remove,
):
    """Replace an object given the configuration instructions"""
    scene_struct = initialize_struct_concepts(split, output_idx, output_img)

    # First loop through the additional constraints
    for shape, config in additional_constraints.items():
        model, name, count = config

        replace_objects(
            args, scene_struct, shape, model, name, count, additional_constraints=True
        )

    # clean up what needs to be cleaned up
    cleanup_scene(elements_to_remove, delete=True)

    # Add the randomicity
    for shape, configs in models_count.items():
        for config in configs:
            model, name, count = config
            if model is None or count == 0:
                continue
            replace_objects(args, scene_struct, shape, model, name, count)

    cleanup_scene(
        ["Cube", "Cylinder", "Cone", "Torus", "Icosphere", "Sphere", "Suzanne"],
        delete=False,
    )
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    return scene_struct


def update_struct_planes(scene_struct):
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object

    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    delete_object(plane)
    return scene_struct


def initialize_struct_concepts(output_split, output_index, output_image):
    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image),
        "objects": [],
        "directions": {},
    }
    scene_struct = update_struct_planes(scene_struct)

    scene_struct["objects"] = list()
    return scene_struct


def apply_knowledge(world, scene_struct, y):
    # apply knowledge to build the label
    scene_struct["label"] = y
    scene_struct["concepts"] = world
    return scene_struct


def generate_world(args, world, idx, split, folder, output_scene, scene, y, instance):
    global CURRENT_CAR_POSITION_MATRIX

    # copy the matrix
    CURRENT_CAR_POSITION_MATRIX = copy.deepcopy(CAR_POSITION_MATRIX)

    output_img = f"{folder}/{args.filename_prefix}_{idx}.png"

    # load the base scene
    load_blend_scene(args, scene)

    # Convert g to camera positions and additional constraints
    positions = from_g_to_camera_position(world, instance)

    (
        camera,
        additional_constraints,
        direction_constraints,
        to_remove,
    ) = from_camera_to_additional_constraints(positions, world, instance, args, split)

    # Remove elements in line of camera
    elements_to_remove = remove_in_line_elements(camera["position"], y, world)

    # if there are other elements to remove
    if len(to_remove) > 0:
        elements_to_remove.extend(to_remove)

    # Randomize default colors
    randomize_default_materials()

    # Replace camera
    replace_camera(camera["position"], camera["camera"]["object"], delete=False)

    # Apply direction constraints
    apply_direction_constraints(direction_constraints)

    # Generate random config counts
    model_counts = generate_random_config()

    # Replace objects given config
    scene_struct = replace_objects_given_config(
        args,
        split,
        idx,
        output_img,
        model_counts,
        additional_constraints,
        elements_to_remove,
    )

    # find the grountruth labels
    scene_struct = apply_knowledge(world, scene_struct, y)

    # render scene
    render_scene(args, output_img)

    with open(output_scene, "w") as f:
        json.dump(scene_struct, f, indent=2)

    return output_scene


def randomize_default_materials():
    material_1_colors = {  # GRASS
        "grey": (0.5, 0.5, 0.5, 1.0),
        "green": (0.2, 0.7, 0.3, 1.0),
        "brown": (0.6, 0.4, 0.2, 1.0),
        "darkyellow": (0.8, 0.6, 0.1, 1.0),
    }
    material_2_colors = {  # STREET
        "black": (0.0, 0.0, 0.0, 1.0),
        "darkgrey": (0.2, 0.2, 0.2, 1.0),
    }
    material_3_colors = {  # SKY
        "light_blue": (0.6, 0.8, 1.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "dark_blue": (0.0, 0.0, 0.5, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
    }

    # Change the color of Material.001 and Material.002
    change_material_color("Plane", "Material.001", material_1_colors)
    change_material_color("Plane", "Material.002", material_2_colors)
    change_material_color("Sky", "Material.004", material_3_colors)


def random_sampling(list_of_values, weights):
    return random.choices(list_of_values, weights=weights)[0]


def apply_direction_constraints(direction_constraints):
    """
    Apply direction constraints to the CONFIG dictionary.
    """
    direction_dict = {"left": 90, "forward": 0, "backward": 180, "right": 270}
    for obj, constraint in direction_constraints.items():
        if obj not in CONFIG:
            CONFIG[obj] = {}
        CONFIG[obj]["direction"] = direction_dict[constraint["direction"]]


def from_g_to_camera_position(g, instance):
    """
    Convert g parameters to camera positions and additional constraints.
    """

    positions = {
        "camera": {(0, 3), (1, 3), (2, 3), (3, 3), (4, 3)},
        "pedestrian": {(2, 2), (2, 3), (3, 2), (3, 3)},
        "car": {(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)},
        "semaphore": {(1, 4)},
        "sign": {(1, 4)},
        "other": {(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)},
    }

    if g["left_lane"] and g["right_lane"]:
        positions["camera"] = {(1, 3)}
    elif g["left_lane"]:
        positions["camera"] = {(1, 3), (3, 3)}
    elif g["right_lane"]:
        positions["camera"] = {(1, 3), (2, 3)}

    # sign should be visibile!
    if (
        not (g["red_light"] == False and g["green_light"] == False)
        or instance["Sign"] != ""
    ):
        positions["camera"] -= {(2, 3), (3, 3), (4, 3)}
        positions["pedestrian"] -= {(3, 2), (3, 3)}
        # the traffic should be followed or there is a car (as obstacle) in the scene
        if g["follow"] or g["car"]:
            positions["car"] -= {(2, 3), (3, 3), (4, 3), (5, 3)}

    if g["person"] or g["rider"]:
        positions["camera"] -= {(0, 3), (3, 3), (4, 3)}

    return positions


def random_sampling(list_of_values, weights):
    """
    Perform random sampling with given weights.
    """
    return random.choices(list_of_values, weights=weights)[0]


def random_sampling_list(config_dict):
    return random_sampling(config_dict["values"], config_dict["weights"])


def deal_with_tl(g, boia_concepts_values):
    if g["TL"] == "G":
        boia_concepts_values["red_light"] = False
        boia_concepts_values["green_light"] = True
    elif g["TL"] == "":
        boia_concepts_values["red_light"] = False
        boia_concepts_values["green_light"] = False
    else:
        boia_concepts_values["red_light"] = True
        boia_concepts_values["green_light"] = False


def deal_with_obs(g, boia_concepts_values):
    if ["car-same-dir"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = True
    elif ["car-diff-dir"] == g["Obs"]:
        boia_concepts_values["car"] = True
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["pedestrian"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["rider"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["other"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = True
        boia_concepts_values["follow"] = False
    elif [""] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["pedestrian", "rider"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["pedestrian", "rider", "car-same-dir"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = True
    elif ["pedestrian", "rider", "car-diff-dir"] == g["Obs"]:
        boia_concepts_values["car"] = True
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["pedestrian", "rider", "other"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = True
        boia_concepts_values["follow"] = False
    elif ["pedestrian", "other"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = True
        boia_concepts_values["follow"] = False
    elif ["pedestrian", "car-same-dir"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = True
    elif ["pedestrian", "car-diff-dir"] == g["Obs"]:
        boia_concepts_values["car"] = True
        boia_concepts_values["person"] = True
        boia_concepts_values["rider"] = False
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    elif ["rider", "other"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = True
        boia_concepts_values["follow"] = False
    elif ["rider", "car-same-dir"] == g["Obs"]:
        boia_concepts_values["car"] = False
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = True
    elif ["rider", "car-diff-dir"] == g["Obs"]:
        boia_concepts_values["car"] = True
        boia_concepts_values["person"] = False
        boia_concepts_values["rider"] = True
        boia_concepts_values["other_obstacle"] = False
        boia_concepts_values["follow"] = False
    else:
        raise ValueError("Invalid observation", g["Obs"])


def deal_with_sign(g, boia_concepts_values):
    if g["Sign"] != "stop":
        boia_concepts_values["stop_sign"] = False
    else:
        boia_concepts_values["stop_sign"] = True


def deal_with_L_R(g, boia_concepts_values):
    if "Left" in g:
        boia_concepts_values["left_lane"] = True
        boia_concepts_values["no_left_lane"] = False
        boia_concepts_values["right_lane"] = False
        boia_concepts_values["no_right_lane"] = True
    if "Right" in g:
        boia_concepts_values["left_lane"] = False
        boia_concepts_values["no_left_lane"] = True
        boia_concepts_values["right_lane"] = True
        boia_concepts_values["no_right_lane"] = False


def deal_with_left_right_tl(g, boia_concepts_values):
    if g["TL"] == "G":
        boia_concepts_values["left_green_light"] = True
        boia_concepts_values["right_green_light"] = True
    else:
        boia_concepts_values["left_green_light"] = False
        boia_concepts_values["right_green_light"] = False


def deal_with_left_right_follow(g, boia_concepts_values):
    if "ObsLeft" in g:
        if ["left-follow"] == g["ObsLeft"]:
            boia_concepts_values["left_follow"] = True
            boia_concepts_values["left_obstacle"] = False
        elif ["left-follow", "left-obstacle"] == g["ObsLeft"]:
            boia_concepts_values["left_follow"] = True
            boia_concepts_values["left_obstacle"] = True
        elif ["left-obstacle"] == g["ObsLeft"]:
            boia_concepts_values["left_obstacle"] = True
            boia_concepts_values["left_follow"] = False
        else:
            boia_concepts_values["left_obstacle"] = False
            boia_concepts_values["left_follow"] = False

    if "ObsRight" in g:
        if "right-follow" in g["ObsRight"]:
            boia_concepts_values["right_follow"] = True
            boia_concepts_values["right_obstacle"] = False
        elif ["right-follow", "right-obstacle"] == g["ObsRight"]:
            boia_concepts_values["right_follow"] = True
            boia_concepts_values["right_obstacle"] = True
        elif ["right-obstacle"] == g["ObsRight"]:
            boia_concepts_values["right_obstacle"] = True
            boia_concepts_values["right_follow"] = False
        else:
            boia_concepts_values["right_obstacle"] = False
            boia_concepts_values["right_follow"] = False


def deal_with_lines(g, boia_concepts_values):
    if not "Line" in g:
        return

    if g["Line"] == "solid":
        boia_concepts_values["left_solid_line"] = True
        boia_concepts_values["right_solid_line"] = True
    else:
        boia_concepts_values["left_solid_line"] = False
        boia_concepts_values["right_solid_line"] = False


def deal_with_left_right(g, boia_concepts_values):
    if "Left" in g:
        boia_concepts_values["left_lane"] = True
        boia_concepts_values["no_left_lane"] = False
    if "Right" in g:
        boia_concepts_values["right_lane"] = True
        boia_concepts_values["no_right_lane"] = False


def boia_concepts_filtered_by_gen(g, boia_concepts_values):
    deal_with_left_right(g, boia_concepts_values)
    deal_with_tl(g, boia_concepts_values)
    deal_with_obs(g, boia_concepts_values)
    deal_with_sign(g, boia_concepts_values)
    deal_with_L_R(g, boia_concepts_values)
    deal_with_left_right_tl(g, boia_concepts_values)
    deal_with_left_right_follow(g, boia_concepts_values)
    deal_with_lines(g, boia_concepts_values)
    return boia_concepts_values


def ensure_consistency(boia_concepts):
    """Ensure consistency among concept"""
    if boia_concepts["left_green_light"] != boia_concepts["right_green_light"]:
        boia_concepts["right_green_light"] = boia_concepts["left_green_light"]
    if boia_concepts["left_green_light"] != boia_concepts["green_light"]:
        boia_concepts["left_green_light"] = boia_concepts["green_light"]
        boia_concepts["right_green_light"] = boia_concepts["green_light"]
    if boia_concepts["left_obstacle"]:
        boia_concepts["left_follow"] = False
    if boia_concepts["right_obstacle"]:
        boia_concepts["right_follow"] = False
    if boia_concepts["follow"]:
        boia_concepts["other_obstacle"] = False
    if boia_concepts["right_solid_line"]:
        boia_concepts["right_follow"] = False
    if boia_concepts["left_solid_line"]:
        boia_concepts["left_follow"] = False

    if not boia_concepts["left_lane"]:
        boia_concepts["left_obstacle"] = False
        boia_concepts["left_solid_line"] = False
        boia_concepts["left_green_light"] = False
        boia_concepts["left_follow"] = False
        boia_concepts["left_solid_line"] = False
        boia_concepts["no_left_lane"] = True
    else:
        boia_concepts["no_left_lane"] = False

    if not boia_concepts["right_lane"]:
        boia_concepts["right_obstacle"] = False
        boia_concepts["right_solid_line"] = False
        boia_concepts["right_green_light"] = False
        boia_concepts["right_follow"] = False
        boia_concepts["right_solid_line"] = False
        boia_concepts["no_right_lane"] = True
    else:
        boia_concepts["no_right_lane"] = False

    return boia_concepts


def get_full_boia_concepts(g):
    """Get the boia full concepts from the boia rules."""

    full_boia_concepts = [
        "red_light",
        "green_light",
        "car",
        "person",
        "rider",
        "other_obstacle",
        "follow",
        "stop_sign",
        "left_lane",
        "left_green_light",
        "left_follow",
        "no_left_lane",
        "left_obstacle",
        "left_solid_line",
        "right_lane",
        "right_green_light",
        "right_follow",
        "no_right_lane",
        "right_obstacle",
        "right_solid_line",
    ]

    # Generate random boolean values for each concept
    boia_concepts_values = {
        concept: bool(random.getrandbits(1)) for concept in full_boia_concepts
    }

    boia_concepts_values = ensure_consistency(boia_concepts_values)

    assert (
        boia_concepts_values["left_lane"] != boia_concepts_values["no_left_lane"]
    ), "1"
    assert (
        boia_concepts_values["right_lane"] != boia_concepts_values["no_right_lane"]
    ), "2"

    boia_concepts_values = boia_concepts_filtered_by_gen(g, boia_concepts_values)

    assert (
        boia_concepts_values["left_lane"] != boia_concepts_values["no_left_lane"]
    ), "3a"
    assert (
        boia_concepts_values["right_lane"] != boia_concepts_values["no_right_lane"]
    ), "3b"

    # get the value of y and the one of clear
    y, clear = apply_sddoiaK(boia_concepts_values)

    # add clear
    boia_concepts_values["clear"] = clear

    assert (
        boia_concepts_values["left_lane"] != boia_concepts_values["no_left_lane"]
    ), "4"
    assert (
        boia_concepts_values["right_lane"] != boia_concepts_values["no_right_lane"]
    ), "5"

    return boia_concepts_values, y


def choose_scene(config):
    if config["left_lane"] and config["right_lane"]:
        return BOIA_FULL
    if config["left_lane"]:
        return BOIA_LEFT
    if config["right_lane"]:
        return BOIA_RIGHT
    return BOIA_STRAIGHT


def generate_g_given_forward():
    """
    Generate g parameters for the forward scenario.
    """
    tl = random_sampling(list(TL_DIST_FORW.keys()), list(TL_DIST_FORW.values()))

    obs = random_sampling_list(OBS_DIST_FORW)

    if tl == "":
        sign = random_sampling(
            list(SIGN_DIST_FORW.keys()), weights=list(SIGN_DIST_FORW.values())
        )
    else:
        sign = ""

    g = {"TL": tl, "Obs": obs, "Sign": sign}

    # get the full config and the y
    config, y = get_full_boia_concepts(g)
    scene = choose_scene(config)

    # for the forward generation, forward should always be true
    assert (
        y[1] == 1 and y[0] == 0
    ), f"Assertion failed: {y}, with world {config}, action: forward"

    return config, scene, y, g


def generate_g_given_stop():
    """
    Generate g parameters for the stop scenario.
    """

    tl = random_sampling(list(TL_DIST.keys()), list(TL_DIST.values()))
    if tl == "R" or tl == "Y":
        obs = random_sampling_list(OBS_DIST)
        sign = ""
    else:
        obs = random_sampling_list(OBS_DIST)
        if any(
            item in obs for item in ["car-diff-dir", "pedestrian", "rider", "other"]
        ):
            if tl == "G":
                sign = ""
            else:
                sign = random_sampling(
                    list(SIGN_DIST.keys()), weights=list(SIGN_DIST.values())
                )
        else:
            sign = random_sampling(["stop"], weights=[1.0])
            tl = random_sampling([""], weights=[1.0])

    g = {"TL": tl, "Obs": obs, "Sign": sign}

    # get the full config and the y
    config, y = get_full_boia_concepts(g)
    scene = choose_scene(config)

    # for the forward generation, forward should always be true
    assert (
        y[0] == 1 and y[1] == 0
    ), f"Assertion failed: {y}, with world {config}, action: stop"

    return config, scene, y, g


def generate_g_given_left():
    """
    Generate g parameters for the turn-left scenario.
    """

    sign = ""
    line = "dashed"
    tl = random_sampling(list(TL_DIST_LEFT.keys()), list(TL_DIST_LEFT.values()))
    obs = random_sampling_list(OBS_DIST)
    obs_left = random_sampling_list(OBS_DIST_LEFT)

    g = {
        "TL": tl,
        "Obs": obs,
        "ObsLeft": obs_left,
        "Sign": sign,
        "Line": line,
        "Left": True,
    }

    # get the full config and the y
    config, y = get_full_boia_concepts(g)
    scene = choose_scene(config)

    # for the forward generation, forward should always be true
    assert y[2] == 1, f"Assertion failed: {y}, with world {config}, action: left"

    return config, scene, y, g


def generate_g_given_right():
    """
    Generate g parameters for the turn-right scenario.
    """

    sign = ""
    line = "dashed"
    tl = random_sampling(list(TL_DIST_RIGHT.keys()), list(TL_DIST_RIGHT.values()))
    obs = random_sampling_list(OBS_DIST)
    obs_right = random_sampling_list(OBS_DIST_RIGHT)

    g = {
        "TL": tl,
        "Obs": obs,
        "ObsRight": obs_right,
        "Sign": sign,
        "Line": line,
        "Right": True,
    }

    # get the full config and the y
    config, y = get_full_boia_concepts(g)
    scene = choose_scene(config)

    # for the forward generation, forward should always be true
    assert y[3] == 1, f"Assertion failed: {y}, with world {config}, action: right"

    return config, scene, y, g


def from_camera_to_additional_constraints(positions, g, instance, args, split):
    global CURRENT_CAR_POSITION_MATRIX

    """
    Convert camera positions and g parameters to additional constraints.
    """

    def convert_format(data):
        """
        Convert additional constraints data to the required format.
        """
        result = {}
        for key, value in data.items():
            name = value["object"]
            if name not in result:
                result[name] = [value["blend"], value["semantic"], 1]
            else:
                if value["semantic"] not in result[name]:
                    result[name].append(value["semantic"])
                result[name][2] += 1
        return result

    def direction_constraints(data):
        """
        Extract direction constraints from additional constraints data.
        """
        result = {}
        for value in data.values():
            if "direction" in value:
                result[value["object"]] = {"direction": value["direction"]}
        return result

    LEFT_DASHED_LINE = {
        "object": "Torus",
        "semantic": "DashedLine",
        "blend": DASHED_LINE,
        "direction": "forward",
    }
    LEFT_LINE = {
        "object": "Torus",
        "semantic": "Line",
        "blend": LINE,
        "direction": "forward",
    }
    RIGHT_DASHED_LINE = {
        "object": "Torus.001",
        "semantic": "DashedLine",
        "blend": DASHED_LINE,
        "direction": "forward",
    }
    RIGHT_LINE = {
        "object": "Torus.001",
        "semantic": "Line",
        "blend": LINE,
        "direction": "forward",
    }

    additional_constraints = {}
    to_remove = []

    # Initialize camera and its position
    camera = {"camera": None, "position": None}

    # Select camera position randomly
    camera_position = random.choice(list(positions["camera"]))

    # Modify the position of other_obstacle
    positions["other"] -= {
        (camera_position[0], camera_position[1])
    }  # remove the current position from other

    # Modify the possible pedestrian position by taking the next positions
    positions["pedestrian"] = {
        (camera_position[0] + 1, camera_position[1] - 1),
        (camera_position[0] + 1, camera_position[1]),
    }

    # Car above the camera
    positions["car"] = {
        (camera_position[0] + 1, camera_position[1]),
    }

    # add the second possibile position
    if camera_position[0] + 2 <= 5:  # limit of the semantic matrix
        positions["car"].add((camera_position[0] + 2, camera_position[1]))

    # follow are basically the position of car
    positions["follow"] = positions["car"]
    positions["left_follow"] = {(3, 2)}
    positions["right_follow"] = {(2, 4)}

    # Define positions for car and other objects relative to camera
    if not g["left_lane"] and not g["right_lane"]:
        positions["other"] = {
            (camera_position[0] + 1, camera_position[1]),
            (camera_position[0] + 2, camera_position[1]),
        }

    # Handle the fact that there may be left and right lane visible
    if g["left_lane"] and camera_position in TURN_IMM_LEFT_POSITION:
        positions["left_follow"] = {
            (camera_position[0], camera_position[1] - 1),
        }

        if g["left_solid_line"]:
            additional_constraints["line"] = LEFT_LINE
        else:
            which_line = random.choice(["dashed", ""])
            if which_line == "dashed":
                additional_constraints["line"] = LEFT_DASHED_LINE
            else:
                to_remove.append(LEFT_LINE["object"])

    if g["right_lane"] and camera_position in TURN_IMM_RIGHT_POSITION:
        positions["right_follow"] = {
            (camera_position[0], camera_position[1] + 1),
        }

        if g["right_solid_line"]:
            additional_constraints["line"] = RIGHT_LINE
        else:
            which_line = random.choice(["dashed", ""])
            if which_line == "dashed":
                additional_constraints["line"] = RIGHT_DASHED_LINE
            else:
                to_remove.append(RIGHT_LINE["object"])

    if g["left_lane"] and camera_position in TURN_EVERY_WHERE_POSITION:
        positions["left_follow"] = {
            (camera_position[0] + 2, camera_position[1] - 1),
        }

        if g["left_solid_line"]:
            additional_constraints["line"] = LEFT_LINE
        else:
            which_line = random.choice(["dashed", ""])
            if which_line == "dashed":
                additional_constraints["line"] = LEFT_DASHED_LINE
            else:
                to_remove.append(LEFT_LINE["object"])

    if g["right_lane"] and camera_position in TURN_EVERY_WHERE_POSITION:
        positions["right_follow"] = {
            (camera_position[0] + 1, camera_position[1] + 1),
        }
        if g["right_solid_line"]:
            additional_constraints["line"] = RIGHT_LINE
        else:
            which_line = random.choice(["dashed", ""])
            if which_line == "dashed":
                additional_constraints["line"] = RIGHT_DASHED_LINE
            else:
                to_remove.append(RIGHT_LINE["object"])

    # Check for the right and left follow:
    if g["left_follow"]:
        obstacle_position = random.choice(list(positions["left_follow"]))
        additional_constraints["left_follow"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["car"]
        )
        # Set car direction
        additional_constraints["left_follow"]["direction"] = "left"
    else:
        for pos in list(positions["left_follow"]):
            to_remove.append(
                CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]["car"][0]["object"]
            )

    if g["right_follow"]:
        obstacle_position = random.choice(list(positions["right_follow"]))
        additional_constraints["right_follow"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["car"]
        )
        # Set car direction
        additional_constraints["right_follow"]["direction"] = "right"
    else:
        for pos in list(positions["left_follow"]):
            to_remove.append(
                CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]["car"][0]["object"]
            )

    # Set camera constraint
    camera["camera"] = CURRENT_CAR_POSITION_MATRIX[camera_position]["camera"]
    camera["position"] = camera_position
    positions["other"] -= {(6, 3)}

    # Handle semaphore constraints
    if not (g["red_light"] == False and g["green_light"] == False):
        obstacle_position = random.choice(list(positions["semaphore"]))

        additional_constraints["semaphore"] = CURRENT_CAR_POSITION_MATRIX[
            obstacle_position
        ]["semaphore"]

        # Determine semaphore type
        if instance["TL"] == "G":
            additional_constraints["semaphore"]["blend"] = TLG
            additional_constraints["semaphore"]["semantic"] = "TLG"
        elif instance["TL"] == "Y":
            additional_constraints["semaphore"]["blend"] = TLY
            additional_constraints["semaphore"]["semantic"] = "TLY"
        elif instance["TL"] == "R":
            additional_constraints["semaphore"]["blend"] = TLR
            additional_constraints["semaphore"]["semantic"] = "TLR"
        else:
            print("No semaphore: impossible?")

    # Handle sign constraints
    if instance["Sign"] != "":
        obstacle_position = random.choice(list(positions["sign"]))
        additional_constraints["sign"] = CURRENT_CAR_POSITION_MATRIX[obstacle_position][
            "sign"
        ]

        # Determine sign type
        if instance["Sign"] == "stop":
            additional_constraints["sign"]["blend"] = STOP
            additional_constraints["sign"]["semantic"] = "Stop"
        elif instance["Sign"] == "nonstop":
            additional_constraints["sign"]["blend"] = MPH30
            additional_constraints["sign"]["semantic"] = "mph30"
        elif instance["Sign"] == "" and (
            g["red_light"] == False and g["green_light"] == False
        ):
            to_remove.append(additional_constraints["semaphore"]["object"])

    # Retrieve object positions
    if g["other_obstacle"]:
        obstacle_position = random.choice(list(positions["other"]))
        additional_constraints["other"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["other"]
        )
        additional_constraints["other"]["direction"] = random.choice(
            additional_constraints["other"]["direction"]
        )
        # remove from car positions
        positions["car"] -= {obstacle_position}

    if g["left_obstacle"]:
        obstacle_position = random.choice([(3, 2)])
        additional_constraints["left_obstacle"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["other"]
        )
        additional_constraints["left_obstacle"]["direction"] = random.choice(
            additional_constraints["left_obstacle"]["direction"]
        )
        # remove from car positions
        positions["left_follow"] -= {obstacle_position}
    else:
        obstacle_positions = [(3, 2)]
        for pos in obstacle_positions:
            to_remove.append(
                CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]["other"][0]["object"]
            )

    if g["right_obstacle"]:
        obstacle_position = random.choice([(2, 4)])
        additional_constraints["right_obstacle"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["other"]
        )
        additional_constraints["right_obstacle"]["direction"] = random.choice(
            additional_constraints["right_obstacle"]["direction"]
        )
        # remove from car positions
        positions["right_follow"] -= {obstacle_position}
    else:
        obstacle_positions = [(2, 4)]
        for pos in obstacle_positions:
            to_remove.append(
                CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]["other"][0]["object"]
            )

    # Handle car constraints (basically every obstacle)
    # important to check for follow before
    if g["follow"]:
        obstacle_position = random.choice(list(positions["follow"]))
        additional_constraints["follow"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["car"]
        )
        # Set car direction
        additional_constraints["follow"]["direction"] = "forward"

    if g["car"]:
        obstacle_position = random.choice(list(positions["car"]))
        additional_constraints["car"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["car"]
        )
        # Set car direction
        additional_constraints["car"]["direction"] = random.choice(
            ["left", "right", "backward"]
        )

    # handle pedestrian and rider constraints
    to_remove_pedestrian = {
        (camera_position[0] + 1, camera_position[1] - 1),
        (camera_position[0] + 1, camera_position[1]),
    }
    to_remove_prev_pedestrian_cars = {
        (camera_position[0], camera_position[1] - 1),
        (camera_position[0], camera_position[1]),
    }

    # IL LMAO
    if g["person"]:
        obstacle_position = random.choice(list(positions["pedestrian"]))
        additional_constraints["pedestrian"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["pedestrian"]
        )
        additional_constraints["pedestrian"]["direction"] = random.choice(
            additional_constraints["pedestrian"]["direction"]
        )
        # remove the position of the chosen pedestrian:
        positions["pedestrian"] -= {obstacle_position}

        to_remove_pedestrian.remove(obstacle_position)
        to_remove_prev_pedestrian_cars.remove(
            (obstacle_position[0] - 1, obstacle_position[1])
        )

    if g["rider"]:
        obstacle_position = random.choice(list(positions["pedestrian"]))
        additional_constraints["rider"] = random.choice(
            CURRENT_CAR_POSITION_MATRIX[obstacle_position]["obstacles"]["pedestrian"]
        )
        additional_constraints["rider"]["direction"] = random.choice(
            additional_constraints["rider"]["direction"]
        )

        to_remove_pedestrian.remove(obstacle_position)
        to_remove_prev_pedestrian_cars.remove(
            (obstacle_position[0] - 1, obstacle_position[1])
        )

    for pos in to_remove_pedestrian:
        if (
            pos not in CURRENT_CAR_POSITION_MATRIX
            or "pedestrian" not in CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]
        ):
            break
        to_remove.append(
            CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]["pedestrian"][0]["object"]
        )
    for pos in to_remove_prev_pedestrian_cars:
        if (
            pos not in CURRENT_CAR_POSITION_MATRIX
            or "car" not in CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]
        ):
            break
        to_remove.append(
            CURRENT_CAR_POSITION_MATRIX[pos]["obstacles"]["car"][0]["object"]
        )

    return (
        camera,
        convert_format(additional_constraints),
        direction_constraints(additional_constraints),
        to_remove,
    )


def remove_in_line_elements(camera_position, y, g):
    """
    Remove in-line elements obstructing the camera's view.
    Remove also things that may or may not change camera opinion.
    Eg: if the move is not stop, then pedestrian and riders should be removed
    """
    global CURRENT_CAR_POSITION_MATRIX

    to_remove = set()

    # Check what to remove based on the g configuration
    if g["clear"]:
        # remove all the cars above!
        for i in range(1, 6):
            pos_to_eval = (camera_position[0] + i, camera_position[1])
            if pos_to_eval in CURRENT_CAR_POSITION_MATRIX:
                if "car" in CURRENT_CAR_POSITION_MATRIX[pos_to_eval]:
                    to_remove.add(
                        CURRENT_CAR_POSITION_MATRIX[pos_to_eval]["car"]["object"]
                    )
                elif "obstacles" in CURRENT_CAR_POSITION_MATRIX[pos_to_eval]:
                    if "car" in CURRENT_CAR_POSITION_MATRIX[pos_to_eval]["obstacles"]:
                        to_remove.add(
                            CURRENT_CAR_POSITION_MATRIX[pos_to_eval]["obstacles"][
                                "car"
                            ][0]["object"]
                        )

        # remove also the pedestrian on that side of the road
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(2, 3)]["obstacles"]["pedestrian"][0]["object"]
        )
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(2, 2)]["obstacles"]["pedestrian"][0]["object"]
        )
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(3, 3)]["obstacles"]["pedestrian"][0]["object"]
        )
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(3, 2)]["obstacles"]["pedestrian"][0]["object"]
        )

    # Remove traffic light if it is not visible
    if (
        g["red_light"] == False
        and g["green_light"] == False
        and g["stop_sign"] == False
    ):
        to_remove.add(CURRENT_CAR_POSITION_MATRIX[(1, 4)]["semaphore"]["object"])

    # car and follow!
    if g["car"] == False and g["follow"] == False:
        # remove the elements two tiles in front of the camera
        camera_position = (camera_position[0], camera_position[1])
        in_front_1 = (camera_position[0] + 1, camera_position[1])
        in_front_2 = (camera_position[0] + 2, camera_position[1])
        if in_front_1 in CURRENT_CAR_POSITION_MATRIX:
            if "car" in CURRENT_CAR_POSITION_MATRIX[in_front_1]:
                to_remove.add(CURRENT_CAR_POSITION_MATRIX[in_front_1]["car"]["object"])
            elif "obstacles" in CURRENT_CAR_POSITION_MATRIX[in_front_1]:
                if "car" in CURRENT_CAR_POSITION_MATRIX[in_front_1]["obstacles"]:
                    to_remove.add(
                        CURRENT_CAR_POSITION_MATRIX[in_front_1]["obstacles"]["car"][0][
                            "object"
                        ]
                    )
        if in_front_2 in CURRENT_CAR_POSITION_MATRIX:
            if "car" in CURRENT_CAR_POSITION_MATRIX[in_front_2]:
                to_remove.add(CURRENT_CAR_POSITION_MATRIX[in_front_2]["car"]["object"])
            elif "obstacles" in CURRENT_CAR_POSITION_MATRIX[in_front_2]:
                if "car" in CURRENT_CAR_POSITION_MATRIX[in_front_2]["obstacles"]:
                    to_remove.add(
                        CURRENT_CAR_POSITION_MATRIX[in_front_2]["obstacles"]["car"][0][
                            "object"
                        ]
                    )

    # Person and Rider!
    if g["person"] == False and g["rider"] == False:
        next_position = (camera_position[0] + 1, camera_position[1])
        next_left_position = (camera_position[0] + 1, camera_position[1] - 1)
        if next_position in CURRENT_CAR_POSITION_MATRIX:
            if "obstacles" in CURRENT_CAR_POSITION_MATRIX[next_position]:
                if (
                    "pedestrian"
                    in CURRENT_CAR_POSITION_MATRIX[next_position]["obstacles"]
                ):
                    to_remove.add(
                        CURRENT_CAR_POSITION_MATRIX[next_position]["obstacles"][
                            "pedestrian"
                        ][0]["object"]
                    )
        if next_left_position in CURRENT_CAR_POSITION_MATRIX:
            if "obstacles" in CURRENT_CAR_POSITION_MATRIX[next_left_position]:
                if (
                    "pedestrian"
                    in CURRENT_CAR_POSITION_MATRIX[next_left_position]["obstacles"]
                ):
                    to_remove.add(
                        CURRENT_CAR_POSITION_MATRIX[next_left_position]["obstacles"][
                            "pedestrian"
                        ][0]["object"]
                    )

    if g["left_follow"] == False:
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(3, 2)]["obstacles"]["car"][0]["object"]
        )
    else:
        # visibile!
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(3, 3)]["obstacles"]["car"][0]["object"]
        )

    if g["right_follow"] == False:
        to_remove.add(
            CURRENT_CAR_POSITION_MATRIX[(2, 4)]["obstacles"]["car"][0]["object"]
        )

    # Add car on the camera position
    to_remove.add(CURRENT_CAR_POSITION_MATRIX[camera_position]["car"]["object"])
    # Obstacle is put only in additional constraints, so no need to remove it

    if y[2] == 1:
        car_positions_on_left_line = [(2, 2), (3, 2)]
        for carpos in car_positions_on_left_line:
            to_remove.add(
                CURRENT_CAR_POSITION_MATRIX[carpos]["obstacles"]["car"][0]["object"]
            )

    if y[2] == 3:
        car_positions_on_right_line = [(2, 3), (3, 3)]
        for carpos in car_positions_on_right_line:
            to_remove.add(
                CURRENT_CAR_POSITION_MATRIX[carpos]["obstacles"]["car"][0]["object"]
            )

    return list(to_remove)


def distribute_samples(n, m):
    avg_samples_per_config = n // m
    remainder = n % m

    samples_distribution = [avg_samples_per_config] * m

    # Distribute remaining samples evenly among configurations
    for i in range(remainder):
        samples_distribution[i] += 1

    return samples_distribution


def groundtruth_given_y_gen():
    generators = [
        generate_g_given_forward,
        generate_g_given_stop,
        generate_g_given_left,
        generate_g_given_right,
    ]
    return generators


def generate_binary_dict(possibilities):
    binary_dict = OrderedDict()

    for i in range(2**possibilities):
        binary_str = format(i, "04b")
        binary_dict[binary_str] = []

    return binary_dict


def check_values_length(dictionary, length):
    for value in dictionary.values():
        if len(value) != length:
            return False
    return True


def configurations_split(args, n_conf, n_samples):
    def shuffle_ordered_dict_keys(d):
        keys = list(d.keys())
        random.shuffle(keys)
        shuffled_dict = OrderedDict((key, d[key]) for key in keys)

        return shuffled_dict

    def reverse_ordered_dict_keys(d):
        keys = list(d.keys())
        reversed_keys = keys[::-1]
        reversed_dict = OrderedDict((key, d[key]) for key in reversed_keys)

        return reversed_dict

    def get_config(
        label_bucket, distribution_samples, n_conf, is_ood=False, use_ood_k=False
    ):
        conf = []
        scenes = []
        ys = []
        instances = []

        # distribution for configuration
        distribution_for_configuration = []

        current_lb = copy.deepcopy(label_bucket)

        if is_ood:
            current_lb = reverse_ordered_dict_keys(label_bucket)

        # loop through all the configurations
        for label_idx, configs in enumerate(current_lb.values()):
            # if I am over the number of label to generate: break
            if label_idx > len(distribution_samples) - 1:
                break

            # how many samples per config
            distributed_samples_per_config = distribute_samples(
                distribution_samples[label_idx], n_conf
            )
            distributed_samples_per_config = list(
                filter(lambda x: x != 0, distributed_samples_per_config)
            )

            # loop over the configurations (n_conf max)
            for i, config in enumerate(configs):
                if i > len(distributed_samples_per_config) - 1:
                    break

                c, scene, y, g, ood_ys = config
                conf.append(c)
                scenes.append(scene)
                # use ood knowledge
                if use_ood_k:
                    ys.append(ood_ys)
                instances.append(g)
                distribution_for_configuration.append(distributed_samples_per_config[i])

        return {
            "conf": conf,
            "y": ys,
            "instance": instances,
            "scene": scenes,
            "num": distribution_for_configuration,
        }

    # bucket of labels
    labels_bucket = generate_binary_dict(4)

    # remove the impossible configuration
    for l in range(2):
        for r in range(2):
            del labels_bucket["11{}{}".format(l, r)]

    for l in range(2):
        for r in range(2):
            del labels_bucket["00{}{}".format(l, r)]

    idx = 0
    budget = 2000

    # fill this configurations
    print("Wait for the config generation...")
    print("This could take a while...")
    while not check_values_length(labels_bucket, n_conf):
        for i, fun in enumerate(groundtruth_given_y_gen()):
            in_dict, scene, ys, g = fun()
            # obtain the ood_ys from the original sampled configurations
            ood_ys = ood_knowledge(in_dict)
            label = "{}{}{}{}".format(ys[0], ys[1], ys[2], ys[3])
            if len(labels_bucket[label]) < n_conf:
                labels_bucket[label].append((in_dict, scene, ys, g, ood_ys))
            idx += 1

        if idx > budget:
            break
    print("Done!")

    # shuffle the label bucket
    for key, value in labels_bucket.items():
        random.shuffle(labels_bucket[key])

    labels_bucket = shuffle_ordered_dict_keys(labels_bucket)

    # n samples per configuration
    train_n_samples = max([int(n_samples * args.train_ratio), 1])
    val_n_samples = max([int(n_samples * args.val_ratio), 1])
    test_n_samples = max([int(n_samples * args.test_ratio), 1])
    ood_n_samples = max([int(n_samples * args.ood_ratio), 1])

    # Split the configurations into training, validation, and test sets
    if args.use_ood_knowledge:
        print(
            "Using OOD Knowledge. I am going to ingnore in distribution and out of distribution ratio"
        )
        in_dist_conf = max([int(1 * len(labels_bucket)), 1])
        ood_conf = max([int(1 * len(labels_bucket)), 1])
    else:
        in_dist_conf = max([int(args.in_dist_ratio * len(labels_bucket)), 1])
        ood_conf = max([int(args.ood_ratio * len(labels_bucket)), 1])

    train_distribution_samples = distribute_samples(train_n_samples, in_dist_conf)
    val_distribution_samples = distribute_samples(val_n_samples, in_dist_conf)
    test_distribution_samples = distribute_samples(test_n_samples, in_dist_conf)
    ood_distribution_samples = distribute_samples(ood_n_samples, ood_conf)

    train_configs = get_config(labels_bucket, train_distribution_samples, n_conf)
    val_configs = get_config(labels_bucket, val_distribution_samples, n_conf)
    test_configs = get_config(labels_bucket, test_distribution_samples, n_conf)
    ood_configs = get_config(
        labels_bucket,
        ood_distribution_samples,
        n_conf,
        is_ood=True,
        use_ood_k=args.use_ood_knowledge,
    )

    return train_configs, val_configs, test_configs, ood_configs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configure_parser(parser)

    print("### START ###")

    argv = extract_args()
    args = parser.parse_args(argv)

    # Use absolute paths
    args.output_scene_dir = os.path.abspath(args.output_scene_dir)
    args.output_image_dir = os.path.abspath(args.output_image_dir)

    # Create the directories
    if not os.path.exists(f"{args.output_image_dir}"):
        os.makedirs(f"{args.output_scene_dir}")
        os.makedirs(f"{args.output_image_dir}/train")
        os.makedirs(f"{args.output_image_dir}/val")
        os.makedirs(f"{args.output_image_dir}/test")
        os.makedirs(f"{args.output_image_dir}/ood")

    # number of sampled configurations
    status_dict = {}
    set_random_seed(args.seed)

    # Load the status log
    if args.load_status_dict:
        status_dict = load_status_log(args.status_log)
        if not check_all_done(status_dict):
            print("Not all done, continuing...")
        else:
            print("All done, exiting...")
            exit(0)

        train_configs = status_dict["train_confs"]
        val_configs = status_dict["val_confs"]
        test_configs = status_dict["test_confs"]
        ood_configs = status_dict["ood_confs"]

        start_idx_train = status_dict["train_done"]
        start_idx_val = status_dict["val_done"]
        start_idx_test = status_dict["test_done"]
        start_idx_ood = status_dict["ood_done"]

    else:
        train_configs, val_configs, test_configs, ood_configs = configurations_split(
            args, args.n_config, args.n_samples
        )

        # Populate the status dict
        status_dict["train_confs"] = train_configs
        status_dict["val_confs"] = val_configs
        status_dict["test_confs"] = test_configs
        status_dict["ood_confs"] = ood_configs

        start_idx_train = [0 for _ in range(len(train_configs["conf"]))]
        start_idx_val = [0 for _ in range(len(val_configs["conf"]))]
        start_idx_test = [0 for _ in range(len(test_configs["conf"]))]
        start_idx_ood = [0 for _ in range(len(ood_configs["conf"]))]

        status_dict["train_done"] = start_idx_train
        status_dict["val_done"] = start_idx_val
        status_dict["test_done"] = start_idx_test
        status_dict["ood_done"] = start_idx_ood

        # Save the status log
        save_status_log(status_dict, args.status_log)

    # all scenes
    all_scene_paths = []

    idx = 0
    for config, folder in zip(
        [train_configs, val_configs, test_configs, ood_configs],
        [
            f"{args.output_image_dir}/train",
            f"{args.output_image_dir}/val",
            f"{args.output_image_dir}/test",
            f"{args.output_image_dir}/ood",
        ],
    ):
        for c_idx, (world, n_sample_per_config, scene, y, instance) in enumerate(
            zip(
                config["conf"],
                config["num"],
                config["scene"],
                config["y"],
                config["instance"],
            )
        ):
            # get the starting index
            if folder == f"{args.output_image_dir}/train":
                start_idx = start_idx_train[c_idx]
            elif folder == f"{args.output_image_dir}/val":
                start_idx = start_idx_val[c_idx]
            elif folder == f"{args.output_image_dir}/test":
                start_idx = start_idx_test[c_idx]
            else:
                start_idx = start_idx_ood[c_idx]

            idx += start_idx

            scenes_paths = []

            print("To be generated", n_sample_per_config)

            # regardless, regenerate all the samples for that config
            n_iterations = n_sample_per_config // args.num_parallel_threads
            n_proc_iter = args.num_parallel_threads

            # starting from the start
            n_i = 0
            auxiliary_counter = 0

            for i in range(0, n_iterations + 1):
                if i == n_iterations:
                    n_proc_iter = n_sample_per_config % args.num_parallel_threads

                expected_samples = n_proc_iter
                for j in range(n_proc_iter):
                    # going to the right amount of samples
                    if (n_i + start_idx) > auxiliary_counter:
                        auxiliary_counter += 1
                        expected_samples -= 1
                        continue

                    res = generate_world(
                        args,
                        world,
                        idx + n_i,
                        folder.split("/")[-1],
                        folder,
                        f"{args.output_scene_dir}/{args.filename_prefix}_{idx + n_i}.json",
                        scene,
                        y,
                        instance,
                    )
                    n_i += 1
                    auxiliary_counter += 1
                    scenes_paths.append(res)

                idx += len(scenes_paths)

                # update the status dict
                status_dict[f"{folder.split('/')[-1]}_done"][c_idx] += len(scenes_paths)

                save_status_log(status_dict, args.status_log)

                scenes_paths.sort()
                all_scene_paths.extend(scenes_paths)

    # Check everything went fine
    if not check_all_done(status_dict):
        print("Not all the images have been generated, please check the logs")
        print(
            "In any case, re-run the script with load_status_dict set to True to generate all data"
        )

    # all scenes
    all_scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, "r") as f:
            all_scenes.append(json.load(f))

    output = {
        "info": {
            "date": args.date,
            "version": args.version,
            "license": args.license,
        },
        "scenes": all_scenes,
    }
    with open(args.output_scene_file, "w") as f:
        json.dump(output, f)

    print("### DONE ###")
