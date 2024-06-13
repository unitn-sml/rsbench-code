# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import numpy as np
from itertools import product
import re
import multiprocessing as mp

sys.path.append("../..")
sys.path.append("/home/x/rss/rss-dataset-gen/rssgen/clevr/")
sys.path.insert(0, "/home/x/rss/rss-dataset-gen/venv/rss/lib/python3.7/site-packages")

from rssgen.utils import log, set_log_level
from rssgen.parsers import clever_parser
import sympy as sp
from clevr_utils import stdout_redirected

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python clevr_renderer.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
CLEVR_UTILS = True

# Constant containing the combinations to generate
POS_SET = None
NEG_SET = None

# Constants which represent the index of the elements within the combination
COLOR_IDX = 0
SHAPE_IDX = 1
MATERIAL_IDX = 2
SIZE_IDX = 3

"""Check whether we are inside blender"""
try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False

if INSIDE_BLENDER:
    try:
        import clevr_utils
    except ImportError as e:
        CLEVR_UTILS = False


def close_unused_files():
    # Get a list of all data blocks in the blend file
    all_data_blocks = (
        set(bpy.data.objects)
        | set(bpy.data.meshes)
        | set(bpy.data.materials)
        | set(bpy.data.images)
    )

    # Get a set of data blocks that are currently in use
    used_data_blocks = set()
    for obj in bpy.context.scene.objects:
        used_data_blocks.add(obj.data)
        for slot in obj.material_slots:
            if slot.material:
                used_data_blocks.add(slot.material)

    # Find unused data blocks and remove them
    unused_data_blocks = all_data_blocks - used_data_blocks
    for data_block in unused_data_blocks:
        bpy.data.objects.remove(data_block, do_unlink=True)
        bpy.data.meshes.remove(data_block, do_unlink=True)
        bpy.data.materials.remove(data_block, do_unlink=True)
        bpy.data.images.remove(data_block, do_unlink=True)


def check_blender_compatibility():
    if not CLEVR_UTILS:
        log(
            "error",
            "Running clevr_renderer.py from Blender and cannot import clevr_utils.py."
            + "\n"
            "You may need to add a .pth file to the site-packages of Blender's" + "\n"
            "bundled python with a command like this:\n" + "\n"
            "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.7/site-packages/clevr.pth"
            + "\n"
            "\nWhere $BLENDER is the directory where Blender is installed, and" + "\n"
            "$VERSION is your Blender version (such as 3.4).",
        )
        exit(1)


def configure_parser(parser) -> None:
    """
    Configure a new subparser for running the Clevr preparation
    Args:
        Subparser (Subparser): argument parser
    """
    # Input options
    parser.add_argument(
        "--base_scene_blendfile",
        default="../../clevr_config/base_scene.blend",
        help="Base blender file on which all scenes are based; includes "
        + "ground plane, lights, and camera.",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        type=str,
        default="../../examples_config/clevr.yml",
        help="YML config file that contains the logic and some specifics of the dataset to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for generation",
    )
    parser.add_argument(
        "--properties_json",
        default="../../clevr_config/properties.json",
        help="JSON file defining objects, materials, sizes, and colors. "
        + 'The "colors" field maps from CLEVR color names to RGB values; '
        + 'The "sizes" field maps from CLEVR size names to scalars used to '
        + 'rescale object models; the "materials" and "shapes" fields map '
        + "from CLEVR material and shape names to .blend files in the "
        + "--object_material_dir and --shape_dir directories respectively.",
    )
    parser.add_argument(
        "--shape_dir",
        default="../../clevr_config/shapes",
        help="Directory where .blend files for object models are stored",
    )
    parser.add_argument(
        "--material_dir",
        default="../../clevr_config/materials",
        help="Directory where .blend files for materials are stored",
    )
    parser.add_argument(
        "--shape_color_combos_json",
        default=None,
        help="Optional path to a JSON file mapping shape names to a list of "
        + "allowed color names for that shape. This allows rendering images "
        + "for CLEVR-CoGenT.",
    )

    # Settings for objects
    parser.add_argument(
        "--min_objects",
        default=2,
        type=int,
        help="The minimum number of objects to place in each scene",
    )
    parser.add_argument(
        "--max_objects",
        default=2,
        type=int,
        help="The maximum number of objects to place in each scene",
    )
    parser.add_argument(
        "--min_dist",
        default=0.25,
        type=float,
        help="The minimum allowed distance between object centers",
    )
    parser.add_argument(
        "--margin",
        default=0.4,
        type=float,
        help="Along all cardinal directions (left, right, front, back), all "
        + "objects will be at least this distance apart. This makes resolving "
        + "spatial relationships slightly less ambiguous.",
    )
    parser.add_argument(
        "--min_pixels_per_object",
        default=200,
        type=int,
        help="All objects will have at least this many visible pixels in the "
        + "final rendered images; this ensures that no objects are fully "
        + "occluded by other objects.",
    )
    parser.add_argument(
        "--max_retries",
        default=50,
        type=int,
        help="The number of times to try placing an object before giving up and "
        + "re-placing all objects in the scene.",
    )

    # Output settings
    parser.add_argument(
        "--start_idx",
        default=0,
        type=int,
        help="The index at which to start for numbering rendered images. Setting "
        + "this to non-zero values allows you to distribute rendering across "
        + "multiple machines and recombine the results later.",
    )
    parser.add_argument(
        "--num_images", default=10, type=int, help="The number of images to render"
    )
    parser.add_argument(
        "--filename_prefix",
        default="CLEVR",
        help="This prefix will be prepended to the rendered images and JSON scenes",
    )
    parser.add_argument(
        "--split",
        default="new",
        help="Name of the split for which we are rendering. This will be added to "
        + "the names of rendered images, and will also be stored in the JSON "
        + "scene structure for each image.",
    )
    parser.add_argument(
        "--output_image_dir",
        default="../../out/clevr_images/",
        help="The directory where output images will be stored. It will be "
        + "created if it does not exist.",
    )
    parser.add_argument(
        "--output_scene_dir",
        default="../../out/scenes/",
        help="The directory where output JSON scene structures will be stored. "
        + "It will be created if it does not exist.",
    )
    parser.add_argument(
        "--output_scene_file",
        default="../../out/CLEVR_scenes.json",
        help="Path to write a single JSON file containing all scene information",
    )
    parser.add_argument(
        "--output_blend_dir",
        default="../../out/blendfiles",
        help="The directory where blender scene files will be stored, if the "
        + "user requested that these files be saved using the "
        + "--save_blendfiles flag; in this case it will be created if it does "
        + "not already exist.",
    )
    parser.add_argument(
        "--save_blendfiles",
        type=int,
        default=0,
        help="Setting --save_blendfiles 1 will cause the blender scene file for "
        + "each generated image to be stored in the directory specified by "
        + "the --output_blend_dir flag. These files are not saved by default "
        + "because they take up ~5-10MB each.",
    )
    parser.add_argument(
        "--version",
        default="1.0",
        help='String to store in the "version" field of the generated JSON file',
    )
    parser.add_argument(
        "--license",
        default="Creative Commons Attribution (CC-BY 4.0)",
        help='String to store in the "license" field of the generated JSON file',
    )
    parser.add_argument(
        "--date",
        default=dt.today().strftime("%m/%d/%Y"),
        help='String to store in the "date" field of the generated JSON file; '
        + "defaults to today's date",
    )

    # Rendering options
    parser.add_argument(
        "--use_gpu",
        default=1,
        type=int,
        help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. "
        + "You must have an NVIDIA GPU with the CUDA toolkit installed for "
        + "to work.",
    )
    parser.add_argument(
        "--width",
        default=320,
        type=int,
        help="The width (in pixels) for the rendered images",
    )
    parser.add_argument(
        "--height",
        default=240,
        type=int,
        help="The height (in pixels) for the rendered images",
    )
    parser.add_argument(
        "--key_light_jitter",
        default=1.0,
        type=float,
        help="The magnitude of random jitter to add to the key light position.",
    )
    parser.add_argument(
        "--fill_light_jitter",
        default=1.0,
        type=float,
        help="The magnitude of random jitter to add to the fill light position.",
    )
    parser.add_argument(
        "--back_light_jitter",
        default=1.0,
        type=float,
        help="The magnitude of random jitter to add to the back light position.",
    )
    parser.add_argument(
        "--camera_jitter",
        default=0.5,
        type=float,
        help="The magnitude of random jitter to add to the camera position",
    )
    parser.add_argument(
        "--render_num_samples",
        default=512,
        type=int,
        help="The number of samples to use when rendering. Larger values will "
        + "result in nicer images but will cause rendering to take longer.",
    )
    parser.add_argument(
        "--render_min_bounces",
        default=8,
        type=int,
        help="The minimum number of bounces to use for rendering.",
    )
    parser.add_argument(
        "--render_max_bounces",
        default=8,
        type=int,
        help="The maximum number of bounces to use for rendering.",
    )
    parser.add_argument(
        "--no-occlusion",
        action="store_true",
        default=False,
        required=False,
        help="Can have occluded objects",
    )
    parser.add_argument(
        "--render_tile_size",
        default=256,
        type=int,
        help="The tile size to use for rendering. This should not affect the "
        + "quality of the rendered image but may affect the speed; CPU-based "
        + "rendering may achieve better performance using smaller tile sizes "
        + "while larger tile sizes may be optimal for GPU-based rendering.",
    )
    parser.add_argument(
        "--num_parallel_threads",
        default=4,
        type=int,
        help="The number of threads to be run in parallel to generate the images",
    )
    parser.add_argument(
        "--use-ood-knowledge",
        default=False,
        type=int,
        help="Uses ood knowledge instead of having unseen configuration with the same knowledge in the ood dataset",
    )


def print_command_line_arguments(args):
    str_to_print = "Command line arguments:"

    # Iterate over the attributes of the args object
    for arg_name, arg_value in vars(args).items():
        str_to_print += f"\n{arg_name}: {arg_value}"

    log("info", str_to_print)


def parse_yaml(args):
    """Parse YAML"""
    parser = clever_parser.ClevrParser(args.config)
    parsed_values = parser.parse()
    log("info", "Parsed values", parsed_values)
    return parsed_values


def _random_combination(colors, shapes, material, size, n_objects):
    """Random combination"""
    sampled_colors = random.choices(colors, k=n_objects)
    sampled_shapes = random.choices(shapes, k=n_objects)
    sampled_material = random.choices(material, k=n_objects)
    sampled_size = random.choices(size, k=n_objects)

    figure = list(zip(sampled_colors, sampled_shapes, sampled_material, sampled_size))
    return figure


def _parse_combination_string(
    combo_figure, clevr_color, clevr_shapes, clevr_materials, clevr_sizes, n_shapes
):
    """Parse combination strings"""
    result_per_figure = []

    for s in combo_figure:
        words = re.findall(r"\b\w+\b", s)

        log("info", "obtained world is", words)

        if (
            len(words) == 4
            and words[0] in clevr_color
            and words[1] in clevr_shapes
            and words[2] in clevr_materials
            and words[3] in clevr_sizes
        ):
            result_per_figure.extend(words)
        else:
            log(
                "error",
                f"Error: Invalid words in string '{s}'. Please use valid colors, shapes, materials and sizes.",
            )
            exit(1)

    if len(result_per_figure) // 4 != n_shapes:
        log(
            "error",
            f"Error: Invalid number of objects in a figure:",
            len(result_per_figure),
        )
        exit(1)

    return tuple(result_per_figure)


def _handle_combinations(
    combinations,
    clevr_color,
    clevr_shapes,
    clevr_materials,
    clevr_sizes,
    n_shapes,
    return_tuple=True,
):
    """Handle logical combinations"""
    if len(combinations[0]) != 1:
        log(
            "error",
            f"Error: Invalid number of combinations. Expected 1, got {len(combinations)}.",
        )
        exit(1)

    result_total = []

    for combo in combinations:
        result_combo = [
            _parse_combination_string(
                combo_figure,
                clevr_color,
                clevr_shapes,
                clevr_materials,
                clevr_sizes,
                n_shapes,
            )
            for combo_figure in combo
        ]
        result_total.append(tuple(result_combo) if return_tuple else result_combo)

    log("info", "total parsed combinations", result_total)

    return result_total


def _evaluate_logic_expression(vector, logic, symbols_names):
    """Evaluate logic expression"""

    values = []
    for i in range(len(vector)):
        values.extend(list(vector[i]))

    # substitution
    substitutions_dict = {
        symbol_name: value for symbol_name, value in zip(symbols_names, values)
    }

    for symbol_name in symbols_names:
        if symbol_name not in substitutions_dict:
            substitutions_dict[symbol_name] = 1000 * int(symbol_name.split("_")[-1])

    log("info", "substituting...", substitutions_dict, "to", logic)
    # Substitute values into the expression
    result = logic.subs(substitutions_dict)
    log("info", "result of the substitution", result)
    return result


"""Mappings"""


def _map_color_to_integer(color, clevr_colors, starting_index=0):
    return clevr_colors.index(color) + starting_index


def _map_shape_to_integer(shape, clevr_shapes, starting_index=10):
    return clevr_shapes.index(shape) + starting_index


def _map_material_to_integer(material, clevr_materials, starting_index=20):
    return clevr_materials.index(material) + starting_index


def _map_shapes_to_integer(size, clevr_sizes, starting_index=30):
    return clevr_sizes.index(size) + starting_index


def _map_vector(combination, clevr_colors, clevr_shapes, clevr_materials, clevr_sizes):
    """Map vector to categorical values"""
    converted_combination = []
    for vect_el in combination:
        vector = list(vect_el)
        log("info", "changed vector in list", vector)
        for i in range(0, len(vector), 4):
            vector[i] = _map_color_to_integer(vector[i], clevr_colors)
            vector[i + 1] = _map_shape_to_integer(vector[i + 1], clevr_shapes)
            vector[i + 2] = _map_material_to_integer(vector[i + 2], clevr_materials)
            vector[i + 3] = _map_shapes_to_integer(vector[i + 3], clevr_sizes)
        log("info", "changed vector in integer", vector)
        converted_combination.append(tuple(vector))
    return converted_combination


def _values_to_one_hot(possible_values, value_to_convert):
    """Values to one hot encoding"""
    value_to_index = {value: index for index, value in enumerate(possible_values)}
    one_hot_representations = (
        np.eye(len(possible_values))[value_to_index[value_to_convert]]
        .astype(int)
        .tolist()
    )
    return one_hot_representations


def _filter_combinations(
    combinations_in_distribution,
    sample_size,
    min_n_objects,
    max_n_objects,
    logic,
    symbols,
    clevr_color,
    clevr_shapes,
    clevr_materials,
    clevr_sizes,
):
    """Filter combinations"""
    global POS_SET, NEG_SET
    POS_SET, NEG_SET = set(), set()

    # basically the combinations if they are passed as input
    to_remove = 0
    if combinations_in_distribution is not None:
        to_remove = len(combinations_in_distribution)
        combinations_in_distribution = _handle_combinations(
            combinations_in_distribution,
            clevr_color,
            clevr_shapes,
            clevr_materials,
            clevr_sizes,
            return_tuple=False,
        )

    # randomly sample some combinations
    combinations = []

    for _ in range(0, sample_size - to_remove):
        # generate random number of objects in the scene
        num_objects = random.randint(min_n_objects, max_n_objects)
        combinations.append(
            _random_combination(
                clevr_color, clevr_shapes, clevr_materials, clevr_sizes, num_objects
            )
        )

    log("info", "combinations", combinations)

    # filtering before combinations_in_distribution
    if combinations_in_distribution is not None:
        combinations.extend(combinations_in_distribution)

    log("debug", f"Figure combinations: {len(combinations)}")
    log("debug", f"Figure combinations: {combinations}")

    # evaluate the logic expression and get the positive and negative sets
    for combo in combinations:
        label = _evaluate_logic_expression(
            _map_vector(combo, clevr_color, clevr_shapes, clevr_materials, clevr_sizes),
            logic,
            symbols,
        )

        if not isinstance(
            label,
            (sp.logic.boolalg.BooleanFalse, sp.logic.boolalg.BooleanTrue),
        ):
            log(
                "error",
                f"Some logic outputs are not boolean values: {label}",
            )
            exit(1)

        if label:
            POS_SET.add(tuple(combo))
        else:
            NEG_SET.add(tuple(combo))

    if not POS_SET or not NEG_SET:
        log(
            "error",
            "Logic is either a contradiction or a tautology or the sampling rate is too low [NO POSITIVE OR NEGATIVE SETS]!",
            "Pos has",
            len(POS_SET),
            "Neg has",
            len(NEG_SET),
        )
        exit(1)

    log(
        "info",
        f"True assignments: {len(POS_SET)}, False assignments: {len(NEG_SET)}",
    )


def get_positive_combinations(
    combinations_in_distribution,
    sample_size,
    min_n_objects,
    max_n_objects,
    logic,
    symbols,
    clevr_color,
    clevr_shapes,
    clevr_materials,
    clevr_sizes,
):
    """Positive combinations"""
    if (POS_SET is None) or (NEG_SET is None):
        _filter_combinations(
            combinations_in_distribution,
            sample_size,
            min_n_objects,
            max_n_objects,
            logic,
            symbols,
            clevr_color,
            clevr_shapes,
            clevr_materials,
            clevr_sizes,
        )
    log("info", "positive set", POS_SET)
    return POS_SET


def get_negative_combinations(
    combinations_in_distribution,
    sample_size,
    min_n_objects,
    max_n_objects,
    logic,
    symbols,
    clevr_color,
    clevr_shapes,
    clevr_materials,
    clevr_sizes,
):
    """Negative combinations"""
    if (not POS_SET) or (not NEG_SET):
        _filter_combinations(
            combinations_in_distribution,
            sample_size,
            min_n_objects,
            max_n_objects,
            logic,
            symbols,
            clevr_color,
            clevr_shapes,
            clevr_materials,
            clevr_sizes,
        )
    log("debug", "negative set", NEG_SET)
    return NEG_SET


def filering_given_combinations(starting_set, given_combinations):
    """Filtering data given combinations"""
    given_combinations_set = set(given_combinations)
    combinations_in_starting = starting_set.intersection(given_combinations_set)
    combinations_not_in_starting = starting_set.difference(combinations_in_starting)
    return list(combinations_in_starting), list(combinations_not_in_starting)


def split_set(set_variable, percentage):
    """Split set"""
    list_set = list(set_variable)
    split_index = int(len(list_set) * percentage)
    first_part = list_set[:split_index]
    second_part = list_set[split_index:]
    return first_part, second_part


def _read_clevr_properties(properties_json):
    """Read CLEVR properties"""
    with open(properties_json, "r") as f:
        properties = json.load(f)
        clevr_color = list(properties["colors"].keys())
        clevr_shapes = list(properties["shapes"].keys())
        clevr_materials = list(properties["materials"].keys())
        clevr_sizes = list(properties["sizes"].keys())

    return clevr_color, clevr_shapes, clevr_materials, clevr_sizes


def choose_templates_by_dataset(
    dataset_name, train_templates, val_templates, test_templates, ood_templates
):
    """Return templates by dataset"""
    if dataset_name == "train":
        return train_templates
    elif dataset_name == "val":
        return val_templates
    elif dataset_name == "test":
        return test_templates
    elif dataset_name == "ood":
        return ood_templates
    else:
        raise ValueError("Invalid dataset name")


def _get_concepts(
    possible_colors,
    possible_shapes,
    possible_materials,
    possible_sizes,
    max_objects,
    world_to_generate,
):
    """Return concepts"""

    concepts = []
    for image in world_to_generate:
        image_concept_vector = []
        image_concept_vector.append(_values_to_one_hot(possible_colors, image[0]))
        image_concept_vector.append(_values_to_one_hot(possible_shapes, image[1]))
        image_concept_vector.append(_values_to_one_hot(possible_materials, image[2]))
        image_concept_vector.append(_values_to_one_hot(possible_sizes, image[3]))
        concepts.append(image_concept_vector)

    for _ in range(len(world_to_generate), max_objects):
        image_concept_vector = []
        image_concept_vector.append([-1 for _ in range(len(possible_colors))])
        image_concept_vector.append([-1 for _ in range(len(possible_shapes))])
        image_concept_vector.append([-1 for _ in range(len(possible_materials))])
        image_concept_vector.append([-1 for _ in range(len(possible_sizes))])
        concepts.append(image_concept_vector)

    return concepts


def generate_world(
    i,
    args,
    total_positive_samples,
    positive_to_sample,
    negative_to_sample,
    already_generated,
    name,
    train_templates,
    val_templates,
    test_templates,
    ood_templates,
    all_scenes_templates,
    possible_colors,
    possible_shapes,
    possible_materials,
    possible_sizes,
    max_objects,
    world_to_generate=None,
):
    """Generate world in blender"""

    label = None
    concepts = None
    world_to_generate = None

    # GET PROPROTIONATE WORLDS
    if i < total_positive_samples and len(positive_to_sample) > 0:
        # generate positive samples
        idx_sampling = i % len(positive_to_sample)
        world_to_generate = positive_to_sample[idx_sampling]
        label = 1
    elif len(negative_to_sample) > 0:
        # generate negative samples
        idx_sampling = i % len(negative_to_sample)
        world_to_generate = negative_to_sample[idx_sampling]
        label = 0

    if world_to_generate is None:
        log("error", "No world to generate!")
        exit(1)

    if not world_to_generate in already_generated:
        log("info", "For", name, "generating world:", world_to_generate, "label", label)

    # Add the current world
    already_generated.add(world_to_generate)

    img_template, scene_template, blend_template = choose_templates_by_dataset(
        name, train_templates, val_templates, test_templates, ood_templates
    )

    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)

    # to absolute
    img_path = os.path.abspath(img_path)
    scene_path = os.path.abspath(scene_path)

    blend_path = None
    if args.save_blendfiles == 1:
        blend_path = blend_template % (i + args.start_idx)
        blend_path = os.path.abspath(blend_path)

    # get one-hot encoding for concepts
    concepts = _get_concepts(
        possible_colors,
        possible_shapes,
        possible_materials,
        possible_sizes,
        max_objects,
        world_to_generate,
    )

    render_scene(
        args,
        num_objects=len(world_to_generate),
        output_index=(i + args.start_idx),
        output_split=name,
        output_image=img_path,
        output_scene=scene_path,
        output_blendfile=blend_path,
        world_to_generate=world_to_generate,
        concepts=concepts,
        label=label,
    )

    log("info", "world to generate submitted", world_to_generate)
    log("info", "index", i)

    return scene_path


def generate_proportionate_dataset(
    args,
    **kwargs,
):
    """Generate proprortionate dataset"""

    all_scene_paths = []

    # setup saving file templates
    train_templates, val_templates, test_templates, ood_templates = setup_directories(
        args
    )

    # read out all the yaml values
    val_prop = kwargs.get("val_prop")
    test_prop = kwargs.get("test_prop")
    ood_prop = kwargs.get("ood_prop")
    prop_in_distribution = kwargs.get("prop_in_distribution")
    num_samples = kwargs.get("n_samples")
    symbols = kwargs.get("symbols")
    logic = kwargs.get("logic")
    combinations_in_distribution = kwargs.get("combinations_in_distribution", None)

    assert len(symbols) == (
        args.max_objects * 4
    ), f"There must be 4 symbols for each object: {args.max_objects}"

    # set the sizes
    train_size = num_samples
    val_size = int(num_samples * val_prop)
    test_size = int(num_samples * test_prop)
    ood_size = int(num_samples * ood_prop)

    log("info", "sizes", train_size, val_size, test_size)

    # read the properties configs
    clevr_color, clevr_shapes, clevr_materials, clevr_sizes = _read_clevr_properties(
        args.properties_json
    )

    log("info", "properites", clevr_color, clevr_shapes, clevr_materials, clevr_sizes)

    # get proportion of in_out_distribution
    positive_combinations = get_positive_combinations(
        combinations_in_distribution,
        num_samples,
        args.min_objects,
        args.max_objects,
        logic,
        symbols,
        clevr_color,
        clevr_shapes,
        clevr_materials,
        clevr_sizes,
    )

    negative_combinations = get_negative_combinations(
        combinations_in_distribution,
        num_samples,
        args.min_objects,
        args.max_objects,
        logic,
        symbols,
        clevr_color,
        clevr_shapes,
        clevr_materials,
        clevr_sizes,
    )

    log("info", "Positive combinations", positive_combinations)
    log("info", "Negative combinations", negative_combinations)

    if combinations_in_distribution is not None:
        log(
            "info",
            "Splitting the dataset according to the given combinations, in_distribution proportion will be ignored",
        )

        log("info", "Given combinations", combinations_in_distribution)

        try:
            combinations_in_distribution = _handle_combinations(
                combinations_in_distribution,
                clevr_color,
                clevr_shapes,
                clevr_materials,
                clevr_sizes,
            )

        except ValueError as e:
            log("error", e)
            exit(1)

        log("info", "Handled combinations", combinations_in_distribution)

        # splitting according to a combination
        positive_id, positive_ood = filering_given_combinations(
            positive_combinations, combinations_in_distribution
        )

        log("debug", "Positive combinations in distribution", positive_id)

        log("debug", "Positive combinations ood distribution", positive_ood)

        negative_id, negative_ood = filering_given_combinations(
            negative_combinations, combinations_in_distribution
        )

        log("debug", "Negative combinations in distribution", negative_id)

        log("debug", "Negative combinations ood distribution", negative_ood)

        log("info", "OOD combinations", (len(positive_ood) + len(positive_ood)))
        log("info", "ID combinations", (len(positive_id) + len(negative_id)))

        if len(positive_id) + len(negative_id) == 0:
            log("error", "There must be at least one combination in distribution!")
            exit(1)

        if len(positive_ood) + len(negative_ood) == 0:
            log(
                "error",
                "There must be at least one combination out of distribution if you specify `combinations_in_distribution`!",
            )
            exit(1)

    else:
        log(
            "info",
            "Splitting the dataset according to the in_distribution proportion",
        )
        # splitting according to a proportion
        positive_id, positive_ood = split_set(
            positive_combinations, prop_in_distribution
        )
        negative_id, negative_ood = split_set(
            negative_combinations, prop_in_distribution
        )

    log("info", "positive samples in distribution", len(positive_id))
    log("info", "negative samples in distribution", len(negative_id))
    log("info", "positive samples out of distribution", len(positive_ood))
    log("info", "negative samples out of distribution", len(negative_ood))

    ood_size = 0 if (len(positive_ood) + len(negative_ood)) == 0 else ood_size

    # parallel pool
    for idx, (name, dataset_size) in enumerate(
        zip(
            ["train", "val", "test", "ood"],
            [train_size, val_size, test_size, ood_size],
        )
    ):
        log("info", "Doing", name, "with", dataset_size, "examples...")

        # positive and negative worlds to sample
        positive_to_sample = positive_id
        negative_to_sample = negative_id
        if name == "ood":
            positive_to_sample = positive_ood
            negative_to_sample = negative_ood

        # count of samples to generate
        total_positive_samples = train_size // 2

        log(
            "info",
            "Currently for",
            name,
            "I have",
            total_positive_samples,
            "positive samples of",
            len(positive_to_sample),
            "positive combinations to sample",
            "and",
            train_size - total_positive_samples,
            "negative samples of",
            len(negative_to_sample),
            "negative combinations to sample",
        )

        # Already Generated
        already_generated = set()

        scenes_paths = []
        for i in range(dataset_size):
            scenes_paths.append(
                generate_world(
                    i,
                    args,
                    total_positive_samples,
                    positive_to_sample,
                    negative_to_sample,
                    already_generated,
                    name,
                    train_templates,
                    val_templates,
                    test_templates,
                    ood_templates,
                    all_scene_paths,
                    clevr_color,
                    clevr_shapes,
                    clevr_materials,
                    clevr_sizes,
                    args.max_objects,
                )
            )

        all_scene_paths.extend(all_scene_paths)

    return all_scene_paths


def generate_templates(args, split_name):
    """Generate templates"""
    num_digits = 6
    prefix = "%s_%s_" % (args.filename_prefix, split_name)
    img_template = "%s%%0%dd.png" % (prefix, num_digits)
    scene_template = "%s%%0%dd.json" % (prefix, num_digits)
    blend_template = "%s%%0%dd.blend" % (prefix, num_digits)

    return img_template, scene_template, blend_template


def generate_directories(args, split_name, scene_subfolder):
    """Generate directories"""

    split_dir = os.path.join(args.output_image_dir, split_name)
    scene_dir = os.path.join(args.output_scene_dir, scene_subfolder, split_name)

    img_template, scene_template, blend_template = generate_templates(args, split_name)

    img_template_path = os.path.join(split_dir, img_template)
    scene_template_path = os.path.join(split_dir, scene_template)
    blend_template_path = os.path.join(split_dir, blend_template)

    _create_folder(split_dir)
    _create_folder(scene_dir)

    return img_template_path, scene_template_path, blend_template_path


def setup_directories(args):
    """Set up directories"""

    train_templates = generate_directories(args, "train", "train")
    val_templates = generate_directories(args, "val", "val")
    test_templates = generate_directories(args, "test", "test")
    ood_templates = generate_directories(args, "ood", "ood")

    if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
        os.makedirs(args.output_blend_dir)

    return train_templates, val_templates, test_templates, ood_templates


def _create_folder(directory):
    """Create folder"""

    if not os.path.isdir(directory):
        os.makedirs(directory)


def generate_dataset(args):
    """Generate datasets"""

    print_command_line_arguments(args)

    # all scenes
    all_scene_paths = []
    # read the logic stuff of the yaml file
    yaml_conf = parse_yaml(args)
    # prepare the combinations in the positive, negative and ood settings
    all_scenes = generate_proportionate_dataset(args, **yaml_conf)
    # After rendering all images, combine the JSON files for each scene into a
    # single JSON file.
    for scene_path in all_scene_paths:
        with open(scene_path, "r") as f:
            all_scenes.append(json.load(f))
    output = {
        "info": {
            "date": args.date,
            "version": args.version,
            "split": args.split,
            "license": args.license,
        },
        "scenes": all_scenes,
    }
    with open(args.output_scene_file, "w") as f:
        json.dump(output, f)


def render_scene(
    args,
    num_objects=5,
    output_index=0,
    output_split="none",
    output_image="render.png",
    output_scene="render_json",
    output_blendfile=None,
    world_to_generate=None,
    concepts=[],
    label=-1,
):
    """Render scene"""

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

    # Load materials
    clevr_utils.load_materials(args.material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.

    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    if args.use_gpu == 1:
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
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.tile_size = args.render_tile_size

    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image),
        "objects": [],
        "directions": {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add()
    plane = bpy.context.object

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    if args.camera_jitter > 0:
        for i in range(3):
            bpy.data.objects["Camera"].location[i] += rand(args.camera_jitter)

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    clevr_utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    # Add random jitter to lamp positions
    if args.key_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Key"].location[i] += rand(args.key_light_jitter)
    if args.back_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Back"].location[i] += rand(args.back_light_jitter)
    if args.fill_light_jitter > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Fill"].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    objects, blender_objects = add_random_objects(
        scene_struct, num_objects, args, camera, world_to_generate
    )

    # Render the scene and dump the scene data structure
    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    # add concepts and labels
    scene_struct["concepts"] = concepts
    scene_struct["label"] = label

    try:
        with stdout_redirected():
            bpy.ops.render.render(write_still=True)
    except Exception as e:
        log("warning", e)
        close_unused_files()

    with open(output_scene, "w") as f:
        json.dump(scene_struct, f, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


class Box:
    """Box function"""

    dim_x = 1
    dim_y = 1

    def __init__(self, min_x, min_y, max_x, max_y, dim_x=dim_x, dim_y=dim_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.dim_x = dim_x
        self.dim_y = dim_y

    @property
    def x(self):
        return round(self.min_x * self.dim_x)

    @property
    def y(self):
        return round(self.dim_y - self.max_y * self.dim_y)

    @property
    def width(self):
        return round((self.max_x - self.min_x) * self.dim_x)

    @property
    def height(self):
        return round((self.max_y - self.min_y) * self.dim_y)

    def __str__(self):
        return "<Box, x=%i, y=%i, width=%i, height=%i>" % (
            self.x,
            self.y,
            self.width,
            self.height,
        )

    def to_tuple(self):
        if self.width == 0 or self.height == 0:
            return (0, 0, 0, 0)
        return (self.x, self.y, self.width, self.height)


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.
    Negative 'z' value means the point is behind the camera.
    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.
    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    me = me_ob.to_mesh(preserve_all_data_layers=True)
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != "ORTHO"

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    # bpy.data.meshes.remove(me)
    me_ob.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    return Box(min_x, min_y, max_x, max_y, dim_x, dim_y)


def clamp(x, minimum, maximum):
    """Clamp"""
    return max(minimum, min(x, maximum))


def add_random_objects(scene_struct, num_objects, args, camera, world_to_generate=None):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(args.properties_json, "r") as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties["colors"].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties["materials"].items()]
        object_mapping = [(v, k) for k, v in properties["shapes"].items()]
        size_mapping = list(properties["sizes"].items())
        size_kv_mapping = {k: v for k, v in properties["sizes"].items()}
        object_kv_mapping = {k: v for k, v in properties["shapes"].items()}
        material_kv_mapping = {k: v for k, v in properties["materials"].items()}

    shape_color_combos = None
    if args.shape_color_combos_json is not None:
        with open(args.shape_color_combos_json, "r") as f:
            shape_color_combos = list(json.load(f).items())

    positions = []
    objects = []
    blender_objects = []

    for i in range(num_objects):
        # Choose a random size
        if world_to_generate is None:
            size_name, r = random.choice(size_mapping)
        else:
            # choose a size from the world to generate
            size_name = world_to_generate[i][SIZE_IDX]
            r = size_kv_mapping[size_name]

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    clevr_utils.delete_object(obj)
                log("warning", "failed to place the object!")
                return add_random_objects(
                    scene_struct, num_objects, args, camera, world_to_generate
                )

            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for xx, yy, rr in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ["left", "right", "front", "behind"]:
                    direction_vec = scene_struct["directions"][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < args.margin:
                        log("warning", margin, args.margin, direction_name)
                        log("warning", "BROKEN MARGIN!")
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # Choose random color and shape
        if shape_color_combos is None:
            obj_name, obj_name_out = random.choice(object_mapping)
            color_name, rgba = random.choice(list(color_name_to_rgba.items()))
        else:
            obj_name_out, color_choices = random.choice(shape_color_combos)
            color_name = random.choice(color_choices)
            obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
            rgba = color_name_to_rgba[color_name]

        # Chose a color and random from the world to generate
        if world_to_generate is not None:
            # Color and shape from the world to generate
            color_name = world_to_generate[i][COLOR_IDX]
            rgba = color_name_to_rgba[color_name]
            obj_name_out = world_to_generate[i][SHAPE_IDX]
            obj_name = object_kv_mapping[obj_name_out]

        # For cube, adjust the size a bit
        if obj_name == "Cube":
            r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        clevr_utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Attach a random material
        if world_to_generate is None:
            # random material
            mat_name, mat_name_out = random.choice(material_mapping)
        else:
            # Choose a color and random from the world to generate
            mat_name_out = world_to_generate[i][MATERIAL_IDX]
            mat_name = material_kv_mapping[mat_name_out]

        clevr_utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = clevr_utils.get_camera_coords(camera, obj.location)

        # Get 2D pixel coordinates for all 8 points in the bounding box
        scene = bpy.context.scene
        cam_ob = scene.camera
        me_ob = bpy.context.object

        bound_box = camera_view_bounds_2d(bpy.context.scene, cam_ob, me_ob)

        objects.append(
            {
                "shape": obj_name_out,
                "size": size_name,
                "material": mat_name_out,
                "3d_coords": tuple(obj.location),
                "rotation": theta,
                "pixel_coords": pixel_coords,
                "color": color_name,
                "x": bound_box.x,
                "y": bound_box.y,
                "width": bound_box.width,
                "height": bound_box.height,
            }
        )

    if args.no_occlusion:
        # Check that all objects are at least partially visible in the rendered image
        all_visible = check_visibility(blender_objects, args.min_pixels_per_object)

        if not all_visible:
            # If any of the objects are fully occluded then start over; delete all
            # objects from the scene and place them all again.
            log("warning", "Some objects are occluded; replacing objects")
            for obj in blender_objects:
                clevr_utils.delete_object(obj)
            return add_random_objects(
                scene_struct, num_objects, args, camera, world_to_generate
            )

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name == "above" or name == "below":
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 == obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix=".png")
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter(
        (p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4)
    )
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path="flat.png"):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    # old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = "CYCLES"

    # render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    # clevr_utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    # clevr_utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    # clevr_utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    # clevr_utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials["Material"]
        mat.name = "Material_%d" % i
        while True:
            r, g, b, a = [random.random() for _ in range(4)]
            if (r, g, b, a) not in object_colors:
                break
        object_colors.add((r, g, b, a))
        mat.diffuse_color = [r, g, b, a]
        # mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    # clevr_utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    # clevr_utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    # clevr_utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    # clevr_utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    # render_args.use_antialiasing = old_use_antialiasing

    return object_colors


def set_random_seed(seed):
    """
    Set the random seed for reproducibility

    Args:
        seed (int): The seed value to be set.
    """
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)


def create_folder_if_not_exists(folder_path):
    """
    Creates a folder if it does not already exist.

    Parameters:
        folder_path (str): The path of the folder to create.

    Returns:
        None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    configure_parser(parser)

    log("info", "### START ###")

    check_blender_compatibility()

    if INSIDE_BLENDER:
        argv = clevr_utils.extract_args()
        args = parser.parse_args(argv)

        # set seed
        set_random_seed(args.seed)

        # create folders
        create_folder_if_not_exists(args.output_image_dir)
        create_folder_if_not_exists(args.output_scene_dir)
        create_folder_if_not_exists(args.output_blend_dir)

        generate_dataset(args)
    elif "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
    else:
        info_str = (
            "This script is intended to be called from blender like this:" + "\n\n"
        )
        info_str += "blender --background --python clevr_renderer.py -- [args]" + "\n\n"
        info_str += "You can also run as a standalone python script to view all" + "\n"
        info_str += "arguments like this:" + "\n\n"
        info_str += "python clevr_renderer.py --help"
        log("info", info_str)

    log("info", "### END ###")


if __name__ == "__main__":
    set_log_level("info")
    main()
