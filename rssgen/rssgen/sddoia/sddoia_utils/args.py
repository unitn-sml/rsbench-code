import argparse
from datetime import datetime as dt


def configure_parser(parser) -> None:
    """
    Configure a new subparser for running the Clevr preparation
    Args:
        Subparser (Subparser): argument parser
    """
    # Input options
    parser.add_argument(
        "--material_dir",
        default="../../boia_config/materials/",
        help="Directory where .blend files for materials are stored",
    )

    # Output settings
    parser.add_argument(
        "--n_samples", default=10000, type=int, help="The number of images to render"
    )
    parser.add_argument(
        "--filename_prefix",
        default="MINI_BOIA",
        help="This prefix will be prepended to the rendered images and JSON scenes",
    )
    parser.add_argument(
        "--output_image_dir",
        default="../../mini_boia_out_2/",
        help="The directory where output images will be stored. It will be "
        + "created if it does not exist.",
    )
    parser.add_argument(
        "--output_scene_dir",
        default="../../mini_boia_out_2/scenes/",
        help="The directory where output JSON scene structures will be stored. "
        + "It will be created if it does not exist.",
    )
    parser.add_argument(
        "--output_scene_file",
        default="../../mini_boia_out_2/mini_boia_scenes.json",
        help="Path to write a single JSON file containing all scene information",
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
        default=469,
        type=int,
        help="The width (in pixels) for the rendered images",
    )
    parser.add_argument(
        "--height",
        default=387,
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
        "--location_jitter",
        default=0.5,
        type=float,
        help="The magnitude of random jitter to add to the objects position",
    )
    parser.add_argument(
        "--rotation_jitter",
        default=0.5,
        type=float,
        help="The magnitude of random jitter to add to the object rotation",
    )
    parser.add_argument(
        "--scale_jitter",
        default=0.1,
        type=float,
        help="The magnitude of random jitter to add to the object scale",
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
        "--render_tile_size",
        default=32,
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

    # Internal settings
    parser.add_argument(
        "--n_config",
        default=100,
        type=int,
        help="The number of configurations to sample from the BN",
    )
    parser.add_argument(
        "--load_status_dict",
        default=False,
        action="store_true",
        help="Load the status dictionary, if an incomplete version has been generated previously",
    )
    parser.add_argument(
        "--status_log",
        default="../../mini_boia_out_2/status_log.json",
        help="Status log file to keep track of the generated images and the chosen combinations",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--train_ratio",
        default=0.7,
        type=float,
        help="The ratio of the dataset to be used for training",
    )
    parser.add_argument(
        "--val_ratio",
        default=0.15,
        type=float,
        help="The ratio of the dataset to be used for validation",
    )
    parser.add_argument(
        "--test_ratio",
        default=0.15,
        type=float,
        help="The ratio of the dataset to be used for testing",
    )
    parser.add_argument(
        "--ood_ratio",
        default=0.1,
        type=float,
        help="The ratio of the dataset to be used for OOD",
    )
    parser.add_argument(
        "--in_dist_ratio",
        default=0.9,
        type=float,
        help="The ratio of the dataset to be used for in-distribution",
    )
    parser.add_argument(
        "--use_ood_knowledge",
        default=True,
        action="store_true",
        help="Use ood knowledge in the ood dataset",
    )
