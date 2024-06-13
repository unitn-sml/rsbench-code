import argparse
import pathlib
import os
from . import utils, parsers, generators

from rssgen.generators.generator import generate_dataset
from rssgen.parsers.parser import parse_config


def check_range(value):
    """Check whether the input value is between 0 and 10**4

    Returns:
        value: integer value

    Raises:
        ArgumentTypeError
    """
    ivalue = int(value)
    if 0 < ivalue < 10**4:
        return ivalue
    else:
        raise argparse.ArgumentTypeError(
            f"Bro, you serious!? {value} is not in valid range (0, 10^4)."
        )


def check_yaml_file(file_path: pathlib.Path):
    """Check whether the file exists and is a YML file

    Args:
        file_path (pathlib.Path): file path

    Returns:
        None: This program does not return stuff

    Raises:
        assertion
    """
    assert os.path.exists(file_path), f"File not found: {file_path}"
    assert file_path.lower().endswith(".yaml") or file_path.lower().endswith(
        ".yml"
    ), f"Not a YAML file: {file_path}"


def get_args():
    """Parse command line arguments.

    Returns:
        argparse (ArgumentParser): argument parser
    """
    parser = argparse.ArgumentParser(
        prog="rssdatasetgen",
        description="RSs Synthetic Dataset Generation.",
    )
    parser = argparse.ArgumentParser(description="Test ArgumentParser")
    parser.add_argument("--test", help="Test argument")

    parser.add_argument(
        "config",
        metavar="FILE",
        type=pathlib.Path,
        help="YML config file that contains the specifics of the dataset to generate",
    )
    parser.add_argument(
        "dataset",
        metavar="DATASET",
        type=str,
        help="Dataset to generate",
        choices=["xor", "mnist", "kandinsky", "other"],
    )
    parser.add_argument(
        "output_dir_path",
        metavar="OUTPUT_DIR",
        type=pathlib.Path,
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--n_samples",
        type=check_range,
        metavar="N_SAMPLES",
        default=1000,
        help="Number of samples for the training set",
    )
    parser.add_argument(
        "--output-compression",
        choices=[None, "bz2", "gzip", "zip", "gzip", "tar.gz"],
        required=False,
        default=None,
        help="Output compression format.",
    )
    parser.add_argument(
        "--keep-only-compressed",
        action="store_true",
        default=False,
        required=False,
        help="Keep only the compressed folders",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "erorr", "critical"],
        required=False,
        default="info",
        help="Log level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random Seed for reproducibility. Default: 0",
    )
    subparsers = parser.add_subparsers(help="sub-commands help")
    generators.generator.configure_subparsers(subparsers)
    parsers.parser.configure_subparsers(subparsers)

    parsed_args = parser.parse_args()

    return parsed_args


def main(args):
    """Main function of the RSs Dataset Generation Program"""

    # Create a folder if it does not exits
    if not args.output_dir_path.exists():
        args.output_dir_path.mkdir(parents=True)

    # sets generation seed
    utils.set_seed(args.seed)

    # Get config instructions from yaml
    config_instructions = parse_config(args.dataset, args.config)

    # Generate dataset
    dset = generate_dataset(
        args.dataset,
        args.output_dir_path,
        config_instructions,
        args.n_samples,
        args.output_compression,
        args.keep_only_compressed,
    )


if __name__ == "__main__":
    args = get_args()

    # logging
    utils.log("info", "### START ###")

    # Setting log level
    utils.set_log_level(args.log_level)

    # main
    main(args)

    utils.log("info", "### CLOSING ###\n")
