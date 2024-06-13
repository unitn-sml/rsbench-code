from rssgen.generators.generator_factory import SyntheticDatasetFactory
from rssgen.utils import log


def configure_subparsers(subparsers) -> None:
    """Configure the subparsers for additional arguments."""
    pass


def generate_dataset(
    dataset,
    output_path,
    config_instruction,
    number_of_samples,
    output_compressed,
    keep_only_compressed,
):
    """Generate the dataset according to the configuration instruction"""

    dataset_generator = SyntheticDatasetFactory.create_dataset(
        dataset, output_path, **config_instruction
    )

    log("info", "Generating dataset...")
    dataset_generator.generate_dataset(
        num_samples=number_of_samples,
        compression_type=output_compressed,
        keep_only_compressed=keep_only_compressed,
        **config_instruction
    )
    log("info", "Done!")
