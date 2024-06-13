from rssgen.generators.kandinksy_generator import SyntheticKandinksyGenerator
from rssgen.generators.xor_generator import SyntheticXORGenerator
from rssgen.generators.mnist_generator import SyntheticMNISTGenerator


class SyntheticDatasetFactory:
    """Factory pattern for synthetic data generation"""

    @staticmethod
    def create_dataset(dataset_type, output_path, **kwargs):
        if dataset_type == "xor":
            return SyntheticXORGenerator(output_path, **kwargs)
        elif dataset_type == "mnist":
            return SyntheticMNISTGenerator(output_path, **kwargs)
        elif dataset_type == "kandinsky":
            return SyntheticKandinksyGenerator(output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
