from rssgen.parsers.mnist_parser import MNISTParser
from rssgen.parsers.xor_parser import XORParser
from rssgen.parsers.kandinksy_parser import KandinksyParser


class ParserFactory:
    """Factory parser for the YAML parser"""

    @staticmethod
    def create_parser(dataset, file_path):
        if dataset == "xor":
            return XORParser(file_path)
        if dataset == "mnist":
            return MNISTParser(file_path)
        if dataset == "kandinsky":
            return KandinksyParser(file_path)
        raise ValueError(f"Unsupported parser type for dataset: {dataset}")
