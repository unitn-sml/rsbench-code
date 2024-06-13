from rssgen.parsers.yaml_parser import YamlParser
from rssgen.parsers.constraints import between_zero_one, len_not_zero, greater_than_one


class ClevrParser(YamlParser):
    """Clevr parser"""

    def __init__(self, file_path):
        super().__init__(file_path)
        self.expected_fields = {
            "val_prop": float,
            "test_prop": float,
            "ood_prop": float,
            "prop_in_distribution": float,
            "n_samples": int,
            "symbols": list,
        }
        self.constraints = {
            "val_prop": between_zero_one,
            "test_prop": between_zero_one,
            "ood_prop": between_zero_one,
            "n_samples": greater_than_one,
            "prop_in_distribution": between_zero_one,
            "symbols": len_not_zero,
        }

    def additional_constraints(self, data):
        pass
