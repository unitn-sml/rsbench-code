from rssgen.parsers.yaml_parser import YamlParser
from rssgen.parsers.constraints import (
    greater_than_zero,
    between_zero_one,
    len_not_zero,
    list_between_zero_nine,
)
from rssgen.utils import log


class MNISTParser(YamlParser):
    """YAML parser for MNISTMATH"""

    def __init__(self, file_path):
        super().__init__(file_path)
        self.expected_fields = {
            "num_digits": int,
            "val_prop": float,
            "test_prop": float,
            "ood_prop": float,
            "digit_values": list,
            "symbols": list,
        }
        self.constraints = {
            "num_digits": greater_than_zero,
            "val_prop": between_zero_one,
            "test_prop": between_zero_one,
            "ood_prop": between_zero_one,
            "digit_values": list_between_zero_nine,
            "prop_in_distribution": between_zero_one,
            "combinations_in_distribution": len_not_zero,
        }

    def additional_constraints(self, data):
        """Additional constraints"""

        # Check for MNIST math
        if data["multiple_labels"]:
            log("INFO", "MNISTMATH selected")
            log("INFO", "Processing additional constraints for MNISTMATH")
            element_in_system = len(data["logic"])
            if len(data["symbols"]) > data["num_digits"]:
                raise ValueError(
                    f"Cannot have more symbols than digits! Each equation ({element_in_system}) should have {data['num_digits']} elements"
                )

            if "combinations_in_distribution" in data:
                if (
                    len(data["combinations_in_distribution"][0])
                    != data["num_digits"] * element_in_system
                ):
                    raise ValueError(
                        f"combinations_in_distribution should have the same number of digits of those ones you specified!  Each equation ({element_in_system}) should have {data['num_digits']} elements"
                    )
        else:
            log("INFO", "MNISTADD selected")
            log("INFO", "Processing additional constraints for MNISTADD")
            if len(data["symbols"]) > data["num_digits"]:
                raise ValueError("Cannot have more symbols than digits!")

            if "combinations_in_distribution" in data:
                if len(data["combinations_in_distribution"][0]) != data["num_digits"]:
                    raise ValueError(
                        "combinations_in_distribution should have the same number of digits of those ones you specified!"
                    )
