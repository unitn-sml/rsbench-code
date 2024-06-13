from rssgen.parsers.yaml_parser import YamlParser
from rssgen.parsers.constraints import (
    greater_than_zero,
    between_zero_one,
    len_not_zero,
    greater_than_one,
)

from rssgen.generators.kandinksy_generator import ALL_SHAPES, ALL_COLORS


class KandinksyParser(YamlParser):
    """Kandinsky parser"""

    def __init__(self, file_path):
        super().__init__(file_path)
        self.expected_fields = {
            "n_figures": int,
            "n_shapes": int,
            "colors": list,
            "shapes": list,
            "val_prop": float,
            "test_prop": float,
            "prop_in_distribution": float,
            "ood_prop": float,
            "symbols": list,
            "logic": str,
            "aggregator_symbols": list,
            "aggregator_logic": str,
            "sample_size": int,
        }
        self.constraints = {
            "n_figures": greater_than_one,
            "n_shapes": greater_than_zero,
            "colors": len_not_zero,
            "shapes": len_not_zero,
            "val_prop": between_zero_one,
            "test_prop": between_zero_one,
            "ood_prop": between_zero_one,
            "prop_in_distribution": between_zero_one,
            "combinations_in_distribution": len_not_zero,
            "sample_size": greater_than_zero,
        }

    def additional_constraints(self, data):
        if not all(element in ALL_COLORS for element in data["colors"]):
            raise ValueError(f"Invalid colors! Valid ones are {ALL_COLORS}")

        if not all(element in ALL_SHAPES for element in data["shapes"]):
            raise ValueError(f"Invalid shapes! Valid ones are {ALL_SHAPES}")

        if len(data["symbols"]) > data["n_shapes"] * 2:
            raise ValueError("Cannot have more symbols than shapes per figure!")

        if len(data["symbols"]) > (len(data["shapes"]) + len(data["colors"])):
            raise ValueError(
                "Cannot have more symbols than shapes and colors in the figure!"
            )

        if len(data["aggregator_symbols"]) > data["n_figures"]:
            raise ValueError("Cannot have more aggregating symbols than figures!")

        if "combinations_in_distribution" in data:
            if len(data["combinations_in_distribution"][0]) != data["n_figures"]:
                raise ValueError(
                    "combinations_in_distribution should have the same number of values of those ones you specified n_figures!"
                )

            if len(data["combinations_in_distribution"][0][0]) != data["n_shapes"]:
                raise ValueError(
                    "combinations_in_distribution items should have the same number of values of those ones you specified n_shapes!"
                )

    def additional_logic(self):
        logic, symbols = self.logic_parser.parse(
            self.data,
            is_multiple_labels=False,
            symbols_name="aggregator_symbols",
            logic_name="aggregator_logic",
        )
        self.data["aggregator_symbols"] = symbols
        self.data["aggregator_logic"] = logic
        return logic, symbols
