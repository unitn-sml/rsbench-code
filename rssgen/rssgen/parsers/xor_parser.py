from rssgen.parsers.yaml_parser import YamlParser
from rssgen.parsers.constraints import greater_than_zero, between_zero_one, len_not_zero


class XORParser(YamlParser):
    """MNIST Logic parser"""

    def __init__(self, file_path):
        super().__init__(file_path)
        self.expected_fields = {
            "n_digits": int,
            "val_prop": float,
            "test_prop": float,
            "ood_prop": float,
            "prop_in_distribution": float,
            "xor_rule": bool,
            "symbols": list,
            "logic": str,
        }
        self.constraints = {
            "n_digits": greater_than_zero,
            "val_prop": between_zero_one,
            "test_prop": between_zero_one,
            "ood_prop": between_zero_one,
            "prop_in_distribution": between_zero_one,
        }
        self.self_define_logic = self._generate_xor_rule_and_symbols

    def additional_constraints(self, data):
        """Check for additional constraints"""
        if len(data["symbols"]) > data["n_digits"]:
            raise ValueError("Cannot have more symbols than digits!")

        if "combinations_in_distribution" in data:
            if len(data["combinations_in_distribution"][0]) != data["n_digits"]:
                raise ValueError(
                    "combinations_in_distribution should have the same number of digits of those ones you specified!"
                )

    def _generate_xor_rule_and_symbols(self, config_vars):
        """Generate the default logic: XOR"""
        n_digits = config_vars["n_digits"]
        xor_rule = config_vars["xor_rule"]

        # DO NOT OVERRIDE IF XOR_RULE IS FALSE
        if not xor_rule:
            return config_vars["logic"], config_vars["symbols"]

        letters = "abcdefghijklmnopqrstuvwxyz"
        xor_rule = f"Xor({letters[0]}, "
        symbols = [letters[0]]

        for i in range(1, n_digits):
            quotient, remainder = divmod(i, 26)
            name = letters[remainder]

            while quotient > 0:
                quotient, remainder = divmod(quotient - 1, 26)
                name = letters[remainder] + name

            # check whether it is the last iteration
            xor_rule += f"{name}, " if i != (n_digits - 1) else f"{name})"

            symbols.append(name)

        return xor_rule, symbols
