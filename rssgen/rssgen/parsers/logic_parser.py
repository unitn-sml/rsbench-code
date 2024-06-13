from sympy import symbols, sympify
import re
from rssgen.utils import log


class LogicParser:
    """Parser of the Logic"""

    def __init__(self):
        pass

    def parse(
        self,
        yaml_config,
        is_multiple_labels=False,
        symbols_name="symbols",
        logic_name="logic",
    ):
        self.symbols = yaml_config.get(symbols_name, [])
        self.logic_expr = (
            yaml_config.get(logic_name, [])
            if is_multiple_labels
            else yaml_config.get(logic_name, "")
        )

        # Multilabel logic
        try:
            if is_multiple_labels:
                for rule in self.logic_expr:
                    self.validate_symbols(self.symbols, rule)
                    self.validate_logic(self.symbols, rule)
            else:
                self.validate_symbols(self.symbols, self.logic_expr)
                self.validate_logic(self.symbols, self.logic_expr)
        except Exception as e:
            log("error", e)
            exit(1)

        # Multilabel logic expression
        if is_multiple_labels:
            logic_expr = [
                self.get_logic_expression(self.symbols, rule)
                for rule in self.logic_expr
            ]
            return [logic_expr, self.symbols]
        else:
            return [
                self.get_logic_expression(self.symbols, self.logic_expr),
                self.symbols,
            ]

    def validate_symbols(self, symbols, logic_expr):
        """Validate symbols"""
        logic_symbols = set(re.findall(r"[A-Za-z_]\w*", logic_expr))

        # Add allowed symbols for sympy to the one provided by the logic
        allowed_symbols = set(symbols) | {
            "Ne",
            "Eq",
            "&",
            "|",
            "~",
            "^",
            "And",
            "Or",
            "Xor",
            "Not",
        }

        for symbol in logic_symbols:
            if symbol not in allowed_symbols:
                raise ValueError(f"Symbol '{symbol}' is not defined.")

    def compile_formula(self, symbols_var, logic_expr):
        """Compile Sympy formula"""
        sym_dict = {s: symbols(s) for s in symbols_var}
        exp = sympify(logic_expr, locals=sym_dict)
        return exp

    def validate_logic(self, symbols, logic_expr):
        """Valdate logic"""
        exp = self.compile_formula(symbols, logic_expr)
        if isinstance(exp, int):
            raise ValueError(
                "The logic expression must result in a valid compiled formula, not a constant value."
            )

    def get_logic_expression(self, symbols, logic_expr):
        """Get logic expression"""
        return self.compile_formula(symbols, logic_expr)
