"""Utils module"""

import sympy as sp


def get_exp(symbols, logic_expression):
    sym_dict = {s: sp.symbols(s) for s in symbols}
    exp = sp.sympify(logic_expression, locals=sym_dict)
    return exp


def evaluate_logic(values, logic_expression, symbols_names):
    substitutions_dict = {
        symbol_name: value for symbol_name, value in zip(symbols_names, values)
    }
    print("Substituting...", substitutions_dict)
    result = logic_expression.subs(substitutions_dict)
    return result
