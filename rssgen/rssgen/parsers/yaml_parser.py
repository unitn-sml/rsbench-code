import yaml
import pathlib
from rssgen.parsers.logic_parser import LogicParser
from rssgen.utils import log


class YamlParser:
    """YAML parser, class which parses the yaml fields and checks whether they comply with
    some constraints
    """

    def __init__(self, file_path: pathlib.Path):
        self.file_path = file_path
        self.data = None
        self.expected_fields = None
        self.constraints = None
        self.self_define_logic = None
        self.logic_parser = LogicParser()

    def parse(self):
        """Parse YAML content

        Returns:
            parsed data
        """
        try:
            with open(self.file_path, "r") as file:
                self.data = yaml.safe_load(file)
            if self.data is not None:
                self.validate_fields()
            log("info", "YAML file successfully parsed.")
        except FileNotFoundError:
            log("error", f"Error: File not found - {self.file_path}")
            exit(1)
        except yaml.YAMLError as e:
            log("error", f"Error parsing YAML file - {e}")
            exit(1)
        except ValueError as ve:
            log("error", f"Validation error in parsing the YML file: {ve}")
            exit(1)

        # deal case in which the logic is already predefined
        if self.self_define_logic is not None:
            logic, symbols = self.self_define_logic(self.data)
            log("info", "Using logic:", logic)
            log("info", "Using symbols", symbols)
            self.data["logic"] = logic
            self.data["symbols"] = symbols

        # fix multilabel when not wanted, maybe mistaken: it is a squeeze
        if isinstance(self.data["logic"], list) and len(self.data["logic"]) == 1:
            self.data["logic"] = self.data["logic"][0]

        is_multiple_labels = False
        if isinstance(self.data["logic"], list):
            is_multiple_labels = True

        # Parse Logic
        try:
            logic, symbols = self.logic_parser.parse(self.data, is_multiple_labels)
        except ValueError as ve:
            log("error", f"Validation error in parsing the logic: {ve}")
            exit(1)

        try:
            additional_logic, additional_symbols = self.additional_logic()
        except NotImplementedError as ne:
            log("info", "No additional rules provided")
        except ValueError as ve:
            log("error", f"Validation error in parsing the additional logic: {ve}")
            exit(1)

        # Update the fields with compiled logic and symbols
        self.data["logic"] = logic
        self.data["symbols"] = symbols
        self.data["multiple_labels"] = is_multiple_labels

        try:
            self.additional_constraints(self.data)
        except ValueError as ve:
            log("error", f"Validation error in checking additional constraints: {ve}")
            exit(1)

        # Return parsed stuff
        return self.data

    def validate_fields(self):
        """Verifies the expected fields are present, have the expected data-types and respect the datatypes"""
        for name, expected_type in self.expected_fields.items():
            if name not in self.data:
                raise ValueError(f"Field '{name}' not found in the YAML file.")

            actual_value = self.data[name]

            if not isinstance(actual_value, expected_type):
                raise ValueError(
                    f"Field '{name}' has unexpected type. Expected {expected_type}, got {actual_value} of type {type(actual_value)}."
                )

            if self.constraints and name in self.constraints:
                constraint_func = self.constraints[name]
                if not constraint_func(self.data[name]):
                    raise ValueError(
                        f"Constraint violation: Field '{name}' does not meet the specified condition."
                    )

    def additional_constraints(self, data):
        pass

    def additional_logic(self):
        raise NotImplementedError("Additional Logic is not implemented")
