# __init__.py
import importlib
import os

# Get a list of all Python files in the current directory (excluding __init__.py)
module_names = [
    file[:-3]
    for file in os.listdir(os.path.dirname(__file__))
    if file.endswith(".py") and file != "__init__.py"
]

# Import functions and constants from modules dynamically and assign them to boia_utils namespace
for module_name in module_names:
    module = importlib.import_module(f"boia_utils.{module_name}")
    for name in dir(module):
        if not name.startswith("__"):  # Exclude private attributes
            globals()[name] = getattr(module, name)
