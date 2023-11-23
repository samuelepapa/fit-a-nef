import importlib
import sys
from pathlib import Path
from typing import Any, Union


def load_module(module_path: Union[Path, str]) -> Any:
    """Load a module from a path.

    Args:
        module_path (Union[Path, str]): The path to the module.

    Returns:
        Any: The module.

    Raises:
        ModuleNotFoundError: If the module is not found.
    """
    module_path = Path(module_path)
    if not module_path.exists():
        raise ModuleNotFoundError(f"Module {module_path} not found.")

    spec = importlib.util.spec_from_file_location(str(module_path.stem), str(module_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[str(module_path)] = module
    spec.loader.exec_module(module)

    # spec = importlib.util.spec_from_file_location(module_name, module_path)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = module
    # spec.loader.exec_module(module)
    return module
