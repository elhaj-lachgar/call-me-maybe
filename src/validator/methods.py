"""Loading and Pydantic validation of the two project input JSON files."""
import json
from typing import List

from src.validator.models import Func, Prompt


def func_validator(path: str, unath: List[str]) -> List[Func]:
    """Load and validate the functions_definition.json file.

    Args:
        path: Filesystem path to the functions definition JSON file.
        unath: Reserved paths that must not be used (currently unused;
            kept for interface symmetry with prompt_validator).

    Returns:
        A list of validated Func models.

    Raises:
        ValueError: If the file is missing or is not valid JSON.
        pydantic.ValidationError: If an entry doesn't match the Func
            schema; propagated to the caller to report precisely.
    """
    try:
        with open(path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        return [Func(**item) for item in data]
    except FileNotFoundError:
        raise ValueError(f"functions definition file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"functions definition file is not valid JSON: {path}")


def prompt_validator(path: str, unath: List[str]) -> List[Prompt]:
    """Load and validate the function_calling_tests.json file.

    Args:
        path: Filesystem path to the test prompts JSON file.
        unath: Reserved paths that must not be used (currently unused;
            kept for interface symmetry with func_validator).

    Returns:
        A list of validated Prompt models.

    Raises:
        ValueError: If the file is missing or is not valid JSON.
        pydantic.ValidationError: If an entry doesn't match the Prompt
            schema; propagated to the caller to report precisely.
    """
    try:
        with open(path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        raise ValueError(f"input file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"input file is not valid JSON: {path}")
