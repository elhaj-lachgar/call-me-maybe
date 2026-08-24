import json
from typing import List
from pydantic import ValidationError
from src.validator.models import Func, Prompt


def func_validator(path: str, unath: List[str]) -> List[Func]:
    try:
        with open(path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        return [Func(**item) for item in data]
    except FileNotFoundError:
        raise ValueError(f"functions definition file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"functions definition file is not valid JSON: {path}")


def prompt_validator(path: str, unath: List[str]) -> List[Prompt]:
    try:
        with open(path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        return [Prompt(**item) for item in data]
    except FileNotFoundError:
        raise ValueError(f"input file not found: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"input file is not valid JSON: {path}")
