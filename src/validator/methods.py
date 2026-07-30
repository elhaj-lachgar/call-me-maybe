import json
from .models import Prompt, Func
from typing import Any, List

def func_validator(path: str, unath: List[str]) -> Any:
    try:
        if path in unath:
            raise ValueError("path is reserve for project file")
        with open(path, encoding="utf-8") as file:
            content = json.load(file)
        return [Func(**dic) for dic in content]
    except json.JSONDecodeError:
        raise ValueError("invalid json file")
    except FileNotFoundError:
        raise ValueError("file not found")
    except IsADirectoryError:
        raise ValueError("path for dir not file")


def prompt_validator(path: str, unath: List[str]) -> Any:
    try:
        if path in unath:
            raise ValueError("path is reserve for project file")
        with open(path, encoding="utf-8") as file:
            content = json.load(file)
        return [Prompt(**dic) for dic in content]
    except json.JSONDecodeError:
        raise ValueError("invalid json file")
    except FileNotFoundError:
        raise ValueError("file not found")
    except IsADirectoryError:
        raise ValueError("path for dir not file")