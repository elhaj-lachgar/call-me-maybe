from pydantic import BaseModel
from typing import Dict, Literal


class Prompt(BaseModel):
    """One test entry: a single natural-language user request."""
    prompt: str


class Parameter(BaseModel):
    """A single function parameter's declared type.

    Restricted to the three types this project actually supports, so an
    invalid or misspelled type (e.g. "str", "int", "bool") is rejected
    at validation time with a clear error, instead of failing silently
    later during generation.
    """
    type: Literal["string", "boolean", "number"]


class Func(BaseModel):
    """One function definition: name, description, parameters, and
    return type, as declared in functions_definition.json."""
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter
