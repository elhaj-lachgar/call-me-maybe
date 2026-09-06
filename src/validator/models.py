from pydantic import BaseModel
from typing import Dict, Literal


class   Prompt(BaseModel):
    """One test entry: a single natural-language user request."""
    prompt: str


class Parameter(BaseModel):
    """A single function parameter's declared type."""
    type: str = Literal["number", "string", "boolean"]


class Func(BaseModel):
    """One function definition: name, description, parameters, and
    return type, as declared in functions_definition.json."""
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter
