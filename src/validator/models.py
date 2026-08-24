"""Pydantic models for the two project input JSON files."""
from pydantic import BaseModel
from typing import Dict


class Prompt(BaseModel):
    """One test entry: a single natural-language user request."""
    prompt: str


class Parameter(BaseModel):
    """A single function parameter's declared type."""
    type: str


class Func(BaseModel):
    """One function definition: name, description, parameters, and
    return type, as declared in functions_definition.json."""
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter
