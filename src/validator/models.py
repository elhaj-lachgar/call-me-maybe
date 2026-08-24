from pydantic import BaseModel
from typing import Dict


class Prompt(BaseModel):
    prompt: str


class Parameter(BaseModel):
    type: str


class Func(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter
