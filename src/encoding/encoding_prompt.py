from llm_sdk import Small_LLM_Model
from typing import List


def encode_prompt(model: Small_LLM_Model, text: str) -> List[int]:
    torser = model.encode(text)
    if torser.numel() == 0:
        raise ValueError('failed to encode prompt')
    result: List[int] = torser[0].tolist()
    return result
