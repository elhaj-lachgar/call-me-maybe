from llm_sdk import Small_LLM_Model
from typing import List


def encode_prompt(model: Small_LLM_Model, text: str) -> List[int]:
    """Load the model's vocabulary file as a token-string -> id mapping.

    Args:
        model: The loaded LLM wrapper, used to locate the vocab file.

    Returns:
        Mapping of token string to token id.

    Raises:
        ValueError: If the vocab file is missing or not valid JSON.
    """
    torser = model.encode(text)
    if torser.numel() == 0:
        raise ValueError('failed to encode prompt')
    result: List[int] = torser[0].tolist()
    return result
