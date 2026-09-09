from llm_sdk import Small_LLM_Model
from typing import List


def encode_prompt(model: Small_LLM_Model, text: str) -> List[int]:
    """Encode text into a list of token ids using the model's tokenizer.

    Args:
        model: The loaded LLM wrapper, used to encode the text.
        text: The raw text to encode.

    Returns:
        The encoded token ids as a plain list of ints.

    Raises:
        ValueError: If the text encodes to an empty tensor.
    """
    torser = model.encode(text)
    if torser.numel() == 0:
        raise ValueError('failed to encode prompt')
    result: List[int] = torser[0].tolist()
    return result
