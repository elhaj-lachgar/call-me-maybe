"""Loading the model's vocabulary and building the id<->token mappings
used throughout constrained decoding."""
from typing import Dict
import json

from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
    """Load the model's vocabulary file as a token-string -> id mapping.

    Args:
        model: The loaded LLM wrapper, used to locate the vocab file.

    Returns:
        Mapping of token string to token id.

    Raises:
        ValueError: If the vocab file is missing or not valid JSON.
    """
    try:
        path = model.get_path_to_vocab_file()
        with open(path, mode='r', encoding='utf-8') as file:
            content: Dict[str, int] = json.load(file)
        return content
    except FileNotFoundError:
        raise ValueError("file not found")
    except json.JSONDecodeError:
        raise ValueError("not valid path for json")


def build_id_to_token(vocab: Dict[str, int]) -> Dict[int, str]:
    """Build the reverse mapping of a vocabulary, once, for O(1) lookups.

    Args:
        vocab: Mapping of token string to token id.

    Returns:
        Mapping of token id to token string.
    """
    return {item_id: item for item, item_id in vocab.items()}
