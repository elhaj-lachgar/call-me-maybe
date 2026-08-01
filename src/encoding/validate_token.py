from typing import Dict, Set, List
from llm_sdk import Small_LLM_Model
from src.decoding.constrained import pick_best_token

def is_token_allowed_as_string(token: str) -> bool:
    return '"' not in token

def compute_allowed_string_tokens(vocab: Dict[str, int]) -> Set[str]:
    allwod = set()
    for key in vocab:
        if not is_token_allowed_as_string(key):
            continue
        allwod.add(key)
    return allwod

def generate_string(model: Small_LLM_Model,
                    vocab: Dict[str, int],
                    id_to_token: Dict[int, str],
                    input_ids: List[int],
                    max_length:int=50) -> str:
    try:
        content = ""
        index = 0
        allowed = compute_allowed_string_tokens(vocab)
        while index < max_length:
            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(allowed, vocab, logits, id_to_token)
            input_ids.append(vocab[token])
            content += token
            index += 1

        return content
    except Exception:
        raise ValueError("failed to generate string.")