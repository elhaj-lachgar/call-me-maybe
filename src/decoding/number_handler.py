from llm_sdk import Small_LLM_Model
from typing import Dict, Set, List
from .constrained import pick_best_token
def is_allowed_number_char(c: str, partial_output: str) -> bool:
    return (
        c.isdigit()
        or ((c == "-" or c == "+") and partial_output == "")
        or (c == "." and any(ch.isdigit() for ch in partial_output) and "." not in partial_output)
    )

def is_token_allowed_as_number(token: str, partial_output: str) -> bool:
    for c in token:
        if not is_allowed_number_char(c, partial_output):
            return False
        partial_output += c
    return True

def get_number_candidate_tokens(vocab: Dict[str, int]) -> Set[str]:
    """Tokens made only of digits/./-/+ -- context-independent, so this can be
    computed once and reused across every step, instead of rescanning the
    full ~150k-token vocab at each generation step."""
    number_chars = set("0123456789.-+")
    return {key for key in vocab if all(c in number_chars for c in key)}

def compute_allowed_number_tokens(candidates: Set[str], vocab: Dict[str, int], partial_output: str) -> Set[str]:
    allowed = set()
    for key in candidates:
        if not is_token_allowed_as_number(key, partial_output):
            continue
        allowed.add(key)

    return allowed

def generate_number(
        model: Small_LLM_Model,
        vocab: Dict[str, int],
        id_to_token: Dict[int, str],
        input_ids: List[int],
        max_len : int = 10
        ) -> str:
    try:
        partial = ""
        alloweds = {"+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", "}", "\n"}
        stop_tokens = {",", "}", "\n"} & set(vocab.keys())
        while len(partial) < max_len:
            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(
                alloweds,
                vocab,
                logits,
                id_to_token
            )
            if token in stop_tokens:
                break
            input_ids.append(vocab[token])
            partial += token
        return partial
    except Exception:
        raise ValueError('failed to constrain the content')