from typing import Set, Dict, List
from llm_sdk import Small_LLM_Model


def compute_allowed(partial_output: str, vocab: dict, legal_words: Set[str]) -> Set[str]:
    allowed = set()
    for word in legal_words:
        if not word.startswith(partial_output):
            continue
        candidate = word[len(partial_output):]
        token = ""
        i = 0
        while len(candidate) > i:
            token += candidate[i]
            if token in vocab:
                allowed.add(token)
            i += 1
    return allowed


def pick_best_token(allowed: Set[str], vocab: Dict[str, int], logits: List[float], id_to_token: Dict[int, str]) -> str:
    allowed_ids = {vocab[token] for token in allowed}
    max_id = max(allowed_ids, key=lambda x: logits[x])
    return id_to_token[max_id]


def generate_constrained(
    model: Small_LLM_Model,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    input_ids: List[int],
    legal_words: Set[str],
) -> str:
    partial = ""
    try:
        while True:
            alloweds = compute_allowed(partial, vocab, legal_words)
            if not alloweds:
                break
            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(alloweds, vocab, logits, id_to_token)
            input_ids.append(vocab[token])
            partial += token
        return partial
    except Exception as e:
        raise ValueError(f"failed to constrain the content: {e}")
