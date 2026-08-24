"""Constrained decoding core: computing which vocab tokens are legal to
generate next, given a fixed set of target strings (used for function
names and boolean values)."""
from typing import Set, Dict, List

from llm_sdk import Small_LLM_Model


def compute_allowed(partial_output: str, vocab: Dict[str, int], legal_words: Set[str]) -> Set[str]:
    """Compute which vocab tokens keep partial_output a valid prefix of at
    least one string in legal_words.

    Args:
        partial_output: The text generated so far for this value.
        vocab: Mapping of token string to token id.
        legal_words: The fixed set of exact target strings (e.g. function
            names, or {"true", "false"} for booleans).

    Returns:
        The set of token strings that would keep partial_output on track
        toward at least one legal word.
    """
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
    """Mask out every id not in `allowed` and return the highest-scoring
    allowed token.

    Args:
        allowed: The set of legal token strings for this step.
        vocab: Mapping of token string to token id.
        logits: Raw next-token scores, indexed by token id.
        id_to_token: Reverse mapping of token id to token string.

    Returns:
        The token string with the highest logit among `allowed`.
    """
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
    """Generate one of a fixed set of exact target strings via constrained
    decoding (e.g. a function name, or "true"/"false" for a boolean).

    Args:
        model: The loaded LLM wrapper providing next-token logits.
        vocab: Mapping of token string to token id.
        id_to_token: Reverse mapping of token id to token string.
        input_ids: Growing list of token ids representing the context so
            far; mutated in place as new tokens are generated.
        legal_words: The fixed set of exact strings the output must match.

    Returns:
        The generated string, guaranteed to be one of `legal_words`.

    Raises:
        ValueError: If generation fails for any reason.
    """
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
