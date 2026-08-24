"""Number value generation via constrained decoding.

A number has no fixed target string, so instead of the
compute_allowed/legal_words approach used for function names, each
character is checked against digit/sign/decimal-point placement rules.
"""
from typing import Set, Dict, List

from llm_sdk import Small_LLM_Model

from src.decoding.constrained import pick_best_token


def is_allowed_number_char(c: str, partial_output: str) -> bool:
    """Check whether a single character is legal at the current position
    of a number being built.

    Args:
        c: The candidate character.
        partial_output: The number text generated so far.

    Returns:
        True if c is a digit, or a leading sign ('-'/'+') with nothing
        generated yet, or a single decimal point placed after at least
        one digit.
    """
    return (
        c.isdigit()
        or ((c == "-" or c == "+") and partial_output == "")
        or (c == "." and any(ch.isdigit() for ch in partial_output) and "." not in partial_output)
    )


def is_token_allowed_as_number(token: str, partial_output: str) -> bool:
    """Check whether every character of a multi-character token is legal,
    applying is_allowed_number_char progressively as partial_output grows.

    Args:
        token: The candidate vocab token (may be more than one character).
        partial_output: The number text generated so far.

    Returns:
        True if the whole token can legally be appended.
    """
    for c in token:
        if not is_allowed_number_char(c, partial_output):
            return False
        partial_output += c
    return True


def get_number_candidate_tokens(vocab: Dict[str, int]) -> Set[str]:
    """Collect the small, fixed set of number-related vocab tokens.

    Only the ~14 possible number characters are checked, via direct
    dictionary lookup, instead of scanning the full ~150k-token vocab.

    Args:
        vocab: Mapping of token string to token id.

    Returns:
        The subset of "0123456789.-+" characters that exist as tokens
        in this vocab.
    """
    number_chars = "0123456789.-+"
    return {c for c in number_chars if c in vocab}


def compute_allowed_number_tokens(candidates: Set[str], vocab: Dict[str, int], partial_output: str) -> Set[str]:
    """Filter the number candidate tokens down to those legal right now.

    Args:
        candidates: The fixed candidate set from get_number_candidate_tokens.
        vocab: Mapping of token string to token id.
        partial_output: The number text generated so far.

    Returns:
        The subset of candidates that are legal to append right now.
    """
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
    max_len: int = 6,
) -> str:
    """Generate a number value via constrained decoding.

    Stops as soon as the model picks a JSON structural stop token
    (comma or closing brace), or after max_len characters as a safety
    net, since the digit/sign/decimal rules alone never produce an
    empty allowed set on their own.

    Args:
        model: The loaded LLM wrapper providing next-token logits.
        vocab: Mapping of token string to token id.
        id_to_token: Reverse mapping of token id to token string.
        input_ids: Growing list of token ids representing the context so
            far; mutated in place as new tokens are generated.
        max_len: Maximum number of characters to generate before forcing
            a stop.

    Returns:
        The generated number as a string (to be parsed by the caller).

    Raises:
        ValueError: If generation fails for any reason.
    """
    try:
        partial = ""
        candidates = get_number_candidate_tokens(vocab)
        stop_tokens = {",", "}"} & set(vocab.keys())
        while len(partial) < max_len:
            alloweds = compute_allowed_number_tokens(candidates, vocab, partial)
            allowed_with_stop = alloweds | stop_tokens
            if not allowed_with_stop:
                break
            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(allowed_with_stop, vocab, logits, id_to_token)
            if token in stop_tokens:
                break
            input_ids.append(vocab[token])
            partial += token
        return partial
    except Exception as e:
        raise ValueError(f"failed to constrain the content: {e}")
