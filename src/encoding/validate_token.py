"""String and regex value generation via constrained decoding.

Handles the two "open-content" parameter types (string and regex): unlike
numbers or function names, there is no fixed target to match against, so
generation is bounded by a closing quote (") instead, with extra structural
rules for regex values.
"""
from typing import Set, Dict, List

from llm_sdk import Small_LLM_Model

from src.decoding.constrained import pick_best_token


def is_token_allowed_as_string(token: str) -> bool:
    """Check whether a vocab token is safe to use inside a JSON string.

    Args:
        token: The raw vocabulary token string to check.

    Returns:
        True if the token contains no double-quote character.
    """
    return '"' not in token


def compute_allowed_string_tokens(vocab: Dict[str, int]) -> Set[str]:
    """Collect every vocab token usable as string content.

    Args:
        vocab: Mapping of token string to token id.

    Returns:
        The set of token strings that do not contain a double quote.
    """
    allowed = set()
    for key in vocab:
        if not is_token_allowed_as_string(key):
            continue
        allowed.add(key)
    return allowed


def generate_string(
    model: Small_LLM_Model,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    input_ids: List[int],
    max_length: int = 20,
) -> str:
    """Generate a generic string value via constrained decoding.

    Content is unconstrained (any non-quote token is allowed); generation
    stops as soon as the model picks a closing double-quote, or after
    max_length tokens as a safety net.

    Args:
        model: The loaded LLM wrapper providing next-token logits.
        vocab: Mapping of token string to token id.
        id_to_token: Reverse mapping of token id to token string.
        input_ids: Growing list of token ids representing the context so
            far; mutated in place as new tokens are generated.
        max_length: Maximum number of tokens to generate before forcing a
            stop, in case the model never picks a closing quote.

    Returns:
        The decoded, stripped string value.

    Raises:
        ValueError: If generation fails for any reason.
    """
    try:
        generated_ids = []
        index = 0
        content_allowed = compute_allowed_string_tokens(vocab)
        quote_allowed = content_allowed | ({'"'} & set(vocab.keys()))

        while index < max_length:
            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(quote_allowed, vocab, logits, id_to_token)

            if token == '"':
                break

            token_id = vocab[token]
            input_ids.append(token_id)
            generated_ids.append(token_id)
            index += 1

        text: str = model.decode(generated_ids)
        if "\n" in text:
            text = text.split("\n")[0]
        return text.strip().strip("'\"")
    except Exception as e:
        raise ValueError(f"failed to generate string: {e}")


# --- regex-specific generation ------------------------------------------
# Idea (adapted from a peer's implementation, kept isolated from the
# generic generate_string above so the core logic we built together stays
# untouched): a valid regex pattern almost always either starts with a
# character class ('[') or an escape sequence ('\'), and never contains a
# literal space. Forcing this structurally, instead of relying only on
# prompting, is far more reliable for a small 0.6B model.


def get_regex_candidate_tokens(vocab: Dict[str, int]) -> Set[str]:
    """Collect vocab tokens usable inside a regex value.

    Args:
        vocab: Mapping of token string to token id.

    Returns:
        The set of token strings containing neither a double quote nor a
        literal space (regex patterns should not contain raw spaces).
    """
    return {key for key in vocab if '"' not in key and ' ' not in key}


def is_valid_regex_start_token(token: str) -> bool:
    """Check whether a token is a valid first character of a regex value.

    Args:
        token: The candidate token string.

    Returns:
        True if the token starts with '[' (character class) or '\\'
        (escape sequence), the two ways a well-formed regex snippet
        typically begins.
    """
    return len(token) > 0 and token[0] in ("[", "\\")


def generate_regex_value(
    model: Small_LLM_Model,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    input_ids: List[int],
    max_length: int = 10,
    max_token_repeats: int = 2,
) -> str:
    """Generate a regex pattern value via constrained decoding.

    Applies several extra rules on top of the generic string generator:
    the first token must start a character class or escape sequence; a
    frequency cap bans any single token once it has already appeared
    max_token_repeats times in this value, which breaks both simple
    repetition and longer oscillation loops (e.g. alternating between
    two tokens many times); and the final decoded text has any run of
    repeated backslashes collapsed to a single one, since the model has
    a strong pretrained bias toward writing JSON-escaped double
    backslashes even outside a JSON encoder.

    Args:
        model: The loaded LLM wrapper providing next-token logits.
        vocab: Mapping of token string to token id.
        id_to_token: Reverse mapping of token id to token string.
        input_ids: Growing list of token ids representing the context so
            far; mutated in place as new tokens are generated.
        max_length: Maximum number of tokens to generate before forcing a
            stop.
        max_token_repeats: Maximum number of times any single token may
            appear in one generated value before being banned for the
            rest of generation.

    Returns:
        The decoded, stripped, backslash-normalized regex pattern value.

    Raises:
        ValueError: If generation fails for any reason.
    """
    try:
        generated_ids = []
        index = 0
        candidates = get_regex_candidate_tokens(vocab)
        quote_token = {'"'} & set(vocab.keys())
        token_counts: Dict[str, int] = {}

        while index < max_length:
            if index == 0:
                allowed = {t for t in candidates if is_valid_regex_start_token(t)}
            else:
                allowed = candidates
            allowed = allowed | quote_token

            # ban any token that has already reached its repeat cap --
            # this breaks both direct repetition (A, A, A, ...) and
            # longer oscillation loops (A, B, A, B, ...) in one rule
            saturated = {t for t, c in token_counts.items() if c >= max_token_repeats}
            if saturated:
                reduced = allowed - saturated
                if reduced:
                    allowed = reduced

            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(allowed, vocab, logits, id_to_token)

            if token == '"':
                break

            token_counts[token] = token_counts.get(token, 0) + 1

            token_id = vocab[token]
            input_ids.append(token_id)
            generated_ids.append(token_id)
            index += 1

        text: str = model.decode(generated_ids)
        if "\n" in text:
            text = text.split("\n")[0]
        text = text.strip().strip("'\"")
        # collapse any run of repeated backslashes (e.g. "\\\\b" -> "\\b"):
        # the model tends to double-escape backslashes as if writing
        # directly into a JSON encoder, even though this raw text IS the
        # final value.
        while "\\\\" in text:
            text = text.replace("\\\\", "\\")
        return text
    except Exception as e:
        raise ValueError(f"failed to generate regex value: {e}")
