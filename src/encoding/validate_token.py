from typing import Set, Dict, List
from llm_sdk import Small_LLM_Model
from src.decoding.constrained import pick_best_token


def is_token_allowed_as_string(token: str) -> bool:
    return '"' not in token


def compute_allowed_string_tokens(vocab: Dict[str, int]) -> Set[str]:
    allowed = set()
    for key in vocab:
        if not is_token_allowed_as_string(key):
            continue
        allowed.add(key)
    return allowed


def generate_string(model: Small_LLM_Model,
                     vocab: Dict[str, int],
                     id_to_token: Dict[int, str],
                     input_ids: List[int],
                     max_length: int = 20) -> str:
    """Generic string generator: any content, stops at a closing quote."""
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

        text = model.decode(generated_ids)
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
    """Tokens usable inside a regex value: no quotes, no literal spaces."""
    return {key for key in vocab if '"' not in key and ' ' not in key}


def is_valid_regex_start_token(token: str) -> bool:
    return len(token) > 0 and token[0] in ("[", "\\")


def generate_regex_value(model: Small_LLM_Model,
                          vocab: Dict[str, int],
                          id_to_token: Dict[int, str],
                          input_ids: List[int],
                          max_length: int = 10) -> str:
    try:
        generated_ids = []
        index = 0
        candidates = get_regex_candidate_tokens(vocab)
        quote_token = {'"'} & set(vocab.keys())
        last_token = None
        repeat_count = 0

        while index < max_length:
            if index == 0:
                allowed = {t for t in candidates if is_valid_regex_start_token(t)}
            else:
                allowed = candidates
            allowed = allowed | quote_token

            # break degenerate repetition loops: if the same token has been
            # picked twice in a row, ban it for this step so the model is
            # forced to choose something else (or the stop token)
            if last_token is not None and repeat_count >= 2 and last_token in allowed:
                allowed = allowed - {last_token}
                if not allowed:
                    break

            logits = model.get_logits_from_input_ids(input_ids)
            token = pick_best_token(allowed, vocab, logits, id_to_token)

            if token == '"':
                break

            if token == last_token:
                repeat_count += 1
            else:
                repeat_count = 1
                last_token = token

            token_id = vocab[token]
            input_ids.append(token_id)
            generated_ids.append(token_id)
            index += 1

        text = model.decode(generated_ids)
        if "\n" in text:
            text = text.split("\n")[0]
        return text.strip().strip("'\"")
    except Exception as e:
        raise ValueError(f"failed to generate regex value: {e}")
