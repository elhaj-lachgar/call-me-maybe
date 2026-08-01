from typing import Dict, Set
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

def compute_allowed_number_tokens(vocab: Dict[str, int], partial_output: str) -> Set[str]:
    allowed = set()
    for key in vocab:
        if not is_token_allowed_as_number(key, partial_output):
            continue
        allowed.add(key)

    return allowed