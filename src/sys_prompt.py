from typing import List, Dict, Optional
import re
from src.validator.models import Func, Prompt
from llm_sdk import Small_LLM_Model
from src.encoding.encoding_prompt import encode_prompt
from src.decoding.constrained import generate_constrained
from src.decoding.number_handler import generate_number
from src.encoding.validate_token import generate_string, generate_regex_value

# --- regex few-shot examples -------------------------------------------
# Kept as data so it's easy to find and extend. A 0.6B model doesn't
# reliably generalize "regex" from a single distant example, so we give
# several short pattern -> meaning pairs, placed close to where the model
# actually has to produce one. Word-match examples use \b...\b so they
# stay consistent with the structural rule enforced in generate_regex_value
# (a regex value must start with '[' or '\').

REGEX_PATTERN_EXAMPLES = [
    ("digits / numbers", r"\d+"),
    ("letters only", r"[a-zA-Z]+"),
    ("vowels", r"[aeiouAEIOU]"),
    ("whitespace", r"\s+"),
    ("punctuation", r"[^\w\s]"),
    ("a specific word, e.g. 'cat'", r"\bcat\b"),
]


def build_regex_hint_block() -> str:
    lines = ["Common regex patterns:"]
    for description, pattern in REGEX_PATTERN_EXAMPLES:
        lines.append(f"  - {description} -> {pattern}")
    return "\n".join(lines)


def build_regex_function_example() -> str:
    text = 'Example for a function with a "regex" parameter:\n'
    text += 'Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n'
    text += 'Request: "Replace all digits in \'I have 3 cats and 7 dogs\' with X"\n'
    text += (
        'Output: {"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "I have 3 cats and 7 dogs", '
        '"regex": "\\\\d+", "replacement": "X"}}\n'
    )
    text += build_regex_hint_block() + "\n"
    return text


def _guess_regex_hint(user_prompt: str) -> Optional[str]:
    """Scan the raw user request for clues about what regex pattern fits.
    This does NOT set the final value -- it only strengthens the context
    given to the model, which still generates the actual token sequence
    itself via constrained decoding (generate_regex_value)."""
    lower = user_prompt.lower()
    if "digit" in lower or "number" in lower:
        return r"\d+"
    if "vowel" in lower:
        return r"[aeiouAEIOU]"
    match = re.search(r"'([a-zA-Z]+)'", user_prompt)
    if match:
        return r"\b" + match.group(1) + r"\b"
    return None


def _extract_prompt_numbers(user_prompt: str) -> List[str]:
    """Numbers written literally in the user request, in order of
    appearance. Used only as a hint in the cue text -- the model still
    generates the actual digits itself via constrained decoding."""
    return re.findall(r"-?\d+(?:\.\d+)?", user_prompt)


def _func_has_regex_param(func: Func) -> bool:
    return any("regex" in name.lower() for name in func.parameters)


# --- prompt construction -------------------------------------------------

def build_prompt_text(user_prompt: str, functions: List[Func]) -> str:
    """Ordering matters for small models: information placed closer to the
    generation point tends to get more weight ('recency'), so the least
    critical text (role description) goes first, and the actual user
    request goes last, right before generation starts."""
    text = "You are a function calling assistant. You must choose exactly one function and produce values for its parameters, as if filling in a JSON object.\n\n"

    text += "Available functions:\n"
    for func in functions:
        line = f"- {func.name}: {func.description}. Parameters: "
        for par in func.parameters:
            line += f"{par} ({func.parameters[par].type}) "
        text += line + "\n"
        if _func_has_regex_param(func):
            text += build_regex_function_example() + "\n"

    text += "Example:\n"
    text += 'Function: fn_add_numbers(a: number, b: number)\n'
    text += 'Request: "What is 4 plus 5?"\n'
    text += 'Output: {"name": "fn_add_numbers", "parameters": {"a": 4, "b": 5}}\n\n'

    text += f"User request: {user_prompt}\nFunction name:"
    return text


def append_text_to_input_ids(model: Small_LLM_Model, input_ids: List[int], text: str) -> None:
    """Encode extra text and append its ids to input_ids in place, so the
    model 'sees' this text as if it had generated/received it as context."""
    extra_ids = encode_prompt(model, text)
    input_ids.extend(extra_ids)


# --- orchestration ---------------------------------------------------------

def orchestrate_one_prompt(
    model: Small_LLM_Model,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    prompt: Prompt,
    functions: List[Func],
) -> Dict[str, object]:
    try:
        text = build_prompt_text(prompt.prompt, functions)
        input_ids = encode_prompt(model, text)
        legal_names = {func.name for func in functions}
        function_name = generate_constrained(
            model, vocab, id_to_token, input_ids, legal_names
        )

        matched_func = None
        for func in functions:
            if func.name == function_name:
                matched_func = func
                break

        if matched_func is None:
            raise ValueError(f"model produced unknown function name: {function_name}")

        parameters: Dict[str, object] = {}
        prompt_numbers = _extract_prompt_numbers(prompt.prompt)
        number_param_index = 0

        for param_name, param_info in matched_func.parameters.items():
            is_regex_param = param_info.type == "string" and "regex" in param_name.lower()

            if is_regex_param:
                hint = _guess_regex_hint(prompt.prompt)
                if hint is not None:
                    cue = (
                        f"\nGive ONLY a regular expression pattern for parameter '{param_name}'.\n"
                        f"Based on the request, the pattern is very likely: {hint}\n"
                        "Value: \""
                    )
                else:
                    cue = (
                        f"\nGive ONLY a regular expression pattern for parameter '{param_name}'.\n"
                        + build_regex_hint_block()
                        + "\nValue: \""
                    )
            elif param_info.type == "string":
                cue = f"\nGive ONLY the string value for parameter '{param_name}', nothing else.\nValue: \""
            elif param_info.type == "number" and number_param_index < len(prompt_numbers):
                hint_number = prompt_numbers[number_param_index]
                cue = (
                    f"\nGive ONLY the number value for parameter '{param_name}'. "
                    f"The request literally contains the number {hint_number}, likely the value needed here. "
                    "Output exactly that number, followed immediately by a comma.\nValue:"
                )
                number_param_index += 1
            else:
                cue = f"\nGive ONLY the {param_info.type} value for parameter '{param_name}', followed immediately by a comma. Do not add extra digits.\nValue:"

            append_text_to_input_ids(model, input_ids, cue)

            if param_info.type == "number":
                raw_value = generate_number(model, vocab, id_to_token, input_ids)
                try:
                    value: object = float(raw_value)
                except ValueError:
                    raise ValueError(f"model produced invalid number for '{param_name}': {raw_value!r}")
            elif is_regex_param:
                value = generate_regex_value(model, vocab, id_to_token, input_ids)
            elif param_info.type == "string":
                value = generate_string(model, vocab, id_to_token, input_ids)
            elif param_info.type == "boolean":
                value = generate_constrained(
                    model, vocab, id_to_token, input_ids, {"true", "false"}
                )
            else:
                raise ValueError(f"unsupported parameter type: {param_info.type}")
            parameters[param_name] = value

        return {"name": function_name, "parameters": parameters}
    except Exception as e:
        raise ValueError(f"failed to orchestrate prompt: {e}")
