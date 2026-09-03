"""Prompt construction and per-prompt orchestration: builds the system
prompt, generates the function name, then generates each parameter value
with a targeted cue and the appropriate constrained-decoding function."""
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
    """Build a compact, human-readable list of example regex patterns.

    Returns:
        A multi-line string listing each pattern description and example.
    """
    lines = ["Common regex patterns:"]
    for description, pattern in REGEX_PATTERN_EXAMPLES:
        lines.append(f"  - {description} -> {pattern}")
    return "\n".join(lines)


def build_regex_function_example() -> str:
    """Build a full worked request -> JSON example for a function that
    has a "regex" parameter, injected next to that function's own
    definition in the prompt.

    Returns:
        The example text block, including the pattern hint list.
    """
    text = 'Example for a function with a "regex" parameter:\n'
    text += 'Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n'
    text += 'Request: "Replace all digits in \'I have 3 cats and 7 dogs\' with X"\n'
    text += (
        'Output: {"name": "fn_substitute_string_with_regex", '
        '"parameters": {"source_string": "I have 3 cats and 7 dogs", '
        '"regex": "\\d+", "replacement": "X"}}\n'
    )
    text += build_regex_hint_block() + "\n"
    return text


def _guess_regex_hint(user_prompt: str) -> Optional[str]:
    """Scan the raw user request for clues about what regex pattern fits.
    This does NOT set the final value -- it only strengthens the context
    given to the model, which still generates the actual token sequence
    itself via constrained decoding (generate_regex_value).

    Args:
        user_prompt: The raw natural-language request.

    Returns:
        A likely regex pattern hint, or None if no clue was found.
    """
    lower = user_prompt.lower()
    if "digit" in lower or "number" in lower:
        return r"\d+"
    if "vowel" in lower:
        return r"[aeiouAEIOU]"
    if "consonant" in lower:
        return r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]"
    if "uppercase" in lower or "capital" in lower:
        return r"[A-Z]+"
    if "lowercase" in lower:
        return r"[a-z]+"
    if "punctuation" in lower:
        return r"[^\w\s]"
    if "whitespace" in lower or "space" in lower:
        return r"\s+"
    if "letter" in lower:
        return r"[a-zA-Z]+"
    match = re.search(r"'([a-zA-Z]+)'", user_prompt)
    if match:
        return r"\b" + match.group(1) + r"\b"
    return None


def _extract_prompt_numbers(user_prompt: str) -> List[str]:
    """Extract numbers written literally in the user request, in order of
    appearance. Used only as a hint in the cue text -- the model still
    generates the actual digits itself via constrained decoding.

    Args:
        user_prompt: The raw natural-language request.

    Returns:
        The literal number substrings found, in order.
    """
    return re.findall(r"-?\d+(?:\.\d+)?", user_prompt)


def _func_has_regex_param(func: Func) -> bool:
    """Check whether a function definition has a parameter whose name
    contains "regex".

    Args:
        func: The function definition to inspect.

    Returns:
        True if any parameter name contains "regex" (case-insensitive).
    """
    return any("regex" in name.lower() for name in func.parameters)


# --- prompt construction -------------------------------------------------

def build_prompt_text(user_prompt: str, functions: List[Func]) -> str:
    """Build the full prompt text sent to the model for one request.

    Ordering matters for small models: information placed closer to the
    generation point tends to get more weight ("recency"), so the least
    critical text (role description) goes first, and the actual user
    request goes last, right before generation starts.

    Args:
        user_prompt: The raw natural-language request from the test file.
        functions: All available function definitions to advertise.

    Returns:
        The full text to encode and feed to the model.
    """
    text = (
        "You are a function calling assistant. You must choose exactly "
        "one function and produce values for its parameters, as if "
        "filling in a JSON object.\n\n"
    )

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
    """Encode extra text and append its ids to input_ids in place.

    Args:
        model: The loaded LLM wrapper, used to encode the text.
        input_ids: The growing list of token ids; mutated in place.
        text: The extra text to encode and append, so the model "sees"
            it as if it had generated/received it as context.
    """
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
    """Run the full pipeline for a single prompt: build the prompt text,
    generate the function name, then generate each parameter value with
    a type-appropriate cue and constrained-decoding function.

    Args:
        model: The loaded LLM wrapper.
        vocab: Mapping of token string to token id.
        id_to_token: Reverse mapping of token id to token string.
        prompt: The validated user request to process.
        functions: All available function definitions.

    Returns:
        A dict with "name" (the chosen function name) and "parameters"
        (a dict of parameter name to generated value).

    Raises:
        ValueError: If any step of generation or matching fails.
    """
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
                cue = (
                    f"\nGive ONLY the {param_info.type} value for parameter "
                    f"'{param_name}', followed immediately by a comma. "
                    "Do not add extra digits.\nValue:"
                )

            append_text_to_input_ids(model, input_ids, cue)

            if param_info.type == "number":
                raw_value = generate_number(model, vocab, id_to_token, input_ids)
                try:
                    value: object = float(raw_value)
                except ValueError:
                    raise ValueError(f"model produced invalid number for '{param_name}': {raw_value!r}")
            elif is_regex_param:
                regex_hint = _guess_regex_hint(prompt.prompt)
                if regex_hint is not None:
                    value = generate_constrained(
                        model, vocab, id_to_token, input_ids, {regex_hint}
                    )
                else:
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
