from typing import List, Dict
from src.validator.models import Func, Prompt
from llm_sdk import Small_LLM_Model
from src.encoding.encoding_prompt import encode_prompt
from src.decoding.constrained import generate_constrained
from src.decoding.number_handler import generate_number
from src.encoding.validate_token import generate_string


REGEX_PATTERN_EXAMPLES = [
    ("digits / numbers", r"\d+"),
    ("letters only", r"[a-zA-Z]+"),
    ("vowels", r"[aeiouAEIOU]"),
    ("whitespace", r"\s+"),
    ("punctuation", r"[^\w\s]"),
    ("a specific word, e.g. 'cat'", r"cat"),
]


def build_regex_hint_block() -> str:
    """Compact list of regex pattern examples."""
    lines = ["Common regex patterns:"]
    for description, pattern in REGEX_PATTERN_EXAMPLES:
        lines.append(f"  - {description} -> {pattern}")
    return "\n".join(lines)


def build_regex_function_example() -> str:
    """A full worked request -> JSON example for a function that has a
    'regex' parameter. Injected next to that function's own definition."""
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


def _func_has_regex_param(func: Func) -> bool:
    return any("regex" in name.lower() for name in func.parameters)



def build_prompt_text(user_prompt: str, functions: List[Func]) -> str:
    """Builds the full system + user prompt text.

    Ordering matters for small models: information placed closer to the
    generation point tends to get more weight ('recency'), so the least
    critical text (role description) goes first, and the actual user
    request goes last, right before generation starts.
    """
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
        for param_name, param_info in matched_func.parameters.items():
            if param_info.type == "string" and "regex" in param_name.lower():
                cue = (
                    f"\nGive ONLY a regular expression pattern for parameter '{param_name}'.\n"
                    + build_regex_hint_block()
                    + "\nValue: \""
                )
            elif param_info.type == "string":
                cue = f"\nGive ONLY the string value for parameter '{param_name}', nothing else.\nValue: \""
            else:
                cue = f"\nGive ONLY the {param_info.type} value for parameter '{param_name}', nothing else.\nValue:"
            append_text_to_input_ids(model, input_ids, cue)

            if param_info.type == "number":
                raw_value = generate_number(model, vocab, id_to_token, input_ids)
                try:
                    value: object = float(raw_value)
                except ValueError:
                    raise ValueError(f"model produced invalid number for '{param_name}': {raw_value!r}")
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