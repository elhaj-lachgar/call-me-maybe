from typing import List, Dict
from src.validator.models import Func, Prompt
from llm_sdk import Small_LLM_Model
from src.encoding.encoding_prompt import encode_prompt
from src.decoding.constrained import generate_constrained
from src.decoding.number_handler import generate_number
from src.encoding.validate_token import generate_string

def build_prompt_text(user_prompt: str, functions: List[Func]) -> str:
    text = "You are a function calling assistant. You must choose exactly one function and produce values for its parameters, as if filling in a JSON object.\n\n"
    text += "Example:\n"
    text += 'Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n'
    text += 'Request: "Replace all digits in \'I have 3 cats and 7 dogs\' with X"\n'
    text += 'Output: {"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "I have 3 cats and 7 dogs", "regex": "\\\\d+", "replacement": "X"}}\n\n'
    text += "Available functions:\n"
    for func in functions:
        line = f"- {func.name}: {func.description}. Parameters: "
        for par in func.parameters:
            line += f"{par} ({func.parameters[par].type}) "
        text += line + "\n"
    text += f"\nUser request: {user_prompt}\nFunction name:"
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
            if param_info.type == "string":
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
