import argparse
from src.validator.methods import func_validator, prompt_validator
from typing import Any
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.decoding.vocab import build_id_to_token, load_vocab


def get_arg() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calls.json")
    parser.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")

    return parser.parse_args()

def main() -> None:
    try: 
        parser = get_arg()
        func = func_validator(parser.functions_definition, [])
        propmt = prompt_validator(parser.input, [])
        model = Small_LLM_Model(model_name=parser.model)
        vocab = load_vocab(model)
        id_to_token = build_id_to_token(vocab)
        print(len(vocab), len(id_to_token))
        print(id_to_token[0], id_to_token[1])
    except ValueError as err:
        print(f"ERROR: {err}")
    except ValidationError as err:
        print(err.errors()[0])
    

main()