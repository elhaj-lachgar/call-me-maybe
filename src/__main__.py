import argparse
from src.validator.methods import func_validator, prompt_validator
from typing import Any
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
from src.decoding.vocab import build_id_to_token, load_vocab
from src.build_result import run_pipeline

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
        funcs = func_validator(parser.functions_definition, [])
        prompts = prompt_validator(parser.input, [])
        model = Small_LLM_Model(model_name=parser.model)
        vocab = load_vocab(model)
        id_to_token = build_id_to_token(vocab)
        run_pipeline(
            model,
            vocab,
            id_to_token,
            prompts,
            funcs,
            parser.output
        )
    except ValueError as err:
        print(f"ERROR: {err}")
    except ValidationError as err:
        print(err.errors()[0])
    

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(f"ERROR: {err}")