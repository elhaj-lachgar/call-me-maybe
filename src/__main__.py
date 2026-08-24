"""Entry point: parses CLI args, validates inputs, and runs the pipeline."""
import argparse

from pydantic import ValidationError
from llm_sdk import Small_LLM_Model

from src.validator.methods import func_validator, prompt_validator
from src.decoding.vocab import load_vocab, build_id_to_token
from src.build_result import run_pipeline


def get_arg() -> argparse.Namespace:
    """Parse command-line arguments for the project.

    Returns:
        The parsed argparse namespace with functions_definition, input,
        output, and model attributes.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - function calling with constrained decoding"
    )
    parser.add_argument("--functions_definition", required=True, help="Path to functions_definition.json")
    parser.add_argument("--input", required=True, help="Path to function_calling_tests.json")
    parser.add_argument("--output", required=True, help="Path to write function_calling_results.json")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model name")
    return parser.parse_args()


def main() -> None:
    """Program entry point: validate inputs, load the model, and run the
    full function-calling pipeline, reporting any error gracefully."""
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
    main()
