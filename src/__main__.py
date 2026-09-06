import argparse
import time
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model

from src.validator.methods import func_validator, prompt_validator
from src.decoding.vocab import load_vocab, build_id_to_token
from src.build_result import run_pipeline


def format_validation_error(source_name: str, err: ValidationError) -> str:
    """Turn a pydantic ValidationError into a clear, human-readable message.

    Args:
        source_name: Which input file this error came from, for context.
        err: The ValidationError raised by pydantic.

    Returns:
        A multi-line message listing each problem: where it is (the field
        path) and what's wrong, instead of the raw pydantic error dump.
    """
    lines = [f"Validation error in {source_name}:"]
    for problem in err.errors():
        location = ".".join(str(part) for part in problem["loc"]) or "(root)"
        lines.append(f"  - field '{location}': {problem['msg']}")
    return "\n".join(lines)


def get_arg() -> argparse.Namespace:
    """Parse command-line arguments for the project.

    Returns:
        The parsed argparse namespace with functions_definition, input,
        output, and model attributes.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - function calling with constrained decoding"
    )
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json",
                        help="Path to functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json",
                        help="Path to function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.jso",
                        help="Path to write function_calling_results.json")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Model name")
    return parser.parse_args()


def main() -> None:
    """Program entry point: validate inputs, load the model, and run the
    full function-calling pipeline, reporting any error gracefully."""
    parser = get_arg()

    try:
        funcs = func_validator(parser.functions_definition, [])
    except ValidationError as err:
        print(format_validation_error(parser.functions_definition, err))
        return
    except ValueError as err:
        print(f"ERROR: {err}")
        return

    try:
        prompts = prompt_validator(parser.input, [])
    except ValidationError as err:
        print(format_validation_error(parser.input, err))
        return
    except ValueError as err:
        print(f"ERROR: {err}")
        return

    try:
        model = Small_LLM_Model(model_name=parser.model)
        start = time.time()
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
        end = time.time()
        minute = (end - start) // 60
        seconds = (end - start) - (minute * 60)
        print(f"Simulation take: {minute} min {seconds} seconds")
    except ValueError as err:
        print(f"ERROR: {err}")
    except ValidationError as err:
        print(format_validation_error("model/pipeline", err))
    except Exception as err:
        print(f"ERROR: {err}")


if __name__ == "__main__":
    main()
