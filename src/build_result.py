"""Assembling per-prompt results into the final output JSON file."""
from typing import Dict, List
import json
import os

from src.validator.models import Prompt, Func
from llm_sdk import Small_LLM_Model
from src.sys_prompt import orchestrate_one_prompt


def build_result_object(prompt: Prompt, orchestration_result: Dict[str, object]) -> Dict[str, object]:
    """Combine a prompt and its orchestration result into one output entry.

    Args:
        prompt: The original validated Prompt.
        orchestration_result: The dict returned by orchestrate_one_prompt,
            containing "name" and "parameters".

    Returns:
        A dict with "prompt", "name", and "parameters" keys, matching the
        required output schema.
    """
    return {
        "prompt": prompt.prompt,
        "name": orchestration_result["name"],
        "parameters": orchestration_result["parameters"]
    }


def run_pipeline(
    model: Small_LLM_Model,
    vocab: Dict[str, int],
    id_to_token: Dict[int, str],
    prompts: List[Prompt],
    functions: List[Func],
    output_path: str,
) -> None:
    """Run the full pipeline over every prompt and write the results file.

    Each prompt is processed independently: if one fails, it is skipped
    with a logged message and the rest continue, so a single bad prompt
    never aborts the whole run.

    Args:
        model: The loaded LLM wrapper.
        vocab: Mapping of token string to token id.
        id_to_token: Reverse mapping of token id to token string.
        prompts: All validated test prompts to process.
        functions: All available function definitions.
        output_path: Filesystem path to write the results JSON file to.

    Raises:
        ValueError: If the output file cannot be written.
    """
    results: List[Dict[str, object]] = []

    for prompt in prompts:
        try:
            obj = orchestrate_one_prompt(model, vocab, id_to_token, prompt, functions)
            res = build_result_object(prompt, obj)
            results.append(res)
        except ValueError as e:
            print(f"skipping prompt {prompt.prompt!r}: {e}")
            continue

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, mode="w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    except OSError as e:
        raise ValueError(f"failed to write output file: {e}")
