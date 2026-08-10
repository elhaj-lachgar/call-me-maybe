from src.validator.models import Prompt, Func
from typing import Dict, List
from llm_sdk import Small_LLM_Model
from src.sys_prompt import orchestrate_one_prompt
import json
import os
def build_result_object(prompt: Prompt, orchestration_result: Dict[str, object]) -> Dict[str, object]:
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