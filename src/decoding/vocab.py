from llm_sdk import Small_LLM_Model
from typing import Dict
import json 
def load_vocab(model: Small_LLM_Model) -> Dict[str, int]:
    try:
        path = model.get_path_to_vocab_file()
        with open(path, mode='r', encoding='utf-8') as file:
            content = json.load(file)
        return content 
    except FileNotFoundError:
        raise ValueError("file not found")
    except json.JSONDecodeError:
        raise ValueError("not valid path for json")
    except Exception:
        raise ValueError("faild to get vocab")

def build_id_to_token(vocab: Dict[str, int]) -> Dict[int, str]:
    return {item_id: item for item, item_id in vocab.items()}
