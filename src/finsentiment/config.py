import warnings
import yaml
import os
from dotenv import load_dotenv
from huggingface_hub import login

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to load YAML: {e}")

def authenticate_hf():
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        warnings.warn("Warning: No HuggingFace token found in .env")


if __name__ == "__main__":
    config = load_config("configs/qlora_mistral7b.yaml")
    print(config)