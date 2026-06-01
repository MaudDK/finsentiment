import yaml
import os

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to load YAML: {e}")


if __name__ == "__main__":
    config = load_config("configs/qlora_mistral7b.yaml")
    print(config)