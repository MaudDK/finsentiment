from datasets import load_dataset
from finsentiment.config import load_config

def load_financial_phrase_bank(config_path):
    config = load_config(config_path)
    dataset_name = config['data']['dataset_name']

    print(f"Loading dataset '{dataset_name}'")
    dataset = load_dataset(dataset_name)
    return dataset

if __name__ == "__main__":
    dataset = load_financial_phrase_bank("configs/qlora_mistral7b.yaml")
    print(dataset)