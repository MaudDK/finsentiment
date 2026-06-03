from finsentiment.evaluation.metrics import load_finetuned_model, evaluate
from finsentiment.data.dataset import load_financial_phrase_bank


if __name__ == "__main__":
    config_path = "configs/qlora_mistral7b.yaml"
    model, tokenizer = load_finetuned_model(config_path)
    dataset = load_financial_phrase_bank(config_path)
    evaluate(model, tokenizer, dataset, config_path)