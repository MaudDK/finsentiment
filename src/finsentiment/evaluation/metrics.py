import torch
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from finsentiment.model.loader import load_model
from finsentiment.config import load_config
from finsentiment.data.dataset import load_financial_phrase_bank

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

def load_finetuned_model(config_path: str):
    """Load base model with trained LoRA adapter for inference"""
    config = load_config(config_path)
    model, tokenizer = load_model(config_path, inference=True)
    model = PeftModel.from_pretrained(model, config['model']['output_dir'])
    model.eval()
    return model, tokenizer

@torch.inference_mode()
def predict(model, tokenizer, text: str) -> str:
    """Run inference on a single sample"""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Classify the sentiment of this financial headline as negative, neutral, or positive.\n\nHeadline: {text}"}],
        tokenize = False,
        add_generation_prompt = True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    new_tokens = output[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate(model, tokenizer, dataset, config_path: str):
    true_labels = []
    pred_labels = []

    for i, sample in tqdm(enumerate(dataset['test']), total=len(dataset['test'])):
        true = LABEL_MAP[sample['label']]
        pred = predict(model, tokenizer, sample['Title']).lower().strip()


        #If Predicts something not in label map
        if pred not in LABEL_MAP.values():
            pred = "unknown"
            print(f"Unknown prediction for sample {i}: {sample['Title']} -> {pred}")

        true_labels.append(true)
        pred_labels.append(pred)
    

    print("\n" + "═" * 50)
    print("CLASSIFICATION REPORT")
    print("═" * 50)
    print(classification_report(true_labels, pred_labels, target_names=list(LABEL_MAP.values())))

    print("CONFUSION MATRIX  (rows=true, cols=predicted)")
    print("             neg    neu    pos")
    cm = confusion_matrix(true_labels, pred_labels, labels=list(LABEL_MAP.values()))
    for label, row in zip(LABEL_MAP.values(), cm):
        print(f"  {label:8s}  {row[0]:5d}  {row[1]:5d}  {row[2]:5d}")

if __name__ == "__main__":
    config_path = "configs/qlora_mistral7b.yaml"
    model, tokenizer = load_finetuned_model(config_path)
    dataset = load_financial_phrase_bank(config_path)
    evaluate(model, tokenizer, dataset, config_path)
