from finsentiment.model.loader import load_model
from finsentiment.config import load_config
from finsentiment.model.loras import build_lora_config, apply_lora
from finsentiment.data.dataset import load_financial_phrase_bank
from datasets import DatasetDict
from trl import SFTTrainer, SFTConfig
import wandb

def apply_chat_template(dataset, tokenizer) -> DatasetDict:
    def format_row(row):
        row['text'] = tokenizer.apply_chat_template(
            row['messages'],
            tokenize=False,
            add_generation_prompt=False
        )

        return row
    return dataset.map(format_row)



def build_trainer(model, tokenizer, dataset, config):

    sft_config = SFTConfig(
        output_dir = config['model']['output_dir'],
        num_train_epochs = config['training']['epochs'],
        per_device_train_batch_size = config['training']['batch_size'],
        gradient_accumulation_steps = config['training']['gradient_accumulation'],
        learning_rate = config['training']['learning_rate'],
        bf16 = True,
        logging_steps = 10,
        save_strategy = "epoch",
        dataset_text_field = "text",
        max_seq_length = config['training']['max_seq_length'],
        report_to = "wandb",
        run_name = config['training']['wandb_run_name'],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer
    )

    return trainer



def train(config_path: str):
    # Load config
    config = load_config(config_path)

    wandb.init(
        project=config['training']['wandb_project'], 
        name=config['training']['wandb_run_name']
    )

    # Load dataset
    dataset = load_financial_phrase_bank(config_path)

    # Load model and tokenizer
    model, tokenizer = load_model(config_path)

    # Build LoRA config and apply to model
    lora_config = build_lora_config(config_path)

    # Apply chat template to dataset
    dataset = apply_chat_template(dataset, tokenizer)


    model = apply_lora(model, lora_config)

    trainer = build_trainer(model, tokenizer, dataset, config)

    trainer.train()
    wandb.finish()

    output_dir = config['model']['output_dir']
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapted model saved to {output_dir}")