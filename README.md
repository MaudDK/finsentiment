# FinSentiment

QLoRA fine-tuning of Mistral-7B for financial sentiment analysis. Classifies financial headlines as **positive**, **neutral**, or **negative** using 4-bit quantization — runs on a single 16GB GPU.

## Results

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Negative | 0.61 | 0.69 | 0.65 |
| Neutral | 0.43 | 0.12 | 0.18 |
| Positive | 0.63 | 0.85 | 0.72 |
| **Overall** | | | **0.61** |

## Stack

- **Base model** — `mistralai/Mistral-7B-Instruct-v0.2`
- **Fine-tuning** — QLoRA (4-bit NF4) via `peft` + `bitsandbytes`
- **Trainer** — `trl.SFTTrainer` with chat template formatting
- **Dataset** — `mltrev23/financial-sentiment-analysis` (8,142 financial headlines)
- **Tracking** — Weights & Biases
- **Hardware** — Single NVIDIA GPU (16GB VRAM)

## Project Structure

```
finsentiment/
├── src/
│   └── finsentiment/
│       ├── config.py               # YAML config loader + HF/W&B auth
│       ├── data/
│       │   └── dataset.py          # Dataset loader with stratified split
│       ├── model/
│       │   ├── loader.py           # Base model + tokenizer loader
│       │   └── loras.py            # LoRA config + adapter application
│       ├── training/
│       │   └── trainer.py          # SFTTrainer setup + train loop
│       ├── evaluation/
│       │   └── metrics.py          # Classification report + confusion matrix
│       └── inference/
│           └── predictor.py        # Interactive inference loop
├── scripts/
│   ├── train.py                    # Run training
│   ├── evaluate.py                 # Run evaluation
│   └── predict.py                  # Interactive predictions
├── configs/
│   └── qlora_mistral7b.yaml        # All hyperparameters
├── .env                            # API keys (not committed)
└── pyproject.toml
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/finsentiment
cd finsentiment
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file in the project root:
```
HUGGING_FACE_TOKEN=your_hf_token
WANDB_API_KEY=your_wandb_key
```

## Usage

### Train
```bash
python scripts/train.py
```

### Evaluate
```bash
python scripts/evaluate.py
```

### Interactive predictions
```bash
python scripts/predict.py
```

```
FinSentiment — Financial Sentiment Classifier
Type a financial headline to classify. Type 'quit' to exit.

Headline: Apple reports record quarterly earnings beating analyst expectations
Sentiment: positive

Headline: Company files for bankruptcy amid mounting debt
Sentiment: negative

Headline: Board of directors announces annual general meeting
Sentiment: neutral
```

## Configuration

All hyperparameters live in `configs/qlora_mistral7b.yaml`. Key settings:

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"

lora:
  r: 16           # rank — higher = more capacity, more VRAM
  alpha: 32       # scaling factor (2× rank)
  dropout: 0.05

quantization:
  bits: 4
  quant_type: "nf4"
  compute_dtype: "bfloat16"

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation: 4    # effective batch = 16
  learning_rate: 0.0001
```

## How it works

1. **Dataset** — Financial headlines are formatted as instruction-tuning prompts using Mistral's chat template and split into 80/20 train/test with stratification
2. **Quantization** — Mistral-7B is loaded in 4-bit NF4 format, reducing VRAM from ~28GB to ~4GB
3. **LoRA** — Small trainable adapter matrices are injected into attention and MLP layers (~0.57% of total parameters trained)
4. **Training** — `SFTTrainer` computes loss only on the label tokens, not the instruction prefix
5. **Best model** — Checkpoint with lowest eval loss is saved automatically

## Extending to other domains

Swap the dataset and update the prompt in `dataset.py`:

```yaml
# configs/qlora_mistral7b.yaml
data:
  dataset_name: "your-dataset"
  text_column: "your_text_column"
  label_column: "your_label_column"
```

```python
# src/finsentiment/data/dataset.py
"Classify the urgency of this medical note as low, medium, or high."
```
