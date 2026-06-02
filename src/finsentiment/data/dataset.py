from datasets import load_dataset, ClassLabel, DatasetDict
from finsentiment.config import load_config, authenticate_hf

authenticate_hf()

def load_financial_phrase_bank(config_path: str) -> DatasetDict:
    """
    Load and prepare the financial sentiment dataset for fine-tuning.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        DatasetDict with 'train' and 'test' splits, containing
        'messages' (list of dicts) and 'label' (ClassLabel) columns.
    """
    config = load_config(config_path)

    test_size = config['data']['test_size']
    shuffle = config['data']['shuffle']
    seed = config['data']['seed']
    dataset_name = config['data']['dataset_name']

    # Map sentiment labels (-1, 0, 1) to string labels
    LABEL_MAP = {
        -1: "negative",
         0: "neutral",
         1: "positive"
    }

    dataset = load_dataset(dataset_name)

    def format_sample(sample: dict) -> dict:
        """
        Format raw dataset sample into an instruction-tuning message format.

        Args:
            sample (dict): Raw dataset row with 'Title' and 'Global Sentiment' fields.

        Returns:
            dict: formated sample with 'messages' (list[dicts]) and 'label' (str)
        """
        label_string = LABEL_MAP[sample['Global Sentiment']]
        sample["messages"] = [
            {
                "role": "user",
                "content": f"Classify the sentiment of this financial headline as negative, neutral, or positive.\n\nHeadline: {sample['Title']}"
            },
            {
                "role": "assistant",
                "content": label_string
            }
        ]

        sample['label'] = label_string
        return sample

    dataset = dataset.map(format_sample)

    # Cast to ClassLabel so train_test_split can stratify by label
    dataset = dataset['train'].cast_column(
        "label",
        ClassLabel(names=["negative", "neutral", "positive"])
    )

    dataset = dataset.train_test_split(
        test_size=test_size,
        shuffle=shuffle,
        seed=seed,
        stratify_by_column="label"
    )

    return dataset