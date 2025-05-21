import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import numpy as np
import config

def load_and_preprocess_data(tokenizer_name: str):
    print(f"Loading dataset: {config.DATASET_NAME}, split: {config.DATASET_SPLIT}")
    try:
        dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
        df = dataset.to_pandas()
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        raise
    print(f"Initial dataset size: {len(df)}")
    df[config.LABEL_COLUMN] = df['model'].apply(lambda x: 0 if x == config.HUMAN_LABEL_VALUE else 1)
    print(f"Value counts for 'model' column:\n{df['model'].value_counts()}")
    print(f"Value counts for '{config.LABEL_COLUMN}' column:\n{df[config.LABEL_COLUMN].value_counts()}")
    if config.MAX_SAMPLES is not None and config.MAX_SAMPLES < len(df):
        print(f"Subsampling to {config.MAX_SAMPLES} samples using strategy: {config.SUBSAMPLE_STRATEGY}")
        if config.SUBSAMPLE_STRATEGY == "first_n":
            df = df.head(config.MAX_SAMPLES)
        elif config.SUBSAMPLE_STRATEGY == "random":
            df = df.sample(n=config.MAX_SAMPLES, random_state=config.SEED)
        elif config.SUBSAMPLE_STRATEGY == "balanced_human_ai":
            human_df = df[df[config.LABEL_COLUMN] == 0]
            ai_df = df[df[config.LABEL_COLUMN] == 1]
            n_human = min(len(human_df), config.MAX_SAMPLES // 2)
            n_ai = min(len(ai_df), config.MAX_SAMPLES - n_human)
            if n_human < config.MAX_SAMPLES // 2 and len(ai_df) > n_ai:
                n_ai = min(len(ai_df), config.MAX_SAMPLES - n_human)
            elif n_ai < (config.MAX_SAMPLES - (config.MAX_SAMPLES // 2)) and len(human_df) > n_human:
                n_human = min(len(human_df), config.MAX_SAMPLES - n_ai)
            sampled_human_df = human_df.sample(n=n_human, random_state=config.SEED)
            sampled_ai_df = ai_df.sample(n=n_ai, random_state=config.SEED)
            df = pd.concat([sampled_human_df, sampled_ai_df]).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown subsampling strategy: {config.SUBSAMPLE_STRATEGY}")
        print(f"Subsampled dataset size: {len(df)}")
        print(f"Subsampled label distribution:\n{df[config.LABEL_COLUMN].value_counts()}")
    df[config.TEXT_COLUMN] = df[config.TEXT_COLUMN].astype(str).fillna('')
    if config.TEST_SPLIT_SIZE > 0:
        train_df, val_df = train_test_split(
            df,
            test_size=config.TEST_SPLIT_SIZE,
            random_state=config.SEED,
            stratify=df[config.LABEL_COLUMN] if config.LABEL_COLUMN in df.columns and len(df[config.LABEL_COLUMN].unique()) > 1 else None
        )
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    else:
        train_dataset = Dataset.from_pandas(df)
        val_dataset = None
        print(f"Train size: {len(train_dataset)}, No validation set.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples):
        return tokenizer(
            examples[config.TEXT_COLUMN],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
        )
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True) if val_dataset else None
    tokenized_train_dataset = tokenized_train_dataset.rename_column(config.LABEL_COLUMN, "labels")
    tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    if tokenized_val_dataset:
        tokenized_val_dataset = tokenized_val_dataset.rename_column(config.LABEL_COLUMN, "labels")
        tokenized_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_train_dataset, tokenized_val_dataset, tokenizer

if __name__ == '__main__':
    print("Testing data_utils.py...")
    tokenizer_name = config.MODEL_NAME
    train_data, val_data, tokenizer = load_and_preprocess_data(tokenizer_name)
    print("\nSample from training data:")
    print(train_data[0])
    if val_data:
        print("\nSample from validation data:")
        print(val_data[0])
    print(f"\nTokenizer pad token ID: {tokenizer.pad_token_id}, EOS token ID: {tokenizer.eos_token_id}")
    print("Data loading and preprocessing test complete.")