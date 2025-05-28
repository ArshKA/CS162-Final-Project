import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import numpy as np
import config
from datasets import load_dataset, concatenate_datasets

def calculate_term_bias(texts, top_n=20):
    word_counts = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        word_counts.update(tokens)
    return word_counts.most_common(top_n)

def calculate_length_stats(texts):
    lengths = [len(word_tokenize(text)) for text in texts]
    if not lengths:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}
    return {
        "mean": np.mean(lengths),
        "median": np.median(lengths),
        "std": np.std(lengths),
        "min": np.min(lengths),
        "max": np.max(lengths),
    }

def calculate_diversity_stats(texts):
    ttrs = []
    for text in texts:
        tokens = word_tokenize(text.lower())
        if tokens:
            ttr = len(set(tokens)) / len(tokens)
            ttrs.append(ttr)
    
    if not ttrs:
        return {"mean": 0, "median": 0, "std": 0}
    return {
        "mean": np.mean(ttrs),
        "median": np.median(ttrs),
        "std": np.std(ttrs),
    }

def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

def load_and_subsample_dataset():
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT, trust_remote_code=True)
    
    if config.MAX_SAMPLES is None or config.MAX_SAMPLES >= len(dataset):
        return dataset
    
    human_data = dataset.filter(lambda x: x['model'] == config.HUMAN_LABEL_VALUE)
    ai_data = dataset.filter(lambda x: x['model'] != config.HUMAN_LABEL_VALUE)
    
    samples_per_class = config.MAX_SAMPLES // 2
    human_samples = min(len(human_data), samples_per_class)
    ai_samples = min(len(ai_data), config.MAX_SAMPLES - human_samples)
    
    datasets_to_concat = []
    if human_samples > 0:
        datasets_to_concat.append(human_data.shuffle(seed=config.SEED).select(range(human_samples)))
    if ai_samples > 0:
        datasets_to_concat.append(ai_data.shuffle(seed=config.SEED).select(range(ai_samples)))
    
    return concatenate_datasets(datasets_to_concat).shuffle(seed=config.SEED) if datasets_to_concat else []

def extract_texts(dataset):
    human_texts, ai_texts = [], []
    
    for item in dataset:
        text = item.get(config.TEXT_COLUMN)
        model = item.get('model')
        
        if not text or not model or not str(text).strip():
            continue
            
        text = str(text).strip()
        if model == config.HUMAN_LABEL_VALUE:
            human_texts.append(text)
        else:
            ai_texts.append(text)
    
    return human_texts, ai_texts

def analyze_texts(texts, label):
    if not texts:
        print(f"\nNo {label} texts found for analysis.")
        return
    
    print(f"\n--- {label} Text Analysis ---")
    print(f"  Length Stats: {calculate_length_stats(texts)}")
    print(f"  Diversity Stats (TTR): {calculate_diversity_stats(texts)}")
    print(f"  Most Common Terms: {calculate_term_bias(texts)}")

def main():
    print("Starting linguistic analysis...")
    
    ensure_nltk_resources()
    
    try:
        dataset = load_and_subsample_dataset()
        print(f"Dataset loaded with {len(dataset)} samples")
        
        human_texts, ai_texts = extract_texts(dataset)
        print(f"Found {len(human_texts)} human texts and {len(ai_texts)} AI texts")
        
        analyze_texts(human_texts, "Human")
        analyze_texts(ai_texts, "AI-Generated")
        
        print("\nAnalysis completed.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 