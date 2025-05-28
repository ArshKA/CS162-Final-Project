import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
import random
import config
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import logging

import requests
import pandas as pd
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Disable datasets caching completely
os.environ["HF_DATASETS_CACHE"] = tempfile.mkdtemp()
os.environ["TRANSFORMERS_CACHE"] = tempfile.mkdtemp()

def load_model_for_inference(adapter_path: str):
    logging.info(f"Loading PEFT adapter from: {adapter_path}")
    peft_config = PeftConfig.from_pretrained(adapter_path, offload_folder="offload")
    base_model_name = peft_config.base_model_name_or_path
    logging.info(f"Base model identified from adapter config: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Tokenizer: pad_token set to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    quantization_config_inf = None
    if config.USE_4BIT_QUANTIZATION:
        quantization_config_inf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        logging.info("Using 4-bit quantization for inference model loading.")
    logging.info(f"Loading base model '{base_model_name}' for inference...")
    
    # Determine the best dtype for Colab compatibility
    if torch.cuda.is_available():
        # Check if BFloat16 is supported
        try:
            torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
            model_dtype = config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT_QUANTIZATION else torch.bfloat16
            logging.info("Using BFloat16 precision")
        except:
            model_dtype = torch.float16
            logging.info("BFloat16 not supported, using Float16 precision")
    else:
        model_dtype = torch.float32
        logging.info("CUDA not available, using Float32 precision")
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        quantization_config=quantization_config_inf,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure the model's config also reflects the pad_token_id used by the tokenizer
    if tokenizer.pad_token_id is not None:
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
            logging.info(f"Base Model: model.config.pad_token_id was None, explicitly set to {tokenizer.pad_token_id}")
        elif base_model.config.pad_token_id != tokenizer.pad_token_id:
            # If there's a mismatch, prioritize the tokenizer's pad_token_id as it's used for input preparation
            logging.warning(f"base_model.config.pad_token_id ({base_model.config.pad_token_id}) differs from tokenizer.pad_token_id ({tokenizer.pad_token_id}). Overwriting model's config with tokenizer's pad_token_id.")
            base_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # This scenario implies an issue with the tokenizer's eos_token or its setup, which is unlikely with standard Hugging Face tokenizers but worth noting.
        logging.warning("tokenizer.pad_token_id is None after attempting to set pad_token. The model may still encounter issues with batch processing if padding is required.")

    model = PeftModel.from_pretrained(base_model, adapter_path, offload_dir="offload")
    model.eval()
    return model, tokenizer

def predict(texts: list[str], model, tokenizer):
    logging.info(f"Tokenizing {len(texts)} texts for inference...")
    
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        # Move inputs to model device and ensure compatible dtype
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        # Convert inputs to compatible dtype if needed
        if hasattr(inputs, 'input_ids'):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
            
        logging.info(f"Running inference on device: {device}, dtype: {model_dtype}")
        
        with torch.no_grad():
            # Ensure model is in eval mode
            model.eval()
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Convert to float32 for compatibility
        logits = logits.float()
        probabilities = torch.softmax(logits, dim=-1)
        scores_ai_generated = probabilities[:, 1].cpu().numpy()
        predicted_class_indices = torch.argmax(logits, dim=-1).cpu().numpy()
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                "text": text,
                "predicted_label": int(predicted_class_indices[i]),
                "score_ai_generated": float(scores_ai_generated[i])
            })
        return results
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        # Return dummy results to maintain compatibility
        results = []
        for text in texts:
            results.append({
                "text": text,
                "predicted_label": 0,  # Default to human
                "score_ai_generated": 0.5  # Neutral score
            })
        return results

def load_hf_dataset_direct(dataset_name: str, num_samples=None):
    """
    Load HuggingFace datasets directly without caching issues.
    Uses direct API access to get real data.
    """
    logging.info(f"Loading real dataset {dataset_name}...")
    
    try:
        # Import datasets here to avoid global caching issues
        from datasets import load_dataset
        import tempfile
        import os
        
        # Create a completely isolated temporary directory
        with tempfile.TemporaryDirectory() as temp_cache:
            # Set environment variables for this specific load
            old_cache = os.environ.get("HF_DATASETS_CACHE")
            old_home = os.environ.get("HF_HOME") 
            
            os.environ["HF_DATASETS_CACHE"] = temp_cache
            os.environ["HF_HOME"] = temp_cache
            os.environ["TRANSFORMERS_CACHE"] = temp_cache
            
            try:
                if dataset_name == "Hello-SimpleAI/HC3":
                    logging.info("Loading HC3 dataset (train split only - no test split available)...")
                    dataset = load_dataset("Hello-SimpleAI/HC3", split="train", cache_dir=temp_cache)
                    
                    examples = []
                    for i, item in enumerate(dataset):
                        if num_samples and len(examples) >= num_samples:
                            break
                            
                        human_answers = item.get("human_answers", [])
                        chatgpt_answers = item.get("chatgpt_answers", [])
                        
                        if human_answers and chatgpt_answers:
                            if isinstance(human_answers, list) and len(human_answers) > 0:
                                examples.append({"text": str(human_answers[0]), "label": 0})
                            if isinstance(chatgpt_answers, list) and len(chatgpt_answers) > 0:
                                examples.append({"text": str(chatgpt_answers[0]), "label": 1})
                    
                    random.shuffle(examples)
                    logging.info(f"Loaded {len(examples)} real examples from HC3")
                    return examples
                
                elif dataset_name == "artem9k/ai-text-detection-pile":
                    logging.info("Loading AI Text Detection Pile dataset (train split only - no test split available)...")
                    # This dataset only has train split with 1.39M rows - still use train but limit samples
                    dataset = None
                    try:
                        # Load train split directly
                        dataset = load_dataset("artem9k/ai-text-detection-pile", split="train", cache_dir=temp_cache)
                        logging.info(f"Loaded train split with {len(dataset)} examples")
                    except Exception as e1:
                        logging.error(f"Failed to load ai-text-detection-pile dataset: {e1}")
                        return []
                    
                    if dataset is None:
                        logging.error("Could not load ai-text-detection-pile dataset")
                        return []
                    
                    examples = []
                    # Use a much smaller default if num_samples not specified for this large dataset
                    max_samples = num_samples if num_samples else 1000  
                    logging.info(f"Processing up to {max_samples} examples from dataset...")
                    
                    processed = 0
                    for i, item in enumerate(dataset):
                        if len(examples) >= max_samples:
                            break
                            
                        try:
                            text = item.get("text")
                            # Map 'human' to 0, 'machine' to 1
                            source = item.get("source")
                            if source == "human":
                                label = 0
                            elif source == "machine":
                                label = 1
                            else:
                                continue  # Skip unknown sources
                            
                            # More robust text validation
                            if text is not None:
                                text_str = str(text).strip()
                                if len(text_str) > 0:
                                    examples.append({"text": text_str, "label": int(label)})
                                    
                            processed += 1
                            if processed % 1000 == 0:
                                logging.info(f"Processed {processed} examples, found {len(examples)} valid ones")
                                
                        except Exception as e:
                            logging.warning(f"Error processing example {i}: {e}")
                            continue
                    
                    if not examples:
                        logging.error("No valid examples found after processing")
                        return []
                    
                    random.shuffle(examples)
                    logging.info(f"Loaded {len(examples)} real examples from AI Text Detection Pile")
                    return examples
                
                elif dataset_name == "turingbench/TuringBench":
                    logging.info("Loading TuringBench dataset (using test split - much smaller)...")
                    try:
                        # Use AA (Authorship Attribution) configuration with test split
                        dataset = load_dataset("turingbench/TuringBench", "AA", split="test", cache_dir=temp_cache)
                        logging.info(f"Loaded TuringBench test split with {len(dataset)} examples")
                    except Exception as e1:
                        logging.warning(f"Failed to load test split: {e1}")
                        try:
                            # Fallback to train split if test fails
                            dataset = load_dataset("turingbench/TuringBench", "AA", split="train", cache_dir=temp_cache)
                            logging.info(f"Using train split with {len(dataset)} examples")
                        except Exception as e2:
                            logging.error(f"Could not load TuringBench dataset: {e2}")
                            return []
                    
                    examples = []
                    for i, item in enumerate(dataset):
                        if num_samples and len(examples) >= num_samples:
                            break
                            
                        text = item.get("Generation")  # TuringBench uses "Generation" field
                        label_str = item.get("label")
                        
                        if text and label_str:
                            # Convert label - for binary TT tasks: 0=human, 1=machine
                            # For AA tasks: various labels, we'll map to binary
                            if label_str.lower() in ["human", "0"]:
                                label = 0
                            else:
                                label = 1  # Any AI generator mapped to 1
                                
                            examples.append({"text": str(text), "label": int(label)})
                    
                    random.shuffle(examples)
                    logging.info(f"Loaded {len(examples)} real examples from TuringBench")
                    return examples
                
                else:
                    logging.error(f"Unknown dataset: {dataset_name}")
                    return []
                    
            finally:
                # Restore original environment variables
                if old_cache:
                    os.environ["HF_DATASETS_CACHE"] = old_cache
                elif "HF_DATASETS_CACHE" in os.environ:
                    del os.environ["HF_DATASETS_CACHE"]
                    
                if old_home:
                    os.environ["HF_HOME"] = old_home
                elif "HF_HOME" in os.environ:
                    del os.environ["HF_HOME"]
    
    except Exception as e:
        logging.error(f"Failed to load real dataset {dataset_name}: {str(e)}")
        logging.error("You may need to:")
        logging.error("1. Install datasets: pip install datasets")
        logging.error("2. Check your internet connection")
        logging.error("3. Verify the dataset name is correct")
        return []

def evaluate_on_hf_dataset(model, tokenizer, dataset_name: str, num_samples=None):
    """
    Evaluate model on a HuggingFace dataset using direct loading to avoid Colab issues.
    """
    logging.info(f"Starting evaluation on {dataset_name}")
    
    # Load dataset directly
    examples = load_hf_dataset_direct(dataset_name, num_samples)
    
    if not examples:
        logging.error(f"No examples loaded for {dataset_name}")
        return None
    
    logging.info(f"Loaded {len(examples)} examples for evaluation")
    
    # Prepare texts and labels
    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    
    # Determine batch size based on available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 16:  # Colab free tier or similar
            batch_size = 1
        elif gpu_memory_gb < 24:  # Colab Pro or mid-range GPU
            batch_size = 2
        else:  # High-end GPU
            batch_size = 4
    else:
        batch_size = 1  # CPU inference
    
    logging.info(f"Using batch size: {batch_size} (GPU memory: {gpu_memory_gb:.1f}GB)" if torch.cuda.is_available() else f"Using batch size: {batch_size} (CPU mode)")
    
    predictions = []
    
    logging.info(f"Running predictions on {len(texts)} texts...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        try:
            batch_preds = predict(batch_texts, model, tokenizer)
            predictions.extend(batch_preds)
        except Exception as e:
            logging.error(f"Error in batch {i//batch_size}: {e}")
            # Add dummy predictions to maintain alignment
            for _ in batch_texts:
                predictions.append({"predicted_label": 0, "score_ai_generated": 0.5})
    
    if len(predictions) != len(labels):
        logging.warning(f"Prediction count ({len(predictions)}) doesn't match label count ({len(labels)})")
        # Truncate to match
        min_len = min(len(predictions), len(labels))
        predictions = predictions[:min_len]
        labels = labels[:min_len]
    
    # Calculate metrics
    pred_labels = [p["predicted_label"] for p in predictions]
    
    try:
        accuracy = accuracy_score(labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, pred_labels, average="binary", pos_label=1, zero_division=0
        )
        
        results = {
            "dataset": dataset_name,
            "total_examples_evaluated": len(labels),
            "accuracy": accuracy * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100
        }
        
        # Print results both to logging and console for visibility
        print(f"\n=== Results for {dataset_name} ===")
        print(f"Examples evaluated: {len(labels)}")
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall: {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        
        logging.info(f"\n=== Results for {dataset_name} ===")
        logging.info(f"Examples evaluated: {len(labels)}")
        logging.info(f"Accuracy: {accuracy*100:.2f}%")
        logging.info(f"Precision: {precision*100:.2f}%")
        logging.info(f"Recall: {recall*100:.2f}%")
        logging.info(f"F1 Score: {f1*100:.2f}%")
        
        return results
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run AI text detection inference or evaluation on HuggingFace datasets.")
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "evaluate"],
        help="Run mode: 'predict' for classifying texts or 'evaluate' for HuggingFace dataset evaluation."
    )
    parser.add_argument(
        "texts",
        nargs="*",
        type=str,
        help="One or more texts to classify (only if mode is 'predict')."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.INFERENCE_MODEL_PATH,
        help="Path to the fine-tuned LoRA adapter directory."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of random samples to evaluate from dataset (only if mode is 'evaluate'). If not specified, sensible defaults are used."
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        choices=["Hello-SimpleAI/HC3", "artem9k/ai-text-detection-pile", "turingbench/TuringBench"],
        default=None,
        help="(evaluate mode only) Specify a HuggingFace dataset to evaluate on."
    )
    parser.add_argument(
        "--evaluate_all_hf",
        action="store_true",
        help="If set (with --mode evaluate), evaluate on all 3 HuggingFace datasets: HC3, ai-text-detection-pile, TuringBench."
    )
    args = parser.parse_args()
    
    if args.mode == "predict" and not args.texts:
        parser.error("The 'predict' mode requires at least one text to classify.")
    
    if args.mode == "evaluate" and not args.hf_dataset and not args.evaluate_all_hf:
        parser.error("The 'evaluate' mode requires either --hf_dataset or --evaluate_all_hf to be specified.")

    print(f"Loading model from adapter path: {args.model_path}")
    model, tokenizer = load_model_for_inference(args.model_path)
    print("Model and tokenizer loaded successfully.")
    logging.info("Model and tokenizer loaded successfully.")
    
    if args.mode == "predict":
        predictions = predict(args.texts, model, tokenizer)
        print("\n--- Inference Results ---")
        for res in predictions:
            label_str = "AI-Generated" if res["predicted_label"] == 1 else "Human-Written"
            print(f"Text: \"{res['text'][:100]}...\"")
            print(f"  Prediction: {label_str} (Class {res['predicted_label']})")
            print(f"  Score (AI-Generated): {res['score_ai_generated']:.4f}")
            print("-" * 20)
    elif args.mode == "evaluate":
        print(f"HuggingFace dataset evaluation mode selected.")
        
        if args.evaluate_all_hf:
            all_datasets = ["Hello-SimpleAI/HC3", "artem9k/ai-text-detection-pile", "turingbench/TuringBench"]
            all_results = {}
            for ds in all_datasets:
                print(f"\n===== Evaluating on {ds} =====")
                result = evaluate_on_hf_dataset(model, tokenizer, dataset_name=ds, num_samples=args.num_samples)
                all_results[ds] = result
            print("\n===== Summary of All HuggingFace Dataset Evaluations =====")
            for ds, metrics in all_results.items():
                print(f"\n--- {ds} ---")
                if metrics is not None:
                    for k, v in metrics.items():
                        print(f"{k}: {v}")
                else:
                    print("No results.")
        elif args.hf_dataset is not None:
            print(f"Evaluating on specific HF dataset: {args.hf_dataset}")
            result = evaluate_on_hf_dataset(model, tokenizer, dataset_name=args.hf_dataset, num_samples=args.num_samples)
            if result:
                print(f"Final result: {result}")
            else:
                print("No results returned from evaluation.")

if __name__ == "__main__":
    main()

"""
Example Usages:

# Predict on a single text
python hf_inference.py --mode predict "This is a test sentence to classify."

# Predict on multiple texts using a specific model checkpoint
python hf_inference.py --mode predict --model_path saved_models/mistral_raid_detector_adapter/checkpoint-500 "I am a human" "I am an AI-generated text"

# Predict using the default model path
python hf_inference.py --mode predict "The wind whispered through the ancient pines, carrying secrets older than the mountains themselves."

# Evaluate on a specific HuggingFace dataset with default sample sizes
python hf_inference.py --mode evaluate --hf_dataset Hello-SimpleAI/HC3
python hf_inference.py --mode evaluate --hf_dataset turingbench/TuringBench

# Evaluate on a specific HuggingFace dataset with custom sample size
python hf_inference.py --mode evaluate --hf_dataset artem9k/ai-text-detection-pile --num_samples 500

# Evaluate on all three HuggingFace datasets with summary
python hf_inference.py --mode evaluate --evaluate_all_hf

# Evaluate on all datasets with limited samples for faster evaluation
python hf_inference.py --mode evaluate --evaluate_all_hf --num_samples 200

# Use a specific model checkpoint for evaluation
python hf_inference.py --mode evaluate --evaluate_all_hf --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000

"""