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

def load_model_for_inference(adapter_path: str):
    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model identified from adapter config: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer: pad_token set to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    quantization_config_inf = None
    if config.USE_4BIT_QUANTIZATION:
        quantization_config_inf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization for inference model loading.")
    print(f"Loading base model '{base_model_name}' for inference...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        quantization_config=quantization_config_inf,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT_QUANTIZATION else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Ensure the model's config also reflects the pad_token_id used by the tokenizer
    if tokenizer.pad_token_id is not None:
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Base Model: model.config.pad_token_id was None, explicitly set to {tokenizer.pad_token_id}")
        elif base_model.config.pad_token_id != tokenizer.pad_token_id:
            # If there's a mismatch, prioritize the tokenizer's pad_token_id as it's used for input preparation
            print(f"Warning: base_model.config.pad_token_id ({base_model.config.pad_token_id}) differs from tokenizer.pad_token_id ({tokenizer.pad_token_id}). Overwriting model's config with tokenizer's pad_token_id.")
            base_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # This scenario implies an issue with the tokenizer's eos_token or its setup, which is unlikely with standard Hugging Face tokenizers but worth noting.
        print("Warning: tokenizer.pad_token_id is None after attempting to set pad_token. The model may still encounter issues with batch processing if padding is required.")

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

def predict(texts: list[str], model, tokenizer):
    print(f"Tokenizing {len(texts)} texts for inference...")
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH
    ).to(model.device)
    print("Performing inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
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

def evaluate_on_dev_set(model, tokenizer, dev_data_path="cs162-final-dev/", output_filepath="evaluation_results.json", num_samples=None):
    all_files_stats = {
        "human_correct": 0, "human_total": 0,
        "ai_correct": 0, "ai_total": 0
    }
    overall_texts_processed = 0
    overall_correct_predictions = 0
    
    evaluation_results = {"files": {}} # To store results grouped by file

    print(f"\n--- Starting Evaluation on Dev Set ({dev_data_path}) ---")

    jsonl_files = []
    if os.path.isfile(dev_data_path) and dev_data_path.endswith(".jsonl"):
        jsonl_files = [os.path.basename(dev_data_path)]
        dev_data_path = os.path.dirname(dev_data_path)
        if not dev_data_path: # Handle case where file is in current directory
            dev_data_path = "."
    elif os.path.isdir(dev_data_path):
        try:
            jsonl_files = [f for f in os.listdir(dev_data_path) if f.endswith(".jsonl") and os.path.isfile(os.path.join(dev_data_path, f))]
        except FileNotFoundError:
            print(f"Error: Directory not found: {dev_data_path}")
            return
        except Exception as e:
            print(f"Error listing files in {dev_data_path}: {e}")
            return
    else:
        print(f"Error: dev_data_path '{dev_data_path}' is not a valid .jsonl file or directory.")
        return

    if not dev_data_path.endswith('/'):
        dev_data_path += '/'

    if not jsonl_files:
        print(f"No .jsonl files found in {dev_data_path}")
        return

    for filename in jsonl_files:
        filepath = os.path.join(dev_data_path, filename)
        print(f"\nProcessing file: {filename}...")
        file_stats = {
            "human_correct": 0, "human_total": 0,
            "ai_correct": 0, "ai_total": 0,
            "errors": 0
        }
        texts_to_predict = []
        ground_truths = []
        original_lines_data = [] # Store original lines with their type

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Reading {filename}")):
                    try:
                        data = json.loads(line.strip())
                        original_lines_data.append(data) # Store the whole data dict
                    except json.JSONDecodeError:
                        file_stats["errors"] += 1
                    except Exception as e:
                        file_stats["errors"] +=1
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}")
            continue
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            continue

        if not original_lines_data:
            print(f"No valid data lines found in {filename} to evaluate.")
            if file_stats["errors"] > 0:
                print(f"  ({file_stats['errors']} lines could not be processed during initial read)")
            continue

        sampled_lines_data = []
        if num_samples and num_samples > 0 and num_samples < len(original_lines_data):
            print(f"Randomly sampling {num_samples} lines from {filename}...")
            sampled_lines_data = random.sample(original_lines_data, num_samples)
        else:
            sampled_lines_data = original_lines_data
            if num_samples and num_samples >= len(original_lines_data):
                print(f"Requested {num_samples} samples, but file only has {len(original_lines_data)}. Using all available lines.")
            elif num_samples is not None and num_samples <=0:
                 print(f"num_samples is {num_samples}, using all available lines.")


        for data in tqdm(sampled_lines_data, desc=f"Preparing texts from {filename}"):
            human_text = data.get("human_text")
            if human_text and isinstance(human_text, str):
                texts_to_predict.append(human_text)
                ground_truths.append(0)
            machine_text = data.get("machine_text")
            if machine_text and isinstance(machine_text, str):
                texts_to_predict.append(machine_text)
                ground_truths.append(1)

        if not texts_to_predict:
            print(f"No valid texts found in the sampled data from {filename} to evaluate.")
            # errors here would already be counted from the initial read if lines were malformed
            # but it's possible that valid json lines didn't contain 'human_text' or 'machine_text'
            if file_stats["errors"] > 0:
                 print(f"  ({file_stats['errors']} lines had JSON parsing errors and were skipped before sampling)")
            evaluation_results["files"][filename] = { # Still record file stats even if no texts post-sampling
                "human_correct": 0, "human_total": 0, "human_accuracy": 0,
                "ai_correct": 0, "ai_total": 0, "ai_accuracy": 0,
                "overall_correct": 0, "overall_total": 0, "overall_accuracy": 0,
                "errors": file_stats["errors"],
                "notes": "No valid texts to predict after sampling (or no texts in original file)."
            }
            continue

        print(f"Predicting {len(texts_to_predict)} texts from {filename} (after sampling if applicable)...")
        batch_size = config.INFERENCE_BATCH_SIZE
        predictions = []
        for i in tqdm(range(0, len(texts_to_predict), batch_size), desc="Predicting batches"):
            batch_texts = texts_to_predict[i:i + batch_size]
            batch_preds = predict(batch_texts, model, tokenizer)
            predictions.extend(batch_preds)
        for i, pred_result in enumerate(predictions):
            predicted_label = pred_result["predicted_label"]
            ground_truth_label = ground_truths[i]
            overall_texts_processed += 1
            if predicted_label == ground_truth_label:
                overall_correct_predictions += 1
            if ground_truth_label == 0:
                file_stats["human_total"] += 1
                all_files_stats["human_total"] +=1
                if predicted_label == 0:
                    file_stats["human_correct"] += 1
                    all_files_stats["human_correct"] += 1
            elif ground_truth_label == 1:
                file_stats["ai_total"] += 1
                all_files_stats["ai_total"] += 1
                if predicted_label == 1:
                    file_stats["ai_correct"] += 1
                    all_files_stats["ai_correct"] +=1
        human_accuracy = (file_stats["human_correct"] / file_stats["human_total"] * 100) if file_stats["human_total"] > 0 else 0
        ai_accuracy = (file_stats["ai_correct"] / file_stats["ai_total"] * 100) if file_stats["ai_total"] > 0 else 0
        file_total_correct = file_stats["human_correct"] + file_stats["ai_correct"]
        file_total_samples = file_stats["human_total"] + file_stats["ai_total"]
        file_overall_accuracy = (file_total_correct / file_total_samples * 100) if file_total_samples > 0 else 0
        print(f"Results for {filename}:")
        print(f"  Human Texts: {file_stats['human_correct']}/{file_stats['human_total']} correct ({human_accuracy:.2f}%)")
        print(f"  AI-Generated Texts: {file_stats['ai_correct']}/{file_stats['ai_total']} correct ({ai_accuracy:.2f}%)")
        print(f"  Overall Accuracy for {filename}: {file_total_correct}/{file_total_samples} correct ({file_overall_accuracy:.2f}%)")
        if file_stats["errors"] > 0:
            print(f"  ({file_stats['errors']} lines had errors and were skipped)")

        evaluation_results["files"][filename] = {
            "human_correct": file_stats["human_correct"],
            "human_total": file_stats["human_total"],
            "human_accuracy": human_accuracy,
            "ai_correct": file_stats["ai_correct"],
            "ai_total": file_stats["ai_total"],
            "ai_accuracy": ai_accuracy,
            "overall_correct": file_total_correct,
            "overall_total": file_total_samples,
            "overall_accuracy": file_overall_accuracy,
            "errors": file_stats["errors"]
        }

    overall_human_accuracy = (all_files_stats["human_correct"] / all_files_stats["human_total"] * 100) if all_files_stats["human_total"] > 0 else 0
    overall_ai_accuracy = (all_files_stats["ai_correct"] / all_files_stats["ai_total"] * 100) if all_files_stats["ai_total"] > 0 else 0
    total_overall_accuracy = (overall_correct_predictions / overall_texts_processed * 100) if overall_texts_processed > 0 else 0
    
    evaluation_results["overall_summary"] = {
        "total_human_texts_evaluated": all_files_stats["human_total"],
        "correct_human_predictions": all_files_stats["human_correct"],
        "overall_human_accuracy": overall_human_accuracy,
        "total_ai_texts_evaluated": all_files_stats["ai_total"],
        "correct_ai_predictions": all_files_stats["ai_correct"],
        "overall_ai_accuracy": overall_ai_accuracy,
        "total_texts_processed": overall_texts_processed,
        "total_correct_predictions": overall_correct_predictions,
        "overall_accuracy": total_overall_accuracy
    }

    print("--- Overall Evaluation Summary ---")
    print(f"Total Human Texts Evaluated: {all_files_stats['human_total']}")
    print(f"  Correct Human Predictions: {all_files_stats['human_correct']} ({overall_human_accuracy:.2f}%)")
    print(f"Total AI-Generated Texts Evaluated: {all_files_stats['ai_total']}")
    print(f"  Correct AI Predictions: {all_files_stats['ai_correct']} ({overall_ai_accuracy:.2f}%)")
    print(f"Total Texts Processed: {overall_texts_processed}")
    print(f"Total Correct Predictions: {overall_correct_predictions}")
    print(f"Overall Accuracy: {total_overall_accuracy:.2f}%")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"\nEvaluation results saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving evaluation results to {output_filepath}: {e}")

def normalize_hf_dataset(dataset, dataset_name):
    """
    Normalize HuggingFace dataset to a list of dicts for evaluation.
    For AITextPile and TuringBench: map 'text' and 'label'.
    For HC3: extract pairs of 'human_answers' and 'chatgpt_answers',
    return list of dicts with 'human_text' and 'machine_text'.
    """
    normalized = []
    if dataset_name in ["AITextPile", "TuringBench"]:
        for ex in dataset:
            text = ex.get("text")
            label = ex.get("label")
            if text is not None and label in [0, 1]:
                normalized.append({"text": text, "label": label})
    elif dataset_name == "HC3":
        for ex in dataset:
            human_answers = ex.get("human_answers")
            chatgpt_answers = ex.get("chatgpt_answers")
            # Both are expected to be lists of strings
            if isinstance(human_answers, list) and isinstance(chatgpt_answers, list):
                for h, m in zip(human_answers, chatgpt_answers):
                    if isinstance(h, str) and isinstance(m, str):
                        normalized.append({"human_text": h, "machine_text": m})
    else:
        raise ValueError(f"Unknown dataset_name for normalization: {dataset_name}")
    return normalized

def evaluate_on_hf_dataset(model, tokenizer, dataset_name: str, num_samples=None):
    """
    Evaluate model on a HuggingFace dataset. Normalizes to list of dicts with 'text' and 'label'.
    Treats label==0 as human, 1 as machine. Computes accuracy, precision, recall, F1.
    Prints and returns results in the same format as evaluate_on_dev_set().
    """
    print(f"\n--- Loading HuggingFace dataset: {dataset_name} ---")
    # You may want to customize the split and field names for each dataset
    dataset = load_dataset(dataset_name, split="test")
    # Normalize to list of dicts with 'text' and 'label'
    normalized = normalize_hf_dataset(dataset, dataset_name)
    if num_samples is not None and num_samples > 0 and num_samples < len(normalized):
        print(f"Randomly sampling {num_samples} examples from dataset...")
        normalized = random.sample(normalized, num_samples)
    elif num_samples is not None and num_samples >= len(normalized):
        print(f"Requested {num_samples} samples, but dataset only has {len(normalized)}. Using all available.")
    elif num_samples is not None and num_samples <= 0:
        print(f"num_samples is {num_samples}, using all available.")
    if not normalized:
        print("No valid examples found in dataset for evaluation.")
        return None
    texts = [ex["text"] for ex in normalized]
    labels = [ex["label"] for ex in normalized]
    print(f"Predicting {len(texts)} texts from HuggingFace dataset '{dataset_name}'...")
    batch_size = getattr(config, "INFERENCE_BATCH_SIZE", 16)
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting batches"):
        batch_texts = texts[i:i+batch_size]
        batch_preds = predict(batch_texts, model, tokenizer)
        predictions.extend(batch_preds)
    pred_labels = [p["predicted_label"] for p in predictions]
    accuracy = accuracy_score(labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_labels, average="binary", pos_label=1)
    print("--- HuggingFace Dataset Evaluation Results ---")
    print(f"Total Texts Evaluated: {len(labels)}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Precision (AI): {precision*100:.2f}%")
    print(f"Recall (AI): {recall*100:.2f}%")
    print(f"F1 Score (AI): {f1*100:.2f}%")
    results = {
        "total_texts_evaluated": len(labels),
        "accuracy": accuracy*100,
        "precision": precision*100,
        "recall": recall*100,
        "f1": f1*100,
    }
    return results

def main():
    parser = argparse.ArgumentParser(description="Run AI text detection inference or evaluation.")
    parser.add_argument(
        "--mode",
        type=str,
        default="predict",
        choices=["predict", "evaluate"],
        help="Run mode: 'predict' for classifying texts or 'evaluate' for dev set evaluation."
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
        "--dev_data_path",
        type=str,
        default="cs162-final-dev/",
        help="Path to the directory containing .jsonl files or a single .jsonl file for evaluation (only if mode is 'evaluate')."
    )
    parser.add_argument(
        "--output_filepath",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results JSON file (only if mode is 'evaluate')."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of random samples to evaluate from each file (only if mode is 'evaluate'). If not specified, all samples are used."
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        choices=["HC3", "AITextPile", "TuringBench"],
        default=None,
        help="(evaluate mode only) If set, use a specific HuggingFace dataset (HC3, AITextPile, or TuringBench) instead of --dev_data_path."
    )
    parser.add_argument(
        "--evaluate_all_hf",
        action="store_true",
        help="If set (with --mode evaluate), evaluate on all 3 HuggingFace datasets: HC3, AITextPile, TuringBench."
    )
    args = parser.parse_args()
    if args.mode == "predict" and not args.texts:
        parser.error("The 'predict' mode requires at least one text to classify.")

    # If --hf_dataset is set and mode is evaluate, override dev_data_path
    if args.mode == "evaluate" and args.hf_dataset is not None:
        # Map dataset names to their corresponding paths (update as needed)
        hf_dataset_paths = {
            "HC3": "dev_data/HC3/",  # Example path, update as needed
            "AITextPile": "dev_data/AITextPile/",  # Example path, update as needed
            "TuringBench": "dev_data/TuringBench/",  # Example path, update as needed
        }
        if args.hf_dataset in hf_dataset_paths:
            args.dev_data_path = hf_dataset_paths[args.hf_dataset]
        else:
            raise ValueError(f"Unknown hf_dataset: {args.hf_dataset}")

    print(f"Loading model from adapter path: {args.model_path}")
    model, tokenizer = load_model_for_inference(args.model_path)
    print("Model and tokenizer loaded successfully.")
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
        if args.evaluate_all_hf:
            all_datasets = ["HC3", "AITextPile", "TuringBench"]
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
            evaluate_on_hf_dataset(model, tokenizer, dataset_name=args.hf_dataset, num_samples=args.num_samples)
        else:
            evaluate_on_dev_set(model, tokenizer, dev_data_path=args.dev_data_path, output_filepath=args.output_filepath, num_samples=args.num_samples)

if __name__ == "__main__":
    main()

"""
Example Usages:

# Predict on a list of texts using a specific model checkpoint
python inference.py --mode predict --model_path saved_models/mistral_raid_detector_adapter/checkpoint-500 "I am a human" "I am an AI-generated text" "I'm a human" "The wind whispered through the ancient pines, carrying secrets older than the mountains themselves. In the valley below, a lone figure trudged through the snow, their cloak a patchwork of faded dreams. Stars blinked faintly above, as if unsure whether to guide or merely watch. A distant howl broke the silence, sharp and fleeting, like a memory that refused to be forgotten. The figure paused, breath clouding in the frigid air, and glanced at the horizon where dawn hesitated, unsure of its welcome."

# Evaluate the model on all .jsonl files in a directory
python inference.py --mode evaluate --dev_data_path cs162-final-dev/ --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000

# Evaluate the model on a single specific .jsonl file
python inference.py --mode evaluate --dev_data_path cs162-final-dev/dataset1.jsonl --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000

# Evaluate the model on all .jsonl files in a directory, using 50 random samples from each file
python inference.py --mode evaluate --dev_data_path cs162-final-dev/ --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000 --num_samples 50

# Evaluate the model on a single specific .jsonl file, using 20 random samples from that file
python inference.py --mode evaluate --dev_data_path cs162-final-dev/arxiv_dolly.jsonl --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000 --num_samples 100

# Evaluate using the default model path (defined in config.py) and default dev data path (cs162-final-dev/)
python inference.py --mode evaluate

# Predict using the default model path
python inference.py --mode predict "This is a test sentence to classify."

# Evaluate on a specific HuggingFace dataset (HC3, AITextPile, or TuringBench)
python inference.py --mode evaluate --hf_dataset HC3
python inference.py --mode evaluate --hf_dataset AITextPile --num_samples 100

# Evaluate on all three HuggingFace datasets and print summary
python inference.py --mode evaluate --evaluate_all_hf

"""