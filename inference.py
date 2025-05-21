import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
import config

def load_model_for_inference(adapter_path: str):
    print(f"Loading PEFT adapter from: {adapter_path}")
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    print(f"Base model identified from adapter config: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

def evaluate_on_dev_set(model, tokenizer, dev_data_path="cs162-final-dev/"):
    all_files_stats = {
        "human_correct": 0, "human_total": 0,
        "ai_correct": 0, "ai_total": 0
    }
    overall_texts_processed = 0
    overall_correct_predictions = 0
    print(f"\n--- Starting Evaluation on Dev Set ({dev_data_path}) ---")
    if not dev_data_path.endswith('/'):
        dev_data_path += '/'
    try:
        jsonl_files = [f for f in os.listdir(dev_data_path) if f.endswith(".jsonl") and os.path.isfile(os.path.join(dev_data_path, f))]
    except FileNotFoundError:
        print(f"Error: Directory not found: {dev_data_path}")
        return
    except Exception as e:
        print(f"Error listing files in {dev_data_path}: {e}")
        return
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
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Reading {filename}")):
                    try:
                        data = json.loads(line.strip())
                        human_text = data.get("human_text")
                        if human_text and isinstance(human_text, str):
                            texts_to_predict.append(human_text)
                            ground_truths.append(0)
                        machine_text = data.get("machine_text")
                        if machine_text and isinstance(machine_text, str):
                            texts_to_predict.append(machine_text)
                            ground_truths.append(1)
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
        if not texts_to_predict:
            print(f"No valid texts found in {filename} to evaluate.")
            if file_stats["errors"] > 0:
                print(f"  ({file_stats['errors']} lines could not be processed)")
            continue
        print(f"Predicting {len(texts_to_predict)} texts from {filename}...")
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
    overall_human_accuracy = (all_files_stats["human_correct"] / all_files_stats["human_total"] * 100) if all_files_stats["human_total"] > 0 else 0
    overall_ai_accuracy = (all_files_stats["ai_correct"] / all_files_stats["ai_total"] * 100) if all_files_stats["ai_total"] > 0 else 0
    total_overall_accuracy = (overall_correct_predictions / overall_texts_processed * 100) if overall_texts_processed > 0 else 0
    print("--- Overall Evaluation Summary ---")
    print(f"Total Human Texts Evaluated: {all_files_stats['human_total']}")
    print(f"  Correct Human Predictions: {all_files_stats['human_correct']} ({overall_human_accuracy:.2f}%)")
    print(f"Total AI-Generated Texts Evaluated: {all_files_stats['ai_total']}")
    print(f"  Correct AI Predictions: {all_files_stats['ai_correct']} ({overall_ai_accuracy:.2f}%)")
    print(f"Total Texts Processed: {overall_texts_processed}")
    print(f"Total Correct Predictions: {overall_correct_predictions}")
    print(f"Overall Accuracy: {total_overall_accuracy:.2f}%")

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
        help="Path to the directory containing .jsonl files for evaluation (only if mode is 'evaluate')."
    )
    args = parser.parse_args()
    if args.mode == "predict" and not args.texts:
        parser.error("The 'predict' mode requires at least one text to classify.")
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
        evaluate_on_dev_set(model, tokenizer, dev_data_path=args.dev_data_path)

if __name__ == "__main__":
    main()