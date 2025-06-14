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

def load_model_for_inference(adapter_path: str = None, use_adapter: bool = True):
    if use_adapter:
        print(f"Loading PEFT adapter from: {adapter_path}")
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path
    else:
        print("Loading base model without adapter...")
        base_model_name = adapter_path

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
    
    if torch.cuda.is_available():
        try:
            torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
            model_dtype = config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT_QUANTIZATION else torch.bfloat16
            print("Using BFloat16 precision")
        except:
            model_dtype = torch.float16
            print("BFloat16 not supported, using Float16 precision")
    else:
        model_dtype = torch.float32
        print("CUDA not available, using Float32 precision")
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        quantization_config=quantization_config_inf,
        torch_dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is not None:
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Base Model: model.config.pad_token_id was None, explicitly set to {tokenizer.pad_token_id}")
        elif base_model.config.pad_token_id != tokenizer.pad_token_id:
            print(f"Warning: base_model.config.pad_token_id ({base_model.config.pad_token_id}) differs from tokenizer.pad_token_id ({tokenizer.pad_token_id}). Overwriting.")
            base_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        print("Warning: tokenizer.pad_token_id is None after attempting to set pad_token.")

    if use_adapter:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    model.eval()
    return model, tokenizer

def predict(texts: list[str], model, tokenizer):
    print(f"Tokenizing {len(texts)} texts for inference...")
    
    try:
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_LENGTH
        )
        
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        
        if hasattr(inputs, 'input_ids'):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)
            
        print(f"Running inference on device: {device}, dtype: {model_dtype}")
        
        with torch.no_grad():
            model.eval()
            outputs = model(**inputs)
            logits = outputs.logits
            
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
        print(f"Error in prediction: {str(e)}")
        # Return dummy results to maintain compatibility
        results = []
        for text in texts:
            results.append({
                "text": text,
                "predicted_label": 0,  # Default to human
                "score_ai_generated": 0.5  # Neutral score
            })
        return results

def evaluate_on_dev_set(model, tokenizer, dev_data_path="dev_data/", output_filepath="evaluation_results.json", num_samples=None):
    all_files_stats = {
        "human_correct": 0, "human_total": 0,
        "ai_correct": 0, "ai_total": 0
    }
    overall_texts_processed = 0
    overall_correct_predictions = 0
    
    # Global confusion matrix for overall metrics
    global_true_positives = 0
    global_true_negatives = 0
    global_false_positives = 0
    global_false_negatives = 0

    evaluation_results = {"files": {}}

    print(f"\n--- Starting Evaluation on Dev Set ({dev_data_path}) ---")

    data_files = []
    if os.path.isfile(dev_data_path) and (dev_data_path.endswith(".jsonl") or dev_data_path.endswith(".json")):
        data_files = [os.path.basename(dev_data_path)]
        dev_data_path = os.path.dirname(dev_data_path)
        if not dev_data_path:
            dev_data_path = "."
    elif os.path.isdir(dev_data_path):
        try:
            data_files = [f for f in os.listdir(dev_data_path) if (f.endswith(".jsonl") or f.endswith(".json")) and os.path.isfile(os.path.join(dev_data_path, f))]
        except FileNotFoundError:
            print(f"Error: Directory not found: {dev_data_path}")
            return
        except Exception as e:
            print(f"Error listing files in {dev_data_path}: {e}")
            return
    else:
        print(f"Error: dev_data_path '{dev_data_path}' is not a valid .json/.jsonl file or directory.")
        return

    if not dev_data_path.endswith('/'):
        dev_data_path += '/'

    if not data_files:
        print(f"No .json/.jsonl files found in {dev_data_path}")
        return

    for filename in data_files:
        filepath = os.path.join(dev_data_path, filename)
        print(f"\nProcessing file: {filename}...")
        file_stats = {
            "human_correct": 0, "human_total": 0,
            "ai_correct": 0, "ai_total": 0,
            "errors": 0
        }
        texts_to_predict = []
        ground_truths = []
        original_lines_data = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filename.endswith('.jsonl'):
                    # JSONL format: one JSON object per line
                    for line_num, line in enumerate(tqdm(f, desc=f"Reading {filename}")):
                        try:
                            data = json.loads(line.strip())
                            original_lines_data.append(data)
                        except json.JSONDecodeError:
                            file_stats["errors"] += 1
                        except Exception as e:
                            file_stats["errors"] +=1
                else:
                    # JSON format: array of objects
                    try:
                        content = f.read()
                        data_array = json.loads(content)
                        if isinstance(data_array, list):
                            original_lines_data.extend(data_array)
                            print(f"Reading {filename}: {len(data_array)} items loaded")
                        else:
                            print(f"Error: {filename} does not contain a JSON array")
                            file_stats["errors"] += 1
                    except json.JSONDecodeError as e:
                        print(f"Error: {filename} is not valid JSON: {e}")
                        file_stats["errors"] += 1
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        file_stats["errors"] += 1
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
            print(f"Subsampling first {num_samples} lines from {filename}...")
            sampled_lines_data = original_lines_data[:num_samples]
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
                ground_truths.append(0)  # 0 = Human
            machine_text = data.get("machine_text")
            if machine_text and isinstance(machine_text, str):
                texts_to_predict.append(machine_text)
                ground_truths.append(1)  # 1 = AI-generated
            
            document_text = data.get("document")
            if document_text and isinstance(document_text, str) and not human_text and not machine_text:
                texts_to_predict.append(document_text)
                ground_truths.append(0)

        if not texts_to_predict:
            print(f"No valid texts found in the sampled data from {filename} to evaluate.")
            if file_stats["errors"] > 0:
                print(f"  ({file_stats['errors']} lines had JSON parsing errors and were skipped before sampling)")
            evaluation_results["files"][filename] = {
                "human_correct": 0, "human_total": 0, "human_accuracy": 0,
                "ai_correct": 0, "ai_total": 0, "ai_accuracy": 0,
                "overall_correct": 0, "overall_total": 0, "overall_accuracy": 0,
                "precision": 0, "recall": 0, "f1_score": 0,
                "confusion_matrix": {
                    "true_positives": 0, "true_negatives": 0,
                    "false_positives": 0, "false_negatives": 0
                },
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

        probs_save_path = os.path.splitext(filename)[0] + "_probabilities.jsonl"
        probs_save_path = os.path.join("probs", probs_save_path)
        try:
            with open(probs_save_path, 'w', encoding='utf-8') as pf:
                for pred in predictions:
                    pf.write(json.dumps(pred) + "\n")
            print(f"  Probabilities for {filename} saved to {probs_save_path}")
        except Exception as e:
            print(f"  Error saving probabilities for {filename} to {probs_save_path}: {e}")


        true_positives = 0   # correct AI predictions
        true_negatives = 0   # correct human predictions
        false_positives = 0  # incorrect human predictions
        false_negatives = 0  # incorrect AI predictions

        for i, pred_result in enumerate(predictions):
            predicted_label = pred_result["predicted_label"]
            ground_truth_label = ground_truths[i]
            overall_texts_processed += 1
            if predicted_label == ground_truth_label:
                overall_correct_predictions += 1
            # update confusion matrix
            if ground_truth_label == 1 and predicted_label == 1:
                true_positives += 1
                global_true_positives += 1
            elif ground_truth_label == 0 and predicted_label == 0:
                true_negatives += 1
                global_true_negatives += 1
            elif ground_truth_label == 0 and predicted_label == 1:
                false_positives += 1
                global_false_positives += 1
            elif ground_truth_label == 1 and predicted_label == 0:
                false_negatives += 1
                global_false_negatives += 1
            
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
        
        precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
        recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        print(f"Results for {filename}:")
        print(f"  Human Texts: {file_stats['human_correct']}/{file_stats['human_total']} correct ({human_accuracy:.2f}%)")
        print(f"  AI-Generated Texts: {file_stats['ai_correct']}/{file_stats['ai_total']} correct ({ai_accuracy:.2f}%)")
        print(f"  Overall Accuracy for {filename}: {file_total_correct}/{file_total_samples} correct ({file_overall_accuracy:.2f}%)")
        print(f"  Precision (AI Detection): {precision:.4f}")
        print(f"  Recall (AI Detection): {recall:.4f}")
        print(f"  F1 Score (AI Detection): {f1_score:.4f}")
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
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "confusion_matrix": {
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives
            },
            "errors": file_stats["errors"]
        }

    overall_human_accuracy = (all_files_stats["human_correct"] / all_files_stats["human_total"] * 100) if all_files_stats["human_total"] > 0 else 0
    overall_ai_accuracy = (all_files_stats["ai_correct"] / all_files_stats["ai_total"] * 100) if all_files_stats["ai_total"] > 0 else 0
    total_overall_accuracy = (overall_correct_predictions / overall_texts_processed * 100) if overall_texts_processed > 0 else 0
    
    overall_precision = (global_true_positives / (global_true_positives + global_false_positives)) if (global_true_positives + global_false_positives) > 0 else 0
    overall_recall = (global_true_positives / (global_true_positives + global_false_negatives)) if (global_true_positives + global_false_negatives) > 0 else 0
    overall_f1_score = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0

    evaluation_results["overall_summary"] = {
        "total_human_texts_evaluated": all_files_stats["human_total"],
        "correct_human_predictions": all_files_stats["human_correct"],
        "overall_human_accuracy": overall_human_accuracy,
        "total_ai_texts_evaluated": all_files_stats["ai_total"],
        "correct_ai_predictions": all_files_stats["ai_correct"],
        "overall_ai_accuracy": overall_ai_accuracy,
        "total_texts_processed": overall_texts_processed,
        "total_correct_predictions": overall_correct_predictions,
        "overall_accuracy": total_overall_accuracy,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1_score": overall_f1_score,
        "overall_confusion_matrix": {
            "true_positives": global_true_positives,
            "true_negatives": global_true_negatives,
            "false_positives": global_false_positives,
            "false_negatives": global_false_negatives
        }
    }

    print("--- Overall Evaluation Summary ---")
    print(f"Total Human Texts Evaluated: {all_files_stats['human_total']}")
    print(f"  Correct Human Predictions: {all_files_stats['human_correct']} ({overall_human_accuracy:.2f}%)")
    print(f"Total AI-Generated Texts Evaluated: {all_files_stats['ai_total']}")
    print(f"  Correct AI Predictions: {all_files_stats['ai_correct']} ({overall_ai_accuracy:.2f}%)")
    print(f"Total Texts Processed: {overall_texts_processed}")
    print(f"Total Correct Predictions: {overall_correct_predictions}")
    print(f"Overall Accuracy: {total_overall_accuracy:.2f}%")
    print(f"Overall Precision (AI Detection): {overall_precision:.4f}")
    print(f"Overall Recall (AI Detection): {overall_recall:.4f}")
    print(f"Overall F1 Score (AI Detection): {overall_f1_score:.4f}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"\nEvaluation results saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving evaluation results to {output_filepath}: {e}")

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
        help="Path to the fine-tuned LoRA adapter directory or base model name/path if not using adapter."
    )
    parser.add_argument(
        "--dev_data_path",
        type=str,
        default="dev_data/",
        help="Path to the directory containing .json/.jsonl files or a single .json/.jsonl file for evaluation (only if mode is 'evaluate')."
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
        "--use_adapter",
        action="store_true",
        help="Whether to use a LoRA adapter (PEFT) checkpoint. If omitted, runs base model only."
    )
    args = parser.parse_args()

    if args.mode == "predict" and not args.texts:
        parser.error("The 'predict' mode requires at least one text to classify.")

    print(f"Loading model from path: {args.model_path} with use_adapter={args.use_adapter}")
    model, tokenizer = load_model_for_inference(args.model_path, use_adapter=args.use_adapter)
    print("Model and tokenizer loaded successfully.")

    if args.mode == "predict":
        predictions = predict(args.texts, model, tokenizer)
        print("\n--- Inference Results ---")
        for res in predictions:
            label_str = "AI-Generated" if res["predicted_label"] == 1 else "Human-Written"
            print(f"Text: \"{res['text'][:100]}...\"")
            print(f"  Prediction: {label_str} (Class {res['predicted_label']})")
            print(f"  Probabilities: {res['probabilities']}")
            print(f"  Score (AI-Generated): {res['score_ai_generated']:.4f}")
            print("-" * 20)
    elif args.mode == "evaluate":
        evaluate_on_dev_set(model, tokenizer, dev_data_path=args.dev_data_path, output_filepath=args.output_filepath, num_samples=args.num_samples)

if __name__ == "__main__":
    main()


"""
Usage:

# Predict on a list of texts using a specific model checkpoint
python inference.py --mode predict --model_path saved_models/mistral_raid_detector_adapter/checkpoint-500 "I am a human" "I am an AI-generated text" "I'm a human" "The wind whispered through the ancient pines, carrying secrets older than the mountains themselves. In the valley below, a lone figure trudged through the snow, their cloak a patchwork of faded dreams. Stars blinked faintly above, as if unsure whether to guide or merely watch. A distant howl broke the silence, sharp and fleeting, like a memory that refused to be forgotten. The figure paused, breath clouding in the frigid air, and glanced at the horizon where dawn hesitated, unsure of its welcome."

    # Evaluate the model on all .json/.jsonl files in a directory
    python inference.py --mode evaluate --dev_data_path dev_data/ --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000

    # Evaluate the model on a single specific .json/.jsonl file
    python inference.py --mode evaluate --dev_data_path dev_data/dataset1.jsonl --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000
    python inference.py --mode evaluate --dev_data_path dev_data/hewlett.json --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000

    # Evaluate the model on all .json/.jsonl files in a directory, using 50 random samples from each file
    python inference.py --mode evaluate --dev_data_path dev_data/ --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000 --num_samples 50

    # Evaluate the model on a single specific file, using 20 random samples from that file
    python inference.py --mode evaluate --dev_data_path dev_data/arxiv_dolly.jsonl --model_path saved_models/mistral_raid_detector_adapter/checkpoint-1000 --num_samples 100

# Evaluate using the default model path (defined in config.py) and default dev data path (dev_data/)
python inference.py --mode evaluate

#
"""
