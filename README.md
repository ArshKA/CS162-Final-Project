# AI-Detector

A fine-tuned language model for detecting AI-generated text vs. human-written text.
Made for CS 162 Final Project—Professor Saadia Gabriel and TA Ashima Suvarna, UCLA.

## Setup Instructions

### 1. Environment Setup

First, ensure you have Python 3.8+ installed. Then install the required dependencies:

```bash
pip install torch transformers peft accelerate bitsandbytes numpy tqdm
```

### 2. Model and Configuration

Ensure you have the following files in your project directory:
- `config.py` - Contains model configuration parameters
- `dev_inference.py` - Main inference script
- `saved_models/` - Directory containing your trained model checkpoints

### 3. Verify Setup

Check that you have access to the trained model by verifying the checkpoint directory exists:
```bash
ls saved_models/
```

## Running Evaluation on Hidden Dev Set

### Data Format Requirements

The hidden dev set should be in JSON Lines (`.jsonl`) or JSON (`.json`) format with one of the following structures:

**Option 1: Paired human/AI text format:**
```json
{"human_text": "Human written text...", "machine_text": "AI generated text..."}
```

**Option 2: Document format (assumes human-written):**
```json
{"document": "Human written text..."}
```

### Basic Evaluation Command

To evaluate the model on a hidden dev set directory:

```bash
python dev_inference.py --mode evaluate --dev_data_path /path/to/hidden/dev/set/ --model_path saved_models/your_checkpoint --use_adapter
```

### Command Line Arguments

- `--mode evaluate`: Run evaluation mode (required for dev set testing)
- `--dev_data_path`: Path to directory containing `.json`/`.jsonl` files OR path to single file
- `--model_path`: Path to your trained model checkpoint
- `--use_adapter`: Include this flag if using a LoRA adapter (PEFT) checkpoint
- `--output_filepath`: Path to save evaluation results (default: `evaluation_results.json`)
- `--num_samples`: Optional - limit number of samples per file for testing

### Example Commands

**Evaluate on entire hidden dev set:**
```bash
python dev_inference.py --mode evaluate --dev_data_path /path/to/hidden/dev/set/ --model_path saved_models/gemma-checkpoint-1400 --use_adapter --output_filepath hidden_dev_results.json
```

**Evaluate on a single file:**
```bash
python dev_inference.py --mode evaluate --dev_data_path /path/to/hidden/dev/set/test_file.jsonl --model_path saved_models/gemma-checkpoint-1400 --use_adapter
```

**Test with limited samples (for quick verification):**
```bash
python dev_inference.py --mode evaluate --dev_data_path /path/to/hidden/dev/set/ --model_path saved_models/gemma-checkpoint-1400 --use_adapter --num_samples 100
```

## Output and Results

### Evaluation Results

The script will generate:

1. **Console output** showing per-file and overall metrics:
   - Human text accuracy
   - AI text accuracy  
   - Overall accuracy
   - Precision, Recall, F1-score
   - Confusion matrix

2. **JSON results file** (default: `evaluation_results.json`) containing:
   - Per-file detailed metrics
   - Overall summary statistics
   - Confusion matrix values

3. **Probability files** in `probs/` directory:
   - Individual prediction probabilities for each text
   - Format: `filename_probabilities.jsonl`

### Interpreting Results

- **Accuracy**: Percentage of correct classifications
- **Precision**: Of texts classified as AI, how many were actually AI
- **Recall**: Of actual AI texts, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Score (AI-Generated)**: Probability that text is AI-generated (0.0 = human, 1.0 = AI)

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: The script automatically handles device placement and dtype compatibility
2. **Memory Issues**: Reduce batch size in `config.py` (INFERENCE_BATCH_SIZE)
3. **Model Loading Errors**: Ensure model path is correct and checkpoint files are present
4. **Data Format Errors**: Check that JSON files are properly formatted and contain required fields

### Requirements

- GPU recommended for faster inference (CUDA compatible)
- Minimum 8GB RAM for model loading
- Python 3.8+
- All dependencies listed in setup section

## Model Configuration

Key settings in `config.py`:
- `INFERENCE_MODEL_PATH`: Default model checkpoint path
- `MAX_LENGTH`: Maximum sequence length for tokenization
- `INFERENCE_BATCH_SIZE`: Batch size for inference
- `USE_4BIT_QUANTIZATION`: Enable memory-efficient quantization

For questions or issues, refer to the inference script usage examples at the bottom of `dev_inference.py`.