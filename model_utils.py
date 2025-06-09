import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import config

def get_model(model_name: str, num_labels: int = 2):
    print(f"Loading base model: {model_name}")
    quantization_config = None
    if config.USE_4BIT_QUANTIZATION:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=config.BNB_4BIT_COMPUTE_DTYPE,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization (QLoRA).")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        quantization_config=quantization_config,
        torch_dtype=config.BNB_4BIT_COMPUTE_DTYPE if config.USE_4BIT_QUANTIZATION else config.BNB_4BIT_COMPUTE_DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
        print(f"Set model.config.pad_token_id to EOS token ID: {model.config.eos_token_id}")
    if config.USE_LORA:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            target_modules=config.LORA_TARGET_MODULES,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        print("PEFT LoRA model configured.")
        model.print_trainable_parameters()
    else:
        print("Skipping LoRA configuration because USE_LORA is set to False.")
    return model

if __name__ == '__main__':
    print("Testing model_utils.py...")
    model = get_model(config.MODEL_NAME)
    print("\nModel architecture:")
    print(model)
    print("\nModel configuration:")
    print(model.config)
    print("Model loading test complete.")