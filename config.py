import torch

MODEL_NAME = "Qwen/Qwen3-4B"

DATASET_NAME = "liamdugan/raid"
DATASET_SPLIT = "train"
TEXT_COLUMN = "generation"
LABEL_COLUMN = "label"
HUMAN_LABEL_VALUE = "human"
MAX_SAMPLES = 50000
SUBSAMPLE_STRATEGY = "balanced_human_ai"

MAX_LENGTH = 512

OUTPUT_DIR = "./saved_models/mistral_raid_detector_adapter_qwen3"
LOGGING_DIR = "./wandb/detector-test-qwen3"
NUM_EPOCHS = 5
BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
OPTIMIZER = "paged_adamw_8bit"
MAX_GRAD_NORM = 1.0

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

USE_4BIT_QUANTIZATION = False
BNB_4BIT_COMPUTE_DTYPE = torch.bfloat16
BNB_4BIT_QUANT_TYPE = "nf4"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 1
TEST_SPLIT_SIZE = 0.1

INFERENCE_MODEL_PATH = OUTPUT_DIR
INFERENCE_BATCH_SIZE = 8

MAX_EVAL_SAMPLES = 100  # Evaluate on 500 samples, or set to None to evaluate on all
