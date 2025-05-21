import torch
import numpy as np
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
# import evaluate

import config
from data_utils import load_and_preprocess_data
from model_utils import get_model

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


def compute_metrics(pred):
    labels = pred.label_ids
    preds_logits = pred.predictions
    preds_indices = np.argmax(preds_logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds_indices, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds_indices)

    probs = torch.softmax(torch.tensor(preds_logits), dim=-1)[:, 1].numpy()
    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = 0.5

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }

# accuracy_metric = evaluate.load("accuracy")
# f1_metric = evaluate.load("f1")
# precision_metric = evaluate.load("precision")
# recall_metric = evaluate.load("recall")
# roc_auc_metric = evaluate.load("roc_auc")

# def compute_metrics_evaluate_lib(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     probs_positive_class = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
#     acc_result = accuracy_metric.compute(predictions=predictions, references=labels)
#     f1_result = f1_metric.compute(predictions=predictions, references=labels, average="binary", pos_label=1)
#     precision_result = precision_metric.compute(predictions=predictions, references=labels, average="binary", pos_label=1)
#     recall_result = recall_metric.compute(predictions=predictions, references=labels, average="binary", pos_label=1)
#     try:
#         roc_auc_result = roc_auc_metric.compute(references=labels, prediction_scores=probs_positive_class)
#     except ValueError:
#         roc_auc_result = {"roc_auc": 0.5}
#     return {
#         "accuracy": acc_result["accuracy"],
#         "f1": f1_result["f1"],
#         "precision": precision_result["precision"],
#         "recall": recall_result["recall"],
#         "roc_auc": roc_auc_result.get("roc_auc", 0.5)
#     }

def main():
    print("Starting training process...")
    print(f"Loading data for model: {config.MODEL_NAME}")
    train_dataset, val_dataset, tokenizer = load_and_preprocess_data(tokenizer_name=config.MODEL_NAME)

    if train_dataset is None or len(train_dataset) == 0:
        print("No training data loaded. Exiting.")
        return

    print("Loading model...")
    model = get_model(model_name=config.MODEL_NAME, num_labels=2)

    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        gradient_accumulation_steps=config.GRAD_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        warmup_ratio=config.WARMUP_RATIO,
        optim=config.OPTIMIZER,
        logging_dir=config.LOGGING_DIR,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=max(1, (len(train_dataset) // (config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS)) // 4),
        save_strategy="steps",
        save_steps=max(1, (len(train_dataset) // (config.BATCH_SIZE * config.GRAD_ACCUMULATION_STEPS)) // 4),
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="roc_auc" if val_dataset else None,
        greater_is_better=True if val_dataset else None,
        report_to="wandb" if "wandb" in config.LOGGING_DIR else "tensorboard",
        fp16=config.BNB_4BIT_COMPUTE_DTYPE == torch.float16,
        bf16=config.BNB_4BIT_COMPUTE_DTYPE == torch.bfloat16,
        seed=config.SEED,
        max_grad_norm=config.MAX_GRAD_NORM,
    )

    callbacks = []
    if val_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001))

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training finished.")

    print(f"Saving LoRA adapter model to {config.OUTPUT_DIR}")
    trainer.save_model(config.OUTPUT_DIR)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if val_dataset:
        print("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        print(f"Validation metrics: {eval_metrics}")

    print(f"Model and tokenizer adapter saved to {config.OUTPUT_DIR}")
    print("Training script completed.")

if __name__ == "__main__":
    main()
