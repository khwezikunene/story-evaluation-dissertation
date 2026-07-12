"""
01_train_qwen.py

Fine-tunes Qwen3-1.7B with a LoRA adapter as a multi-label sequence classifier
over six custom story-quality dimensions (Narrative Structure & Quality,
Character & Emotion, Originality, Immersion, Thematic Depth, Writing Style).

Pipeline:
    1. Load hand-annotated sentence dataset.
    2. Split into train / val / test.
    3. Wrap in a Dataset + attach a NaN-safe Trainer subclass (gradient/loss
       guarding was needed to stabilise training given severe class imbalance).
    4. Train with LoRA adapters, evaluate on held-out test set.
    5. Save the adapter + tokenizer for downstream scoring (see
       02_score_reviews.py).

Cleaned up from the original notebook: dead/duplicate code removed, all
logic organised into functions, single entry point via `main()`.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

################################################# Configuration ################################################# 

SEED = 42
MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_LEN = 128

BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 5
LR = 2e-4
WARMUP_RATIO = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# NOTE: update these paths for the environment the script runs in.
ANNOTATED_FILE = Path("/scratch/kk01697/data/processed/annotation_sample_annotated.csv")
OUTPUT_DIR = Path("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/scripts/objective1/outputs")
MODEL_DIR = OUTPUT_DIR / "qwen3_lora_classifier"

CUSTOM_DIM = ["Narrative Structure & Quality","Character & Emotion","Originality","Immersion","Thematic Depth","Writing Style",]

#################################################  Dataset ################################################# 

class SentenceDataset(Dataset):
    """Tokenised sentence-level dataset with multi-label float targets."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }

################################################# Trainer with NaN/Inf gradient guarding ################################################# 

class NaNGuardTrainer(Trainer):
    """
    Trainer subclass that:
      - computes weighted BCE loss for multi-label classification,
      - replaces NaN losses with 0 (with a warning) instead of crashing,
      - zeros any NaN/Inf gradients before the optimizer step.

    This was required to stabilise training given some dimensions have
    positive rates as low as ~3%.
    """

    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight.to(DEVICE) if pos_weight is not None else None
        self._nan_batches = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()  # force fp32 even in bf16 runs
        labels = labels.float()

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight)
        loss = loss_fn(logits, labels)

        if torch.isnan(loss):
            self._nan_batches += 1
            print(f"[NaNGuard] NaN loss detected (batch #{self._nan_batches}) — skipping")
            loss = torch.tensor(0.0, requires_grad=True, device=DEVICE)

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Zero any NaN/Inf gradients before the optimizer touches them.
        # With LoRA this only scans the small set of trainable adapter params.
        nan_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                bad = torch.isnan(param.grad) | torch.isinf(param.grad)
                if bad.any():
                    param.grad[bad] = 0.0
                    nan_params.append(name)
        if nan_params:
            shown = nan_params[:3]
            suffix = "..." if len(nan_params) > 3 else ""
            print(f"[NaNGuard] Zeroed NaN/Inf grads in: {shown}{suffix}")

        return loss

#################################################  Metrics ################################################# 

def build_compute_metrics(threshold: float = 0.5):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
        preds = (probs >= threshold).astype(int)

        micro = f1_score(labels, preds, average="micro", zero_division=0)
        macro = f1_score(labels, preds, average="macro", zero_division=0)
        per_dim = f1_score(labels, preds, average=None, zero_division=0)

        metrics = {"f1_micro": micro, "f1_macro": macro}
        for dim, score in zip(CUSTOM_DIM, per_dim):
            metrics[f"f1_{dim}"] = score
        return metrics

    return compute_metrics

################################################# Data loading / splitting ################################################# 
def load_annotations() -> pd.DataFrame:
    df = pd.read_csv(ANNOTATED_FILE)
    df = df.dropna(subset=["sentence"])
    print("Shape:", df.shape)
    return df


def make_splits(ann_df: pd.DataFrame, tokenizer):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sentences = ann_df["sentence"].tolist()
    labels = ann_df[CUSTOM_DIM].values.astype(np.float32)

    idx = np.arange(len(sentences))
    idx_trainval, idx_test = train_test_split(idx, test_size=0.15, random_state=SEED)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15, random_state=SEED)

    train_labels = labels[idx_train]
    pos_counts = train_labels.sum(axis=0).clip(min=1)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)

    print(f"Split — train: {len(idx_train)}  val: {len(idx_val)}  test: {len(idx_test)}\n")
    print(f"{'Dimension':<35} {'Pos':>5}  {'pos_weight':>10}")
    print("-" * 55)
    for dim, p, w in zip(CUSTOM_DIM, pos_counts.astype(int), pos_weight.tolist()):
        print(f"{dim:<35} {p:>5}  {w:>10.1f}")

    train_ds = SentenceDataset([sentences[i] for i in idx_train], train_labels, tokenizer)
    val_ds = SentenceDataset([sentences[i] for i in idx_val], labels[idx_val], tokenizer)
    test_ds = SentenceDataset([sentences[i] for i in idx_test], labels[idx_test], tokenizer)

    print("Datasets created.")
    print("Pad token:", tokenizer.pad_token, "| id:", tokenizer.pad_token_id)
    print("Sample labels:", train_ds[0]["labels"])

    return train_ds, val_ds, test_ds, pos_weight

################################################# Model construction ################################################# 

def build_model(tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CUSTOM_DIM),
        torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # the classification head is randomly initialised and small; train it in full
        modules_to_save=["score"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print("Model on   :", next(model.parameters()).device)
    return model


def sanity_check(model, train_ds, pos_weight):
    sample = {k: v.unsqueeze(0).to(DEVICE) for k, v in train_ds[0].items() if k != "labels"}
    label = train_ds[0]["labels"].unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(**sample)
    model.train()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    check_loss = loss_fn(out.logits.float(), label.float())

    print(f"Logits : {out.logits.float().cpu()}")
    print(f"Loss   : {check_loss.item():.4f}")

    assert not torch.isnan(check_loss), "NaN loss before training starts — check data!"
    assert check_loss.item() < 20, "Loss is unreasonably large — check pos_weight!"
    print("\nSanity check passed \u2713")


def build_training_args(train_ds) -> TrainingArguments:
    steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    print(f"Total optimizer steps: {total_steps}  |  Warmup steps: {warmup_steps}")

    return TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=False,
        bf16=USE_BF16,
        # gradient / memory stability
        max_grad_norm=0.5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # paged_adamw_8bit keeps optimizer state memory down
        optim="paged_adamw_8bit",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        label_names=["labels"],
        save_total_limit=1,
        report_to="none",
    )


def print_env_info():
    import accelerate
    import peft
    import transformers

    print("torch        :", torch.__version__)
    print("transformers :", transformers.__version__)
    print("accelerate   :", accelerate.__version__)
    print("peft         :", peft.__version__)
    print("CUDA         :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU          :", torch.cuda.get_device_name(0))
        print("VRAM (GB)    :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
        print("bf16 support :", torch.cuda.is_bf16_supported())


################################################# Main ################################################# 

def main():
    print_env_info()
    print(f"Compute dtype: {'bf16' if USE_BF16 else 'fp32'}")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    ann_df = load_annotations()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, val_ds, test_ds, pos_weight = make_splits(ann_df, tokenizer)

    model = build_model(tokenizer)
    sanity_check(model, train_ds, pos_weight)

    training_args = build_training_args(train_ds)
    trainer = NaNGuardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=build_compute_metrics(),
        pos_weight=pos_weight,
    )

    trainer.train()
    print(f"\nNaN batches caught by guard: {trainer._nan_batches}")

    # Saves only the LoRA adapter + classification head.
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print("Adapter + tokenizer saved to", MODEL_DIR)

    model.config.num_labels = len(CUSTOM_DIM)
    model.config.id2label = {i: label for i, label in enumerate(CUSTOM_DIM)}
    model.config.label2id = {label: i for i, label in enumerate(CUSTOM_DIM)}

    test_results = trainer.evaluate(test_ds)
    print("\n\u2500\u2500 Test-set evaluation \u2500\u2500")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()