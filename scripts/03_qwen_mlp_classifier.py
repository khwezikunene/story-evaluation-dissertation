"""
Train the Qwen3-1.7B + LoRA classifier that maps Goodreads review sentences
onto the six custom dimensions (Narrative_Structure_Quality, Character_Emotion,
Originality, Immersion, Thematic_Depth, Writing_Style).

Each dimension is a binary presence/absence label: 1 if the sentence
expresses that dimension, 0 if it doesn't. This is a multi-label problem
(a sentence can express several dimensions at once, or none), so the model
head produces one logit per dimension and is trained with independent
binary cross-entropy per dimension, sharing the same Qwen3 + LoRA encoder.
This is what classifier_mlp.py's predict_proportions (sigmoid over 6
logits) already expects, so no changes are needed downstream.

Class imbalance handling: some dimensions have very low positive rates
(as low as ~3% in earlier annotation passes). BCEWithLogitsLoss's
pos_weight parameter is used per-dimension, computed from the training
data's own label frequencies, to avoid the model collapsing to predicting
the majority (absent) class everywhere.

This script only touches the classifier's own labeled training data (the
annotated Goodreads sentences). It never sees HANNA stories, so it cannot
leak into the mlp_train / few_shot / test splits used later in the
pipeline by train_mlp.py.

Output artifacts (paths come from config.py):
  - config.CLASSIFIER_LORA_PATH: saved LoRA adapter + tokenizer

This is loaded downstream by classifier_mlp.py's ClassifierMLPPipeline,
so this script's save format must match what that module expects.

Usage:
    python 03_qwen_mlp_classifier.py --data-path path/to/annotations.csv
    python 03_qwen_mlp_classifier.py --data-path path/to/annotations.csv --limit 200   # smoke test
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_classifier")

N_DIMS = len(config.CUSTOM_DIMENSIONS)


class SentenceLabelDataset(Dataset):
    """Wraps the annotated sentence dataframe.

    Expects df to have a `text` column plus one binary (0/1) column per
    config.CUSTOM_DIMENSIONS, indicating whether that sentence expresses
    the corresponding dimension.
    """

    def __init__(self, df):
        self.texts = df["text"].tolist()
        self.labels = df[config.CUSTOM_DIMENSIONS].astype(float).values  # (n, N_DIMS)
        if not np.isin(self.labels, [0.0, 1.0]).all():
            raise ValueError(
                "Found dimension labels outside {0, 1}. This script expects "
                "binary presence/absence labels per dimension, not discrete "
                "or continuous scores."
            )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch, tokenizer, max_length):
    texts, labels = zip(*batch)
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    labels = torch.tensor(np.stack(labels), dtype=torch.float32)  # (batch, N_DIMS)
    return encoded, labels



# Maps this project's actual annotation export column names to the
# `text` + config.CUSTOM_DIMENSIONS names this script works with
# internally. Update this if your export's column headers change.
COLUMN_RENAME_MAP = {
    "sentence": "text",
    "Narrative Structure & Quality": "Narrative_Structure_Quality",
    "Character & Emotion": "Character_Emotion",
    "Thematic Depth": "Thematic_Depth",
    "Writing Style": "Writing_Style",
}


def load_training_data(path, limit=None):
    """Load the annotated Goodreads sentence dataset.

    The raw export uses `sentence` for text and spaced/ampersand dimension
    names (e.g. `Narrative Structure & Quality`); COLUMN_RENAME_MAP
    normalizes these to `text` plus the underscored config.CUSTOM_DIMENSIONS
    names used throughout this script. Update COLUMN_RENAME_MAP if your
    export's column headers change.
    """
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    required = {"text"} | set(config.CUSTOM_DIMENSIONS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Training data is missing expected columns: {missing}")
    if limit:
        df = df.head(limit)
    return df


def compute_pos_weights(train_df):
    """Per-dimension pos_weight for BCEWithLogitsLoss, so that the loss
    penalizes missing rare-positive dimensions more heavily than the
    majority (absent) class. weight = n_negative / n_positive, so a
    dimension with 3% positives gets a much larger weight than one at 40%.
    """
    weights = []
    for dim in config.CUSTOM_DIMENSIONS:
        pos = train_df[dim].sum()
        neg = len(train_df) - pos
        if pos == 0:
            logger.warning(
                f"Dimension '{dim}' has zero positive examples in the "
                "training split. Its pos_weight is undefined; using 1.0, "
                "but this dimension's classifier head will likely not learn "
                "anything useful. Check your annotation data."
            )
            weights.append(1.0)
        else:
            weights.append(float(neg / pos))
    logger.info("Per-dimension pos_weight (to counter class imbalance):")
    for dim, w in zip(config.CUSTOM_DIMENSIONS, weights):
        logger.info(f"  {dim}: {w:.2f}")
    return torch.tensor(weights, dtype=torch.float32)


def build_model(device):
    logger.info(f"Loading base model: {config.CLASSIFIER_BASE_MODEL}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_BASE_MODEL,
        num_labels=N_DIMS,
        problem_type="multi_label_classification",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    lora_config = LoraConfig(
        r=getattr(config, "LORA_R", 16),
        lora_alpha=getattr(config, "LORA_ALPHA", 32),
        lora_dropout=getattr(config, "LORA_DROPOUT", 0.05),
        target_modules=getattr(config, "LORA_TARGET_MODULES", ["q_proj", "v_proj"]),
        task_type="SEQ_CLS",
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    return model


class NaNGuard:
    """Tracks batches skipped due to NaN/Inf loss, so persistent
    instability is visible rather than silently corrupting training."""

    def __init__(self):
        self.skipped_batches = 0

    def step_is_valid(self, loss):
        if torch.isnan(loss) or torch.isinf(loss):
            self.skipped_batches += 1
            return False
        return True


def run_epoch(model, loader, device, pos_weight, nan_guard, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    total_loss = 0.0
    total_seen = 0

    all_preds = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for encoded, labels in loader:
            encoded = {k: v.to(device) for k, v in encoded.items()}
            labels = labels.to(device)  # (batch, N_DIMS)

            if is_train:
                optimizer.zero_grad()

            outputs = model(**encoded)
            logits = outputs.logits  # (batch, N_DIMS)

            loss = loss_fn(logits, labels)

            if is_train:
                if not nan_guard.step_is_valid(loss):
                    logger.warning(
                        f"NaN/Inf loss encountered, skipping batch "
                        f"(total skipped so far: {nan_guard.skipped_batches})."
                    )
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)

            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    mean_loss = total_loss / max(total_seen, 1)
    all_preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, N_DIMS))
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0, N_DIMS))
    per_dim_acc = (all_preds == all_labels).mean(axis=0) if len(all_labels) else np.zeros(N_DIMS)
    return mean_loss, per_dim_acc


def main():
    parser = argparse.ArgumentParser(description="Train the Qwen3 LoRA multi-label classifier over 6 custom dimensions")
    parser.add_argument(
        "--data-path",
        type=str,
        default=getattr(config, "CLASSIFIER_TRAIN_DATA_PATH", None),
        help="CSV with a `text` column and one binary (0/1) column per custom dimension",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows, for smoke testing")
    parser.add_argument("--epochs", type=int, default=getattr(config, "CLASSIFIER_EPOCHS", 10))
    parser.add_argument("--batch-size", type=int, default=getattr(config, "CLASSIFIER_BATCH_SIZE", 16))
    parser.add_argument("--lr", type=float, default=getattr(config, "CLASSIFIER_LR", 2e-4))
    parser.add_argument("--val-fraction", type=float, default=getattr(config, "CLASSIFIER_VAL_FRACTION", 0.15))
    args = parser.parse_args()

    if not args.data_path:
        raise ValueError(
            "No training data path provided. Pass --data-path, or set "
            "config.CLASSIFIER_TRAIN_DATA_PATH to your annotated sentence CSV."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    df = load_training_data(args.data_path, limit=args.limit)
    logger.info(f"Loaded {len(df)} annotated sentences.")
    for dim in config.CUSTOM_DIMENSIONS:
        rate = df[dim].mean()
        logger.info(f"  {dim}: {rate:.1%} positive rate")

    train_df, val_df = train_test_split(
        df, test_size=args.val_fraction, random_state=config.RANDOM_SEED
    )
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    pos_weight = compute_pos_weights(train_df)

    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_BASE_MODEL)

    train_dataset = SentenceLabelDataset(train_df)
    val_dataset = SentenceLabelDataset(val_df)

    collate = lambda batch: collate_fn(batch, tokenizer, max_length=512)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    nan_guard = NaNGuard()

    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, device, pos_weight, nan_guard, optimizer=optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, pos_weight, nan_guard, optimizer=None)

        logger.info(
            f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )
        for dim, acc in zip(config.CUSTOM_DIMENSIONS, val_acc):
            logger.info(f"  val accuracy [{dim}]: {acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    else:
        raise RuntimeError(
            "Training produced no valid (non-NaN) validation loss at any "
            "epoch. Lower --lr and re-run."
        )

    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    if nan_guard.skipped_batches:
        logger.warning(f"Skipped {nan_guard.skipped_batches} batches total due to NaN/Inf loss.")

    # Save the LoRA adapter + tokenizer to CLASSIFIER_LORA_PATH, matching
    # what classifier_mlp.py's ClassifierMLPPipeline expects to load: it
    # calls AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_LORA_PATH)
    # then PeftModel.from_pretrained on top of it.
    logger.info(f"Saving LoRA adapter to {config.CLASSIFIER_LORA_PATH}")
    model.save_pretrained(config.CLASSIFIER_LORA_PATH)
    tokenizer.save_pretrained(config.CLASSIFIER_LORA_PATH)


if __name__ == "__main__":
    main()