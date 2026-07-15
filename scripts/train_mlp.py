"""
Train the MLP that maps classifier proportion vectors onto HANNA scores.

This replaces the previous fixed 1-to-1 positional mapping between custom
dimensions and HANNA dimensions. Instead, the MLP is trained end-to-end on
the frozen mlp_train split of HANNA stories: the classifier (frozen) is
run over each story's text to produce a 6-dim proportion vector, and the
MLP is trained to regress that vector onto the story's gold HANNA scores.

This script must only ever touch the mlp_train split. It must never see
few_shot or test stories, since mlp_train and test are required to be
disjoint (enforced in common.get_mlp_train_split) and training on test
stories would invalidate every arm evaluation result.

Usage:
    python train_mlp.py
    python train_mlp.py --limit 50   # smoke test
    python train_mlp.py --skip-feature-cache   # force re-running classifier
"""

import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mlp
import common
import config

logger = logging.getLogger("train_mlp")


def extract_classifier_features(pipeline, mlp_train_df, cache_filename, skip_cache=False):
    """Run the frozen classifier over each mlp_train story to get its
    6-dim proportion vector, used as MLP input features. Cached to disk
    so re-running MLP training with different hyperparameters does not
    require re-running the classifier every time.
    """
    if not skip_cache:
        try:
            cached = common.load_predictions(cache_filename)
            cached_ids = set(cached["story_id"])
            needed_ids = set(mlp_train_df["story_id"])
            if needed_ids.issubset(cached_ids):
                logger.info(f"Using cached classifier features from {cache_filename}")
                return cached[cached["story_id"].isin(needed_ids)].reset_index(drop=True)
            logger.info("Cache exists but does not cover all requested stories, regenerating.")
        except FileNotFoundError:
            logger.info("No feature cache found, running classifier over mlp_train split.")

    rows = []
    for i, row in mlp_train_df.iterrows():
        logger.info(f"Extracting features {i + 1}/{len(mlp_train_df)}: story_id={row['story_id']}")
        proportions = pipeline.predict_proportions(row["story"])
        record = {"story_id": row["story_id"]}
        record.update(proportions)
        rows.append(record)

    features_df = pd.DataFrame(rows)
    common.save_predictions(features_df, cache_filename)
    return features_df


def build_tensors(features_df, gold_df):
    merged = features_df.merge(gold_df, on="story_id", suffixes=("_feat", "_gold"))
    x_cols = [f"{d}_feat" if f"{d}_feat" in merged.columns else d for d in config.CUSTOM_DIMENSIONS]
    y_cols = [f"{d}_gold" if f"{d}_gold" in merged.columns else d for d in config.HANNA_DIMENSIONS]
    x = torch.tensor(merged[x_cols].values, dtype=torch.float32)
    y = torch.tensor(merged[y_cols].values, dtype=torch.float32)
    return x, y, merged["story_id"].values


def train_val_split(x, y, val_fraction, seed):
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def train_mlp(x_train, y_train, x_val, y_val):
    model = mlp.MLPRegressor(
        input_dim=len(config.CUSTOM_DIMENSIONS),
        hidden_dim=32,
        output_dim=len(config.HANNA_DIMENSIONS),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.MLP_LEARNING_RATE, weight_decay=config.MLP_WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.MLP_BATCH_SIZE, shuffle=True)

    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(config.MLP_MAX_EPOCHS):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)

            if torch.isnan(preds).any():
                logger.warning(
                    f"NaN detected in predictions at epoch {epoch}, skipping batch. "
                    "If this persists, lower MLP_LEARNING_RATE."
                )
                continue

            loss = loss_fn(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val)
            val_loss = loss_fn(val_preds, y_val).item()

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        logger.info(
            f"Epoch {epoch + 1}/{config.MLP_MAX_EPOCHS}: "
            f"train_loss={mean_train_loss:.4f} val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.MLP_EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.4f}")
            break

    if best_state_dict is None:
        raise RuntimeError(
            "Training produced no valid (non-NaN) validation loss at any "
            "epoch. Lower the learning rate and re-run."
        )

    model.load_state_dict(best_state_dict)
    return model, best_val_loss


def report_learned_mapping(model):
    """Log the learned weight structure so it can be inspected and
    reported in the dissertation, e.g. to check whether the learned
    mapping ends up close to the originally assumed 1-to-1 mapping or
    meaningfully different from it.
    """
    first_layer_weights = model.net[0].weight.detach().cpu().numpy()
    logger.info("Learned first-layer weight matrix (rows=hidden units, cols=custom dims):")
    logger.info(f"Custom dimension order: {config.CUSTOM_DIMENSIONS}")
    logger.info(f"Weight matrix shape: {first_layer_weights.shape}")

    input_importance = np.abs(first_layer_weights).mean(axis=0)
    for dim, importance in zip(config.CUSTOM_DIMENSIONS, input_importance):
        logger.info(f"  {dim}: mean absolute weight magnitude = {importance:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the learned custom-to-HANNA MLP mapping")
    parser.add_argument("--limit", type=int, default=None, help="Limit mlp_train stories, for smoke testing")
    parser.add_argument("--skip-feature-cache", action="store_true")
    parser.add_argument("--output-mlp", type=str, default=None, help="Override output path for mlp_regressor.pt")
    args = parser.parse_args()

    hanna_df = common.load_hanna_data()
    split_df = common.load_hanna_split()
    mlp_train_df = common.get_mlp_train_split(hanna_df, split_df)

    if args.limit:
        mlp_train_df = mlp_train_df.head(args.limit)

    logger.info(f"Training MLP on {len(mlp_train_df)} mlp_train stories.")

    pipeline = mlp.ClassifierMLPPipeline(load_mlp=False)

    features_df = extract_classifier_features(
        pipeline, mlp_train_df, config.MLP_TRAIN_FEATURES_CACHE, skip_cache=args.skip_feature_cache
    )

    gold_df = mlp_train_df[["story_id"] + config.HANNA_DIMENSIONS]
    x, y, story_ids = build_tensors(features_df, gold_df)

    x_train, y_train, x_val, y_val = train_val_split(
        x, y, config.MLP_INTERNAL_VAL_FRACTION, config.RANDOM_SEED
    )
    logger.info(f"Internal split: {x_train.shape[0]} train, {x_val.shape[0]} val stories.")

    model, best_val_loss = train_mlp(x_train, y_train, x_val, y_val)
    logger.info(f"Training complete. Best validation MSE: {best_val_loss:.4f}")

    report_learned_mapping(model)

    output_path = args.output_mlp or config.MLP_WEIGHTS_PATH
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved trained MLP weights to {output_path}")


if __name__ == "__main__":
    main()