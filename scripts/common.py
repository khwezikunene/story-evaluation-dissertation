"""
Shared utilities for the story evaluation pipeline.

Other notebooks (04, 05, 06) will import from this module so that data
loading, few-shot example selection, and evaluation logic stay identical
across the pipeline.
"""

import json
import logging
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("common")

######################### Data loading #########################

def load_hanna_data():
    if not os.path.exists(config.HANNA_PATH):
        raise FileNotFoundError(...)
    df = pd.read_csv(config.HANNA_PATH)
    df = df.rename(columns={"Story ID": "story_id", "Story": "story"})
    required_cols = {"story_id", "story"} | set(config.HANNA_DIMENSIONS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"HANNA data is missing expected columns: {missing}")
    return df


def load_hanna_split():
    """Load the frozen few_shot / test split for HANNA stories.
    Returns a DataFrame with columns: story_id, split.
    """
    if not os.path.exists(config.HANNA_SPLIT_PATH):
        raise FileNotFoundError(
            f"HANNA split file not found at {config.HANNA_SPLIT_PATH}. "
            "Create this file once and freeze it across all arms so "
            "every arm evaluates on the same held-out stories."
        )
    split_df = pd.read_csv(config.HANNA_SPLIT_PATH)
    required_cols = {"story_id", "split"}
    missing = required_cols - set(split_df.columns)
    if missing:
        raise ValueError(f"Split file is missing expected columns: {missing}")
    return split_df


def get_few_shot_examples(hanna_df, split_df, n_examples=3):
    """Select few-shot examples from the frozen few_shot split.
    These stories must never overlap with the test split, and must never
    be stories used to train the classifier or MLP.
    """
    few_shot_ids = split_df.loc[split_df["split"] == "few_shot", "story_id"]
    pool = hanna_df[hanna_df["story_id"].isin(few_shot_ids)]
    if len(pool) < n_examples:
        raise ValueError(
            f"Only {len(pool)} few_shot stories available, need {n_examples}. "
            "Add more stories to the few_shot split."
        )
    return pool.sample(n=n_examples, random_state=config.RANDOM_SEED)


def get_test_split(hanna_df, split_df):
    """Return the frozen test split of HANNA stories with gold labels."""
    test_ids = split_df.loc[split_df["split"] == "test", "story_id"]
    test_df = hanna_df[hanna_df["story_id"].isin(test_ids)].reset_index(drop=True)
    if test_df.empty:
        raise ValueError("Test split is empty. Check config.HANNA_SPLIT_PATH.")
    return test_df


def get_mlp_train_split(hanna_df, split_df):
    """Return the frozen mlp_train split of HANNA stories with gold labels.

    This split is used exclusively to fit the MLP's learned mapping from
    classifier proportion vectors to HANNA scores. It must be disjoint from
    both few_shot and test. If any story_id appears in both mlp_train and
    test, MLP training labels leak into arm evaluation and every
    downstream correlation result becomes invalid, so this is checked
    explicitly below rather than assumed.
    """
    mlp_train_ids = split_df.loc[split_df["split"] == "mlp_train", "story_id"]
    test_ids = set(split_df.loc[split_df["split"] == "test", "story_id"])
    overlap = set(mlp_train_ids) & test_ids
    if overlap:
        raise ValueError(
            f"{len(overlap)} story_id(s) appear in both mlp_train and test "
            "splits. This would leak MLP training labels into arm "
            "evaluation. Fix hanna_split.csv before proceeding."
        )
    mlp_train_df = hanna_df[hanna_df["story_id"].isin(mlp_train_ids)].reset_index(drop=True)
    if mlp_train_df.empty:
        raise ValueError(
            "mlp_train split is empty. Add story_ids with split='mlp_train' "
            "to hanna_split.csv before running train_mlp.py."
        )
    return mlp_train_df

######################### Prompt construction #########################

def build_dimension_description_block():
    """Render the HANNA dimension descriptions as a prompt-ready block."""
    lines = []
    for dim in config.HANNA_DIMENSIONS:
        lines.append(f"- {dim}: {config.HANNA_DIMENSION_DESCRIPTIONS[dim]}")
    return "\n".join(lines)

def build_few_shot_block(few_shot_df):
    """Render few-shot examples as story plus gold scores, for the baseline prompt."""
    blocks = []
    for _, row in few_shot_df.iterrows():
        scores_json = json.dumps(
            {dim: float(row[dim]) for dim in config.HANNA_DIMENSIONS}
        )
        blocks.append(
            f"Story:\n{row['story']}\n\nScores:\n{scores_json}"
        )
    return "\n\n---\n\n".join(blocks)

 
def build_single_dimension_few_shot_block(few_shot_df, dimension):
    blocks = []
    for _, row in few_shot_df.iterrows():
        blocks.append(
            f"Story:\n{row['story']}\n\n{dimension} score: {float(row[dimension])}"
        )
    return "\n\n---\n\n".join(blocks)
 
 
def parse_single_score(raw_text):
    match = re.search(r"-?\d+(\.\d+)?", raw_text)
    if not match:
        raise ValueError(f"No numeric score found in model output: {raw_text[:200]}")
    return float(match.group(0))
 


def parse_json_scores(raw_text, expected_keys):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw_text[:200]}")
    parsed = json.loads(match.group(0))
    missing = set(expected_keys) - set(parsed.keys())
    if missing:
        raise ValueError(f"Model output missing keys {missing}: {raw_text[:200]}")
    return {k: float(parsed[k]) for k in expected_keys}


def clip_scores(scores_dict):
    """Clip predicted scores into the valid HANNA score range."""
    return {
        k: float(np.clip(v, config.HANNA_SCORE_MIN, config.HANNA_SCORE_MAX))
        for k, v in scores_dict.items()
    }

######################### Evaluation #########################

def compute_correlations(pred_df, gold_df, dimensions=None):
    """Compute per-dimension and aggregate Spearman and Kendall correlations.

    pred_df and gold_df must both be joinable on story_id, and contain one
    column per dimension. Returns a DataFrame with one row per dimension
    plus an "Aggregate" row (mean across dimensions).
    """
    dimensions = dimensions or config.HANNA_DIMENSIONS
    merged = pred_df.merge(gold_df, on="story_id", suffixes=("_pred", "_gold"))
    if merged.empty:
        raise ValueError(
            "No overlapping story_id values between predictions and gold "
            "labels. Check that both cover the same test split."
        )

    rows = []
    for dim in dimensions:
        pred_col = f"{dim}_pred"
        gold_col = f"{dim}_gold"
        spearman_r, spearman_p = spearmanr(merged[pred_col], merged[gold_col])
        kendall_r, kendall_p = kendalltau(merged[pred_col], merged[gold_col])
        rows.append(
            {
                "dimension": dim,
                "spearman": spearman_r,
                "spearman_p": spearman_p,
                "kendall": kendall_r,
                "kendall_p": kendall_p,
                "n": len(merged),
            }
        )

    result = pd.DataFrame(rows)
    aggregate = {
        "dimension": "Aggregate",
        "spearman": result["spearman"].mean(),
        "spearman_p": np.nan,
        "kendall": result["kendall"].mean(),
        "kendall_p": np.nan,
        "n": len(merged),
    }
    result = pd.concat([result, pd.DataFrame([aggregate])], ignore_index=True)
    return result


def summarize_variance_across_samples(pred_dfs, dimensions=None):
    """Given a list of prediction DataFrames from repeated generations of
    the same stories (e.g. N reviews per story in Arm B), compute the
    standard deviation of predicted scores per story per dimension.

    Each DataFrame in pred_dfs must have identical story_id coverage and
    one column per dimension.
    """
    dimensions = dimensions or config.HANNA_DIMENSIONS
    stacked = pd.concat(pred_dfs, keys=range(len(pred_dfs)), names=["sample_idx"])
    variance = stacked.groupby("story_id")[dimensions].std().reset_index()
    variance = variance.rename(columns={d: f"{d}_std" for d in dimensions})
    return variance

######################### IO helper functions #########################

def ensure_results_dir():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
 

def save_predictions(df, filename):
    ensure_results_dir()
    out_path = os.path.join(config.RESULTS_DIR, filename)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")
    return out_path


def load_predictions(filename):
    path = os.path.join(config.RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Predictions file not found: {path}")
    return pd.read_parquet(path)