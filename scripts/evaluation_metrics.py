"""
Comprehensive evaluation metrics, shared across the arm comparison and classifier-level evaluation.

1. Rank-based correlation (Spearman, Kendall) - already used in
   common.py, kept there for backward compatibility with existing arm
   scripts. 
   Answers: does the predicted ordering of stories match human
   judgment ordering.

2. Regression metrics (MAE, RMSE, R^2) 
   Answers: how close are predicted scores to gold scores in absolute terms, not just rank order. A model
   can have perfect rank correlation while being systematically miscalibrated
   (e.g. always predicting one point too high), and only regression metrics
   surface that.

3. Ordinal classification metrics 
   (accuracy, precision, recall, F1 macro and micro, per-class breakdown, confusion matrix, quadratic weighted kappa) 
   answers: if you treat each HANNA score as a discrete class (on the 1-5 scale), how well do
   predictions agree with gold labels as a classification problem. 

4. Multi-label classification metrics (per-dimension precision, recall, F1, accuracy, ROC-AUC, plus macro and micro aggregates) 
   used for evaluating the classifier's six custom dimensions directly against
   their own binary labels, a separate evaluation target from HANNA scores.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
import config

logger = logging.getLogger("evaluation_metrics")


###############  Regression metrics ##############

def compute_regression_metrics(pred_df, gold_df, dimensions=None):
    """MAE, RMSE, and R^2 per dimension plus an aggregate row (mean cross dimensions). 
    Complements rank correlation by capturing absolute calibration, not just ordering.
    """
    dimensions = dimensions or config.HANNA_DIMENSIONS
    merged = pred_df.merge(gold_df, on="story_id", suffixes=("_pred", "_gold"))
    if merged.empty:
        raise ValueError("No overlapping story_id values between predictions and gold labels.")

    rows = []
    for dim in dimensions:
        pred_col = f"{dim}_pred"
        gold_col = f"{dim}_gold"
        y_pred = merged[pred_col].values
        y_gold = merged[gold_col].values

        mae = mean_absolute_error(y_gold, y_pred)
        rmse = np.sqrt(mean_squared_error(y_gold, y_pred))
        r2 = r2_score(y_gold, y_pred)

        rows.append({"dimension": dim, "mae": mae, "rmse": rmse, "r2": r2, "n": len(merged)})

    result = pd.DataFrame(rows)
    aggregate = {
        "dimension": "Aggregate",
        "mae": result["mae"].mean(),
        "rmse": result["rmse"].mean(),
        "r2": result["r2"].mean(),
        "n": len(merged),
    }
    result = pd.concat([result, pd.DataFrame([aggregate])], ignore_index=True)
    return result

############## Ordinal classification metrics (HANNA scores treated as discrete classes) ##############

def scores_to_classes(scores, score_min=None, score_max=None):
    """Round continuous predicted or gold scores to the nearest integer
    class on the HANNA scale, clipped to the valid range. Gold HANNA
    scores are themselves means across three annotators, so they are
    fractional even before this step; rounding is necessary to treat
    this as a classification problem rather than regression.
    """
    score_min = score_min if score_min is not None else config.HANNA_SCORE_MIN
    score_max = score_max if score_max is not None else config.HANNA_SCORE_MAX
    rounded = np.round(scores).astype(int)
    return np.clip(rounded, score_min, score_max)


def compute_ordinal_classification_metrics(pred_df, gold_df, dimensions=None):
    """Accuracy, macro/micro precision, recall, F1, quadratic weighted
    kappa, and confusion matrix per dimension, treating each HANNA
    dimension as an ordinal multi-class classification problem after
    rounding both predictions and gold scores to integer classes.

    Returns a dict with two keys:
      - "summary": DataFrame, one row per dimension plus an Aggregate row
      - "confusion_matrices": dict of dimension -> confusion matrix (as
        DataFrame, rows=gold class, cols=predicted class), for inclusion
        as figures/appendix tables in the dissertation
    """
    dimensions = dimensions or config.HANNA_DIMENSIONS
    merged = pred_df.merge(gold_df, on="story_id", suffixes=("_pred", "_gold"))
    if merged.empty:
        raise ValueError("No overlapping story_id values between predictions and gold labels.")

    class_labels = list(range(config.HANNA_SCORE_MIN, config.HANNA_SCORE_MAX + 1))

    rows = []
    confusion_matrices = {}
    for dim in dimensions:
        pred_col = f"{dim}_pred"
        gold_col = f"{dim}_gold"
        y_pred = scores_to_classes(merged[pred_col].values)
        y_gold = scores_to_classes(merged[gold_col].values)

        accuracy = accuracy_score(y_gold, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_gold, y_pred, labels=class_labels, average="macro", zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_gold, y_pred, labels=class_labels, average="micro", zero_division=0
        )
        qwk = cohen_kappa_score(y_gold, y_pred, labels=class_labels, weights="quadratic")

        rows.append(
            {
                "dimension": dim,
                "accuracy": accuracy,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "precision_micro": precision_micro,
                "recall_micro": recall_micro,
                "f1_micro": f1_micro,
                "quadratic_weighted_kappa": qwk,
                "n": len(merged),
            }
        )

        cm = confusion_matrix(y_gold, y_pred, labels=class_labels)
        cm_df = pd.DataFrame(
            cm,
            index=[f"gold_{c}" for c in class_labels],
            columns=[f"pred_{c}" for c in class_labels],
        )
        confusion_matrices[dim] = cm_df

    summary = pd.DataFrame(rows)
    numeric_cols = [c for c in summary.columns if c not in ("dimension", "n")]
    aggregate = {"dimension": "Aggregate", "n": len(merged)}
    for col in numeric_cols:
        aggregate[col] = summary[col].mean()
    summary = pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)

    return {"summary": summary, "confusion_matrices": confusion_matrices}


############## Multi-label classification metrics (custom classifier dimensions) ############## 

def compute_multilabel_classification_metrics(y_true, y_pred_binary, y_pred_proba, label_names):
    """Per-label and aggregate metrics for a multi-label binary
    classification problem, i.e. the custom-dimension classifier's own
    output evaluated against its own binary gold labels (not HANNA
    scores). This is a distinct evaluation target from the arm-level
    HANNA metrics above and should be reported separately in the
    dissertation as classifier validation, prior to and independent of
    how well the classifier's output helps predict HANNA scores.

    Args:
        y_true: array (n_samples, n_labels) of binary gold labels
        y_pred_binary: array (n_samples, n_labels) of binary predictions
            (e.g. thresholded at 0.5)
        y_pred_proba: array (n_samples, n_labels) of predicted
            probabilities, used for ROC-AUC
        label_names: list of dimension names, length n_labels

    Returns a dict with "summary" (DataFrame, one row per label plus
    macro/micro aggregate rows) and "per_label_confusion" (dict of
    label -> 2x2 confusion matrix as DataFrame).
    """
    y_true = np.asarray(y_true)
    y_pred_binary = np.asarray(y_pred_binary)
    y_pred_proba = np.asarray(y_pred_proba)

    rows = []
    confusions = {}
    for i, label in enumerate(label_names):
        true_col = y_true[:, i]
        pred_col = y_pred_binary[:, i]
        proba_col = y_pred_proba[:, i]

        accuracy = accuracy_score(true_col, pred_col)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_col, pred_col, average="binary", zero_division=0
        )

        n_positive = int(true_col.sum())
        if n_positive == 0 or n_positive == len(true_col):
            auc = float("nan")
            logger.warning(
                f"{label}: cannot compute ROC-AUC, only one class present "
                f"in gold labels ({n_positive}/{len(true_col)} positive). "
                "Report this explicitly rather than omitting the row."
            )
        else:
            auc = roc_auc_score(true_col, proba_col)

        rows.append(
            {
                "dimension": label,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
                "n_positive": n_positive,
                "n_total": len(true_col),
                "positive_rate": n_positive / len(true_col),
            }
        )

        cm = confusion_matrix(true_col, pred_col, labels=[0, 1])
        confusions[label] = pd.DataFrame(
            cm, index=["gold_negative", "gold_positive"], columns=["pred_negative", "pred_positive"]
        )

    summary = pd.DataFrame(rows)

    macro_row = {
        "dimension": "Macro average",
        "accuracy": summary["accuracy"].mean(),
        "precision": summary["precision"].mean(),
        "recall": summary["recall"].mean(),
        "f1": summary["f1"].mean(),
        "roc_auc": summary["roc_auc"].mean(),
        "n_positive": summary["n_positive"].sum(),
        "n_total": summary["n_total"].sum(),
        "positive_rate": np.nan,
    }

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred_binary.ravel(), average="binary", zero_division=0
    )
    micro_accuracy = accuracy_score(y_true.ravel(), y_pred_binary.ravel())
    micro_row = {
        "dimension": "Micro average",
        "accuracy": micro_accuracy,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1,
        "roc_auc": np.nan,
        "n_positive": int(y_true.sum()),
        "n_total": y_true.size,
        "positive_rate": float(y_true.sum()) / y_true.size,
    }

    summary = pd.concat([summary, pd.DataFrame([macro_row, micro_row])], ignore_index=True)
    return {"summary": summary, "per_label_confusion": confusions}

################### Combined report builder ###################

def build_full_arm_report(pred_df, gold_df, correlation_df, dimensions=None):
    """Combine rank correlation (passed in from common.compute_correlations),
    regression metrics, and ordinal classification metrics into one
    dictionary, for a single arm. Used by 05_evaluate_arms.py to build a
    complete per-arm evaluation without repeating merge logic three times.
    """
    dimensions = dimensions or config.HANNA_DIMENSIONS
    regression_df = compute_regression_metrics(pred_df, gold_df, dimensions)
    ordinal_result = compute_ordinal_classification_metrics(pred_df, gold_df, dimensions)

    combined = correlation_df.merge(regression_df, on=["dimension", "n"])
    combined = combined.merge(ordinal_result["summary"], on=["dimension", "n"])

    return {
        "combined_summary": combined,
        "confusion_matrices": ordinal_result["confusion_matrices"],
    }