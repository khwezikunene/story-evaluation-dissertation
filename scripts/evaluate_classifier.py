"""
Evaluate the trained custom-dimension classifier directly on its own
held-out Goodreads test set, independent of HANNA and independent of the MLP. 

This answers a "is the classifier itself accurate at the task it was actually trained to do."

Expects a labeled Goodreads test set at config.CLASSIFIER_TEST_SET_PATH
with columns: review_text (or sentence_text, see --text-column), and one binary (0/1) 
column per dimension in config.CUSTOM_DIMENSIONS.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

import classifier_mlp
import config
import evaluation_metrics

logger = logging.getLogger("classifier_eval")

def load_test_set(path, text_column):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Classifier test set not found at {path}. Set "
            "--test-set-path to your held-out labeled Goodreads test file."
        )
    df = pd.read_csv(path)
    required_cols = {text_column} | set(config.CUSTOM_DIMENSIONS)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Test set is missing expected columns: {missing}")
    return df


def run_classifier_on_test_set(pipeline, test_df, text_column):
    proba_rows = []
    for i, row in test_df.iterrows():
        if i % 50 == 0:
            logger.info(f"Scoring {i + 1}/{len(test_df)}")
        proportions = pipeline.predict_proportions(row[text_column])
        proba_rows.append([proportions[dim] for dim in config.CUSTOM_DIMENSIONS])
    return np.array(proba_rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the classifier on its own held-out Goodreads test set")
    parser.add_argument(
        "--test-set-path",
        type=str,
        default=os.path.join(config.DATA_DIR, "classifier_test_set.csv"),
    )
    parser.add_argument("--text-column", type=str, default="review_text")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary predictions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default="classifier_evaluation_summary.csv")
    args = parser.parse_args()

    test_df = load_test_set(args.test_set_path, args.text_column)
    if args.limit:
        test_df = test_df.head(args.limit)
    logger.info(f"Loaded {len(test_df)} labeled test examples.")

    for dim in config.CUSTOM_DIMENSIONS:
        positive_rate = test_df[dim].mean()
        logger.info(f"  {dim}: positive rate = {positive_rate:.3f}")
        if positive_rate < 0.05 or positive_rate > 0.95:
            logger.warning(
                f"  {dim} has a severely imbalanced positive rate "
                f"({positive_rate:.3f}). ROC-AUC and F1 for this "
                "dimension should be interpreted cautiously, small "
                "sample sizes at the minority class inflate variance in "
                "these estimates."
            )

    pipeline = classifier_mlp.ClassifierMLPPipeline(load_mlp=False)
    y_pred_proba = run_classifier_on_test_set(pipeline, test_df, args.text_column)
    y_pred_binary = (y_pred_proba >= args.threshold).astype(int)
    y_true = test_df[config.CUSTOM_DIMENSIONS].values

    result = evaluation_metrics.compute_multilabel_classification_metrics(
        y_true, y_pred_binary, y_pred_proba, config.CUSTOM_DIMENSIONS
    )

    summary = result["summary"]
    out_path = os.path.join(config.RESULTS_DIR, args.output)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info(f"Saved classifier evaluation summary to {out_path}")

    confusion_dir = os.path.join(config.RESULTS_DIR, "classifier_confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    for dim, cm_df in result["per_label_confusion"].items():
        cm_df.to_csv(os.path.join(confusion_dir, f"{dim.lower()}.csv"))
    logger.info(f"Saved per-dimension confusion matrices to {confusion_dir}")

    print("\nClassifier evaluation (threshold={:.2f}):\n".format(args.threshold))
    print(summary.round(3).to_string(index=False))

    print(
        "\nNote: This evaluates the classifier against its own Goodreads test labels"
    )

if __name__ == "__main__":
    main()