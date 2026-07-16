"""
Evaluate and compare Arm A, Arm B, and Arm C against gold HANNA labels,
with a full evaluation battery: rank correlation, regression metrics, and
ordinal classification metrics (accuracy, precision, recall, F1 macro and
micro, quadratic weighted kappa, confusion matrices).

Produces, per arm and combined:
  - results/arm_comparison_full.csv: every metric, every dimension, every
    arm, in one table. This is the master results table for the
    dissertation results chapter.
  - results/arm_comparison_aggregate_only.csv: just the Aggregate row per
    arm, for a compact summary table or figure.
  - results/confusion_matrices/{arm}_{dimension}.csv: one confusion
    matrix per arm per dimension, suitable for appendix inclusion or for
    plotting.
  - results/arm_b_variance.csv: standard deviation of Arm B predictions
    across the N generated review samples per story (unchanged from
    previous version).

Run this after 01, 02, 03, and 04 have all produced their output files.

Usage:
    python 05_evaluate_arms.py
"""

import argparse
import logging
import os

import pandas as pd

import common
import config
import evaluation_metrics

logger = logging.getLogger("evaluate_arms")

ARM_FILES = {
    "Arm A (direct Qwen baseline)": "arm_a_baseline_predictions.parquet",
    "Arm B (review -> classifier -> MLP)": "arm_b_predictions.parquet",
    "Arm C (story -> classifier -> MLP)": "arm_c_predictions.parquet",
}


def slugify(text):
    return text.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("->", "to").replace("--", "-")


def save_confusion_matrices(arm_name, confusion_matrices):
    out_dir = os.path.join(config.RESULTS_DIR, "confusion_matrices")
    os.makedirs(out_dir, exist_ok=True)
    arm_slug = slugify(arm_name)
    for dim, cm_df in confusion_matrices.items():
        out_path = os.path.join(out_dir, f"{arm_slug}_{dim.lower()}.csv")
        cm_df.to_csv(out_path)
    logger.info(f"Saved {len(confusion_matrices)} confusion matrices for {arm_name} to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full detailed comparison of all arms against gold HANNA labels")
    parser.add_argument("--output", type=str, default="arm_comparison_full.csv")
    parser.add_argument("--aggregate-output", type=str, default="arm_comparison_aggregate_only.csv")
    args = parser.parse_args()

    hanna_df = common.load_hanna_data()
    gold_df = hanna_df[["story_id"] + config.HANNA_DIMENSIONS]

    all_results = []
    for arm_name, filename in ARM_FILES.items():
        path = os.path.join(config.RESULTS_DIR, filename)
        if not os.path.exists(path):
            logger.warning(f"Skipping {arm_name}: {filename} not found yet.")
            continue

        pred_df = common.load_predictions(filename)
        pred_df = pred_df[["story_id"] + config.HANNA_DIMENSIONS]

        correlation_df = common.compute_correlations(pred_df, gold_df)
        report = evaluation_metrics.build_full_arm_report(pred_df, gold_df, correlation_df)

        combined = report["combined_summary"]
        combined.insert(0, "arm", arm_name)
        all_results.append(combined)

        save_confusion_matrices(arm_name, report["confusion_matrices"])

    if not all_results:
        logger.error(
            "No arm prediction files found in results/. Run the arm "
            "scripts (01, 02+03, 04) before evaluating."
        )
        return

    full_df = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(config.RESULTS_DIR, args.output)
    full_df.to_csv(out_path, index=False)
    logger.info(f"Saved full metric table to {out_path}")

    aggregate_df = full_df[full_df["dimension"] == "Aggregate"].copy()
    aggregate_path = os.path.join(config.RESULTS_DIR, args.aggregate_output)
    aggregate_df.to_csv(aggregate_path, index=False)
    logger.info(f"Saved aggregate-only table to {aggregate_path}")

    display_cols = [
        "arm", "spearman", "kendall", "mae", "rmse", "r2",
        "accuracy", "f1_macro", "f1_micro", "quadratic_weighted_kappa", "n",
    ]
    print("\nAggregate metrics by arm (rounded to 3dp):\n")
    print(aggregate_df[display_cols].round(3).to_string(index=False))

    per_dimension_cols = ["arm", "dimension", "spearman", "mae", "accuracy", "f1_macro", "quadratic_weighted_kappa"]
    print("\nPer-dimension breakdown (rounded to 3dp):\n")
    print(full_df[full_df["dimension"] != "Aggregate"][per_dimension_cols].round(3).to_string(index=False))

    # Arm B variance across generated review samples, if available.
    per_sample_path = os.path.join(config.RESULTS_DIR, "arm_b_per_sample_predictions.parquet")
    if os.path.exists(per_sample_path):
        per_sample_df = common.load_predictions("arm_b_per_sample_predictions.parquet")
        variance_rows = []
        for sample_idx, group in per_sample_df.groupby("sample_idx"):
            variance_rows.append(group.set_index("story_id")[config.HANNA_DIMENSIONS])
        variance_df = common.summarize_variance_across_samples(variance_rows)
        variance_out_path = os.path.join(config.RESULTS_DIR, "arm_b_variance.csv")
        variance_df.to_csv(variance_out_path, index=False)
        logger.info(f"Saved Arm B generation variance to {variance_out_path}")

        mean_std = variance_df[[c for c in variance_df.columns if c.endswith("_std")]].mean()
        print("\nArm B: mean std dev of predicted score across generated review samples:\n")
        print(mean_std.round(3).to_string())
    else:
        logger.info(
            "arm_b_per_sample_predictions.parquet not found, skipping "
            "variance report. Run 03_arm_b_review_classifier_mlp.py to "
            "produce it."
        )

    logger.info(
        "Reminder for the dissertation: report quadratic_weighted_kappa "
        "alongside accuracy and F1 for the ordinal classification view, "
        "since plain accuracy and unweighted F1 treat a prediction that "
        "is off by one point the same as one that is off by four points, "
        "which understates how badly a model is doing on large errors "
        "and overstates how well it is doing on small ones."
    )


if __name__ == "__main__":
    main()