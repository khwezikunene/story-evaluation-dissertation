import logging

import pandas as pd

import common
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("postprocess_arm_a1")


RAW_FILE = "arm_a1_per_dimension_raw.parquet"
MEAN_OUTPUT_FILE = "arm_a1_single_dimension_predictions_mean.parquet"
ROUNDED_OUTPUT_FILE = "arm_a1_single_dimension_predictions.parquet"


def main():
    # ------------------------------------------------------------------
    # Load raw generation outputs
    # ------------------------------------------------------------------
    raw_df = common.load_predictions(RAW_FILE)

    n_failed = raw_df["parse_error"].notna().sum()

    logger.info(
        f"Loaded {len(raw_df)} rows with {n_failed} parse failures."
    )

    # ------------------------------------------------------------------
    # Aggregate the three story predictions for each prompt
    # ------------------------------------------------------------------
    mean_df = (
        raw_df
        .groupby(["story_id", "dimension"], as_index=False)["predicted_score"]
        .mean()
    )

    logger.info(
        f"Aggregated to {len(mean_df)} prompt/dimension predictions."
    )

    # ------------------------------------------------------------------
    # Save continuous mean predictions
    # ------------------------------------------------------------------
    mean_wide_df = (
        mean_df
        .pivot(
            index="story_id",
            columns="dimension",
            values="predicted_score",
        )
        .reset_index()
    )

    mean_wide_df.columns.name = None

    common.save_predictions(
        mean_wide_df,
        MEAN_OUTPUT_FILE,
    )

    logger.info(
        f"Saved mean predictions to results/{MEAN_OUTPUT_FILE}"
    )

    # ------------------------------------------------------------------
    # Round means to nearest integer for evaluation
    # ------------------------------------------------------------------
    rounded_df = mean_df.copy()

    # Round the valid predictions
    rounded_df["predicted_score"] = (
        rounded_df["predicted_score"]
        .round()
        .clip(
            lower=config.HANNA_SCORE_MIN,
            upper=config.HANNA_SCORE_MAX,
        )
    )

    # Count prompts where every prediction failed #146
    n_missing = rounded_df["predicted_score"].isna().sum()
    logger.info(f"{n_missing} prompt/dimension combinations have no valid prediction.")
    rounded_df["predicted_score"] = rounded_df["predicted_score"].astype("Int64")  # Leave NaNs as NaN (do not convert them)

    # ------------------------------------------------------------------
    # Convert to wide format
    # ------------------------------------------------------------------
    wide_df = (
        rounded_df
        .pivot(
            index="story_id",
            columns="dimension",
            values="predicted_score",
        )
        .reset_index()
    )

    wide_df.columns.name = None

    missing_dims = set(config.HANNA_DIMENSIONS) - set(wide_df.columns)

    if missing_dims:
        logger.warning(
            f"Missing dimensions: {missing_dims}"
        )

    common.save_predictions(
        wide_df,
        ROUNDED_OUTPUT_FILE,
    )

    logger.info(
        f"Saved rounded predictions to results/{ROUNDED_OUTPUT_FILE}"
    )

    logger.info("Post-processing complete.")


if __name__ == "__main__":
    main()