import argparse
import logging

import pandas as pd

import classifier_mlp
import common
import config

logger = logging.getLogger("arm_b")


def main():
    parser = argparse.ArgumentParser(description="Arm B: review-mediated classifier + MLP pipeline")
    parser.add_argument("--reviews-file", type=str, default="generated_reviews.parquet")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of reviews, for smoke testing")
    parser.add_argument("--per-sample-output", type=str, default="arm_b_per_sample_predictions.parquet")
    parser.add_argument("--output", type=str, default="arm_b_predictions.parquet")
    args = parser.parse_args()

    reviews_df = common.load_predictions(args.reviews_file)
    if args.limit:
        reviews_df = reviews_df.head(args.limit)

    pipeline = classifier_mlp.ClassifierMLPPipeline()

    rows = []
    for i, row in reviews_df.iterrows():
        logger.info(
            f"Scoring review {i + 1}/{len(reviews_df)} "
            f"(story_id={row['story_id']}, sample={row['sample_idx']})"
        )
        scores = pipeline.predict_hanna_scores(row["review_text"])
        record = {"story_id": row["story_id"], "sample_idx": row["sample_idx"]}
        record.update(scores)
        rows.append(record)

    per_sample_df = pd.DataFrame(rows)
    common.save_predictions(per_sample_df, args.per_sample_output)

    mean_df = (
        per_sample_df.groupby("story_id")[config.HANNA_DIMENSIONS]
        .mean()
        .reset_index()
    )
    common.save_predictions(mean_df, args.output)

    logger.info(
        f"Done. Scored {len(per_sample_df)} review samples across "
        f"{mean_df['story_id'].nunique()} stories."
    )


if __name__ == "__main__":
    main()