"""
Arm A.1: single-dimension prompting ablation of the Arm A baseline.
This tests whether narrowing the prompt's focus to one dimension at a time changes prediction quality: 
for each of the six dimensions, a separate prompt is built containing only that dimension's description 
and few-shot examples showing only that dimension's gold score, and the model returns a single number rather
than a six-key JSON object.

This isolates a specific hypothesis: joint multi-dimension prompting may cause the model 
to average or conflate distinct dimensions and asking about one dimension in isolation 
may reduce that cross-dimension interference, at the cost of six times the inference calls per story.
"""
import argparse
import logging
 
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
import common
import config
 
logger = logging.getLogger("arm_a1")
 
 
SINGLE_DIMENSION_PROMPT_TEMPLATE = """You are an expert literary evaluator. Score the following \
story on a single dimension: {dimension}, on a scale from {min_score} to {max_score}.
 
Dimension definition:
{dimension_description}
 
Here are some worked examples of stories and their gold scores for this dimension only:
 
{few_shot_block}
 
---
 
Now score this new story on {dimension} only. Respond with ONLY a single \
number, and nothing else.
 
Story:
{story}
 
{dimension} score:"""
 
 
def build_prompt(story_text, dimension, few_shot_block):
    return SINGLE_DIMENSION_PROMPT_TEMPLATE.format(
        dimension=dimension,
        min_score=config.HANNA_SCORE_MIN,
        max_score=config.HANNA_SCORE_MAX,
        dimension_description=config.HANNA_DIMENSION_DESCRIPTIONS[dimension],
        few_shot_block=few_shot_block,
        story=story_text,
    )
 
 
def load_generation_model():
    logger.info(f"Loading generation model: {config.GENERATION_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.GENERATION_MODEL,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model 
 
def generate_single_dimension_score(story_text, dimension, few_shot_block, tokenizer, model):
    prompt = build_prompt(story_text, dimension, few_shot_block)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
 
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=config.BASELINE_TEMPERATURE > 0,
            temperature=max(config.BASELINE_TEMPERATURE, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
 
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
 
    try:
        score = common.parse_single_score(raw_text)
        score = float(min(max(score, config.HANNA_SCORE_MIN), config.HANNA_SCORE_MAX))
        return score, raw_text, None
    except Exception as exc:
        logger.warning(f"Failed to parse model output, will record as NaN: {exc}")
        return float("nan"), raw_text, str(exc)
 
 
def main():
    parser = argparse.ArgumentParser(description="Arm A.1: single-dimension prompting ablation")
    parser.add_argument("--n-few-shot", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test stories, for smoke testing")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=config.HANNA_DIMENSIONS,
        help="Which dimensions to run this pass. Defaults to all six.",
    )
    parser.add_argument("--raw-output", type=str, default="arm_a1_per_dimension_raw.parquet")
    parser.add_argument("--output", type=str, default="arm_a1_single_dimension_predictions.parquet")
    args = parser.parse_args()
 
    invalid_dims = set(args.dimensions) - set(config.HANNA_DIMENSIONS)
    if invalid_dims:
        raise ValueError(f"Unknown dimension(s): {invalid_dims}. Must be a subset of {config.HANNA_DIMENSIONS}")
 
    hanna_df = common.load_hanna_data()
    split_df = common.load_hanna_split()
    test_df = common.get_test_split(hanna_df, split_df)
    few_shot_df = common.get_few_shot_examples(hanna_df, split_df, n_examples=args.n_few_shot)

    print(test_df.columns.tolist())
 
    if args.limit:
        test_df = test_df.head(args.limit)
 
    tokenizer, model = load_generation_model()
 
    rows = []
    total = len(test_df) * len(args.dimensions)
    done = 0
    for dimension in args.dimensions:
        few_shot_block = common.build_single_dimension_few_shot_block(few_shot_df, dimension)
        for _, row in test_df.iterrows():
            done += 1
            logger.info(
                f"Scoring {done}/{total} (story_id={row['story_id']}, dimension={dimension})"
            )
            score, raw_text, error = generate_single_dimension_score(
                row["story"], dimension, few_shot_block, tokenizer, model
            )
            rows.append(
                {
                    "story_id": row["story_id"],
                    "dimension": dimension,
                    "story_row": row.name,   ## for a new unique ID
                    "predicted_score": score,
                    "raw_output": raw_text,
                    "parse_error": error,
                }
            )
 
    raw_df = pd.DataFrame(rows)
    common.save_predictions(raw_df, args.raw_output)
 
    n_failed = raw_df["parse_error"].notna().sum()
    logger.info(f"Done generating. {len(raw_df)} calls made, {n_failed} parse failures.")
 
    ##wide_df = raw_df.pivot(index="story_id", columns="dimension", values="predicted_score").reset_index()
    wide_df = raw_df.pivot(index=["story_id", "story_row"], columns="dimension", values="predicted_score").reset_index()

    wide_df.columns.name = None
 
    try:
        existing_df = common.load_predictions(args.output)
        new_dim_cols = [c for c in wide_df.columns if c != "story_id"]
        existing_df = existing_df.drop(columns=[c for c in new_dim_cols if c in existing_df.columns])
        wide_df = existing_df.merge(wide_df, on=["story_id", "story_row"], how="outer")
        logger.info(
            f"Merged newly-run dimension(s) {new_dim_cols} into existing "
            f"{args.output}, preserving previously-run dimensions."
        )
    except FileNotFoundError:
        pass
 
    missing_dims = set(config.HANNA_DIMENSIONS) - set(wide_df.columns)
    if missing_dims:
        logger.warning(
            f"Dimensions {missing_dims} have not been run yet (see --dimensions) "
            "and are absent from the combined output. This file cannot be used "
            "for a full arm evaluation until all six dimensions have been run "
            "across one or more passes, since 05_evaluate_arms.py expects all "
            "six columns."
        )
 
    common.save_predictions(wide_df, args.output)
    logger.info(f"Saved wide-format predictions to results/{args.output}")
 
 
if __name__ == "__main__":
    main()