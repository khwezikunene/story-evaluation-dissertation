"""
Arm A: Baseline - direct Qwen prompting on HANNA story text.

Prompts Qwen directly with a HANNA story, the six dimension descriptions,
and a set of few-shot examples drawn from the frozen few_shot split. Qwen
is asked to return a JSON object of predicted scores for all six HANNA
dimensions. Predictions are saved to results/arm_a_baseline_predictions.parquet
for comparison against the other arms in 05_evaluate_arms.py.

Usage:
    python 01_baseline_direct_qwen.py --n-few-shot 3
    python 01_baseline_direct_qwen.py --limit 20   # quick smoke test
"""

import argparse
import logging

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
import config

logger = logging.getLogger("arm_a")


PROMPT_TEMPLATE = """You are an expert literary evaluator. Score the following \
story on six dimensions, each on a scale from {min_score} to {max_score}.

Dimensions:
{dimension_block}

Here are some worked examples of stories and their gold scores:

{few_shot_block}

---

Now score this new story. Respond with ONLY a JSON object mapping each \
dimension name to a numeric score, and nothing else.

Story:
{story}

JSON scores:"""


def build_prompt(story_text, dimension_block, few_shot_block):
    return PROMPT_TEMPLATE.format(
        min_score=config.HANNA_SCORE_MIN,
        max_score=config.HANNA_SCORE_MAX,
        dimension_block=dimension_block,
        few_shot_block=few_shot_block,
        story=story_text,
    )


def load_generation_model():
    logger.info(f"Loading generation model: {config.GENERATION_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.GENERATION_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return tokenizer, model


def generate_scores_for_story(story_text, dimension_block, few_shot_block, tokenizer, model):
    prompt = build_prompt(story_text, dimension_block, few_shot_block)
    messages = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,##400
            do_sample=config.BASELINE_TEMPERATURE > 0,
            temperature=max(config.BASELINE_TEMPERATURE, 1e-5),
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    try:
        scores = common.parse_json_scores(raw_text, config.HANNA_DIMENSIONS)
        scores = common.clip_scores(scores)
        return scores, raw_text, None
    except Exception as exc:
        logger.warning(f"Failed to parse model output, will record as NaN: {exc}")
        return {dim: float("nan") for dim in config.HANNA_DIMENSIONS}, raw_text, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Arm A: direct Qwen baseline")
    parser.add_argument("--n-few-shot", type=int, default=3, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test stories, for smoke testing")
    parser.add_argument("--output", type=str, default="arm_a_baseline_predictions.parquet")
    args = parser.parse_args()

    hanna_df = common.load_hanna_data()
    split_df = common.load_hanna_split()
    test_df = common.get_test_split(hanna_df, split_df)
    few_shot_df = common.get_few_shot_examples(hanna_df, split_df, n_examples=args.n_few_shot)

    if args.limit:
        test_df = test_df.head(args.limit)

    dimension_block = common.build_dimension_description_block()
    few_shot_block = common.build_few_shot_block(few_shot_df)

    tokenizer, model = load_generation_model()

    rows = []
    for i, row in test_df.iterrows():
        logger.info(f"Scoring story {i + 1}/{len(test_df)}: story_id={row['story_id']}")
        scores, raw_text, error = generate_scores_for_story(
            row["story"], dimension_block, few_shot_block, tokenizer, model
        )
        record = {"story_id": row["story_id"], "raw_output": raw_text, "parse_error": error}
        record.update(scores)
        rows.append(record)

    predictions_df = pd.DataFrame(rows)
    common.save_predictions(predictions_df, args.output)

    n_failed = predictions_df["parse_error"].notna().sum()
    logger.info(
        f"Done. {len(predictions_df)} stories scored, {n_failed} parse failures."
    )


if __name__ == "__main__":
    main()