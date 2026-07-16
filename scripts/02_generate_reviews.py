"""
Review generation step, feeding into Arm B.

For each HANNA test story, generates N Goodreads-style reviews using Qwen.
Generating multiple reviews per story (config.N_REVIEWS_PER_STORY) lets you
report variance downstream rather than relying on a single sample, since
review generation is stochastic.

Output is saved to results/generated_reviews.parquet with columns:
story_id, sample_idx, review_text

Usage:
    python 02_generate_reviews.py
    python 02_generate_reviews.py --n-samples 5 --limit 10
"""

import argparse
import logging

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import common
import config

logger = logging.getLogger("generate_reviews")


REVIEW_PROMPT_TEMPLATE = """You are a reader writing a review for Goodreads. \
Write a review of the following short story in the informal, first-person \
style typical of a Goodreads review: personal reactions, some opinions \
about the plot and characters, and an overall impression. Do not write a \
formal literary critique. Keep it to one paragraph.

Story:
{story}

Review:"""


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


def generate_review(story_text, tokenizer, model):
    prompt = REVIEW_PROMPT_TEMPLATE.format(story=story_text)
    messages = [{"role": "user", "content": prompt}]
    #input_ids = tokenizer.apply_chat_template(
    #    messages, add_generation_prompt=True, return_tensors="pt"
    #).to(model.device)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,##input_ids,
            max_new_tokens=config.GENERATION_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=config.GENERATION_TEMPERATURE,
            top_p=config.GENERATION_TOP_P,
            pad_token_id=tokenizer.eos_token_id,
        )

    #generated_tokens = output[0][inputs.shape[1]:]
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    review_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return review_text



def main():
    parser = argparse.ArgumentParser(description="Generate Goodreads-style reviews for HANNA stories")
    parser.add_argument("--n-samples", type=int, default=config.N_REVIEWS_PER_STORY)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test stories, for smoke testing")
    parser.add_argument("--output", type=str, default="generated_reviews.parquet")
    args = parser.parse_args()

    hanna_df = common.load_hanna_data()
    split_df = common.load_hanna_split()
    test_df = common.get_test_split(hanna_df, split_df)

    if args.limit:
        test_df = test_df.head(args.limit)

    tokenizer, model = load_generation_model()

    rows = []
    total = len(test_df) * args.n_samples
    done = 0
    for _, row in test_df.iterrows():
        for sample_idx in range(args.n_samples):
            done += 1
            logger.info(
                f"Generating review {done}/{total} "
                f"(story_id={row['story_id']}, sample={sample_idx})"
            )
            review_text = generate_review(row["story"], tokenizer, model)
            rows.append(
                {
                    "story_id": row["story_id"],
                    "sample_idx": sample_idx,
                    "review_text": review_text,
                }
            )

    reviews_df = pd.DataFrame(rows)
    common.save_predictions(reviews_df, args.output)

    empty_count = (reviews_df["review_text"].str.len() < 20).sum()
    if empty_count > 0:
        logger.warning(
            f"{empty_count} generated reviews are suspiciously short "
            "(under 20 characters). Spot-check these before proceeding "
            "to the classifier step."
        )

    logger.info(f"Done. Generated {len(reviews_df)} reviews.")


if __name__ == "__main__":
    main()