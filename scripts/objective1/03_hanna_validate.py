"""
03_hanna_validate.py

Takes the review-level proportion vectors produced by 02_score_reviews.py and:
    1. Maps them onto the six HANNA dimensions (Coherence, Empathy, Surprise,
       Engagement, Complexity, Relevance) via a fixed custom->HANNA mapping.
    2. Scales the raw HANNA scores onto a 1-5 range using two candidate
       methods: corpus-relative min-max, and percentile/quintile-based.
    3. Runs a held-out sample of reviews through an LLM few-shot scorer as an
       independent HANNA scoring signal.
    4. Computes Spearman correlations between each scaling method and the LLM
       scores, to empirically choose the better-performing scaling approach
       for the dissertation methodology.

Cleaned up from the original notebook: dead/duplicate code removed, all
logic organised into functions, single entry point via `main()`.

Note on scaling methods: two percentile-scaling variants were explored.
`percentile_scale_basic` splits the full distribution (including zeros) into
five quintiles. `percentile_scale_zero_aware` treats zero as its own bucket
and quintiles only the positive values, which behaves better for sparse
dimensions like Surprise/Originality. Both are computed and plotted below;
the final correlation-vs-LLM comparison matches the original methodology and
uses the min-max result vs. the basic percentile result (`hanna_percentile`),
with the zero-aware variant (`hanna_percentile_zero_aware`) kept for
inspection/plotting.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from scipy.stats import spearmanr
from tqdm.auto import tqdm

################################################# Configuration ################################################# 

REVIEW_SCORES_CSV = ("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/""scripts/data/review_scores.csv")
HANNA_SCORES_CSV = "/scratch/kk01697/data/raw/hanna/hanna_stories_annotations.csv"
FILTERED_REVIEWS_CSV = ("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/""data/processed/filtered_reviews.csv")
OUT_DIR = Path("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/outputs")

HANNA_DIMS = ["Coherence", "Empathy", "Surprise", "Engagement", "Complexity", "Relevance"]

HELD_OUT_SAMPLE_SIZE = 500
SEED = 42

LLM_MODEL = "qwen/qwen3-17b"  

#################################################  Step 1: raw HANNA scores from custom-dimension proportions ################################################# 

def compute_hanna_raw(props: pd.DataFrame) -> pd.DataFrame:
    """Map the six custom classifier dimensions onto the six HANNA dimensions."""
    raw = {
        "Coherence": props["Narrative Structure & Quality"],
        "Empathy": props["Character & Emotion"],
        "Surprise": props["Originality"],
        "Engagement": (props["Immersion"] + props["Character & Emotion"]) / 2,
        "Complexity": (props["Thematic Depth"] + props["Writing Style"]) / 2,
        "Relevance": (props["Narrative Structure & Quality"] + props["Thematic Depth"]) / 2,
    }
    return pd.DataFrame(raw)

#################################################  Step 2: scaling methods ################################################# 

def minmax_scale_to_hanna(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Scale each HANNA dimension to 1-5 using min/max observed in this corpus."""
    scaled = df.copy()
    for dim in dims:
        vals = df[dim].values
        dmin, dmax = vals.min(), vals.max()
        scaled[dim] = 1 + (vals - dmin) / (dmax - dmin + 1e-8) * 4
    return scaled


def percentile_scale_basic(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """Scale each HANNA dimension to 1-5 using quintiles of the corpus distribution."""
    scaled = df.copy()
    for dim in dims:
        ranks = df[dim].rank(method="first")
        scaled[dim] = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return scaled


def percentile_scale_zero_aware(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    """
    Percentile scaling that preserves zero-heavy dimensions.
    Zeros remain 1; positive values are split into four percentile groups.
    More robust than the basic version when a dimension (e.g. Surprise /
    Originality) is sparse and dominated by zeros.
    """
    scaled = df.copy()
    for dim in dims:
        x = df[dim]
        scores = pd.Series(1, index=x.index)
        positive = x > 0
        if positive.sum() >= 4:
            scores.loc[positive] = pd.qcut(
                x.loc[positive], q=4, labels=[2, 3, 4, 5], duplicates="drop"
            ).astype(int)
        scaled[dim] = scores
    return scaled


def plot_scaling_comparison(hanna_minmax, hanna_percentile, hanna_percentile_zero_aware, dims):
    fig, axes = plt.subplots(3, len(dims), figsize=(20, 6), sharey="row")
    for i, dim in enumerate(dims):
        axes[0, i].hist(hanna_minmax[dim], bins=5, range=(1, 5))
        axes[0, i].set_title(f"{dim}\n(min-max)")
        axes[1, i].hist(hanna_percentile[dim], bins=5, range=(1, 5))
        axes[1, i].set_title(f"{dim}\n(percentile)")
        axes[2, i].hist(hanna_percentile_zero_aware[dim], bins=5, range=(1, 5))
        axes[2, i].set_title(f"{dim}\n(zero-aware percentile)")
    plt.tight_layout()
    plt.show()

#################################################  Step 3: LLM few-shot scoring ################################################# 

SYSTEM_PROMPT = """You are an expert literary critic evaluating book reviews according to the HANNA \
story evaluation framework. For a given review, rate it on each of the following six dimensions, \
using a scale of 1 (very poor) to 5 (excellent):

- Coherence: Is the review logically structured and easy to follow?
- Empathy: Does the review engage with characters' emotions and inner lives?
- Surprise: Does the review highlight originality, twists, or unexpected elements?
- Engagement: Does the review convey how immersive or gripping the story is?
- Complexity: Does the review discuss thematic depth or stylistic sophistication?
- Relevance: Does the review stay focused and relevant to the story's core content?

Respond with ONLY a JSON object, no preamble, no markdown fences, in this exact format:
{"Coherence": <1-5>, "Empathy": <1-5>, "Surprise": <1-5>, "Engagement": <1-5>, "Complexity": <1-5>, "Relevance": <1-5>}
"""

FEW_SHOT_EXAMPLES = [
    {
        "review": (
            "This book completely swept me away. The characters felt so real I cried with them "
            "in the final chapters. The plot twist at the 2/3 mark genuinely shocked me — I did not "
            "see it coming at all. Beautifully written, with themes of loss and redemption woven "
            "throughout. A perfect blend of heart and craft."
        ),
        "scores": {"Coherence": 5, "Empathy": 5, "Surprise": 5, "Engagement": 5, "Complexity": 4, "Relevance": 5},
    },
    {
        "review": "It was fine. Nothing special, kind of predictable honestly. I skimmed the middle third.",
        "scores": {"Coherence": 3, "Empathy": 1, "Surprise": 1, "Engagement": 2, "Complexity": 1, "Relevance": 2},
    },
]


def build_messages(review_text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["review"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["scores"])})
    messages.append({"role": "user", "content": review_text})
    return messages


def parse_json_response(text: str) -> dict | None:
    # Strip markdown fences if the model adds them despite instructions.
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        parsed = json.loads(cleaned)
        if all(dim in parsed for dim in HANNA_DIMS):
            return {dim: float(parsed[dim]) for dim in HANNA_DIMS}
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def score_review_llm(client: OpenAI, review_text: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=build_messages(review_text[:3000]),
                temperature=0,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            parsed = parse_json_response(content)
            if parsed is not None:
                return parsed
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def build_held_out_sample(hanna_raw: pd.DataFrame, review_scores: pd.DataFrame) -> pd.DataFrame:
    filtered_df = pd.read_csv(FILTERED_REVIEWS_CSV)

    hanna_df = pd.concat(
        [review_scores[["review_id"]].reset_index(drop=True), hanna_raw.reset_index(drop=True)],
        axis=1,
    )
    held_out = hanna_df.sample(n=HELD_OUT_SAMPLE_SIZE, random_state=SEED).copy()
    held_out = held_out.merge(filtered_df[["review_id", "review_text"]], on="review_id", how="left")
    held_out = held_out[["review_id", "review_text", *HANNA_DIMS]]
    return held_out


def run_llm_scoring(held_out: pd.DataFrame) -> pd.DataFrame:
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

    llm_sample = held_out[["review_id", "review_text"]]
    llm_scores, failed_ids = [], []

    for _, row in tqdm(llm_sample.iterrows(), total=len(llm_sample), desc="LLM scoring"):
        result = score_review_llm(client, row["review_text"])
        if result is not None:
            result["review_id"] = row["review_id"]
            llm_scores.append(result)
        else:
            failed_ids.append(row["review_id"])
        time.sleep(0.1)  # light rate-limit courtesy

    llm_scores_df = pd.DataFrame(llm_scores)
    print(f"Scored {len(llm_scores_df)}/{len(llm_sample)} reviews ({len(failed_ids)} failed)")
    return llm_scores_df

#############################################  Step 4: correlation validation ############################################# 

def correlate_with_llm(classifier_df: pd.DataFrame, llm_df: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = classifier_df.merge(llm_df, on="review_id", suffixes=("_clf", "_llm"))
    rows = []
    for dim in HANNA_DIMS:
        rho, pval = spearmanr(merged[f"{dim}_clf"], merged[f"{dim}_llm"])
        rows.append({"dimension": dim, "spearman_rho": rho, "p_value": pval, "method": label})
    return pd.DataFrame(rows)

#################################################  Main ################################################# 

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    review_scores = pd.read_csv(REVIEW_SCORES_CSV)
    # hanna_scores.csv (ground-truth HANNA annotations) is loaded for reference
    # but not directly used in this validation pass.
    _hanna_scores = pd.read_csv(HANNA_SCORES_CSV)

    # --- raw HANNA scores from proportions ---
    hanna_raw = compute_hanna_raw(review_scores)

    # --- scaling comparison ---
    hanna_minmax = minmax_scale_to_hanna(hanna_raw, HANNA_DIMS)
    hanna_percentile = percentile_scale_basic(hanna_raw, HANNA_DIMS)
    hanna_percentile_zero_aware = percentile_scale_zero_aware(hanna_raw, HANNA_DIMS)

    plot_scaling_comparison(hanna_minmax, hanna_percentile, hanna_percentile_zero_aware, HANNA_DIMS)

    hanna_raw.to_parquet(OUT_DIR / "hanna_raw.parquet", index=False)
    hanna_minmax.to_parquet(OUT_DIR / "hanna_minmax.parquet", index=False)
    hanna_percentile_zero_aware.to_parquet(OUT_DIR / "hanna_percentile.parquet", index=False)
    print(hanna_raw.shape, hanna_minmax.shape, hanna_percentile_zero_aware.shape)

    # --- validate against few-shot LLM scores ---
    held_out = build_held_out_sample(hanna_raw, review_scores)
    llm_scores_df = run_llm_scoring(held_out)
    llm_scores_df.to_parquet(OUT_DIR / "llm_hanna_scores.parquet", index=False)

    corr_minmax = correlate_with_llm(hanna_minmax, llm_scores_df, "min-max")
    corr_percentile = correlate_with_llm(hanna_percentile, llm_scores_df, "percentile")

    correlation_results = pd.concat([corr_minmax, corr_percentile], ignore_index=True)
    correlation_results.to_csv(OUT_DIR / "hanna_scaling_correlation.csv", index=False)

    pivot = correlation_results.pivot(index="dimension", columns="method", values="spearman_rho")
    print("\nSpearman correlation vs. LLM scores by scaling method:")
    print(pivot)


if __name__ == "__main__":
    main()