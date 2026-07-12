"""
02_score_reviews.py

Loads the raw Goodreads fantasy/paranormal reviews + books data, filters down
to a usable English-language subset, runs the fine-tuned Qwen3 LoRA multi-label
classifier (trained in 01_train_qwen.py) over every sentence, and aggregates
sentence-level predictions into per-review proportion vectors across the six
custom dimensions.

Pipeline:
    1. Load + merge raw books/reviews JSON.
    2. Filter (language, length, rating, dedupe) and sample down to a
       manageable working set.
    3. Sentence-tokenise each review.
    4. Load the base Qwen3 model + LoRA adapter, run batched inference.
    5. Aggregate sentence predictions to review-level mean proportions.
    6. Save sentence-level and review-level outputs to CSV.

Cleaned up from the original notebook: dead/duplicate code removed, all
logic organised into functions, single entry point via `main()`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#################################################  Configuration ################################################# 

REVIEWS_FILE = "/scratch/kk01697/data/raw/goodreads/goodreads_reviews_fantasy_paranormal.json"
BOOKS_FILE = "/scratch/kk01697/data/raw/goodreads/goodreads_books_fantasy_paranormal.json"

MODEL_DIR = ("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/""scripts/objective1/outputs/qwen3_lora_classifier")
BASE_MODEL = "Qwen/Qwen3-1.7B"

FILTERED_REVIEWS_CSV = ("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/""data/processed/filtered_reviews.csv")
SENTENCE_PREDICTIONS_CSV = ("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/""scripts/data/sentence_predictions.csv")
REVIEW_SCORES_CSV = ("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/""scripts/data/review_scores.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 50_000
SEED = 42

CUSTOM_DIM = ["Narrative Structure & Quality","Character & Emotion","Originality","Immersion","Thematic Depth","Writing Style",]

#################################################  Data loading + filtering ################################################# 

def load_json_lines(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r") as f:
        for line in tqdm(f, desc=f"Loading {Path(path).name}"):
            records.append(json.loads(line))
    return pd.DataFrame(records)


def basic_filter(
    df: pd.DataFrame, min_words: int = 30, min_rating: int = 1, max_rating: int = 5
) -> pd.DataFrame:
    """Remove empty / very short reviews, non-English reviews, and invalid ratings."""
    english_codes = ["eng", "en-UK", "en-US", "en-AUS"]

    df = df[df["language_code"].isin(english_codes)]
    df = df.dropna(subset=["review_text", "rating"])
    df = df[df["review_text"].str.split().str.len() >= min_words]
    df = df[df["rating"].between(min_rating, max_rating)]
    df = df.drop_duplicates(subset=["review_text"])
    df = df.reset_index(drop=True)
    return df


def build_filtered_sample(books_file: str, reviews_file: str) -> pd.DataFrame:
    book_df = load_json_lines(books_file)
    print("Books shape:", book_df.shape)

    reviews_df = load_json_lines(reviews_file)
    print("Reviews shape:", reviews_df.shape)

    merged = reviews_df.merge(book_df, on="book_id", how="inner")
    print("Unique reviews after merge:", merged["review_id"].nunique())

    df_filtered = basic_filter(merged)
    print(f"Loaded {len(merged):,} rows -> {len(df_filtered):,} after filtering")

    # Working with the full ~1.9M filtered reviews is impractical here, so we
    # sample down to a fixed working subset.
    sample_ids = df_filtered["review_id"].drop_duplicates().sample(
        n=SAMPLE_SIZE, random_state=SEED
    )
    filter_df = df_filtered[df_filtered["review_id"].isin(sample_ids)]
    print("Sampled shape:", filter_df.shape)

    Path(FILTERED_REVIEWS_CSV).parent.mkdir(parents=True, exist_ok=True)
    filter_df.to_csv(FILTERED_REVIEWS_CSV)

    filter_df = filter_df[["review_id", "review_text"]].dropna()
    print("Rows with text:", len(filter_df))
    return filter_df


#################################################  Sentence tokenisation ################################################# 

def tokenise_reviews(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Tokenising reviews"):
        sents = sent_tokenize(r.review_text)
        for i, s in enumerate(sents):
            rows.append({"review_id": r.review_id, "sentence_idx": i, "sentence": s})
    return pd.DataFrame(rows)

#################################################  Model loading + inference ################################################# 

def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CUSTOM_DIM),
        torch_dtype=torch.bfloat16,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_sentences(
    sentences: list[str],
    tokenizer,
    model,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> np.ndarray:
    all_preds = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Predicting", unit="batch"):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            logits = model(**enc).logits.float()

        probs = torch.sigmoid(logits).float().cpu().numpy()
        preds = (probs >= threshold).astype(int)
        all_preds.append(preds)

    return np.vstack(all_preds)


#################################################  Main ################################################# 

def main():
    filter_df = build_filtered_sample(BOOKS_FILE, REVIEWS_FILE)

    sentence_df = tokenise_reviews(filter_df)

    tokenizer, model = load_classifier()
    print("num_labels:", model.config.num_labels)

    preds = predict_sentences(sentence_df["sentence"].tolist(), tokenizer, model)
    pred_df = pd.DataFrame(preds, columns=CUSTOM_DIM)
    sentence_predictions = pd.concat([sentence_df, pred_df], axis=1)

    # Review-level proportion vectors: fraction of sentences per review that
    # were flagged positive for each dimension.
    review_scores = (
        sentence_predictions.groupby("review_id")[CUSTOM_DIM].mean().reset_index()
    )

    Path(SENTENCE_PREDICTIONS_CSV).parent.mkdir(parents=True, exist_ok=True)
    sentence_predictions.to_csv(SENTENCE_PREDICTIONS_CSV, index=False)
    review_scores.to_csv(REVIEW_SCORES_CSV, index=False)

    print("\nReview-level proportions (head):")
    print(review_scores.head())
    print("\nSentence-level predictions (head):")
    print(sentence_predictions.head())


if __name__ == "__main__":
    main()