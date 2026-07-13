import json
import random
import pandas as pd


reviews = []
review_file = "/scratch/kk01697/data/raw/goodreads/goodreads_reviews_fantasy_paranormal.json"

with open(review_file, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["rating"] >= 4:
            reviews.append(r)

sample = random.sample(reviews, 1000)

with open("/scratch/kk01697/data/processed/sample_good_reviews.jsonl", "w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r) + "\n")
sample_file = "/scratch/kk01697/data/processed/sample_good_reviews.jsonl"
output_file = "/scratch/kk01697/data/processed/reviews_only.jsonl"

with open(sample_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        review = json.loads(line)

        out = {
            "review_id": review["review_id"],
            "review_text": review["review_text"]
        }

        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

# Read the JSONL file
df = pd.read_json(output_file, lines=True)
df.to_csv("/scratch/kk01697/data/processed/output.csv", index=False)