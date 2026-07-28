from transformers import AutoModelForSequenceClassification

import time


print("Starting...")
t0 = time.time()

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-1.7B",
    num_labels=6,
)

print(f"Loaded in {time.time() - t0:.1f} seconds")