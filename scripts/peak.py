import pandas as pd

reviews_df = pd.read_parquet("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/results/generated_reviews.parquet")

print(reviews_df.dtypes)
print(reviews_df.info())
print(reviews_df.head())