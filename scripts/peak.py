import pandas as pd

reviews_df = pd.read_parquet("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/results/generated_reviews.parquet")

#print(reviews_df.dtypes)
#print(reviews_df.info())
#print(reviews_df.head())

reviews_df.to_csv("/scratch/kk01697/outputs/generated_reviews.csv", index=False)

base_scores_df = pd.read_parquet("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/results/arm_a_baseline_predictions.parquet")

#print(base_scores_df.dtypes)
#print(base_scores_df.info())
print(base_scores_df.head())
print(base_scores_df.columns)
unique_stories = base_scores_df.drop_duplicates(subset="story_id")

print("Base OG", len(base_scores_df))
#print("Base New", len(unique_stories))
print(unique_stories.head())
print(unique_stories[['story_id','Coherence', 'Empathy','Surprise', 'Engagement', 'Complexity', 'Relevance']].head(20))