import os
from pathlib import Path

########################### Paths ########################### 
###
PROJECT_ROOT = os.environ.get("PROJECT_ROOT","/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation",)
DATA_DIR = Path("/scratch/kk01697/data/raw/hanna") #os.path.join(PROJECT_ROOT, "data") 
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
#MODELS_DIR = os.path.join(PROJECT_ROOT, "scripts/models")
MODELS_DIR = Path("/user/HS402/kk01697/Documents/dissertation/story-evaluation-dissertation/scripts/models")

# HANNA dataset. Expected columns: story_id, story, and one column per HANNA dimension 
# containing the gold human-annotated score.
HANNA_PATH = os.path.join(DATA_DIR, "hanna_stories_annotations.csv")

# Held-out split file listing story_ids reserved for evaluation, for few-shot example selection
# and for training the MLP's mapping between custom dimensions and HANNA dimensions. 
# Expected columns: story_id, split
# where split is one of {"few_shot", "mlp_train", "test"}. These three
# subsets must be disjoint. mlp_train stories must never appear in test,
# or MLP training labels will leak into arm evaluation.
HANNA_SPLIT_PATH = os.path.join(DATA_DIR, "hanna_split.csv")

# Trained classifier and MLP checkpoints from the main proposed dissertation pipeline.
CLASSIFIER_BASE_MODEL = os.environ.get("CLASSIFIER_BASE_MODEL", "Qwen/Qwen3-1.7B")
CLASSIFIER_LORA_PATH = os.path.join(MODELS_DIR, "classifier_lora_adapter")
CLASSIFIER_HEAD_PATH = os.path.join(MODELS_DIR, "classifier_head.pt")
MLP_WEIGHTS_PATH = os.path.join(MODELS_DIR, "mlp_regressor.pt")

# Generation model used for both the baseline prompting arm and the Goodreads-style review generation step.
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "Qwen/Qwen3-1.7B")

###########################  HANNA dimensions ########################### 

HANNA_DIMENSIONS = ["Coherence","Empathy","Surprise","Engagement","Complexity","Relevance",]
HANNA_DIMENSION_DESCRIPTIONS = {
    "Coherence": (
        "How much the story reads as a logically consistent whole, with "
        "events and details that fit together rather than contradicting "
        "one another."
    ),
    "Empathy": (
        "How much the story allows the reader to understand and feel for "
        "the emotions and motivations of its characters."
    ),
    "Surprise": (
        "How much the story contains unexpected turns, twists, or "
        "developments that diverge from a predictable path."
    ),
    "Engagement": (
        "How much the story holds the reader's interest and makes them "
        "want to keep reading."
    ),
    "Complexity": (
        "How much the story develops an intricate plot or rich, "
        "multi-layered characters rather than staying simple."
    ),
    "Relevance": (
        "How well the story stays on topic and follows the given writing "
        "prompt or premise without drifting away from it."
    ),
}

# HANNA scores are on a 1 to 5 scale in the original benchmark.
HANNA_SCORE_MIN = 1
HANNA_SCORE_MAX = 5

###########################  Custom Goodreads-derived dimensions used by the trained classifier ########################### 

CUSTOM_DIMENSIONS = ["Narrative_Structure_Quality", "Character_Emotion","Originality","Immersion","Thematic_Depth","Writing_Style",]

# The MLP is now trained to learn its own weighted combination of all six custom
# dimensions for each HANNA dimension, rather than assuming a 1-to-1
# correspondence. See train_mlp.py for how the MLP is fit, and HANNA_MLP_TRAIN_PATH 
# below for the labeled data it trains on.

###########################  Generation settings ########################### 

# Number of Goodreads-style reviews to generate per story.
# Generating more than one lets you report variance rather than a single
# point estimate, as flagged in the design critique.
N_REVIEWS_PER_STORY = int(os.environ.get("N_REVIEWS_PER_STORY", "3"))

GENERATION_TEMPERATURE = float(os.environ.get("GENERATION_TEMPERATURE", "0.8"))
GENERATION_TOP_P = float(os.environ.get("GENERATION_TOP_P", "0.9"))
GENERATION_MAX_NEW_TOKENS = int(os.environ.get("GENERATION_MAX_NEW_TOKENS", "300"))

# Baseline prompting uses a lower temperature since we want a stable, repeatable score rather than diverse output.
BASELINE_TEMPERATURE = float(os.environ.get("BASELINE_TEMPERATURE", "0.2"))

RANDOM_SEED = 42

 ########################### MLP training settings (learned custom-dimension -> HANNA mapping) ########################### 

# Fraction of the mlp_train split held out internally for early stopping.
MLP_INTERNAL_VAL_FRACTION = float(os.environ.get("MLP_INTERNAL_VAL_FRACTION", "0.15"))

MLP_LEARNING_RATE = float(os.environ.get("MLP_LEARNING_RATE", "1e-3"))
MLP_WEIGHT_DECAY = float(os.environ.get("MLP_WEIGHT_DECAY", "1e-4"))
MLP_BATCH_SIZE = int(os.environ.get("MLP_BATCH_SIZE", "16"))
MLP_MAX_EPOCHS = int(os.environ.get("MLP_MAX_EPOCHS", "300"))
MLP_EARLY_STOPPING_PATIENCE = int(os.environ.get("MLP_EARLY_STOPPING_PATIENCE", "20"))

# Where the classifier's proportion vectors over mlp_train stories are cached
# so train_mlp.py does not need to re-run the classifier 
MLP_TRAIN_FEATURES_CACHE = "mlp_train_classifier_proportions.parquet"