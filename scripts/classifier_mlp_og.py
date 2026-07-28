"""
Shared classifier + MLP inference logic, used by both:
  - (03_arm_b_review_classifier_mlp.py): input is generated reviews
  - (04_arm_c_story_classifier_mlp.py): input is raw HANNA story text

Keeping this logic in one module guarantees that Arm B and Arm C differ
only in what text is fed in, not in how the classifier or MLP is applied.
This is the isolation that makes the B vs C comparison a fair test of
the review-representation hypothesis.
"""

import logging
import os

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

import config

logger = logging.getLogger("classifier_mlp")


class MLPRegressor(nn.Module):
    """6 -> 32 -> 32 -> 6 regression head mapping custom dimension
    proportions to predicted HANNA scores."""

    def __init__(self, input_dim=6, hidden_dim=32, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ClassifierMLPPipeline:
    """Loads the trained Qwen3-1.7B + LoRA classifier and exposes prediction methods used by Arm B, Arm C,
    and train_mlp.py.

    Set load_mlp=False when using this class purely to extract classifier
    proportion vectors for MLP training data (train_mlp.py).
    """

    def __init__(self, device=None, load_mlp=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_classifier()
        self.mlp = None
        if load_mlp:
            self._load_mlp()

    def _load_classifier(self):
        logger.info(f"Loading classifier base model: {config.CLASSIFIER_BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_BASE_MODEL)

        #base_model = AutoModel.from_pretrained(
        #    config.CLASSIFIER_BASE_MODEL,
        #    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        #)


        print("CUDA available:", torch.cuda.is_available())
        print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

        logger.info("About to load base model...")

        base_model = AutoModel.from_pretrained(
            config.CLASSIFIER_BASE_MODEL,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        logger.info("Base model successfully loaded.")

        logger.info(f"Loading LoRA adapter from {config.CLASSIFIER_LORA_PATH}")
        print(config.CLASSIFIER_LORA_PATH) 
        print(os.path.exists(config.CLASSIFIER_LORA_PATH))
        self.encoder = PeftModel.from_pretrained(base_model, config.CLASSIFIER_LORA_PATH)
        self.encoder.to(self.device)
        self.encoder.eval()

        hidden_size = self.encoder.config.hidden_size
        n_dims = len(config.CUSTOM_DIMENSIONS)
        self.classification_head = nn.Linear(hidden_size, n_dims)

        logger.info(f"Loading classifier head weights from {config.CLASSIFIER_HEAD_PATH}")
        state_dict = torch.load(config.CLASSIFIER_HEAD_PATH, map_location=self.device)
        self.classification_head.load_state_dict(state_dict)
        self.classification_head.to(self.device)
        self.classification_head.eval()

        print(config.CLASSIFIER_HEAD_PATH)
        print(os.path.exists(config.CLASSIFIER_HEAD_PATH))

        print(config.MLP_WEIGHTS_PATH)
        print(os.path.exists(config.MLP_WEIGHTS_PATH))

    def _load_mlp(self):
        n_dims = len(config.CUSTOM_DIMENSIONS)
        self.mlp = MLPRegressor(input_dim=n_dims, hidden_dim=32, output_dim=n_dims)
        logger.info(f"Loading MLP weights from {config.MLP_WEIGHTS_PATH}")
        state_dict = torch.load(config.MLP_WEIGHTS_PATH, map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()

    def _pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = torch.sum(hidden_states * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def predict_proportions(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state
            pooled = self._pool(last_hidden, inputs["attention_mask"])
            logits = self.classification_head(pooled)
            proportions = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        return {dim: float(p) for dim, p in zip(config.CUSTOM_DIMENSIONS, proportions)}

    def predict_hanna_scores(self, text):
        """Full pipeline: text -> custom dimension proportions -> MLP ->
        predicted HANNA scores, scaled to config.HANNA_SCORE_MIN/MAX.

        The MLP output is no longer interpreted via a fixed positional
        mapping between custom and HANNA dimensions. Instead the MLP was
        trained end-to-end in train_mlp.py to output a 6-vector in the
        exact order of config.HANNA_DIMENSIONS, learning its own weighted
        combination of all six input proportions for each HANNA dimension.
        """
        if self.mlp is None:
            raise RuntimeError(
                "MLP was not loaded (load_mlp=False). Instantiate "
                "ClassifierMLPPipeline with load_mlp=True, or use "
                "predict_proportions() directly if you only need "
                "classifier output."
            )

        proportions = self.predict_proportions(text)
        ordered = torch.tensor(
            [[proportions[dim] for dim in config.CUSTOM_DIMENSIONS]],
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            mlp_output = self.mlp(ordered).squeeze(0).cpu().numpy()

        # MLP output is assumed to already be in HANNA score units, since
        # train_mlp.py trains directly against gold HANNA scores on that
        # scale. If you change the training target scale, rescale here to
        # match config.HANNA_SCORE_MIN / config.HANNA_SCORE_MAX and note
        # the rescaling in your methodology section.
        return {
            dim: float(score)
            for dim, score in zip(config.HANNA_DIMENSIONS, mlp_output)
        }