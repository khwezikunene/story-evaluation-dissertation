import logging

import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

logger = logging.getLogger("classifier_mlp")


class MLPRegressor(nn.Module):
    """Maps the six custom dimension proportions to six HANNA scores."""

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
    """
    Loads the trained Qwen3 LoRA sequence classifier and the trained MLP.

    Pipeline:

        review text
             ↓
        LoRA classifier
             ↓
      six custom dimension probabilities
             ↓
             MLP
             ↓
       six HANNA predictions
    """

    def __init__(self, device=None, load_mlp=True):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self._load_classifier()

        self.mlp = None
        if load_mlp:
            self._load_mlp()

    def _load_classifier(self):

        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.CLASSIFIER_LORA_PATH
        )

        logger.info("Loading base sequence-classification model...")

        #base_model = AutoModelForSequenceClassification.from_pretrained(
        #    config.CLASSIFIER_BASE_MODEL,
        #    num_labels=len(config.CUSTOM_DIMENSIONS),
        #    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        #)

        base_model = AutoModelForSequenceClassification.from_pretrained(config.CLASSIFIER_LORA_PATH,torch_dtype=torch.float32,)

        logger.info(f"Loading LoRA adapter from {config.CLASSIFIER_LORA_PATH}")

        self.model = PeftModel.from_pretrained(
            base_model,
            config.CLASSIFIER_LORA_PATH,
        )

        self.model.to(self.device)
        self.model.eval()

        logger.info("Classifier loaded successfully.")

    def _load_mlp(self):

        logger.info(f"Loading MLP weights from {config.MLP_WEIGHTS_PATH}")

        self.mlp = MLPRegressor(
            input_dim=len(config.CUSTOM_DIMENSIONS),
            hidden_dim=32,
            output_dim=len(config.HANNA_DIMENSIONS),
        )

        state_dict = torch.load(
            config.MLP_WEIGHTS_PATH,
            map_location=self.device,
        )

        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device)
        self.mlp.eval()

        logger.info("MLP loaded successfully.")

    def predict_proportions(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():

            outputs = self.model(**inputs)

            logits = outputs.logits

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        return {
            dim: float(prob)
            for dim, prob in zip(config.CUSTOM_DIMENSIONS, probs)
        }

    def predict_hanna_scores(self, text):

        if self.mlp is None:
            raise RuntimeError(
                "MLP not loaded. Initialise with load_mlp=True."
            )

        proportions = self.predict_proportions(text)

        x = torch.tensor(
            [[proportions[d] for d in config.CUSTOM_DIMENSIONS]],
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad():
            scores = self.mlp(x).squeeze(0).cpu().numpy()

        return {
            dim: float(score)
            for dim, score in zip(config.HANNA_DIMENSIONS, scores)
        }