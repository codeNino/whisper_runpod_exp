import torch
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer

from helpers import logger
from config import settings
import os

EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
os.environ["HF_TOKEN"] == settings.HF_TOKEN
WHISPER_MODEL_SIZE = "distil-large-v3"


def whisper_models():
    """
    Load and return the models
    """
    if torch.cuda.is_available():
        # Run on GPU with INT8_FP16
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="int8_float16")
        return model
    else:
        # or run on CPU with INT8
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        return model


class EmbeddingModel(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(self):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if self.model == None:
            if torch.cuda.is_available():
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
            else:
                self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        return self.model

    @classmethod
    def predict(self, input):
        model = self.get_model()
        return model.encode(input, normalize_embeddings=True)
