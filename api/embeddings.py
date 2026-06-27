"""Shared text embedder for RAG and semantic cache."""

from __future__ import annotations

import os

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

_embedder = None


def embedding_model_name() -> str:
    return (
        os.getenv("RAG_EMBEDDING_MODEL")
        or os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )


def embedding_dimension() -> int:
    return int(
        os.getenv("RAG_EMBEDDING_DIMENSION")
        or os.getenv("SEMANTIC_CACHE_EMBEDDING_DIMENSION")
        or DEFAULT_EMBEDDING_DIMENSION
    )


class TransformerEmbedder:
    """Mean-pooled embeddings via transformers."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        attention = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask = attention.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = self._torch.sum(token_embeddings * mask, dim=1)
        counts = self._torch.clamp(mask.sum(dim=1), min=1e-9)
        vector = (summed / counts).squeeze(0).numpy().astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector


def get_embedder() -> TransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TransformerEmbedder(embedding_model_name())
    return _embedder


def embed_text(text: str) -> np.ndarray:
    return get_embedder().embed(text)


def warm_embedder() -> None:
    get_embedder()
