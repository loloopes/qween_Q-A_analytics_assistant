"""In-memory semantic cache backed by embedding similarity and TTL."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_SIMILARITY_THRESHOLD = 0.88
DEFAULT_MAX_ENTRIES = 2000
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_last_error: str | None = None


@dataclass
class _CacheEntry:
    entry_id: str
    namespace: str
    query_text: str
    value: Any
    embedding: np.ndarray
    created_at: float
    expires_at: float


class _TransformerEmbedder:
    """Mean-pooled embeddings via transformers (no sentence-transformers package)."""

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


class SemanticCache:
    def __init__(
        self,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.embedding_model_name = embedding_model
        self._entries: list[_CacheEntry] = []
        self._exact: dict[tuple[str, str], tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._embedder: _TransformerEmbedder | None = None
        self._embedder_ready = False
        self._embedder_failed = False
        self._hits = 0
        self._exact_hits = 0
        self._misses = 0

    def warm(self) -> None:
        """Load the embedding model at startup; exact-match cache works even if this fails."""
        if self._embedder_ready or self._embedder_failed:
            return
        try:
            self._get_embedder()
        except Exception as exc:
            global _last_error
            self._embedder_failed = True
            _last_error = f"embedder init failed: {exc}"
            logger.warning("Semantic cache embedder unavailable; exact-match only: %s", exc)

    def _get_embedder(self) -> _TransformerEmbedder:
        global _last_error
        if self._embedder is None:
            self._embedder = _TransformerEmbedder(self.embedding_model_name)
            self._embedder_ready = True
            self._embedder_failed = False
            _last_error = None
            logger.info("Semantic cache embedder loaded: %s", self.embedding_model_name)
        return self._embedder

    def _try_embed(self, text: str) -> np.ndarray | None:
        if self._embedder_failed:
            return None
        try:
            return self._get_embedder().embed(text)
        except Exception as exc:
            global _last_error
            self._embedder_failed = True
            self._embedder = None
            self._embedder_ready = False
            _last_error = f"embed failed: {exc}"
            logger.warning("Semantic cache embed failed: %s", exc)
            return None

    @staticmethod
    def _normalize_query(text: str) -> str:
        return " ".join(text.lower().split())

    def _purge_expired(self, now: float | None = None) -> None:
        now = now or time.time()
        self._entries = [entry for entry in self._entries if entry.expires_at > now]
        self._exact = {
            key: value
            for key, value in self._exact.items()
            if value[1] > now
        }

    def _evict_if_needed(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        self._entries.sort(key=lambda entry: entry.created_at)
        overflow = len(self._entries) - self.max_entries
        del self._entries[:overflow]

    def lookup(self, namespace: str, query: str) -> tuple[bool, Any | None, str]:
        """Return (hit, value, match_type) where match_type is exact|semantic|miss."""
        with self._lock:
            global _last_error
            self._purge_expired()
            if not query.strip():
                self._misses += 1
                return False, None, "miss"

            exact_key = (namespace, self._normalize_query(query))
            if exact_key in self._exact:
                value, _expires = self._exact[exact_key]
                self._hits += 1
                self._exact_hits += 1
                return True, value, "exact"

            try:
                query_vec = self._try_embed(query)
            except Exception as exc:
                _last_error = f"embed failed on lookup: {exc}"
                logger.warning("Semantic cache lookup embed failed: %s", exc)
                self._misses += 1
                return False, None, "miss"

            if query_vec is None:
                self._misses += 1
                return False, None, "miss"

            best_score = -1.0
            best_entry: _CacheEntry | None = None
            for entry in self._entries:
                if entry.namespace != namespace:
                    continue
                score = float(np.dot(query_vec, entry.embedding))
                if score > best_score:
                    best_score = score
                    best_entry = entry

            if best_entry is not None and best_score >= self.similarity_threshold:
                self._hits += 1
                _last_error = None
                return True, best_entry.value, "semantic"

            self._misses += 1
            return False, None, "miss"

    def store(self, namespace: str, query: str, value: Any) -> bool:
        with self._lock:
            global _last_error
            self._purge_expired()
            now = time.time()
            expires_at = now + self.ttl_seconds
            normalized = self._normalize_query(query)
            self._exact[(namespace, normalized)] = (value, expires_at)

            try:
                embedding = self._try_embed(query)
            except Exception as exc:
                _last_error = f"embed failed on store: {exc}"
                logger.warning("Semantic cache store embed failed: %s", exc)
                return True

            if embedding is None:
                return True

            self._entries.append(
                _CacheEntry(
                    entry_id=str(uuid.uuid4()),
                    namespace=namespace,
                    query_text=query,
                    value=value,
                    embedding=embedding,
                    created_at=now,
                    expires_at=expires_at,
                )
            )
            self._evict_if_needed()
            _last_error = None
            return True

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._purge_expired()
            by_namespace: dict[str, int] = {}
            for entry in self._entries:
                by_namespace[entry.namespace] = by_namespace.get(entry.namespace, 0) + 1
            total = self._hits + self._misses
            return {
                "enabled": True,
                "embedder_ready": self._embedder_ready,
                "entries": len(self._entries),
                "exact_entries": len(self._exact),
                "namespaces": by_namespace,
                "hits": self._hits,
                "exact_hits": self._exact_hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "ttl_hours": self.ttl_seconds / 3600,
                "similarity_threshold": self.similarity_threshold,
                "embedding_model": self.embedding_model_name,
                "last_error": _last_error,
            }


_cache: SemanticCache | None = None


def semantic_cache_enabled() -> bool:
    return os.getenv("SEMANTIC_CACHE_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def get_semantic_cache() -> SemanticCache | None:
    global _cache
    if not semantic_cache_enabled():
        return None
    if _cache is None:
        ttl_seconds = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS)))
        threshold = float(
            os.getenv("SEMANTIC_CACHE_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
        )
        max_entries = int(os.getenv("SEMANTIC_CACHE_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES)))
        embedding_model = os.getenv("SEMANTIC_CACHE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        _cache = SemanticCache(
            ttl_seconds=ttl_seconds,
            similarity_threshold=threshold,
            max_entries=max_entries,
            embedding_model=embedding_model,
        )
    return _cache


def warm_semantic_cache() -> None:
    cache = get_semantic_cache()
    if cache is not None:
        cache.warm()


def cache_lookup(namespace: str, query: str) -> tuple[bool, Any | None]:
    cache = get_semantic_cache()
    if cache is None:
        return False, None
    hit, value, _match_type = cache.lookup(namespace, query)
    return hit, value


def cache_store(namespace: str, query: str, value: Any) -> None:
    cache = get_semantic_cache()
    if cache is not None:
        cache.store(namespace, query, value)


def cache_stats() -> dict[str, Any]:
    cache = get_semantic_cache()
    if cache is None:
        return {"enabled": False}
    return cache.stats()
