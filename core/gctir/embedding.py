from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_BACKENDS = ("sentence_transformers",)


def l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def compute_cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("embedding vectors must have the same nonzero dimension")
    if not all(math.isfinite(x) for x in (*left, *right)):
        raise ValueError("embedding vectors must contain finite values")
    left_norm = math.sqrt(sum(x * x for x in left))
    right_norm = math.sqrt(sum(x * x for x in right))
    if not left_norm or not right_norm:
        raise ValueError("cosine similarity is undefined for zero vectors")
    value = sum(x * y for x, y in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def compute_similarity(left: list[float], right: list[float]) -> float:
    return (1.0 + compute_cosine_similarity(left, right)) / 2.0


@dataclass
class SemanticEmbedder:
    backend: str = "sentence_transformers"
    model: str = DEFAULT_EMBEDDING_MODEL
    normalize: bool = True
    _sentence_transformer: object | None = None

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported embedding backend {self.backend!r}; "
                             f"expected 'sentence_transformers'")
        if not self.model:
            raise ValueError("an embedding model is required")
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "the embedding backend requires the sentence-transformers package") from error
        self._sentence_transformer = SentenceTransformer(self.model)

    def encode(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("cannot embed empty text")
        vector = [float(value) for value in self._sentence_transformer.encode(
            text, normalize_embeddings=self.normalize)]
        return l2_normalise(vector) if self.normalize else vector
