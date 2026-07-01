"""
Embedding service using sentence-transformers.

Loads a local model (all-MiniLM-L6-v2) for generating embeddings.
Falls back to TF-IDF if sentence-transformers is not installed.

The model is loaded once (singleton) and cached in memory.
Embeddings are numpy arrays suitable for cosine similarity.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model — fast, good quality, 384-dim
_DEFAULT_MODEL = "all-MiniLM-L6-v2"

_embedder = None
_model_name: str = _DEFAULT_MODEL
_use_fallback = False


def init_embedder(model_name: str = _DEFAULT_MODEL) -> None:
    """Initialise the embedding model.

    Call once at startup, or it will be lazily loaded.
    """
    global _embedder, _model_name, _use_fallback

    _model_name = model_name
    _use_fallback = False

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformer model: %s", model_name)
        _embedder = SentenceTransformer(model_name)
        logger.info("Embedder ready (dim=%d)", _embedder.get_embedding_dimension())
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Falling back to TF-IDF embeddings."
        )
        _embedder = _TfidfFallback()
        _use_fallback = True
    except Exception as e:
        logger.error("Failed to load sentence-transformer: %s. Using TF-IDF fallback.", e)
        _embedder = _TfidfFallback()
        _use_fallback = True


def get_embedder():
    """Return the embedder, loading it lazily if needed."""
    global _embedder
    if _embedder is None:
        init_embedder()
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into a 2D numpy array.

    For TF-IDF fallback, fits the vocabulary on the first call so that
    subsequent embed_query() calls share the same vocabulary.

    Returns shape (n_texts, embedding_dim).
    """
    global _use_fallback
    embedder = get_embedder()
    if not texts:
        return np.zeros((0, embedder_dim()))
    # For TF-IDF fallback, ensure vocabulary is fitted on corpus
    if _use_fallback and hasattr(embedder, "fit") and not getattr(embedder, "_fitted", False):
        embedder.fit(texts)
    return embedder.encode(texts)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns 1D array."""
    embedder = get_embedder()
    return embedder.encode([text])[0]


def embedder_dim() -> int:
    """Return the embedding dimension."""
    embedder = get_embedder()
    if hasattr(embedder, "get_sentence_embedding_dimension"):
        return embedder.get_sentence_embedding_dimension()
    return getattr(embedder, "_dim", 384)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors a and b.

    a: shape (n, d) or (d,)
    b: shape (m, d) or (d,)

    Returns shape (n, m) or (n,) or scalar.
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)

    a_normalized = a / a_norm
    b_normalized = b / b_norm

    return np.dot(a_normalized, b_normalized.T)


# ---------------------------------------------------------------------------
# TF-IDF Fallback (no external dependency beyond numpy)
# ---------------------------------------------------------------------------

class _TfidfFallback:
    """Minimal TF-IDF embedding fallback when sentence-transformers is unavailable.

    Produces lower-quality but functional embeddings for basic semantic matching.
    Uses character n-gram TF-IDF vectors.

    IMPORTANT: Call fit(texts) first, then encode() uses the fitted vocabulary.
    This ensures query encoding uses the same vocabulary as the corpus.
    """

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._vocabulary: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary and IDF from corpus texts. Call once before encode."""
        if not texts:
            return

        tokenized = [self._tokenize(t) for t in texts]

        # Build vocabulary
        all_tokens: set[str] = set()
        for tokens in tokenized:
            all_tokens.update(tokens)

        vocab_list = sorted(all_tokens)[: self._dim]
        self._vocabulary = {t: i for i, t in enumerate(vocab_list)}

        # Document frequency
        n_docs = len(texts)
        df: dict[str, int] = {}
        for tokens in tokenized:
            seen = set(tokens)
            for t in seen:
                df[t] = df.get(t, 0) + 1

        # IDF
        self._idf = {
            t: np.log((1 + n_docs) / (1 + df_t)) + 1 for t, df_t in df.items()
        }
        self._fitted = True

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into TF-IDF feature vectors using fitted vocabulary.

        If not yet fitted, fits on these texts (backward compat for single-call usage).
        """
        if not texts:
            return np.zeros((0, len(self._vocabulary) if self._fitted else self._dim))

        # Backward compat: if not fitted, fit on these texts
        if not self._fitted:
            self.fit(texts)

        tokenized = [self._tokenize(t) for t in texts]
        n_features = len(self._vocabulary)
        matrix = np.zeros((len(texts), n_features), dtype=np.float32)

        for i, tokens in enumerate(tokenized):
            for t in tokens:
                if t in self._vocabulary:
                    tf = tokens.count(t) / max(len(tokens), 1)
                    matrix[i, self._vocabulary[t]] = tf * self._idf.get(t, 1.0)

        # L2 normalize each row
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        matrix = matrix / norms

        return matrix

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple word tokenization with lowercasing."""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim