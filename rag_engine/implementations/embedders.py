"""Embedding provider implementations."""

from typing import List
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from ..interfaces import EmbeddingProvider
from ..factory import ComponentRegistry
from ..config import EmbeddingConfig


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize with configuration."""
        self.config = config
        self.model = SentenceTransformer(config.model_name or "all-MiniLM-L6-v2")

    async def embed(self, texts: List[str]) -> List[npt.NDArray[np.float32]]:
        """Get embeddings for a list of texts."""
        if not texts:
            return []

        # Generate embeddings
        embeddings = self.model.encode(
            texts, convert_to_tensor=False, normalize_embeddings=True
        )

        # Convert to list of numpy arrays
        return [np.array(emb, dtype=np.float32) for emb in embeddings]


# Register implementation
ComponentRegistry.register_embedder("sentence_transformer", SentenceTransformerEmbedder)
