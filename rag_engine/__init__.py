"""RAG Engine - A modular Retrieval-Augmented Generation pipeline."""

from .config import (
    RAGConfig,
    ConfigBuilder,
    DocumentConfig,
    ChunkingConfig,
    EmbeddingConfig,
    EnhancementConfig,
    VectorStoreConfig,
    MetadataConfig,
)

from .pipeline import RAGPipeline, DefaultRAGPipeline

from .interfaces import VectorStore, SearchResult, Document, Chunk

from .implementations import MilvusStore
from .factory import RAGFactory, ComponentRegistry, load_implementations

__all__ = [
    "RAGConfig",
    "ConfigBuilder",
    "DocumentConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "EnhancementConfig",
    "VectorStoreConfig",
    "MetadataConfig",
    "RAGPipeline",
    "DefaultRAGPipeline",
    "VectorStore",
    "MilvusStore",
    "SearchResult",
    "Document",
    "Chunk",
    "RAGFactory",
    "ComponentRegistry",
    "load_implementations",
]

# Load implementations when module is imported
load_implementations()
