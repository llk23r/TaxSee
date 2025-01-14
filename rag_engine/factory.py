"""Factory for creating RAG engine components."""

import importlib
from pathlib import Path
from typing import Type, Dict, Any

from .config import RAGConfig
from .interfaces import (
    DocumentParser,
    TextChunker,
    ChunkEnhancer,
    EmbeddingProvider,
    VectorStore,
    MetadataExtractor,
    RAGPipeline,
)
from .pipeline import DefaultRAGPipeline


class ComponentRegistry:
    """Registry for component implementations."""

    _parsers: Dict[str, Type[DocumentParser]] = {}
    _chunkers: Dict[str, Type[TextChunker]] = {}
    _enhancers: Dict[str, Type[ChunkEnhancer]] = {}
    _embedders: Dict[str, Type[EmbeddingProvider]] = {}
    _vector_stores: Dict[str, Type[VectorStore]] = {}
    _metadata_extractors: Dict[str, Type[MetadataExtractor]] = {}

    @classmethod
    def register_parser(cls, name: str, parser_cls: Type[DocumentParser]) -> None:
        """Register a document parser implementation."""
        cls._parsers[name] = parser_cls

    @classmethod
    def register_chunker(cls, name: str, chunker_cls: Type[TextChunker]) -> None:
        """Register a text chunker implementation."""
        cls._chunkers[name] = chunker_cls

    @classmethod
    def register_enhancer(cls, name: str, enhancer_cls: Type[ChunkEnhancer]) -> None:
        """Register a chunk enhancer implementation."""
        cls._enhancers[name] = enhancer_cls

    @classmethod
    def register_embedder(
        cls, name: str, embedder_cls: Type[EmbeddingProvider]
    ) -> None:
        """Register an embedding provider implementation."""
        cls._embedders[name] = embedder_cls

    @classmethod
    def register_vector_store(cls, name: str, store_cls: Type[VectorStore]) -> None:
        """Register a vector store implementation."""
        cls._vector_stores[name] = store_cls

    @classmethod
    def register_metadata_extractor(
        cls, name: str, extractor_cls: Type[MetadataExtractor]
    ) -> None:
        """Register a metadata extractor implementation."""
        cls._metadata_extractors[name] = extractor_cls

    @classmethod
    def get_parser(cls, name: str) -> Type[DocumentParser]:
        """Get a registered document parser implementation."""
        if name not in cls._parsers:
            raise KeyError(f"Parser implementation '{name}' not found")
        return cls._parsers[name]

    @classmethod
    def get_chunker(cls, name: str) -> Type[TextChunker]:
        """Get a registered text chunker implementation."""
        if name not in cls._chunkers:
            raise KeyError(f"Chunker implementation '{name}' not found")
        return cls._chunkers[name]

    @classmethod
    def get_enhancer(cls, name: str) -> Type[ChunkEnhancer]:
        """Get a registered chunk enhancer implementation."""
        if name not in cls._enhancers:
            raise KeyError(f"Enhancer implementation '{name}' not found")
        return cls._enhancers[name]

    @classmethod
    def get_embedder(cls, name: str) -> Type[EmbeddingProvider]:
        """Get a registered embedding provider implementation."""
        if name not in cls._embedders:
            raise KeyError(f"Embedder implementation '{name}' not found")
        return cls._embedders[name]

    @classmethod
    def get_vector_store(cls, name: str) -> Type[VectorStore]:
        """Get a registered vector store implementation."""
        if name not in cls._vector_stores:
            raise KeyError(f"Vector store implementation '{name}' not found")
        return cls._vector_stores[name]

    @classmethod
    def get_metadata_extractor(cls, name: str) -> Type[MetadataExtractor]:
        """Get a registered metadata extractor implementation."""
        if name not in cls._metadata_extractors:
            raise KeyError(f"Metadata extractor implementation '{name}' not found")
        return cls._metadata_extractors[name]


class RAGFactory:
    """Factory for creating RAG pipeline components."""

    def __init__(self, config: RAGConfig):
        self.config = config

    def create_parser(self) -> DocumentParser:
        """Create document parser instance."""
        parser_cls = ComponentRegistry.get_parser(self.config.document.pdf_parser)
        try:
            return parser_cls(self.config.document)
        except TypeError:
            return parser_cls()

    def create_chunker(self) -> TextChunker:
        """Create text chunker instance."""
        chunker_cls = ComponentRegistry.get_chunker(self.config.chunking.implementation)
        try:
            return chunker_cls(self.config.chunking)
        except TypeError:
            return chunker_cls()

    def create_enhancer(self) -> ChunkEnhancer:
        """Create chunk enhancer instance."""
        enhancer_cls = ComponentRegistry.get_enhancer(
            self.config.enhancement.implementation
        )
        try:
            return enhancer_cls(self.config.enhancement)
        except TypeError:
            return enhancer_cls()

    def create_embedder(self) -> EmbeddingProvider:
        """Create embedding provider instance."""
        embedder_cls = ComponentRegistry.get_embedder(
            self.config.embedding.implementation
        )
        try:
            return embedder_cls(self.config.embedding)
        except TypeError:
            return embedder_cls()

    def create_vector_store(self) -> VectorStore:
        """Create vector store instance."""
        store_cls = ComponentRegistry.get_vector_store(
            self.config.vector_store.implementation
        )
        try:
            return store_cls(self.config.vector_store)
        except TypeError:
            return store_cls()

    def create_metadata_extractor(self) -> MetadataExtractor:
        """Create metadata extractor instance."""
        extractor_cls = ComponentRegistry.get_metadata_extractor(
            self.config.metadata.implementation
        )
        try:
            return extractor_cls(self.config.metadata)
        except TypeError:
            return extractor_cls()

    def create_pipeline(self) -> RAGPipeline:
        """Create a complete RAG pipeline."""
        return DefaultRAGPipeline(
            parser=self.create_parser(),
            chunker=self.create_chunker(),
            enhancer=self.create_enhancer(),
            embedder=self.create_embedder(),
            vector_store=self.create_vector_store(),
            metadata_extractor=self.create_metadata_extractor(),
            config=self.config,
        )


def load_implementations():
    """Load all available implementations from the implementations directory."""
    base_dir = Path(__file__).parent

    # Load from implementations directory
    implementations_dir = base_dir / "implementations"
    for module_path in implementations_dir.glob("*.py"):
        if module_path.stem.startswith("_"):
            continue
        module_name = f"rag_engine.implementations.{module_path.stem}"
        importlib.import_module(module_name)

    # Load from enhancers directory
    enhancers_dir = base_dir / "enhancers"
    for module_path in enhancers_dir.glob("*.py"):
        if module_path.stem.startswith("_"):
            continue
        module_name = f"rag_engine.enhancers.{module_path.stem}"
        importlib.import_module(module_name)
