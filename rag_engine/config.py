"""Configuration classes for RAG pipeline."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Self


@dataclass
class DocumentConfig:
    """Configuration for document handling."""

    pdf_parser: str = "default"
    max_file_size_mb: int = 10
    supported_types: Set[str] = field(default_factory=lambda: {".pdf", ".txt", ".md"})


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    implementation: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    chunking_method: str = "semantic"
    min_chunk_size: int = 100


@dataclass
class EnhancementConfig:
    """Configuration for chunk enhancement."""

    implementation: str = "gemini"
    model: str = "gemini-1.5-pro-exp"
    batch_size: int = 5
    max_retries: int = 3
    api_key: Optional[str] = None
    prompt_template: Optional[str] = None  # The actual prompt template to use

    def __post_init__(self):
        """Load prompt template if not directly provided."""
        if self.prompt_template is None:
            from .prompts import TAX_EXPERT_PROMPT, SQL_CONVERTER_PROMPT

            # Map of available prompts
            PROMPT_TEMPLATES = {
                "tax_expert": TAX_EXPERT_PROMPT,
                "sql_converter": SQL_CONVERTER_PROMPT,
            }
            # Default to tax expert prompt
            self.prompt_template = PROMPT_TEMPLATES.get("tax_expert", TAX_EXPERT_PROMPT)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    implementation: str = "sentence_transformer"
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class VectorStoreConfig:
    """Configuration for vector storage."""

    implementation: str = "milvus"
    collection_name: str = "tax_docs"
    host: str = "localhost"
    port: int = 19530
    dim: int = 384


@dataclass
class MetadataConfig:
    """Configuration for metadata extraction."""

    implementation: str = "default"
    required_fields: Set[str] = field(default_factory=lambda: {"doc_id", "filename"})
    default_doc_type: str = "Tax Document"
    tax_years: List[str] = field(default_factory=lambda: ["2023", "2024"])


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""

    document: DocumentConfig = field(default_factory=DocumentConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    workspace: str = "./workspace"


class ConfigBuilder:
    """Builder for creating RAG pipeline configuration."""

    def __init__(self):
        self._config = RAGConfig()

    def with_document_config(
        self,
        pdf_parser: str = "default",
        max_file_size_mb: int = 10,
        supported_types: Optional[Set[str]] = None,
    ) -> Self:
        """Configure document handling."""
        if supported_types is None:
            supported_types = {".pdf", ".txt", ".md"}
        self._config.document = DocumentConfig(
            pdf_parser=pdf_parser,
            max_file_size_mb=max_file_size_mb,
            supported_types=supported_types,
        )
        return self

    def with_chunking_config(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        chunking_method: str = "semantic",
    ) -> Self:
        """Configure text chunking."""
        self._config.chunking = ChunkingConfig(
            implementation=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_method=chunking_method,
        )
        return self

    def with_enhancement_config(
        self,
        provider: str = "gemini",
        api_key: Optional[str] = None,
        batch_size: int = 5,
        max_retries: int = 3,
        model: str = "gemini-1.5-pro",
        prompt_template: Optional[str] = None,
    ) -> Self:
        """Configure chunk enhancement."""
        self._config.enhancement = EnhancementConfig(
            implementation=provider,
            model=model,
            batch_size=batch_size,
            max_retries=max_retries,
            api_key=api_key,
            prompt_template=prompt_template,
        )
        return self

    def with_embedding_config(
        self,
        provider: str = "sentence_transformer",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ) -> Self:
        """Configure embedding generation."""
        self._config.embedding = EmbeddingConfig(
            implementation=provider, model_name=model_name, batch_size=batch_size
        )
        return self

    def with_vector_store_config(
        self,
        provider: str = "milvus",
        host: str = "localhost",
        port: int = 19530,
        collection_prefix: str = "tax_docs",
        dim: int = 384,
    ) -> Self:
        """Configure vector store."""
        self._config.vector_store = VectorStoreConfig(
            implementation=provider,
            collection_name=collection_prefix,
            host=host,
            port=port,
            dim=dim,
        )
        return self

    def with_metadata_config(
        self,
        implementation: str = "default",
        required_fields: Optional[Set[str]] = None,
        default_doc_type: str = "Tax Document",
        tax_years: Optional[List[str]] = None,
    ) -> Self:
        """Configure metadata extraction."""
        if required_fields is None:
            required_fields = {"doc_id", "filename"}
        if tax_years is None:
            tax_years = ["2023", "2024"]
        self._config.metadata = MetadataConfig(
            implementation=implementation,
            required_fields=required_fields,
            default_doc_type=default_doc_type,
            tax_years=tax_years,
        )
        return self

    def with_workspace(self, workspace: str = "./workspace") -> Self:
        """Configure workspace directory."""
        self._config.workspace = workspace
        return self

    def build(self) -> RAGConfig:
        """Build and return the configuration."""
        return self._config
