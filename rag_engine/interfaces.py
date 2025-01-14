"""Core interfaces for the RAG engine components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import numpy.typing as npt


@dataclass
class Document:
    """Represents a document with its content and metadata."""

    content: str
    metadata: Dict[str, Any]
    source_path: Path


@dataclass
class Chunk:
    """Represents a chunk of text with its metadata."""

    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None


@dataclass
class SearchResult:
    """Represents a search result with its metadata and match context."""

    chunk: Chunk
    similarity: float
    match_context: Dict[str, Any]


class DocumentParser(ABC):
    """Interface for document parsers."""

    @abstractmethod
    async def parse(self, file_path: Path) -> Document:
        """Parse a document file into text and metadata."""
        pass


class TextChunker(ABC):
    """Interface for text chunking strategies."""

    @abstractmethod
    async def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks while maintaining context."""
        pass


class ChunkEnhancer(ABC):
    """Interface for enhancing chunks with LLM processing."""

    @abstractmethod
    async def enhance_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enhance a batch of chunks with additional context and metadata."""
        pass

    @abstractmethod
    async def process_stream(
        self, chunk_stream: AsyncIterator[Chunk]
    ) -> AsyncIterator[Chunk]:
        """Process chunks as they become available."""
        pass


class PromptBasedEnhancer(ChunkEnhancer):
    """Base class for enhancers that use prompt templates."""

    def __init__(self, config: Any, default_prompt: str):
        """Initialize with configuration and default prompt."""
        self.config = config
        self.prompt_template = default_prompt

    def set_prompt_template(self, template: str) -> None:
        """Set a custom prompt template."""
        self.prompt_template = template

    @abstractmethod
    async def generate_completion(self, prompt: str) -> str:
        """Generate completion from the LLM."""
        pass

    async def enhance_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enhance a batch of chunks using the prompt template."""
        # Combine chunks into a single text with separator
        combined_text = "\n---\n".join(chunk.text for chunk in chunks)

        # Format prompt with the text
        prompt = self.prompt_template.format(text=combined_text)

        try:
            # Get response from LLM
            response_text = await self.generate_completion(prompt)

            try:
                # Parse JSON response
                result = json.loads(response_text)

                # Update each chunk with enhanced text and metadata
                for chunk in chunks:
                    # If enhanced text is empty or None, keep original text
                    enhanced_text = result.get("enhanced_text")
                    if not enhanced_text:
                        enhanced_text = chunk.text

                    chunk.text = enhanced_text

                    # Update metadata with all extracted information
                    metadata = result.get("metadata", {})
                    metadata.update(
                        {
                            "entities": result.get("entities", []),
                            "relationships": result.get("relationships", []),
                            "cypher_query": result.get("cypher_query"),
                        }
                    )
                    chunk.metadata.update(metadata)

            except json.JSONDecodeError as e:
                print(
                    f"Warning: Could not parse LLM response as JSON: {str(e)}\nResponse: {response_text}"
                )
                # Keep original text if JSON parsing fails
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "entities": [],
                            "relationships": [],
                            "cypher_query": None,
                            "error": "JSON parsing failed",
                        }
                    )

        except Exception as e:
            print(f"Warning: Error enhancing chunks with LLM: {str(e)}")
            # Keep original text and add empty metadata if enhancement fails
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "entities": [],
                        "relationships": [],
                        "cypher_query": None,
                        "error": str(e),
                    }
                )

        return chunks

    async def process_stream(
        self, chunk_stream: AsyncIterator[Chunk]
    ) -> AsyncIterator[Chunk]:
        """Process chunks in a streaming fashion."""
        buffer = []
        async for chunk in chunk_stream:
            buffer.append(chunk)

            # Process buffer when it reaches batch size
            if len(buffer) >= self.config.batch_size:
                enhanced = await self.enhance_batch(buffer)
                for chunk in enhanced:
                    yield chunk
                buffer = []

        # Process any remaining chunks
        if buffer:
            enhanced = await self.enhance_batch(buffer)
            for chunk in enhanced:
                yield chunk


class EmbeddingProvider(ABC):
    """Interface for embedding generation."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass


class VectorStore(ABC):
    """Interface for vector storage and retrieval."""

    @abstractmethod
    async def store(self, chunks: List[Chunk]) -> None:
        """Store chunks with their embeddings."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: npt.NDArray[np.float32],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar chunks using a query embedding."""
        pass

    @abstractmethod
    async def delete(self, filter_expr: Dict[str, Any]) -> None:
        """Delete chunks matching the filter expression."""
        pass


class MetadataExtractor(ABC):
    """Interface for metadata extraction."""

    @abstractmethod
    async def extract(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract metadata from a chunk."""
        pass

    @abstractmethod
    def validate(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata against schema."""
        pass


class RAGPipeline(ABC):
    """Interface for the complete RAG pipeline."""

    @abstractmethod
    async def process_document(self, file_path: Path) -> str:
        """Process a document through the entire pipeline."""
        pass

    @abstractmethod
    async def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a processed document."""
        pass

    @abstractmethod
    async def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """Get chunks for a processed document."""
        pass

    @abstractmethod
    async def search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search through processed documents."""
        pass

    @abstractmethod
    async def update_document(self, file_path: Path) -> str:
        """Update an existing document in the pipeline."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and its chunks from the pipeline."""
        pass
