"""Text chunking implementations."""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..interfaces import TextChunker, Document, Chunk
from ..factory import ComponentRegistry
from ..config import ChunkingConfig


class SemanticChunker(TextChunker):
    """Semantic text chunker implementation."""

    def __init__(self, config: ChunkingConfig):
        """Initialize chunker with configuration."""
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def chunk(self, document: Document) -> List[Chunk]:
        """Split document into semantic chunks."""
        if not document.content:
            return []

        # Split text into chunks
        texts = self.splitter.split_text(document.content)

        # Create chunks with metadata
        chunks = []
        for i, text in enumerate(texts):
            chunk = Chunk(
                text=text,
                metadata={
                    "doc_id": document.metadata.get("doc_id", ""),
                    "chunk_index": i,
                    "total_chunks": len(texts),
                    **document.metadata,
                },
            )
            chunks.append(chunk)

        return chunks


# Register implementation
ComponentRegistry.register_chunker("semantic", SemanticChunker)
