"""Tests for text chunking components."""

import pytest
from pathlib import Path
from typing import List

from rag_engine.interfaces import Document, Chunk, TextChunker
from rag_engine.config import ChunkingConfig


class TestChunker(TextChunker):
    """Test implementation of TextChunker."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    async def chunk(self, document: Document) -> List[Chunk]:
        """Simple test chunking implementation."""
        chunks = []
        text = document.content
        size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for i in range(0, len(text), size - overlap):
            chunk_text = text[i : i + size]
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={**document.metadata, "chunk_index": len(chunks)},
                )
            )
        return chunks


@pytest.fixture
def test_chunker(test_config) -> TextChunker:
    """Create test chunker instance."""
    return TestChunker(test_config.chunking)


@pytest.fixture
def sample_document() -> Document:
    """Create sample document for testing."""
    return Document(
        content="This is a test document. " * 100,
        metadata={"doc_type": "test", "tax_year": "2023"},
        source_path=Path("test.txt"),
    )


@pytest.mark.asyncio
async def test_chunking_basic(test_chunker: TestChunker, sample_document: Document):
    """Test basic chunking functionality."""
    chunks = await test_chunker.chunk(sample_document)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(
        len(chunk.text) >= test_chunker.config.min_chunk_size for chunk in chunks
    )


@pytest.mark.asyncio
async def test_chunking_metadata(test_chunker: TestChunker, sample_document: Document):
    """Test that chunks preserve document metadata."""
    chunks = await test_chunker.chunk(sample_document)

    for chunk in chunks:
        assert chunk.metadata["doc_type"] == sample_document.metadata["doc_type"]
        assert chunk.metadata["tax_year"] == sample_document.metadata["tax_year"]
        assert "chunk_index" in chunk.metadata


@pytest.mark.asyncio
async def test_chunking_overlap(test_chunker: TestChunker, sample_document: Document):
    """Test chunk overlap behavior."""
    chunks = await test_chunker.chunk(sample_document)

    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i].text
            next_chunk = chunks[i + 1].text

            overlap_size = test_chunker.config.chunk_overlap
            assert current_chunk[-overlap_size:] == next_chunk[:overlap_size]


@pytest.mark.asyncio
async def test_chunking_empty_document(test_chunker: TextChunker):
    """Test chunking with empty document."""
    empty_doc = Document(content="", metadata={}, source_path=Path("empty.txt"))

    chunks = await test_chunker.chunk(empty_doc)
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_chunking_small_document(test_chunker: TextChunker):
    """Test chunking with document smaller than chunk size."""
    small_doc = Document(
        content="Small document", metadata={}, source_path=Path("small.txt")
    )

    chunks = await test_chunker.chunk(small_doc)
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_chunking_config_changes(test_config):
    """Test chunking behavior with different configurations."""
    configs = [
        ChunkingConfig(chunk_size=500, chunk_overlap=50),
        ChunkingConfig(chunk_size=1000, chunk_overlap=100),
        ChunkingConfig(chunk_size=2000, chunk_overlap=200),
    ]

    document = Document(
        content="Test document. " * 200, metadata={}, source_path=Path("test.txt")
    )

    chunk_counts = []
    for config in configs:
        chunker = TestChunker(config)
        chunks = await chunker.chunk(document)
        chunk_counts.append(len(chunks))

    assert chunk_counts[0] > chunk_counts[1] > chunk_counts[2]
