"""Tests for vector store components."""

import pytest
from typing import List, Dict, Any, Optional
import numpy as np

from rag_engine.interfaces import Chunk, SearchResult, VectorStore
from rag_engine.config import VectorStoreConfig


class TestVectorStore(VectorStore):
    """Test implementation of VectorStore."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.chunks: Dict[str, Chunk] = {}  # Simple in-memory storage

    async def store(self, chunks: List[Chunk]) -> None:
        """Store chunks in memory."""
        for i, chunk in enumerate(chunks):
            if not chunk.chunk_id:
                chunk.chunk_id = f"chunk_{len(self.chunks) + i}"
            self.chunks[chunk.chunk_id] = chunk

    async def search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Simple mock search that returns chunks in order."""
        results = []
        for chunk in list(self.chunks.values())[:limit]:
            # Mock similarity score
            similarity = np.random.random()

            # Apply filters if any
            if filters:
                matches = all(chunk.metadata.get(k) == v for k, v in filters.items())
                if not matches:
                    continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    similarity=similarity,
                    match_context={
                        "score": similarity,
                        "filter_matched": bool(filters),
                    },
                )
            )
        return results

    async def delete(self, filter_expr: Dict[str, Any]) -> None:
        """Delete chunks matching the filter."""
        to_delete = []
        for chunk_id, chunk in self.chunks.items():
            if all(chunk.metadata.get(k) == v for k, v in filter_expr.items()):
                to_delete.append(chunk_id)

        for chunk_id in to_delete:
            del self.chunks[chunk_id]


@pytest.fixture
def test_store(test_config) -> VectorStore:
    """Create test vector store instance."""
    return TestVectorStore(test_config.vector_store)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Create sample chunks with embeddings."""
    return [
        Chunk(
            text="Test chunk 1",
            metadata={"doc_id": "test1", "tax_year": "2023"},
            embedding=[0.1, 0.2, 0.3] * 128,  # 384-dim vector
        ),
        Chunk(
            text="Test chunk 2",
            metadata={"doc_id": "test1", "tax_year": "2023"},
            embedding=[0.2, 0.3, 0.4] * 128,
        ),
        Chunk(
            text="Test chunk 3",
            metadata={"doc_id": "test2", "tax_year": "2022"},
            embedding=[0.3, 0.4, 0.5] * 128,
        ),
    ]


@pytest.mark.asyncio
async def test_store_basic(test_store: VectorStore, sample_chunks: List[Chunk]):
    """Test basic storage functionality."""
    await test_store.store(sample_chunks)

    # Search should return results
    results = await test_store.search("test query", limit=5)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_search_with_filters(test_store: VectorStore, sample_chunks: List[Chunk]):
    """Test search with metadata filters."""
    await test_store.store(sample_chunks)

    # Search with tax year filter
    results = await test_store.search("test query", filters={"tax_year": "2023"})

    assert len(results) > 0
    assert all(r.chunk.metadata["tax_year"] == "2023" for r in results)


@pytest.mark.asyncio
async def test_delete_chunks(test_store: VectorStore, sample_chunks: List[Chunk]):
    """Test chunk deletion."""
    await test_store.store(sample_chunks)

    # Delete chunks for specific document
    await test_store.delete({"doc_id": "test1"})

    # Search should only return remaining chunks
    results = await test_store.search("test query")
    assert all(r.chunk.metadata["doc_id"] != "test1" for r in results)


@pytest.mark.asyncio
async def test_search_limit(test_store: VectorStore, sample_chunks: List[Chunk]):
    """Test search result limiting."""
    await test_store.store(sample_chunks)

    limit = 2
    results = await test_store.search("test query", limit=limit)
    assert len(results) <= limit


@pytest.mark.asyncio
async def test_search_empty_store(test_store: VectorStore):
    """Test search on empty store."""
    results = await test_store.search("test query")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_store_duplicate_chunks(
    test_store: VectorStore, sample_chunks: List[Chunk]
):
    """Test storing duplicate chunks."""
    # Store same chunks twice
    await test_store.store(sample_chunks)
    await test_store.store(sample_chunks)

    # Search should handle duplicates gracefully
    results = await test_store.search("test query")
    assert len(results) <= len(sample_chunks)


@pytest.mark.asyncio
async def test_search_similarity_scores(
    test_store: VectorStore, sample_chunks: List[Chunk]
):
    """Test similarity scores in search results."""
    await test_store.store(sample_chunks)

    results = await test_store.search("test query")
    assert all(0 <= r.similarity <= 1 for r in results)
    assert all("score" in r.match_context for r in results)
