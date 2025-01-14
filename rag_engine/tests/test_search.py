"""Tests for vector search functionality with different tax-related queries."""

import pytest
from typing import List, Dict, Any
from pathlib import Path

from rag_engine.interfaces import Chunk, SearchResult, VectorStore
from rag_engine.config import VectorStoreConfig


class TestVectorStore(VectorStore):
    """Test implementation of vector store with sample tax document chunks."""

    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.chunks: Dict[str, Chunk] = {}
        self._initialize_test_chunks()

    def _initialize_test_chunks(self):
        """Initialize with sample tax-related chunks."""
        test_chunks = [
            Chunk(
                text="The standard deduction for single filers in 2023 is $13,850. "
                "This amount is adjusted annually for inflation. Taxpayers who are 65 or older "
                "or blind get an additional standard deduction.",
                metadata={
                    "doc_id": "standard_deduction_2023",
                    "main_topic": "Standard Deduction",
                    "tax_year": "2023",
                    "doc_type": "Tax Guide",
                },
                embedding=[0.1, 0.2, 0.3] * 128,  # Mock embedding
            ),
            Chunk(
                text="Itemized deductions include mortgage interest, state and local taxes (SALT) "
                "up to $10,000, charitable contributions, and medical expenses exceeding 7.5% "
                "of adjusted gross income (AGI).",
                metadata={
                    "doc_id": "itemized_deductions",
                    "main_topic": "Itemized Deductions",
                    "tax_year": "2023",
                    "doc_type": "Tax Guide",
                },
                embedding=[0.2, 0.3, 0.4] * 128,
            ),
            Chunk(
                text="The tax filing deadline for 2023 returns is April 15, 2024. "
                "Taxpayers can request a 6-month extension until October 15, 2024, "
                "but must pay any estimated taxes owed by the original deadline.",
                metadata={
                    "doc_id": "filing_deadlines",
                    "main_topic": "Tax Filing Deadlines",
                    "tax_year": "2023",
                    "doc_type": "Tax Calendar",
                },
                embedding=[0.3, 0.4, 0.5] * 128,
            ),
            Chunk(
                text="Adjusted Gross Income (AGI) is calculated by taking total income and "
                "subtracting specific adjustments like student loan interest, self-employment "
                "tax, and contributions to qualified retirement accounts.",
                metadata={
                    "doc_id": "agi_calculation",
                    "main_topic": "AGI",
                    "tax_year": "2023",
                    "doc_type": "Tax Guide",
                },
                embedding=[0.4, 0.5, 0.6] * 128,
            ),
        ]

        for chunk in test_chunks:
            if chunk.metadata and "doc_id" in chunk.metadata:
                self.chunks[chunk.metadata["doc_id"]] = chunk

    async def store(self, chunks: List[Chunk]) -> None:
        """Store new chunks."""
        for chunk in chunks:
            if chunk.metadata and "doc_id" in chunk.metadata:
                self.chunks[chunk.metadata["doc_id"]] = chunk

    async def search(
        self, query: str, limit: int = 5, filters: Dict[str, Any] | None = None
    ) -> List[SearchResult]:
        """Search for chunks based on query and filters."""
        results = []

        # Simple keyword matching for demonstration
        query_terms = query.lower().split()
        for chunk in self.chunks.values():
            # Apply filters first if any
            if filters:
                if not all(chunk.metadata.get(k) == v for k, v in filters.items()):
                    continue

            # Calculate relevance score based on term matches
            text_matches = sum(
                2.0 if term in chunk.text.lower() else 0.0 for term in query_terms
            )
            metadata_matches = sum(
                1.0
                if any(term in str(v).lower() for v in chunk.metadata.values())
                else 0.0
                for term in query_terms
            )

            # Normalize score
            total_matches = text_matches + metadata_matches
            if total_matches > 0:
                relevance = total_matches / (
                    2.0 * len(query_terms)
                )  # Normalize to 0-1 range
                results.append(
                    SearchResult(
                        chunk=chunk,
                        similarity=relevance,
                        match_context={"score": relevance},
                    )
                )

        # Sort by relevance and limit results
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:limit]

    async def delete(self, filter_expr: Dict[str, Any]) -> None:
        """Delete chunks matching the filter expression."""
        to_delete = []
        for chunk_id, chunk in self.chunks.items():
            if all(chunk.metadata.get(k) == v for k, v in filter_expr.items()):
                to_delete.append(chunk_id)

        for chunk_id in to_delete:
            del self.chunks[chunk_id]


@pytest.fixture
def vector_store(test_config) -> VectorStore:
    """Create a test vector store with sample data."""
    return TestVectorStore(test_config.vector_store)


@pytest.mark.asyncio
async def test_search_standard_deduction(vector_store: VectorStore):
    """Test searching for standard deduction information."""
    results = await vector_store.search("What is the standard deduction for 2023?")

    assert len(results) > 0
    top_result = results[0].chunk
    assert "standard deduction" in top_result.text.lower()
    assert "13,850" in top_result.text
    assert top_result.metadata["tax_year"] == "2023"


@pytest.mark.asyncio
async def test_search_itemized_deductions(vector_store: VectorStore):
    """Test searching for itemized deductions information."""
    results = await vector_store.search("What expenses can I itemize on my taxes?")

    assert len(results) > 0
    top_result = results[0].chunk
    assert "itemized deductions" in top_result.text.lower()
    assert "mortgage interest" in top_result.text.lower()
    assert "charitable contributions" in top_result.text.lower()


@pytest.mark.asyncio
async def test_search_filing_deadline(vector_store: VectorStore):
    """Test searching for tax filing deadline."""
    results = await vector_store.search("When is the tax filing deadline for 2023?")

    assert len(results) > 0
    top_result = results[0].chunk
    assert "filing deadline" in top_result.text.lower()
    assert "April 15, 2024" in top_result.text


@pytest.mark.asyncio
async def test_search_agi(vector_store: VectorStore):
    """Test searching for AGI information."""
    results = await vector_store.search("How is adjusted gross income calculated?")

    assert len(results) > 0
    top_result = results[0].chunk
    assert "adjusted gross income" in top_result.text.lower()
    assert "AGI" in top_result.metadata["main_topic"]


@pytest.mark.asyncio
async def test_search_with_year_filter(vector_store: VectorStore):
    """Test searching with tax year filter."""
    results = await vector_store.search(
        "standard deduction", filters={"tax_year": "2023"}
    )

    assert len(results) > 0
    for result in results:
        assert result.chunk.metadata["tax_year"] == "2023"


@pytest.mark.asyncio
async def test_search_with_doc_type_filter(vector_store: VectorStore):
    """Test searching with document type filter."""
    results = await vector_store.search("deductions", filters={"doc_type": "Tax Guide"})

    assert len(results) > 0
    for result in results:
        assert result.chunk.metadata["doc_type"] == "Tax Guide"
