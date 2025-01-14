"""Integration tests for the RAG pipeline, ensuring all components work together seamlessly."""

import pytest
from pathlib import Path
from typing import List, Dict, Any, AsyncIterator, Optional
from unittest.mock import patch
import asyncio
import numpy as np
import numpy.typing as npt

from rag_engine.interfaces import (
    Document,
    Chunk,
    SearchResult,
    DocumentParser,
    TextChunker,
    ChunkEnhancer,
    EmbeddingProvider,
    VectorStore,
    MetadataExtractor,
    RAGPipeline,
)
from rag_engine.config import RAGConfig
from rag_engine.pipeline import DefaultRAGPipeline


class MockParser(DocumentParser):
    async def parse(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return Document(
            content="Test document content\n" * 10,
            metadata={"doc_type": "test", "tax_year": "2023"},
            source_path=file_path,
        )


class MockChunker(TextChunker):
    async def chunk(self, document: Document) -> List[Chunk]:
        return [
            Chunk(
                text=f"Chunk {i}\n{document.content[i*100:(i+1)*100]}",
                metadata=document.metadata,
                doc_id=document.source_path.stem,
            )
            for i in range(3)
        ]


class MockEnhancer(ChunkEnhancer):
    async def enhance_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        await asyncio.sleep(0.1)
        for chunk in chunks:
            chunk.metadata.update(
                {"main_topic": "Test Topic", "subtopics": ["Subtopic 1", "Subtopic 2"]}
            )
        return chunks

    async def process_stream(
        self, chunk_stream: AsyncIterator[Chunk]
    ) -> AsyncIterator[Chunk]:
        async for chunk in chunk_stream:
            enhanced_chunks = await self.enhance_batch([chunk])
            yield enhanced_chunks[0]


class MockEmbedder(EmbeddingProvider):
    async def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2, 0.3] * 128 for _ in texts]


class MockVectorStore(VectorStore):
    def __init__(self):
        self.chunks: Dict[str, Chunk] = {}

    async def store(self, chunks: List[Chunk]) -> None:
        for chunk in chunks:
            if chunk.doc_id is not None:
                self.chunks[chunk.doc_id] = chunk

    async def search(
        self,
        query_embedding: npt.NDArray[np.float32],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        results = []
        for chunk in list(self.chunks.values())[:limit]:
            if filters is not None and not all(
                chunk.metadata.get(k) == v for k, v in filters.items()
            ):
                continue
            results.append(
                SearchResult(chunk=chunk, similarity=0.9, match_context={"score": 0.9})
            )
        return results

    async def delete(self, filter_expr: Dict[str, Any]) -> None:
        if "doc_id" in filter_expr:
            doc_id = filter_expr["doc_id"]
            self.chunks = {k: v for k, v in self.chunks.items() if v.doc_id != doc_id}
        else:
            to_delete = []
            for chunk_id, chunk in self.chunks.items():
                if all(chunk.metadata.get(k) == v for k, v in filter_expr.items()):
                    to_delete.append(chunk_id)
            for chunk_id in to_delete:
                del self.chunks[chunk_id]


class MockMetadataExtractor(MetadataExtractor):
    async def extract(self, chunk: Chunk) -> Dict[str, Any]:
        return {"extracted_topic": "Test", "confidence": 0.9}

    def validate(self, metadata: Dict[str, Any]) -> bool:
        return True


@pytest.fixture
def mock_pipeline(test_config: RAGConfig) -> RAGPipeline:
    """Creates a test pipeline with mock components for isolated testing."""
    return DefaultRAGPipeline(
        parser=MockParser(),
        chunker=MockChunker(),
        enhancer=MockEnhancer(),
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        metadata_extractor=MockMetadataExtractor(),
        config=test_config,
    )


@pytest.fixture
def test_data_dir() -> Path:
    """Sets up a clean test data directory for storing temporary test files."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def sample_pdf(test_data_dir: Path) -> Path:
    """Creates a simple PDF file with test content for pipeline processing."""
    pdf_path = test_data_dir / "test.pdf"
    pdf_path.write_text("Test PDF content")
    return pdf_path


@pytest.mark.asyncio
async def test_pipeline_process_document(mock_pipeline: RAGPipeline, sample_pdf: Path):
    """Verifies that a document can be processed end-to-end through the pipeline."""
    await mock_pipeline.process_document(sample_pdf)

    results = await mock_pipeline.search("test query")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)

    for result in results:
        assert "main_topic" in result.chunk.metadata
        assert "extracted_topic" in result.chunk.metadata
        assert result.chunk.embedding is not None


@pytest.mark.asyncio
async def test_pipeline_update_document(mock_pipeline: RAGPipeline, sample_pdf: Path):
    """Ensures documents can be updated without breaking the search functionality."""
    await mock_pipeline.process_document(sample_pdf)
    await mock_pipeline.update_document(sample_pdf)

    results = await mock_pipeline.search("test query")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_pipeline_delete_document(mock_pipeline: RAGPipeline, sample_pdf: Path):
    """Confirms that deleted documents are properly removed from the search index."""
    await mock_pipeline.process_document(sample_pdf)
    await mock_pipeline.delete_document(sample_pdf.stem)

    results = await mock_pipeline.search("test query")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_pipeline_search_with_filters(
    mock_pipeline: RAGPipeline, sample_pdf: Path
):
    """Validates that metadata filters correctly narrow down search results."""
    await mock_pipeline.process_document(sample_pdf)

    results = await mock_pipeline.search("test query", filters={"tax_year": "2023"})

    assert len(results) > 0
    assert all(r.chunk.metadata["tax_year"] == "2023" for r in results)


@pytest.mark.asyncio
async def test_pipeline_error_handling(mock_pipeline: RAGPipeline):
    """Verifies the pipeline properly handles and reports errors for missing files."""
    with pytest.raises(Exception):
        await mock_pipeline.process_document(Path("nonexistent.pdf"))


@pytest.mark.asyncio
async def test_pipeline_rate_limiting(mock_pipeline: RAGPipeline, sample_pdf: Path):
    """Confirms that rate limiting is properly enforced during document processing."""
    with patch("asyncio.sleep") as mock_sleep:
        await mock_pipeline.process_document(sample_pdf)
        assert mock_sleep.called
