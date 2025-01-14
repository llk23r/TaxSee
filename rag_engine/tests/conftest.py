"""Test configuration and shared fixtures."""

import os
import pytest
from pathlib import Path
from typing import AsyncGenerator, List, Dict, Any, AsyncIterator, Optional
import asyncio
import numpy as np
import numpy.typing as npt

from rag_engine.config import (
    ConfigBuilder,
    RAGConfig,
    DocumentConfig,
    ChunkingConfig,
    EnhancementConfig,
    VectorStoreConfig,
)
from rag_engine.factory import ComponentRegistry
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
)


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


# Register mock implementations
ComponentRegistry.register_parser("mock_parser", MockParser)
ComponentRegistry.register_chunker("mock_chunker", MockChunker)
ComponentRegistry.register_enhancer("mock_enhancer", MockEnhancer)
ComponentRegistry.register_embedder("mock_embedder", MockEmbedder)
ComponentRegistry.register_vector_store("mock_store", MockVectorStore)
ComponentRegistry.register_metadata_extractor("mock_extractor", MockMetadataExtractor)


@pytest.fixture
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_pdf(test_data_dir: Path) -> Path:
    """Get path to sample PDF file."""
    return test_data_dir / "sample.pdf"


@pytest.fixture
def test_config() -> RAGConfig:
    """Create test configuration."""
    return (
        ConfigBuilder()
        .with_document_config(pdf_parser="mock_parser", max_file_size_mb=10)
        .with_chunking_config(
            chunk_size=1000, chunk_overlap=100, chunking_method="mock_chunker"
        )
        .with_enhancement_config(
            provider="mock_enhancer", api_key="test_key", batch_size=5
        )
        .with_embedding_config(model_name="mock_embedder", batch_size=32)
        .with_vector_store_config(
            provider="mock_store",
            host="localhost",
            port=19530,
            collection_prefix="test_collection",
        )
        .with_metadata_config(
            default_doc_type="Test Document", implementation="mock_extractor"
        )
        .with_workspace("./test_workspace")
        .build()
    )


@pytest.fixture
async def clean_workspace(test_config: RAGConfig) -> AsyncGenerator[None, None]:
    """Ensure clean workspace before and after tests."""
    workspace = Path(test_config.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    yield

    if workspace.exists():
        for file in workspace.glob("*"):
            file.unlink()
        workspace.rmdir()


@pytest.fixture
def mock_gemini_response() -> List[Dict[str, Any]]:
    """Mock response from Gemini API."""
    return [
        {
            "enhanced_text": "Sample enhanced text that provides more context about tax deductions...",
            "metadata": {
                "main_topic": "Tax Deductions",
                "subtopics": ["Standard Deduction", "Itemized Deductions"],
                "entities": ["deduction", "taxpayer", "income"],
                "relationships": [
                    "taxpayer_claims_deduction",
                    "deduction_reduces_income",
                ],
                "tax_year": "2023",
                "jurisdiction": ["federal"],
                "doc_type": "1040",
            },
            "cypher_query": 'MERGE (d:Deduction {name: "Standard Deduction", year: "2023"}) MERGE (t:Taxpayer {type: "Individual"}) MERGE (i:Income {type: "Adjusted Gross"}) MERGE (t)-[:CLAIMS]->(d) MERGE (d)-[:REDUCES]->(i)',
        }
    ]


@pytest.fixture
def mock_embeddings() -> list:
    """Mock embeddings for testing."""
    return [[0.1, 0.2, 0.3] * 128]
