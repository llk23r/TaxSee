"""This module contains tests for components that enhance document chunks with additional metadata and information."""

import json
import pytest
from typing import List, AsyncIterator, Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

from rag_engine.interfaces import Chunk, ChunkEnhancer
from rag_engine.config import EnhancementConfig


class MockGeminiResponse:
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text


class GeminiEnhancer(ChunkEnhancer):
    """Test implementation of Gemini-based chunk enhancement."""

    def __init__(self, config: EnhancementConfig):
        self.config = config
        self.model: Any = None  # Type hint to avoid linter error
        self.setup_gemini()

    def setup_gemini(self):
        """Set up the Gemini model with API key from config."""
        import google.generativeai as genai

        genai.configure(api_key=self.config.api_key)
        self.model = genai.GenerativeModel(self.config.model)

    async def enhance_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Enhance chunks using Gemini API."""
        if not chunks:
            return []

        # Combine chunks into a single prompt
        combined_text = "\n---\n".join(chunk.text for chunk in chunks)
        prompt = """Analyze the following text segments and for each one:
1. Extract key topics and concepts
2. Identify tax-related entities and relationships
3. Generate a Cypher query to represent these relationships

Text segments:
{text}

Respond in JSON format with the following structure for each segment:
{{
    "enhanced_text": "improved text with added context",
    "metadata": {{
        "main_topic": "primary topic",
        "subtopics": ["list", "of", "subtopics"],
        "entities": ["extracted", "entities"],
        "relationships": ["identified", "relationships"]
    }},
    "cypher_query": "MERGE (n:Entity {{name: \\"example\\"}}) ..."
}}""".format(text=combined_text)

        try:
            response = await self.model.generate_content_async(prompt)
            result = json.loads(str(response.text))

            # If result is a single dict, convert it to a list
            if isinstance(result, dict):
                result = [result]

            # Update each chunk with the enhanced content
            for chunk in chunks:
                # Use the first result for all chunks if not enough results
                chunk_result = result[0] if result else None
                if chunk_result:
                    chunk.text = chunk_result["enhanced_text"]
                    chunk.metadata.update(chunk_result["metadata"])
                    chunk.metadata["cypher_query"] = chunk_result["cypher_query"]

            return chunks
        except Exception as e:
            # In case of API error, return original chunks
            return chunks

    async def process_stream(
        self, chunk_stream: AsyncIterator[Chunk]
    ) -> AsyncIterator[Chunk]:
        """Process chunks in streaming fashion."""
        buffer = []
        async for chunk in chunk_stream:
            buffer.append(chunk)
            if len(buffer) >= self.config.batch_size:
                enhanced = await self.enhance_batch(buffer)
                for chunk in enhanced:
                    yield chunk
                buffer = []

        if buffer:
            enhanced = await self.enhance_batch(buffer)
            for chunk in enhanced:
                yield chunk


@pytest.fixture
def gemini_enhancer(test_config) -> ChunkEnhancer:
    """Create a Gemini enhancer instance for testing."""
    return GeminiEnhancer(test_config.enhancement)


@pytest.mark.asyncio
async def test_gemini_enhancement(
    gemini_enhancer: GeminiEnhancer,
    sample_chunks: List[Chunk],
    mock_gemini_response: Dict[str, Any],
):
    """Test enhancement using Gemini API."""
    mock_response = MockGeminiResponse(json.dumps(mock_gemini_response))
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)

    with patch.object(gemini_enhancer, "model", mock_model):
        enhanced_chunks = await gemini_enhancer.enhance_batch(sample_chunks)

        assert len(enhanced_chunks) == len(sample_chunks)
        for chunk in enhanced_chunks:
            assert "main_topic" in chunk.metadata
            assert "subtopics" in chunk.metadata
            assert isinstance(chunk.metadata["subtopics"], list)
            assert "cypher_query" in chunk.metadata


@pytest.mark.asyncio
async def test_gemini_enhancement_error_handling(
    gemini_enhancer: GeminiEnhancer, sample_chunks: List[Chunk]
):
    """Test error handling when Gemini API fails."""
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(side_effect=Exception("API Error"))

    with patch.object(gemini_enhancer, "model", mock_model):
        enhanced_chunks = await gemini_enhancer.enhance_batch(sample_chunks)

        # Should return original chunks on error
        assert len(enhanced_chunks) == len(sample_chunks)
        assert all(chunk in enhanced_chunks for chunk in sample_chunks)


@pytest.mark.asyncio
async def test_gemini_streaming(
    gemini_enhancer: GeminiEnhancer,
    sample_chunks: List[Chunk],
    mock_gemini_response: Dict[str, Any],
):
    """Test streaming enhancement with Gemini."""
    mock_response = MockGeminiResponse(json.dumps(mock_gemini_response))
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)

    with patch.object(gemini_enhancer, "model", mock_model):
        enhanced_chunks = []
        async for chunk in gemini_enhancer.process_stream(
            chunk_generator(sample_chunks)
        ):
            enhanced_chunks.append(chunk)

        assert len(enhanced_chunks) == len(sample_chunks)
        for chunk in enhanced_chunks:
            assert "main_topic" in chunk.metadata
            assert "subtopics" in chunk.metadata
            assert "cypher_query" in chunk.metadata


# Keep the existing TestEnhancer and its tests below for basic interface testing
class TestEnhancer(ChunkEnhancer):
    """A simple test implementation of the ChunkEnhancer interface that adds mock metadata."""

    def __init__(self, config: EnhancementConfig):
        self.config = config

    async def enhance_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """Takes a batch of chunks and adds some test metadata like topics and tax year."""
        for chunk in chunks:
            chunk.text = "Enhanced text"
            chunk.metadata.update(
                {
                    "main_topic": "Test Topic",
                    "tax_year": "2023",
                    "subtopics": ["Test Subtopic 1", "Test Subtopic 2"],
                }
            )
        return chunks

    async def process_stream(
        self, chunk_stream: AsyncIterator[Chunk]
    ) -> AsyncIterator[Chunk]:
        """Processes chunks one at a time in a streaming fashion for memory efficiency."""
        async for chunk in chunk_stream:
            enhanced = await self.enhance_batch([chunk])
            yield enhanced[0]


@pytest.fixture
def test_enhancer(test_config) -> ChunkEnhancer:
    """Sets up a fresh test enhancer instance for each test."""
    return TestEnhancer(test_config.enhancement)


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    """Creates a couple of sample chunks that we can use in our tests."""
    return [
        Chunk(
            text="Test chunk 1",
            metadata={"doc_id": "test1"},
        ),
        Chunk(
            text="Test chunk 2",
            metadata={"doc_id": "test1"},
        ),
    ]


@pytest.mark.asyncio
async def test_enhancement_batch(
    test_enhancer: ChunkEnhancer, sample_chunks: List[Chunk]
):
    """Makes sure we can enhance multiple chunks at once with the right metadata."""
    enhanced_chunks = await test_enhancer.enhance_batch(sample_chunks)

    assert len(enhanced_chunks) == len(sample_chunks)
    for chunk in enhanced_chunks:
        assert chunk.text == "Enhanced text"
        assert chunk.metadata["main_topic"] == "Test Topic"
        assert "tax_year" in chunk.metadata


@pytest.mark.asyncio
async def test_enhancement_empty_batch(test_enhancer: ChunkEnhancer):
    """Checks that the enhancer handles empty batches gracefully without crashing."""
    enhanced = await test_enhancer.enhance_batch([])
    assert len(enhanced) == 0


@pytest.mark.asyncio
async def test_enhancement_metadata_preservation(
    test_enhancer: ChunkEnhancer, sample_chunks: List[Chunk]
):
    """Verifies that the enhancer keeps existing metadata while adding new fields."""
    original_doc_ids = [chunk.metadata["doc_id"] for chunk in sample_chunks]
    enhanced_chunks = await test_enhancer.enhance_batch(sample_chunks)

    for chunk, original_id in zip(enhanced_chunks, original_doc_ids):
        assert chunk.metadata["doc_id"] == original_id
        assert "main_topic" in chunk.metadata


async def chunk_generator(chunks: List[Chunk]) -> AsyncIterator[Chunk]:
    """A helper that turns a list of chunks into an async stream for testing."""
    for chunk in chunks:
        yield chunk
