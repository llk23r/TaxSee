"""Gemini-based enhancer implementation."""

import os
import json
import logging
from typing import List, AsyncIterator

import google.generativeai as genai
from google.generativeai import GenerativeModel, GenerationConfig

from ..interfaces import ChunkEnhancer, Chunk
from ..config import EnhancementConfig
from ..factory import ComponentRegistry
from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class GeminiEnhancer(ChunkEnhancer):
    def __init__(self, config: EnhancementConfig):
        self.config = config

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        if not self.config.model:
            raise ValueError("Model name not specified in configuration")

        genai.configure(api_key=api_key)
        self._setup_model()
        self._token_counter = TokenCounter(config.model)

    def _setup_model(self) -> None:
        config = GenerationConfig(
            temperature=0,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )

        if self.config.prompt_template:
            self.model = GenerativeModel(
                self.config.model,
                generation_config=config,
                system_instruction=self.config.prompt_template,
            )
        else:
            self.model = GenerativeModel(
                self.config.model,
                generation_config=config,
            )

    async def enhance_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return []

        combined_text = "\n---\n".join(chunk.text for chunk in chunks)

        try:
            token_info = self._token_counter.count_tokens(
                combined_text, system_prompt=self.config.prompt_template
            )
            logger.info(f"Token usage for batch: {token_info}")

            if not self._token_counter.will_fit_in_context(combined_text):
                raise ValueError(
                    f"Content exceeds token limits. "
                    f"Total tokens: {token_info['total_tokens']}, "
                    f"Limit: {token_info['input_token_limit']}"
                )

            response = await self.model.generate_content_async(combined_text)

            if hasattr(response, "usage"):
                logger.info(f"Actual token usage: {response.usage}")

            try:
                result = json.loads(response.text)

                for chunk in chunks:
                    enhanced_text = result.get("enhanced_text")
                    if enhanced_text:
                        chunk.text = enhanced_text

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
                logger.warning(f"Failed to parse JSON response: {e}")
                for chunk in chunks:
                    chunk.metadata.update(
                        {
                            "entities": [],
                            "relationships": [],
                            "cypher_query": None,
                            "error": "JSON parsing failed",
                        }
                    )

        except ValueError as ve:
            logger.error(f"Token limit exceeded: {ve}")
            for chunk in chunks:
                chunk.metadata.update(
                    {
                        "entities": [],
                        "relationships": [],
                        "cypher_query": None,
                        "error": str(ve),
                    }
                )
        except Exception as e:
            logger.error(f"Error enhancing chunks: {e}")
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


ComponentRegistry.register_enhancer("gemini", GeminiEnhancer)
