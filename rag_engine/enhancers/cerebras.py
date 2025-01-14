"""Cerebras-based enhancer implementation."""

import os
import json
from typing import List, AsyncIterator
from cerebras.cloud.sdk import Cerebras

from ..interfaces import ChunkEnhancer, Chunk
from ..config import EnhancementConfig
from ..factory import ComponentRegistry


class CerebrasEnhancer(ChunkEnhancer):
    """Enhances chunks using Cerebras API."""

    def __init__(self, config: EnhancementConfig):
        """Initialize the enhancer with API key and configuration."""
        self.config = config

        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY environment variable not set")

        self.client = Cerebras(api_key=api_key)

    async def enhance_batch(
        self, chunks: List[Chunk], task_name: str = "default"
    ) -> List[Chunk]:
        """Enhance a batch of chunks using Cerebras API."""
        if not chunks:
            return []

        # Use the model and prompt template directly from the config
        model_name = self.config.model
        prompt_template = self.config.prompt_template or ""

        # Combine chunks into a single text with separator
        combined_text = "\n---\n".join(chunk.text for chunk in chunks)

        # Format prompt with text
        prompt = prompt_template.format(text=combined_text)

        try:
            # Get response from Cerebras using the model
            stream = self.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model=model_name,
                stream=True,
                max_completion_tokens=self.config.batch_size,  # Assuming batch_size is used for tokens
                temperature=0.7,  # Default temperature
                top_p=0.9,  # Default top_p
            )

            # Collect response
            response_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content

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
                            "task": task_name,
                            "model": model_name,
                            "provider": "cerebras",
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
                            "task": task_name,
                            "model": model_name,
                            "provider": "cerebras",
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
                        "task": task_name,
                        "model": model_name,
                        "provider": "cerebras",
                        "entities": [],
                        "relationships": [],
                        "cypher_query": None,
                        "error": str(e),
                    }
                )

        return chunks

    async def process_stream(
        self, chunk_stream: AsyncIterator[Chunk], task_name: str = "default"
    ) -> AsyncIterator[Chunk]:
        """Process chunks in a streaming fashion."""
        buffer = []
        async for chunk in chunk_stream:
            buffer.append(chunk)

            # Process buffer when it reaches batch size
            if len(buffer) >= self.config.batch_size:
                enhanced = await self.enhance_batch(buffer, task_name=task_name)
                for chunk in enhanced:
                    yield chunk
                buffer = []

        # Process any remaining chunks
        if buffer:
            enhanced = await self.enhance_batch(buffer, task_name=task_name)
            for chunk in enhanced:
                yield chunk


# Register implementation
ComponentRegistry.register_enhancer("cerebras", CerebrasEnhancer)
