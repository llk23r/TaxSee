"""Base class for prompt-based enhancers."""

from typing import List, Dict, Any, AsyncIterator
import json

from ..interfaces import ChunkEnhancer, Chunk


class PromptBasedEnhancer(ChunkEnhancer):
    """Base class for enhancers that use prompt templates."""

    def __init__(self, config: Any, default_prompt: str):
        """Initialize with configuration and default prompt."""
        self.config = config
        self.prompt_template = default_prompt

    def set_prompt_template(self, template: str) -> None:
        """Set a custom prompt template."""
        self.prompt_template = template

    async def generate_completion(self, prompt: str) -> str:
        """Generate completion from the LLM."""
        raise NotImplementedError("Subclasses must implement generate_completion")

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
