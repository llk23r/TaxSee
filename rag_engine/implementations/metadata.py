"""Metadata extractor implementations."""

from typing import Dict, Any
from pathlib import Path

from ..interfaces import MetadataExtractor, Chunk
from ..factory import ComponentRegistry
from ..config import MetadataConfig


class DefaultMetadataExtractor(MetadataExtractor):
    """Default metadata extractor implementation."""

    def __init__(self, config: MetadataConfig):
        """Initialize the metadata extractor with configuration."""
        self.config = config
        self.required_fields = config.required_fields or {"doc_id", "filename"}

    async def extract(self, source: Path | Chunk) -> Dict[str, Any]:
        """Extract metadata from a file or chunk."""
        if isinstance(source, Path):
            # Extract metadata from file
            metadata = {
                "doc_id": source.stem,
                "filename": source.name,
                "extension": source.suffix.lstrip("."),
                "size_bytes": source.stat().st_size,
                "doc_type": self.config.default_doc_type or "Tax Document",
            }

            # Extract year from filename if present
            if any(
                year in source.stem
                for year in self.config.tax_years or ["2023", "2024"]
            ):
                for year in self.config.tax_years or ["2023", "2024"]:
                    if year in source.stem:
                        metadata["tax_year"] = year
                        break

            return metadata
        else:
            # Extract metadata from chunk
            # For now, just return existing metadata
            # This will be enhanced by the Gemini enhancer
            return source.metadata

    async def validate(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata structure."""
        return all(field in metadata for field in self.required_fields)


# Register implementation
ComponentRegistry.register_metadata_extractor("default", DefaultMetadataExtractor)
