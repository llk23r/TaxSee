"""Vector store implementations."""

from typing import List, Dict, Any, Optional, cast
import json
import logging
import numpy as np
import numpy.typing as npt
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from ..interfaces import VectorStore, Chunk, SearchResult
from ..factory import ComponentRegistry
from ..config import VectorStoreConfig

# Configure logging
logger = logging.getLogger(__name__)


class MilvusStore(VectorStore):
    """Vector store implementation using Milvus."""

    def __init__(self, config: VectorStoreConfig):
        """Initialize with configuration."""
        self.config = config
        self.collection: Optional[Collection] = None
        self.collection_name = config.collection_name

        # Connect to Milvus
        try:
            connections.connect(alias="default", host=config.host, port=config.port)
            logger.info(f"Connected to Milvus at {config.host}:{config.port}")
        except Exception as e:
            logger.error(f"Could not connect to Milvus: {e}")
            raise

    def _ensure_collection(self) -> Collection:
        """Ensure collection exists and is loaded."""
        if self.collection is None:
            # Create collection if it doesn't exist
            if not utility.has_collection(self.collection_name):
                logger.info(f"Creating collection: {self.collection_name}")
                fields = [
                    FieldSchema(
                        name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                    ),
                    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=self.config.dim,
                    ),
                ]
                schema = CollectionSchema(fields=fields, description="Document chunks")
                self.collection = Collection(name=self.collection_name, schema=schema)

                # Create vector index
                logger.info("Creating vector index...")
                self.collection.create_index(
                    field_name="embedding",
                    index_params={
                        "metric_type": "COSINE",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128},
                    },
                )
                logger.info("Created collection and index successfully")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
                self.collection = Collection(self.collection_name)

            # Load collection
            self.collection.load()

        return cast(Collection, self.collection)

    async def store(self, chunks: List[Chunk]) -> None:
        """Store chunks in Milvus."""
        if not chunks:
            return

        collection = self._ensure_collection()

        # Prepare data as list of dictionaries
        data = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(
                    f"Chunk has no embedding, skipping: {chunk.text[:100]}..."
                )
                continue

            # Create individual record
            record = {
                "doc_id": chunk.metadata.get("doc_id", ""),
                "text": chunk.text,
                "metadata": json.dumps(chunk.metadata),  # Milvus requires JSON string
                "embedding": chunk.embedding.tolist()
                if isinstance(chunk.embedding, np.ndarray)
                else chunk.embedding,
            }
            data.append(record)

        if data:  # Only insert if we have valid chunks
            try:
                collection.insert(data)
                collection.flush()
                logger.info(f"Successfully stored {len(data)} chunks")
            except Exception as e:
                logger.error(f"Failed to store chunks: {e}")
                raise

    async def search(
        self,
        query_embedding: npt.NDArray[np.float32],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        collection = self._ensure_collection()

        # Build filter expression if needed
        expr = None
        if filters:
            try:
                conditions = []
                for key, value in filters.items():
                    # Convert to JSON string and wrap in quotes for Milvus
                    json_str = json.dumps({key: value})
                    conditions.append(f"JSON_CONTAINS(metadata, '{json_str}')")
                if conditions:
                    expr = " && ".join(conditions)
            except Exception as e:
                logger.warning(f"Failed to build filter expression: {e}")

        try:
            # Perform search
            milvus_results = collection.search(
                data=[query_embedding.tolist()],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                expr=expr,
                output_fields=["text", "metadata", "doc_id"],
            )

            # Format results
            search_results = []

            # Handle both SearchResult and SearchFuture types
            if hasattr(milvus_results, "wait"):
                milvus_results = milvus_results.wait()

            if milvus_results and len(milvus_results) > 0:
                hits = milvus_results[0]  # First query's results
                for hit in hits:
                    metadata = json.loads(hit.fields.get("metadata", "{}"))
                    chunk = Chunk(
                        text=hit.fields.get("text", ""),
                        metadata=metadata,
                        doc_id=hit.fields.get("doc_id", ""),
                    )
                    result = SearchResult(
                        chunk=chunk,
                        similarity=float(hit.score),
                        match_context={"score": float(hit.score)},
                    )
                    search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete(self, filter_expr: Dict[str, Any]) -> None:
        """Delete documents by ID."""
        if not filter_expr:
            return

        collection = self._ensure_collection()
        try:
            # Handle doc_id specially since it's a direct field
            if "doc_id" in filter_expr:
                expr = f'doc_id == "{filter_expr["doc_id"]}"'
            else:
                # For metadata fields, use JSON_CONTAINS with string literals
                conditions = []
                for key, value in filter_expr.items():
                    json_str = json.dumps({key: value})
                    conditions.append(f"JSON_CONTAINS(metadata, '{json_str}')")
                expr = " && ".join(conditions)

            collection.delete(expr)
            collection.flush()
            logger.info(f"Successfully deleted documents matching: {expr}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise


# Register implementation
ComponentRegistry.register_vector_store("milvus", MilvusStore)
