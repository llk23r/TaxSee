"""Default implementation of the RAG pipeline."""

import asyncio
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Any, Optional, Sequence, cast, Awaitable
from pathlib import Path
from neo4j import Record

from .interfaces import (
    RAGPipeline,
    DocumentParser,
    TextChunker,
    ChunkEnhancer,
    EmbeddingProvider,
    VectorStore,
    MetadataExtractor,
    Document,
    Chunk,
    SearchResult,
)
from .config import RAGConfig
from neo4j_utils import Neo4jExecutor


class DefaultRAGPipeline(RAGPipeline):
    """Default implementation of the RAG pipeline."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: TextChunker,
        enhancer: ChunkEnhancer,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        metadata_extractor: MetadataExtractor,
        config: RAGConfig,
    ):
        self.parser = parser
        self.chunker = chunker
        self.enhancer = enhancer
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_extractor = metadata_extractor
        self.config = config
        self._document_cache: Dict[str, Document] = {}
        self._chunk_cache: Dict[str, List[Chunk]] = {}
        self.neo4j_executor = Neo4jExecutor()

    async def neo4j_query(self, query: str) -> List[Record]:
        """Execute a Neo4j query."""
        result = await self.neo4j_executor.run_query(query)
        return result

    async def create_neo4j_node(self, node_name: str, node_type: str, properties: dict) -> None:
        """Create a node in Neo4j."""
        create_query = f"CREATE ({node_name}:{node_type} {{ {', '.join(f'{k}: {repr(v)}' for k, v in properties.items())} }})"
        await self.neo4j_executor.run_query(create_query)

    async def process_document(self, file_path: Path) -> str:
        """Process a document through the entire pipeline."""
        try:
            print("Pipeline: Starting document processing")

            # Parse document
            print("Pipeline: Parsing document")
            document = await self.parser.parse(file_path)
            doc_id = str(file_path.stem)
            document.metadata["doc_id"] = doc_id
            self._document_cache[doc_id] = document

            print(f"Pipeline: Document parsed, content length: {len(document.content)}")
            print(f"Pipeline: Document metadata: {document.metadata}")

            # Create chunks
            print("Pipeline: Creating chunks")
            chunks = await self.chunker.chunk(document)
            self._chunk_cache[doc_id] = chunks
            print(f"Pipeline: Created {len(chunks)} chunks")
            if chunks:
                print(f"Pipeline: First chunk metadata: {chunks[0].metadata}")

            # Process chunks in batches
            batch_size = self.config.enhancement.batch_size
            print(f"Pipeline: Processing chunks in batches of {batch_size}")

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                print(f"Pipeline: Processing batch {i//batch_size + 1}")

                # Enhance chunks with LLM to add additional context/metadata
                print("Pipeline: Enhancing chunks")
                enhanced_chunks = await self.enhancer.enhance_batch(batch)
                print(f"Pipeline: Enhanced {len(enhanced_chunks)} chunks")

                # Extract metadata and generate embeddings for each chunk
                print("Pipeline: Processing enhanced chunks")
                for echunk in enhanced_chunks:
                    # Extract structured metadata
                    metadata = await self.metadata_extractor.extract(echunk)
                    echunk.metadata.update(metadata)

                    # Generate embeddings for vector search
                    print("Pipeline: Generating embeddings")
                    embeddings = await self.embedder.embed([echunk.text])
                    if embeddings and len(embeddings) > 0:
                        # Convert the embedding to the correct type
                        embedding_array = np.array(embeddings[0], dtype=np.float32)
                        echunk.embedding = embedding_array.tolist()

                    # Store relationships in knowledge graph if available
                    if "cypher_query" in echunk.metadata:
                        await self.neo4j_query(echunk.metadata["cypher_query"])

                    if "entities" in echunk.metadata:
                        for node in echunk.metadata["entities"]:
                            await self.create_neo4j_node(
                                node["entity"], 
                                node["type"], 
                                node["properties"]
                            )

                # Store chunks in vector database
                print("Pipeline: Storing chunks in vector store")
                await self.vector_store.store(enhanced_chunks)
                print("Pipeline: Chunks stored successfully")

                # Rate limiting between batches
                if i + batch_size < len(chunks):
                    print("Pipeline: Applying rate limit")
                    await asyncio.sleep(1)

            print("Pipeline: Document processing completed successfully")
            return doc_id

        except Exception as e:
            print(f"Pipeline Error: {str(e)}")
            print(f"Pipeline Error Type: {type(e)}")
            import traceback
            print(f"Pipeline Traceback: {traceback.format_exc()}")
            raise

    async def get_document_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get metadata for a processed document."""
        if doc_id in self._document_cache:
            return self._document_cache[doc_id].metadata
        return {}

    async def get_document_chunks(self, doc_id: str) -> List[Chunk]:
        """Get chunks for a processed document."""
        return self._chunk_cache.get(doc_id, [])

    async def search(
        self, query: str, limit: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for relevant chunks."""
        # Generate query embedding
        embeddings = await self.embedder.embed([query])
        if not embeddings:
            return []

        # Convert embedding to numpy array for vector store
        query_embedding = np.array(embeddings[0], dtype=np.float32)

        # Search vector store
        results = await self.vector_store.search(
            query_embedding=query_embedding, limit=limit, filters=filters
        )
        return list(results)

    async def update_document(self, file_path: Path) -> str:
        """Update an existing document."""
        doc_id = str(file_path.stem)
        # Delete existing document chunks
        await self.delete_document(doc_id)
        # Process document again
        return await self.process_document(file_path)

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and its chunks."""
        await self.vector_store.delete({"doc_id": doc_id})
        self._document_cache.pop(doc_id, None)
        self._chunk_cache.pop(doc_id, None)
