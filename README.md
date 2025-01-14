# Minimal RAG Chatbot with Hybrid Search

This implementation demonstrates a cutting-edge approach to Retrieval-Augmented Generation (RAG) combining multiple SOTA techniques for maximum accuracy and performance.

### 🎯 Key Features

- **Hybrid Search Architecture**
  - FAISS vector search with cosine similarity
  - Neo4j graph database for relationship understanding
  - Cross-encoder re-ranking for precision
  - Parallel processing for high-performance ingestion

- **Advanced Document Processing**
  - Multi-format support (PDF, CSV, PPTX)
  - Intelligent chunking with overlap
  - Parallel batch processing
  - Memory-efficient streaming

- **Production-Ready Features**
  - Session isolation for security
  - Auto-cleanup for privacy
  - Progress monitoring
  - Error handling and recovery

### 🏗️ Architecture Deep Dive

#### 1. Document Ingestion Pipeline
- **Parallel Processing**: Uses ProcessPoolExecutor for CPU-intensive tasks
- **Batch Optimization**: Dynamic batch sizing based on available cores
- **Memory Management**: Streaming approach for large files
- **Vector Normalization**: L2 normalization for stable cosine similarity

#### 2. Search Implementation
- **Primary Search**: FAISS with IndexFlatIP for fast cosine similarity
- **Secondary Search**: Neo4j graph traversal for relationship context
- **Re-ranking**: Cross-encoder for high-precision result refinement
- **Result Fusion**: Weighted combination of vector and graph results

#### 3. Performance Optimizations
- **Caching**: Strategic use of Streamlit caching for models
- **Resource Management**: Dynamic worker allocation
- **Batch Processing**: Optimal chunk sizes for parallel processing
- **Memory Efficiency**: Stream processing for large documents

### 🤔 Key Design Decisions

1. **Vector Search Implementation**
   - Chose FAISS over Milvus (tried but didn't like)
     * Better performance at scale
     * Memory efficiency
     * Cosine similarity support
   
2. **Graph Database Choice**
   - Selected Neo4j for:
     * Native graph operations
     * Relationship modeling
     * Query flexibility

3. **Re-ranking Strategy**
   - Implemented cross-encoder because:
     * Higher accuracy than bi-encoders
     * Better semantic understanding
     * Worth the computational trade-off

4. **Processing Architecture**
   - Parallel processing with:
     * Process-based parallelism for CPU tasks
     * Thread-based for I/O operations
     * Dynamic batch sizing
