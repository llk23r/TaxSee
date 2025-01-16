import streamlit as st
import pdfplumber
import pandas as pd
from pptx import Presentation
from io import StringIO, BytesIO
import os
import time
from datetime import datetime
import io
import openai
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
from typing import List, Tuple
import multiprocessing
from functools import partial
import concurrent.futures
from threading import Lock

from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer, CrossEncoder


import faiss
import numpy as np

from neo4j import GraphDatabase

from fast_processor import FastDocumentProcessor

load_dotenv()

HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "taxrag_dev_password")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log_lock = Lock()


def log_status(message: str, level: str = "info"):
    with log_lock:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"{timestamp} - {message}"
        print(log_msg)

        try:
            if level == "error":
                st.sidebar.error(f"❌ {message}")
            elif level == "warning":
                st.sidebar.warning(f"⚠️ {message}")
            elif level == "success":
                st.sidebar.success(f"✅ {message}")
            else:
                st.sidebar.info(f"ℹ️ {message}")
        except:
            pass


@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    """Load and cache the sentence transformer model for embeddings."""
    log_status(f"Loading embedding model: {model_name}")
    start_time = time.time()
    embedder = SentenceTransformer(model_name)
    load_time = time.time() - start_time
    log_status(f"Embedding model loaded in {load_time:.2f}s", level="success")
    return embedder


@st.cache_resource(show_spinner=False)
def load_cross_encoder(model_name: str):
    """Load and cache a cross-encoder model for re-ranking."""
    log_status(f"Loading cross-encoder model: {model_name}")
    start_time = time.time()
    cross_encoder = CrossEncoder(model_name)
    load_time = time.time() - start_time
    log_status(f"Cross-encoder loaded in {load_time:.2f}s", level="success")
    return cross_encoder


# Initialize models
embedder = load_embedder(HF_EMBED_MODEL)
cross_encoder = load_cross_encoder(CROSS_ENCODER_MODEL)
EMBED_DIM = embedder.get_sentence_embedding_dimension()


def initialize_session_resources():
    """Initialize or get FAISS resources from session state."""
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    if "faiss_docs" not in st.session_state:
        st.session_state.faiss_docs = []
    return st.session_state.faiss_index, st.session_state.faiss_docs



@st.cache_resource(show_spinner=False)
def init_neo4j_driver():
    """Initialize and return the Neo4j driver instance."""
    log_status("Connecting to Neo4j...")
    start_time = time.time()
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            if result.single()["num"] == 1:
                setup_time = time.time() - start_time
                log_status(f"Neo4j connected in {setup_time:.2f}s", level="success")
                return driver
    except Exception as e:
        log_status(f"Failed to connect to Neo4j: {str(e)}", level="error")
        return None

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        chunk_text = " ".join(chunk)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks


def embed_chunks_batch(chunks: List[str], model_name: str) -> List[np.ndarray]:
    """Embed a batch of chunks using a local embedder instance."""
    local_embedder = SentenceTransformer(model_name)
    return [local_embedder.encode(chunk).astype(np.float32) for chunk in chunks]


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors for cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]


def ingest_into_faiss(
    faiss_index: faiss.Index,
    docstore: list,
    content: str,
    source: str,
    batch_size: int = 1000,
):
    """
    Parallel ingestion of text into FAISS index using process pool for embedding.
    """
    start_time = time.time()
    log_status(f"Ingesting content from: {source}")

    text_chunks = chunk_text(content)
    total_chunks = len(text_chunks)

    log_status(f"Created {total_chunks} chunks for {source}")

    num_cores = multiprocessing.cpu_count()
    num_processes = min(num_cores - 1 or 1, 4)
    batch_size = min(batch_size, max(100, total_chunks // (num_processes * 2)))
    chunk_batches = [
        text_chunks[i : i + batch_size] for i in range(0, len(text_chunks), batch_size)
    ]
    all_vectors = []
    new_docstore_entries = []
    embed_fn = partial(embed_chunks_batch, model_name=HF_EMBED_MODEL)
    processed_chunks_count = 0
    total_batches = len(chunk_batches)

    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_batch = {
                executor.submit(embed_fn, batch): (i, batch)
                for i, batch in enumerate(chunk_batches)
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                i, batch = future_to_batch[future]
                try:
                    batch_vectors = future.result()
                    all_vectors.extend(batch_vectors)

                    for chunk in batch:
                        new_docstore_entries.append(
                            {"content": chunk, "source": source}
                        )

                    processed_chunks_count += len(batch)
                    progress = f"[{processed_chunks_count}/{total_chunks}] "
                    progress += f"Batch {i + 1}/{total_batches} complete"
                    log_status(
                        f"{progress} ({processed_chunks_count / total_chunks * 100:.1f}%)"
                    )

                except Exception as e:
                    log_status(f"Error processing batch {i}: {str(e)}", level="error")
                    continue

        if all_vectors:
            vectors_np = np.vstack(all_vectors)
            vectors_np = normalize_vectors(vectors_np)
            faiss_index.add(vectors_np)
            docstore.extend(new_docstore_entries)

        total_time = time.time() - start_time
        chunks_per_second = total_chunks / total_time
        log_status(
            f"Finished FAISS ingestion in {total_time:.2f}s ({chunks_per_second:.1f} chunks/sec)",
            level="success",
        )

    except Exception as e:
        log_status(f"Error during parallel processing: {str(e)}", level="error")
        raise


def ingest_into_neo4j(driver, content: str, source: str):
    if not driver:
        log_status(
            "Neo4j driver not available, skipping graph ingestion", level="warning"
        )
        return

    start_time = time.time()
    log_status(f"Ingesting into Neo4j: {source}")

    try:
        text_chunks = chunk_text(content)
        with driver.session() as session:
            session.run("MERGE (d:Document {name: $source}) RETURN d", source=source)
            for idx, chunk in enumerate(text_chunks, start=1):
                session.run(
                    """
                    MERGE (c:Chunk {content: $chunk, chunk_id: $chunk_id})
                    WITH c
                    MATCH (d:Document {name: $source})
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    chunk=chunk[:1000],
                    chunk_id=idx,
                    source=source,
                )
                if idx % 10 == 0:
                    log_status(
                        f"Processed {idx}/{len(text_chunks)} Neo4j chunks for {source}"
                    )

        process_time = time.time() - start_time
        log_status(f"Neo4j ingestion complete in {process_time:.2f}s", level="success")

    except Exception as e:
        log_status(f"Error during Neo4j ingestion: {str(e)}", level="error")
        raise


def vector_search_faiss(
    faiss_index: faiss.Index, docstore: list, query: str, top_k: int = 10
):
    """Perform cosine similarity search in FAISS using normalized vectors with inner product."""
    query_emb = embedder.encode(query).astype(np.float32)
    query_emb = normalize_vectors(query_emb.reshape(1, -1))
    similarities, indexes = faiss_index.search(query_emb, top_k)
    results = []
    if indexes.shape[1] > 0:
        for i, idx in enumerate(indexes[0]):
            if idx < 0 or idx >= len(docstore):
                continue
            content = docstore[idx]["content"]
            source = docstore[idx]["source"]
            score = float(similarities[0][i])
            results.append((content, source, score))
    return results


def neo4j_graph_search(driver, query: str, limit: int = 10):
    if not driver:
        log_status("Neo4j driver not available, skipping graph search", level="warning")
        return []

    start_time = time.time()
    try:
        query_pattern = f".*{query}.*"
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.content =~ $regex
                RETURN c.content AS content
                LIMIT $limit
                """,
                regex=query_pattern,
                limit=limit,
            )
            found = [record["content"] for record in result]

        search_time = time.time() - start_time
        log_status(f"Graph search completed in {search_time:.2f}s", level="success")
        # Add a default score of 0.5 for graph results
        return [(txt, "graph_db", 0.5) for txt in found]
    except Exception as e:
        log_status(f"Error during graph search: {str(e)}", level="error")
        return []


def cross_encoder_rerank(
    query: str, candidates: list[tuple[str, str, float]], max_candidates: int = 10
):
    """
    Rerank candidates using cross-encoder.
    Candidates should be list of tuples (content, source, score).
    """
    if not candidates:
        return []

    pair_inputs = [(query, content) for (content, source, _) in candidates]
    scores = cross_encoder.predict(pair_inputs)

    combined = []
    for (content, source, _), score in zip(candidates, scores):
        combined.append((content, source, float(score)))

    combined.sort(key=lambda x: x[2], reverse=True)
    return combined[:max_candidates]


def generate_answer(
    query: str, context: list[tuple[str, str, float]], conversation_history: str = ""
) -> str:
    formatted_context = "\n\n".join(
        [
            f"Source: {source}\nContent: {content}\nRelevance: {score:.4f}"
            for content, source, score in context
        ]
    )

    system_prompt = """You are a friendly and helpful tax expert assistant. Use ONLY the provided context to answer questions.
If the context doesn't contain enough information to answer confidently, acknowledge this politely and:
1. Explain what specific information you would need to provide a complete answer
2. Suggest related questions the user might want to ask instead
3. Offer to help if they can provide more details or upload relevant documents

Format your response with clear citations:
- Use [Source: filename] to cite your sources inline
- Keep the response clear and well-structured with markdown
- Use bullet points and headers where appropriate
- Place citations right after the relevant information

Maintain a helpful and encouraging tone throughout your responses."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"""Context:
{formatted_context}

Conversation History:
{conversation_history}

Question: {query}

Please provide a detailed answer based on the context above. Include relevant citations.""",
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.3, max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        log_status(f"Error generating answer: {str(e)}", level="error")
        return f"Error generating answer. Retrieved context:\n\n{formatted_context}"


def rag_pipeline(
    faiss_index: faiss.Index,
    docstore: list,
    driver,
    user_query: str,
    conversation_history: str = "",
):
    # 1. Vector retrieval
    vector_results = vector_search_faiss(faiss_index, docstore, user_query, top_k=10)

    # 2. Graph retrieval
    graph_results = neo4j_graph_search(driver, user_query, limit=10)

    # 3. Combine & re-rank
    all_candidates = vector_results + graph_results
    if not all_candidates:
        return "No relevant information found. (No vector or graph matches.)"

    reranked = cross_encoder_rerank(user_query, all_candidates, max_candidates=10)

    # 4. Generate answer
    answer = generate_answer(user_query, reranked, conversation_history)

    if os.getenv("DEBUG", "false").lower() == "true":
        answer += "\n\n---\n### Debug Information\n"
        answer += f"**Query**: {user_query}\n\n"
        answer += "**Top Re-ranked Results**:\n"
        for i, (text, source, score) in enumerate(reranked[:3], 1):
            snippet = text[:150].replace("\n", " ")
            answer += (
                f"{i}. (score={score:.4f}) Source: {source} | Snippet: {snippet}...\n"
            )

    return answer


@st.cache_resource(show_spinner=False)
def get_document_processor():
    return FastDocumentProcessor(chunk_size=300, chunk_overlap=50)


def process_file_with_progress(
    processor: FastDocumentProcessor, file_obj, filename: str
) -> str:
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_container = st.empty()

        file_bytes = file_obj.read()
        file_stream = io.BytesIO(file_bytes)
        file_type = filename.split(".")[-1].lower()

        chunks = []
        total_chunks = 0

        for i, chunk in enumerate(processor.process_stream(file_stream, file_type), 1):
            chunks.append(chunk)
            total_chunks = i
            if i % 10 == 0:
                progress = min(0.95, i / (i + 10))
                progress_bar.progress(progress)
                status_text.text(f"Processing chunk {i}...")

                stats = processor.get_stats()
                stats_md = f"""
                **Processing Stats**:
                - Time: {stats["total_time_seconds"]:.1f}s
                - Speed: {stats["processing_speed_chars_per_sec"]:.0f} chars/sec
                - Memory: {stats["peak_memory_mb"]:.1f} MB
                - Chunks: {stats["num_chunks"]}
                - Method: {stats["extraction_method"]}
                """
                stats_container.markdown(stats_md)

        progress_bar.progress(1.0)
        status_text.text("Processing complete!")

        final_stats = processor.get_stats()
        stats_md = f"""
        **Final Processing Stats**:
        - Total Time: {final_stats["total_time_seconds"]:.1f}s
        - Extraction Time: {final_stats["extraction_time_seconds"]:.1f}s
        - Chunking Time: {final_stats["chunking_time_seconds"]:.1f}s
        - File Size: {final_stats["file_size_mb"]:.1f} MB
        - Total Chunks: {final_stats["num_chunks"]}
        - Avg Chunk Size: {final_stats["avg_chunk_size_chars"]:.0f} chars
        - Method: {final_stats["extraction_method"]}
        """
        stats_container.markdown(stats_md)

        return "\n\n".join(chunks)

    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()


def check_session_timeout():
    """Clear session data if it's too old."""
    timeout_minutes = 60  # Adjust this value
    if "last_activity" in st.session_state:
        elapsed = (time.time() - st.session_state.last_activity) / 60
        if elapsed > timeout_minutes:
            clear_session_data()
    st.session_state.last_activity = time.time()


def main():
    check_session_timeout()
    st.title("Advanced RAG Chatbot (FAISS + Neo4j) with Cross-Encoder Re-ranking")
    st.markdown(
        "This chatbot ingests PDF/CSV/PPTX files, performs a vector search with FAISS, "
        "an optional graph search with Neo4j, and then re-ranks results using a cross-encoder + GPT."
    )
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="batch_size",
        help="Number of chunks to process in parallel. Higher values use more memory but may be faster.",
    )
    faiss_index, faiss_docs = initialize_session_resources()
    driver = init_neo4j_driver()
    processor = get_document_processor()

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload multiple files (PDF, CSV, PPTX)",
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded_files and st.button("Process Files", key="process_files"):
            for uploaded_file in uploaded_files:
                filename = uploaded_file.name
                if filename in st.session_state.processed_files:
                    st.info(f"Already processed: {filename}")
                    continue

                st.write(f"Processing file: {filename}")
                try:
                    # 1. Extract + chunk text
                    content = process_file_with_progress(
                        processor, uploaded_file, filename
                    )

                    # 2. Ingest into FAISS
                    with st.spinner("Storing in FAISS index..."):
                        batch_size = st.session_state.batch_size
                        ingest_into_faiss(
                            faiss_index,
                            faiss_docs,
                            content,
                            filename,
                            batch_size=batch_size,
                        )

                    # 3. Ingest into Neo4j (optional)
                    with st.spinner("Storing in graph database..."):
                        ingest_into_neo4j(driver, content, source=filename)

                    st.session_state.processed_files.add(filename)
                    st.success(f"File {filename} processed and ingested.")

                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
            st.rerun()

        if st.session_state.processed_files:
            st.write("---")
            st.write("Processed Files:")
            for file in st.session_state.processed_files:
                st.write(f"✅ {file}")

        if st.button("Clear All Data", type="secondary"):
            clear_session_data()
            st.rerun()

        st.write("---")
        st.write("### Session Info")
        if "faiss_docs" in st.session_state:
            st.write(f"Vectors stored: {len(st.session_state.faiss_docs)}")
        if "last_activity" in st.session_state:
            elapsed = (time.time() - st.session_state.last_activity) / 60
            st.write(f"Session age: {elapsed:.1f} minutes")

    st.write("---")
    st.subheader("Chat Interface")

    for msg in st.session_state.conversation:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f"**You**: {content}")
        else:
            msg_col, source_col = st.columns([3, 1])
            with msg_col:
                st.markdown(f"**Assistant**: {content}")
            with source_col:
                if "Source:" in content:
                    with st.expander("📚 Sources", expanded=False):
                        sources = set()
                        for line in content.split("\n"):
                            if "Source:" in line:
                                s = line.split("Source:")[1].split("|")[0].strip()
                                sources.add(s)
                        for s in sorted(sources):
                            st.markdown(f"- `{s}`")

    user_query = st.text_input("Enter your question:", key="user_input")

    if st.button("Send", key="send_query"):
        if not st.session_state.processed_files:
            st.warning("Please upload and process some files first.")
        elif user_query.strip():
            # Add user message
            st.session_state.conversation.append(
                {"role": "user", "content": user_query}
            )

            with st.spinner("Retrieving relevant information..."):
                start_time = time.time()
                conversation_history_str = "\n".join(
                    [
                        f"{msg['role'].upper()}: {msg['content']}"
                        for msg in st.session_state.conversation
                    ]
                )

                # Run the advanced RAG pipeline
                answer = rag_pipeline(
                    faiss_index,
                    faiss_docs,
                    driver,
                    user_query=user_query,
                    conversation_history=conversation_history_str,
                )
                query_time = time.time() - start_time
                log_status(f"Query processed in {query_time:.2f}s", level="success")

            st.session_state.conversation.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        else:
            st.warning("Please enter a question.")

    if len(uploaded_files) > 0:
        total_size_mb = sum(file.size / (1024 * 1024) for file in uploaded_files)
        if total_size_mb > 1000:  # 1GB
            st.warning(
                f"Warning: Processing {total_size_mb:.1f}MB of files may require significant memory and time."
            )


def clear_session_data():
    """Clear all vectors and documents from the current session."""
    if "faiss_index" in st.session_state:
        # Reset FAISS index
        st.session_state.faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    if "faiss_docs" in st.session_state:
        st.session_state.faiss_docs = []
    if "processed_files" in st.session_state:
        st.session_state.processed_files = set()
    if "conversation" in st.session_state:
        st.session_state.conversation = []
    st.success("All session data cleared!")


if __name__ == "__main__":
    main()
