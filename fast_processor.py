"""
High-performance document processor for fast reading and chunking of various file types.
Minimal implementation focused on speed and efficiency.
"""

import io
import time
import psutil
import os
from pathlib import Path
from typing import Iterator, Union, List, Dict, Any
from dataclasses import dataclass
import pypdf
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.ppt import partition_ppt
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs
from tempfile import NamedTemporaryFile


@dataclass
class ProcessingStats:
    """Statistics for document processing."""

    total_time: float = 0.0
    extraction_time: float = 0.0
    chunking_time: float = 0.0
    peak_memory_mb: float = 0.0
    num_chunks: int = 0
    avg_chunk_size: float = 0.0
    total_chars: int = 0
    chars_per_second: float = 0.0
    file_size_mb: float = 0.0
    extraction_method: str = ""


class FastDocumentProcessor:
    """Fast document processor for reading and chunking files."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ".",
                "!",
                "?",
                ";",
                ":",
                ". ",
                " - ",
                "--",
                "\t",
                "  ",
                " ",
                "",
            ],
        )
        self.stats = ProcessingStats()
        self._start_time = 0.0
        self._extraction_start = 0.0
        self._chunking_start = 0.0

    def _start_benchmark(self) -> None:
        """Start benchmarking."""
        self.stats = ProcessingStats()
        self._start_time = time.time()
        self.stats.peak_memory_mb = self._get_memory_usage()

    def _update_memory_usage(self) -> None:
        """Update peak memory usage."""
        current_memory = self._get_memory_usage()
        self.stats.peak_memory_mb = max(self.stats.peak_memory_mb, current_memory)

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _update_stats(self, text: str, chunks: List[str]) -> None:
        """Update processing statistics."""
        self.stats.total_time = time.time() - self._start_time
        self.stats.extraction_time = time.time() - self._extraction_start
        self.stats.chunking_time = time.time() - self._chunking_start
        self.stats.num_chunks = len(chunks)
        self.stats.total_chars = len(text)
        self.stats.avg_chunk_size = (
            sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        )
        self.stats.chars_per_second = (
            self.stats.total_chars / self.stats.total_time
            if self.stats.total_time > 0
            else 0
        )
        self._update_memory_usage()

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics as a dictionary."""
        return {
            "total_time_seconds": round(self.stats.total_time, 3),
            "extraction_time_seconds": round(self.stats.extraction_time, 3),
            "chunking_time_seconds": round(self.stats.chunking_time, 3),
            "peak_memory_mb": round(self.stats.peak_memory_mb, 2),
            "num_chunks": self.stats.num_chunks,
            "avg_chunk_size_chars": round(self.stats.avg_chunk_size, 2),
            "total_chars": self.stats.total_chars,
            "processing_speed_chars_per_sec": round(self.stats.chars_per_second, 2),
            "file_size_mb": round(self.stats.file_size_mb, 2),
            "extraction_method": self.stats.extraction_method,
        }

    def _extract_text_from_pdf_fast(
        self, file_stream: Union[io.BytesIO, io.StringIO]
    ) -> Union[str, None]:
        """Extract text from PDF using pypdf (faster but less accurate for complex PDFs)."""
        try:
            reader = pypdf.PdfReader(file_stream)
            text_parts = []
            total_pages = len(reader.pages)

            # Process pages in chunks to manage memory
            chunk_size = 50  # Process 50 pages at a time
            for start_idx in range(0, total_pages, chunk_size):
                end_idx = min(start_idx + chunk_size, total_pages)
                chunk_text = []

                for i in range(start_idx, end_idx):
                    try:
                        page_text = reader.pages[i].extract_text() or ""
                        if page_text.strip():
                            chunk_text.append(page_text)
                    except Exception as e:
                        print(f"Error extracting page {i}: {e}")
                        continue

                if chunk_text:
                    text_parts.extend(chunk_text)

                # Free up memory
                chunk_text.clear()

            if not text_parts:
                return None  # Signal that we need to fall back to unstructured

            self.stats.extraction_method = "pypdf"
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"pypdf extraction failed: {e}")
            return None

    def process_stream(
        self, file_stream: Union[io.BytesIO, io.StringIO], file_type: str
    ) -> Iterator[str]:
        """Process a file stream and yield chunks."""
        self._start_benchmark()
        self._extraction_start = time.time()
        text = self._extract_text(file_stream, file_type)

        self._chunking_start = time.time()
        chunks = list(self._chunk_text(text))
        self._update_stats(text, chunks)

        yield from chunks

    def process_file(self, file_path: Union[str, Path]) -> Iterator[str]:
        """Process a file and yield chunks."""
        file_path = Path(file_path)
        file_type = file_path.suffix.lower()[1:]  # Remove the dot

        # Get file size
        self.stats.file_size_mb = file_path.stat().st_size / 1024 / 1024

        with open(file_path, "rb") as f:
            file_stream = io.BytesIO(f.read())
            yield from self.process_stream(file_stream, file_type)

    def _extract_text(
        self, file_stream: Union[io.BytesIO, io.StringIO], file_type: str
    ) -> str:
        """Extract text from various file types."""
        if file_type == "pdf":
            # Try fast extraction first
            text = self._extract_text_from_pdf_fast(file_stream)
            if text:
                return text

            # Fall back to unstructured for complex PDFs
            self.stats.extraction_method = "unstructured"
            with NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_file:
                if isinstance(file_stream, io.BytesIO):
                    tmp_file.write(file_stream.getvalue())
                else:
                    tmp_file.write(file_stream.getvalue().encode("utf-8"))
                tmp_file.flush()
                file_stream.seek(0)  # Reset stream position

                # Process large PDFs in chunks
                elements = []
                try:
                    elements = partition_pdf(
                        filename=tmp_file.name,
                        strategy="hi_res",
                        include_page_breaks=True,
                        extract_images_in_pdf=False,  # Skip images for speed
                        infer_table_structure=False,  # Skip tables for speed
                        max_partition=50,  # Process max 50 pages at a time
                    )
                except Exception as e:
                    print(f"Error in unstructured PDF processing: {e}")
                    # Try fast mode as last resort
                    elements = partition_pdf(
                        filename=tmp_file.name,
                        strategy="fast",
                        include_page_breaks=True,
                        extract_images_in_pdf=False,
                        infer_table_structure=False,
                    )

                text = "\n\n".join(str(element) for element in elements)

                # Clear elements to free memory
                elements.clear()

        elif file_type in ("pptx", "ppt"):
            # Always use unstructured for presentations - more reliable
            self.stats.extraction_method = "unstructured"
            suffix = ".pptx" if file_type == "pptx" else ".ppt"
            partition_func = partition_pptx if file_type == "pptx" else partition_ppt

            with NamedTemporaryFile(suffix=suffix, delete=True) as tmp_file:
                if isinstance(file_stream, io.BytesIO):
                    tmp_file.write(file_stream.getvalue())
                else:
                    tmp_file.write(file_stream.getvalue().encode("utf-8"))
                tmp_file.flush()
                file_stream.seek(0)

                try:
                    elements = partition_func(
                        filename=tmp_file.name, include_page_breaks=True
                    )
                    text = "\n\n".join(str(element) for element in elements)
                    if not text.strip():
                        raise ValueError("No text extracted")
                except Exception as e:
                    print(f"Presentation extraction failed: {e}")
                    text = "Error: Could not extract text from presentation file."

        elif file_type == "csv":
            self.stats.extraction_method = "pandas"
            df = pd.read_csv(file_stream)
            text = df.to_string()

        elif file_type == "txt":
            self.stats.extraction_method = "direct"
            if isinstance(file_stream, io.BytesIO):
                text = file_stream.read().decode("utf-8")
            else:
                text = file_stream.read()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Clean the extracted text
        text = clean_extra_whitespace(text)
        text = group_broken_paragraphs(text)
        return text

    def _chunk_text(self, text: str) -> Iterator[str]:
        """Chunk text using langchain's text splitter."""
        chunks = self.text_splitter.split_text(text)
        yield from chunks
