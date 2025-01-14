"""Document parser implementations."""

from pathlib import Path
from typing import List
import logging

from ..interfaces import DocumentParser, Document
from ..factory import ComponentRegistry
from ..config import DocumentConfig


class PDFParser(DocumentParser):
    """Basic PDF parser implementation using PyPDF2."""

    def __init__(self, config: DocumentConfig):
        """Initialize parser with configuration."""
        self.config = config
        try:
            import PyPDF2

            self.PyPDF2 = PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required. Install it with: pip install PyPDF2")

    async def parse(self, file_path: Path) -> Document:
        """Parse a PDF file into a Document object."""
        print(f"PDFParser: Parsing file {file_path}")
        print(f"PDFParser: File exists? {file_path.exists()}")
        print(f"PDFParser: File type: {type(file_path)}")
        print(f"PDFParser: File absolute path: {file_path.absolute()}")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size if configured
        if self.config.max_file_size_mb:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                raise ValueError(
                    f"File size {file_size_mb:.1f}MB exceeds maximum allowed size of {self.config.max_file_size_mb}MB"
                )

        text_content = []
        with open(file_path, "rb") as file:
            print("PDFParser: File opened successfully")
            pdf_reader = self.PyPDF2.PdfReader(file)
            print(f"PDFParser: PDF has {len(pdf_reader.pages)} pages")

            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                print(f"PDFParser: Extracted {len(text)} chars from page {i+1}")
                text_content.append(text)

        doc = Document(
            content="\n\n".join(text_content),
            source_path=file_path,
            metadata={
                "doc_id": file_path.stem,
                "filename": file_path.name,
                "num_pages": len(text_content),
            },
        )
        print(f"PDFParser: Created document with {len(doc.content)} chars")
        return doc


# Register implementations
ComponentRegistry.register_parser("pdf_parser", PDFParser)
