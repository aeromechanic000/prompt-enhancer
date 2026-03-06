"""File parsing utilities for PDF, MD, and TXT files."""

import os
from pathlib import Path
from typing import Optional

import tiktoken
from pypdf import PdfReader


def parse_file(file_path: str) -> str:
    """Extract text content from a file.

    Supports PDF, Markdown, and plain text files.

    Args:
        file_path: Path to the file to parse.

    Returns:
        Extracted text content.

    Raises:
        ValueError: If file type is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_path)
    elif suffix in (".md", ".markdown"):
        return _parse_markdown(file_path)
    elif suffix == ".txt":
        return _parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n\n".join(text_parts)


def _parse_markdown(file_path: str) -> str:
    """Read markdown file content."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def _parse_text(file_path: str) -> str:
    """Read plain text file content."""
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split text into chunks based on character count with overlap.

    Args:
        text: Text to split into chunks.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.
        encoding_name: Tiktoken encoding name for token counting.

    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []

    # Clean up the text
    text = text.strip()

    # Simple character-based chunking with overlap
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to find a good break point (newline or space)
        if end < text_len:
            # Look for a natural break point within the last 100 chars
            break_region = text[end - 100 : end]
            newline_pos = break_region.rfind("\n")
            space_pos = break_region.rfind(" ")

            # Prefer newline break, then space break
            if newline_pos > 50:
                end = end - 100 + newline_pos + 1
            elif space_pos > 50:
                end = end - 100 + space_pos + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < text_len else end

    return chunks


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count the number of tokens in text.

    Args:
        text: Text to count tokens for.
        encoding_name: Tiktoken encoding name.

    Returns:
        Number of tokens.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate count
        return len(text) // 4


def scan_directory(
    directory: str,
    recursive: bool = True,
    extensions: Optional[list[str]] = None,
) -> list[str]:
    """Scan directory for supported files.

    Args:
        directory: Path to directory to scan.
        recursive: Whether to scan subdirectories.
        extensions: File extensions to include (default: .pdf, .md, .txt).

    Returns:
        List of file paths found.
    """
    if extensions is None:
        extensions = [".pdf", ".md", ".markdown", ".txt"]

    dir_path = Path(directory)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    files = []
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for ext in extensions:
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                files.append(str(file_path))

    return sorted(set(files))
