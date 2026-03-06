"""Tests for RAG service."""

import pytest
import tempfile
from pathlib import Path

from app.services.rag_service import RAGService
from app.utils.file_parser import chunk_text, parse_file, scan_directory


class TestFileParser:
    """Tests for file parsing utilities."""

    def test_parse_text_file(self, tmp_path):
        """Test parsing a plain text file."""
        text_file = tmp_path / "test.txt"
        text_file.write_text("Hello, this is a test file.\nWith multiple lines.")

        content = parse_file(str(text_file))
        assert "Hello, this is a test file" in content
        assert "With multiple lines" in content

    def test_parse_markdown_file(self, tmp_path):
        """Test parsing a markdown file."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nThis is **bold** text.")

        content = parse_file(str(md_file))
        assert "# Title" in content
        assert "**bold**" in content

    def test_parse_unsupported_file(self, tmp_path):
        """Test parsing an unsupported file type raises error."""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_file(str(unsupported_file))

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "This is a test. " * 100  # Long text
        chunks = chunk_text(text, chunk_size=200, overlap=20)

        assert len(chunks) > 1
        assert all(len(chunk) <= 250 for chunk in chunks)  # Allow some flexibility for word boundaries

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("")
        assert chunks == []

    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_scan_directory(self, tmp_path):
        """Test scanning directory for supported files."""
        # Create test files
        (tmp_path / "doc1.txt").write_text("content1")
        (tmp_path / "doc2.md").write_text("content2")
        (tmp_path / "ignore.xyz").write_text("ignored")

        # Create subdirectory with file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc3.txt").write_text("content3")

        files = scan_directory(str(tmp_path), recursive=True)

        assert len(files) == 3
        assert any("doc1.txt" in f for f in files)
        assert any("doc2.md" in f for f in files)
        assert any("doc3.txt" in f for f in files)
        assert not any("ignore.xyz" in f for f in files)

    def test_scan_directory_non_recursive(self, tmp_path):
        """Test non-recursive directory scanning."""
        (tmp_path / "doc1.txt").write_text("content1")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "doc2.txt").write_text("content2")

        files = scan_directory(str(tmp_path), recursive=False)

        assert len(files) == 1
        assert "doc1.txt" in files[0]

    def test_scan_nonexistent_directory(self):
        """Test scanning non-existent directory raises error."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            scan_directory("/nonexistent/path")


class TestRAGService:
    """Tests for RAG service."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self):
        """Test getting stats from empty knowledge base."""
        rag = RAGService()
        stats = await rag.get_stats()

        assert stats["total_documents"] >= 0
        assert stats["total_chunks"] >= 0
        assert "embedding_model" in stats

    @pytest.mark.asyncio
    async def test_clear_knowledge_base(self):
        """Test clearing the knowledge base."""
        rag = RAGService()
        await rag.clear()

        stats = await rag.get_stats()
        assert stats["total_chunks"] == 0
