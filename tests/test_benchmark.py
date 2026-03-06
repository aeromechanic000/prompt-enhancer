"""Tests for knowledge endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
import pytest_asyncio
import tempfile
from pathlib import Path

from app.main import app


@pytest_asyncio.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


class TestKnowledgeEndpoints:
    """Tests for knowledge base management endpoints."""

    @pytest.mark.asyncio
    async def test_get_knowledge_status(self, async_client):
        """Test getting knowledge base status."""
        response = await async_client.get("/api/v1/knowledge/status")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "stats" in data
        assert "total_documents" in data["stats"]
        assert "total_chunks" in data["stats"]

    @pytest.mark.asyncio
    async def test_clear_knowledge_base(self, async_client):
        """Test clearing knowledge base."""
        response = await async_client.delete("/api/v1/knowledge")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_directory(self, async_client):
        """Test ingesting from non-existent directory."""
        response = await async_client.post(
            "/api/v1/knowledge/ingest",
            json={
                "directory": "/nonexistent/path",
                "recursive": True,
                "chunk_size": 500
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_ingest_empty_directory(self, async_client):
        """Test ingesting from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            response = await async_client.post(
                "/api/v1/knowledge/ingest",
                json={
                    "directory": tmpdir,
                    "recursive": True,
                    "chunk_size": 500
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["result"]["indexed_files"] == 0
            assert data["result"]["total_chunks"] == 0

    @pytest.mark.asyncio
    async def test_ingest_with_files(self, async_client):
        """Test ingesting from directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("This is test content for the knowledge base. " * 10)

            response = await async_client.post(
                "/api/v1/knowledge/ingest",
                json={
                    "directory": tmpdir,
                    "recursive": True,
                    "chunk_size": 200
                }
            )

            # This test requires Ollama to be running for embeddings
            # If Ollama is not available, it will return an error
            assert response.status_code == 200
            data = response.json()
            # Either success with indexed files or failure due to Ollama
            if data["success"]:
                assert data["result"]["indexed_files"] >= 1

    @pytest.mark.asyncio
    async def test_submit_rating(self, async_client):
        """Test submitting a rating."""
        response = await async_client.post(
            "/api/v1/knowledge/rate",
            json={
                "original_prompt": "test prompt",
                "enhanced_prompt": "enhanced test prompt",
                "rating": 4,
                "feedback": "Good enhancement"
            }
        )

        # This test requires Ollama for embeddings
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_rating_invalid(self, async_client):
        """Test submitting an invalid rating."""
        response = await async_client.post(
            "/api/v1/knowledge/rate",
            json={
                "original_prompt": "test",
                "enhanced_prompt": "enhanced",
                "rating": 6,  # Invalid: must be 1-5
            }
        )

        assert response.status_code == 422  # Validation error
