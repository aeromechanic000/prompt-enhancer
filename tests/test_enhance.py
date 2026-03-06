"""Tests for enhance endpoint."""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
import pytest_asyncio

from app.main import app


@pytest_asyncio.fixture
async def async_client():
    """Create async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client


class TestEnhanceEndpoint:
    """Tests for /api/v1/enhance endpoint."""

    @pytest.mark.asyncio
    async def test_enhance_prompt_success(self, async_client):
        """Test successful prompt enhancement."""
        # This test requires Ollama to be running
        # Skip if Ollama is not available
        response = await async_client.post(
            "/api/v1/enhance",
            json={"prompt": "write a function to sort a list", "use_rag": False}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["result"]["original"] == "write a function to sort a list"
        assert len(data["result"]["enhanced"]) > 0
        assert len(data["result"]["intent"]) > 0

    @pytest.mark.asyncio
    async def test_enhance_prompt_empty(self, async_client):
        """Test enhancement with empty prompt fails validation."""
        response = await async_client.post(
            "/api/v1/enhance",
            json={"prompt": "", "use_rag": False}
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_enhance_prompt_with_rag(self, async_client):
        """Test enhancement with RAG enabled."""
        response = await async_client.post(
            "/api/v1/enhance",
            json={"prompt": "explain the project structure", "use_rag": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestBenchmarkEndpoint:
    """Tests for /api/v1/enhance/benchmark and /api/v1/benchmark endpoints."""

    @pytest.mark.asyncio
    async def test_benchmark_without_doubao(self, async_client):
        """Test benchmark fails gracefully when Doubao is not configured."""
        response = await async_client.post(
            "/api/v1/benchmark",
            json={"prompt": "test prompt", "use_rag": False}
        )

        # Should return 503 if Doubao is not configured
        # or 200 with success if it is configured
        assert response.status_code in [200, 503]

    @pytest.mark.asyncio
    async def test_benchmark_via_enhance_route(self, async_client):
        """Test benchmark via /enhance/benchmark route."""
        response = await async_client.post(
            "/api/v1/enhance/benchmark",
            json={"prompt": "test prompt", "use_rag": False}
        )

        assert response.status_code in [200, 503]


class TestHealthEndpoint:
    """Tests for health check endpoints."""

    @pytest.mark.asyncio
    async def test_root_health(self, async_client):
        """Test root endpoint returns health status."""
        response = await async_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client):
        """Test /health endpoint."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ollama_connected" in data
        assert "doubao_configured" in data
