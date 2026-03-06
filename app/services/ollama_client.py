"""Ollama API client for local SLM inference."""

from typing import Optional

import httpx

from app.config import get_settings


class OllamaClient:
    """Async client for Ollama API."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_base_url
        self.default_model = self.settings.ollama_model
        self.embedding_model = self.settings.embedding_model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,  # Long timeout for generation
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate text using Ollama.

        Args:
            prompt: Input prompt for generation.
            model: Model to use (default from settings).
            system: System prompt for context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text response.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        client = await self._get_client()
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system:
            payload["system"] = system

        response = await client.post("/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()

        return data.get("response", "").strip()

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed.
            model: Embedding model to use (default from settings).

        Returns:
            Embedding vector.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        client = await self._get_client()
        model = model or self.embedding_model

        payload = {
            "model": model,
            "input": text,
        }

        response = await client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]
        return []

    async def embed_batch(
        self,
        texts: list[str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            model: Embedding model to use.

        Returns:
            List of embedding vectors.
        """
        client = await self._get_client()
        model = model or self.embedding_model

        payload = {
            "model": model,
            "input": texts,
        }

        response = await client.post("/api/embed", json=payload)
        response.raise_for_status()
        data = response.json()

        return data.get("embeddings", [])

    async def check_health(self) -> bool:
        """Check if Ollama server is healthy.

        Returns:
            True if server is responding.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names.
        """
        client = await self._get_client()
        response = await client.get("/api/tags")
        response.raise_for_status()
        data = response.json()

        return [model["name"] for model in data.get("models", [])]


# Singleton instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get singleton Ollama client instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
