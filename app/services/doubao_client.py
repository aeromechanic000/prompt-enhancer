"""Doubao API client for LLM inference."""

from typing import Optional

import httpx

from app.config import get_settings


class DoubaoClient:
    """Async client for Doubao API (Volcengine)."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.doubao_base_url
        self.api_key = self.settings.doubao_api_key
        self.model = self.settings.doubao_model
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def chat(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate chat completion using Doubao.

        Args:
            prompt: User message/prompt.
            system: System message for context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response text.

        Raises:
            httpx.HTTPError: If API request fails.
        """
        client = await self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return ""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text using Doubao.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        client = await self._get_client()

        payload = {
            "model": self.model,
            "input": text,
        }

        response = await client.post("/embeddings", json=payload)
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("data", [])
        if embeddings:
            return embeddings[0].get("embedding", [])
        return []

    def is_configured(self) -> bool:
        """Check if Doubao API is properly configured.

        Returns:
            True if API key is set.
        """
        return bool(self.api_key and self.api_key != "your_doubao_api_key_here")


# Singleton instance
_doubao_client: Optional[DoubaoClient] = None


def get_doubao_client() -> DoubaoClient:
    """Get singleton Doubao client instance."""
    global _doubao_client
    if _doubao_client is None:
        _doubao_client = DoubaoClient()
    return _doubao_client
