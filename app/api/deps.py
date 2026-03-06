"""Dependency injection for API routes."""

from app.services.doubao_client import get_doubao_client
from app.services.enhancer import get_enhancer_service
from app.services.ollama_client import get_ollama_client
from app.services.rag_service import get_rag_service


async def get_enhancer():
    """Get enhancer service dependency."""
    return get_enhancer_service()


async def get_ollama():
    """Get Ollama client dependency."""
    return get_ollama_client()


async def get_doubao():
    """Get Doubao client dependency."""
    return get_doubao_client()


async def get_rag():
    """Get RAG service dependency."""
    return get_rag_service()
