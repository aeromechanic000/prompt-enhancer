"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import router
from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services.doubao_client import get_doubao_client
from app.services.ollama_client import get_ollama_client

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    yield
    # Shutdown - close clients
    ollama = get_ollama_client()
    doubao = get_doubao_client()
    await ollama.close()
    await doubao.close()


app = FastAPI(
    title="Prompt Enhancer",
    description="Local-first prompt enhancement middleware using SLM + RAG",
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic health info."""
    return HealthResponse(
        status="healthy",
        version=__version__,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with service status."""
    ollama = get_ollama_client()
    doubao = get_doubao_client()

    ollama_connected = await ollama.check_health()
    doubao_configured = doubao.is_configured()

    status = "healthy" if ollama_connected else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        ollama_connected=ollama_connected,
        doubao_configured=doubao_configured,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
