"""Pydantic models for API request/response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class EnhanceRequest(BaseModel):
    """Request for prompt enhancement."""

    prompt: str = Field(..., min_length=1, description="Raw user prompt to enhance")
    use_rag: bool = Field(True, description="Whether to use RAG for context injection")


class EnhancedPromptResult(BaseModel):
    """Result of prompt enhancement."""

    original: str
    enhanced: str
    intent: str
    contexts_used: list[str] = Field(default_factory=list)


class EnhanceResponse(BaseModel):
    """Response for prompt enhancement."""

    success: bool = True
    result: Optional[EnhancedPromptResult] = None
    error: Optional[str] = None


class BenchmarkRequest(BaseModel):
    """Request for benchmark comparison."""

    prompt: str = Field(..., min_length=1, description="Prompt to benchmark")
    use_rag: bool = Field(True, description="Whether to use RAG enhancement")


class BenchmarkResult(BaseModel):
    """Result of benchmark comparison."""

    original_prompt: str
    enhanced_prompt: str
    raw_response: str
    enhanced_response: str
    intent: str


class BenchmarkResponse(BaseModel):
    """Response for benchmark comparison."""

    success: bool = True
    result: Optional[BenchmarkResult] = None
    error: Optional[str] = None


class DocumentUploadRequest(BaseModel):
    """Request to ingest documents from a directory."""

    directory: str = Field(..., description="Path to directory containing documents")
    recursive: bool = Field(True, description="Whether to scan subdirectories")
    chunk_size: int = Field(500, ge=100, le=2000, description="Chunk size in characters")


class IngestResult(BaseModel):
    """Result of document ingestion."""

    indexed_files: int
    total_chunks: int
    files_processed: list[str]
    errors: list[str] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Response for document ingestion."""

    success: bool = True
    result: Optional[IngestResult] = None
    error: Optional[str] = None


class KnowledgeStats(BaseModel):
    """Statistics about the knowledge base."""

    total_documents: int
    total_chunks: int
    last_ingest: Optional[datetime] = None
    embedding_model: str


class KnowledgeStatusResponse(BaseModel):
    """Response for knowledge base status."""

    success: bool = True
    stats: Optional[KnowledgeStats] = None
    error: Optional[str] = None


class ClearKnowledgeResponse(BaseModel):
    """Response for clearing knowledge base."""

    success: bool = True
    message: str = "Knowledge base cleared"
    error: Optional[str] = None


class RatingRequest(BaseModel):
    """Request to submit a rating for an enhancement."""

    original_prompt: str
    enhanced_prompt: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (bad) to 5 (excellent)")
    feedback: Optional[str] = None


class RatingResponse(BaseModel):
    """Response for rating submission."""

    success: bool = True
    message: str = "Rating recorded"
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str = "healthy"
    version: str = "0.1.0"
    ollama_connected: bool = False
    doubao_configured: bool = False
