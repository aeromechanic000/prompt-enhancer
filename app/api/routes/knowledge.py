"""Knowledge base management endpoints."""

from fastapi import APIRouter, Depends

from app.models.schemas import (
    ClearKnowledgeResponse,
    DocumentResponse,
    DocumentUploadRequest,
    IngestResult,
    KnowledgeStats,
    KnowledgeStatusResponse,
    RatingRequest,
    RatingResponse,
)
from app.services.rag_service import RAGService, get_rag_service

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/ingest", response_model=DocumentResponse)
async def ingest_documents(
    request: DocumentUploadRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """Ingest documents from a directory into the knowledge base.

    Scans the specified directory for PDF, Markdown, and text files,
    parses their content, chunks it, and indexes it into LanceDB.

    Args:
        request: DocumentUploadRequest with directory path and options.
        rag: RAG service instance.

    Returns:
        DocumentResponse with ingestion results.
    """
    try:
        files_indexed, total_chunks, files_processed, errors = await rag.ingest_directory(
            directory=request.directory,
            recursive=request.recursive,
            chunk_size=request.chunk_size,
        )

        result = IngestResult(
            indexed_files=files_indexed,
            total_chunks=total_chunks,
            files_processed=files_processed,
            errors=errors,
        )

        return DocumentResponse(success=True, result=result)

    except Exception as e:
        return DocumentResponse(success=False, error=str(e))


@router.get("/status", response_model=KnowledgeStatusResponse)
async def get_knowledge_status(
    rag: RAGService = Depends(get_rag_service),
):
    """Get current knowledge base statistics.

    Args:
        rag: RAG service instance.

    Returns:
        KnowledgeStatusResponse with stats.
    """
    try:
        stats = await rag.get_stats()
        return KnowledgeStatusResponse(
            success=True,
            stats=KnowledgeStats(**stats),
        )
    except Exception as e:
        return KnowledgeStatusResponse(success=False, error=str(e))


@router.delete("", response_model=ClearKnowledgeResponse)
async def clear_knowledge_base(
    rag: RAGService = Depends(get_rag_service),
):
    """Clear all documents from the knowledge base.

    Args:
        rag: RAG service instance.

    Returns:
        ClearKnowledgeResponse confirming deletion.
    """
    try:
        await rag.clear()
        return ClearKnowledgeResponse(success=True)
    except Exception as e:
        return ClearKnowledgeResponse(success=False, error=str(e))


@router.post("/rate", response_model=RatingResponse)
async def submit_rating(
    request: RatingRequest,
    rag: RAGService = Depends(get_rag_service),
):
    """Submit a rating for an enhanced prompt.

    Stores ratings for future analysis and potential fine-tuning.

    Args:
        request: RatingRequest with prompt, enhancement, and rating.
        rag: RAG service instance.

    Returns:
        RatingResponse confirming rating was recorded.
    """
    try:
        await rag.add_rating(
            prompt=request.original_prompt,
            enhanced_prompt=request.enhanced_prompt,
            rating=request.rating,
            feedback=request.feedback,
        )
        return RatingResponse(success=True)
    except Exception as e:
        return RatingResponse(success=False, error=str(e))
