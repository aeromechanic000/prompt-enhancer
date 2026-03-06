"""Enhance endpoint for prompt improvement."""

from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    BenchmarkResult,
    EnhanceRequest,
    EnhanceResponse,
)
from app.services.doubao_client import DoubaoClient, get_doubao_client
from app.services.enhancer import EnhancerService, get_enhancer_service

router = APIRouter(prefix="/enhance", tags=["enhance"])


@router.post("", response_model=EnhanceResponse)
async def enhance_prompt(
    request: EnhanceRequest,
    enhancer: EnhancerService = Depends(get_enhancer_service),
):
    """Enhance a raw prompt using SLM and RAG.

    Args:
        request: EnhanceRequest with prompt and options.
        enhancer: Enhancer service instance.

    Returns:
        EnhanceResponse with enhanced prompt and metadata.
    """
    try:
        result = await enhancer.enhance(
            prompt=request.prompt,
            use_rag=request.use_rag,
        )
        return EnhanceResponse(success=True, result=result)
    except Exception as e:
        return EnhanceResponse(success=False, error=str(e))


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark_prompt(
    request: BenchmarkRequest,
    enhancer: EnhancerService = Depends(get_enhancer_service),
    doubao: DoubaoClient = Depends(get_doubao_client),
):
    """Benchmark raw vs enhanced prompt with Doubao.

    Performs a shadow test by sending both raw and enhanced prompts
    to Doubao and returning side-by-side comparison.

    Args:
        request: BenchmarkRequest with prompt.
        enhancer: Enhancer service instance.
        doubao: Doubao client instance.

    Returns:
        BenchmarkResponse with both responses for comparison.
    """
    if not doubao.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Doubao API is not configured. Set DOUBAO_API_KEY in environment.",
        )

    try:
        # Enhance the prompt
        enhanced_result = await enhancer.enhance(
            prompt=request.prompt,
            use_rag=request.use_rag,
        )

        # Get responses from Doubao for both versions
        raw_response = await doubao.chat(request.prompt)
        enhanced_response = await doubao.chat(enhanced_result.enhanced)

        result = BenchmarkResult(
            original_prompt=request.prompt,
            enhanced_prompt=enhanced_result.enhanced,
            raw_response=raw_response,
            enhanced_response=enhanced_response,
            intent=enhanced_result.intent,
        )

        return BenchmarkResponse(success=True, result=result)

    except Exception as e:
        return BenchmarkResponse(success=False, error=str(e))
