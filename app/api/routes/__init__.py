"""API routes package."""

from fastapi import APIRouter

from app.api.routes.benchmark import router as benchmark_router
from app.api.routes.enhance import router as enhance_router
from app.api.routes.knowledge import router as knowledge_router

# Main router that includes all sub-routers
router = APIRouter()

router.include_router(enhance_router)
router.include_router(benchmark_router)
router.include_router(knowledge_router)

__all__ = ["router"]
