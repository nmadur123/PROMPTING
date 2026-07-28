"""OpenRouter model katalogi va ML holati."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.ml.engine import engine
from app.services import openrouter

router = APIRouter(prefix="/api", tags=["Models"])


@router.get("/models", summary="AI modellar ro'yxati")
async def list_models(
    only_popular: bool = Query(True, description="Faqat mashhur modellar"),
    family: str | None = Query(None, description="claude | openai | google | ..."),
) -> dict:
    """OpenRouter'dagi modellarni qaytaradi — foydalanuvchi shundan tanlaydi."""
    try:
        models = (
            await openrouter.client.popular_models()
            if only_popular
            else await openrouter.client.list_models()
        )
    except openrouter.OpenRouterError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if family:
        models = [m for m in models if m.family == family]

    return {
        "count": len(models),
        "families": sorted({m.family for m in models}),
        "models": [m.to_dict() for m in models],
    }


@router.get("/ml/stats", summary="ML yadro holati")
async def ml_stats() -> dict:
    """Korpus va model hajmi — 'ML haqiqatan bormi' degan savolga javob."""
    return engine.stats()
