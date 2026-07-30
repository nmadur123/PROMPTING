"""Raqobat tahlili endpointi."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.ml.engine import engine, extract_signals
from app.models.user import User
from app.routers.auth import current_user
from app.services import competitors

router = APIRouter(prefix="/api", tags=["Competitors"])


class CompetitorRequest(BaseModel):
    description: str = Field(min_length=20, max_length=6000)
    lang: str = Field(default="uz", pattern="^(uz|ru|en)$")
    top_k: int = Field(default=5, ge=1, le=10)


@router.post("/competitors", summary="O'xshash startuplar va ustunlik yo'llari")
async def analyze_competitors(
    body: CompetitorRequest,
    # Kirish talab qilinadi: bu tahlil pullik mahsulotning qismi va
    # ochiq qoldirilsa dataset butunlay ko'chirib olinishi mumkin.
    user: User = Depends(current_user),
) -> dict:
    """G'oyaga o'xshash o'zbek startuplarini topadi.

    Kvota yechilmaydi — bu chaqiruv tashqi xizmatga pul to'lamaydi, hammasi
    lokal indeks ustida ishlaydi.
    """
    if competitors.get_index() is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Raqobat dataseti hozircha yo'q. `python -m scripts.fetch_competitors`.",
        )

    prediction = engine.classifier.predict(body.description)
    result = competitors.analyze(
        description=body.description,
        project_type=prediction.label,
        signals=extract_signals(body.description),
        top_k=body.top_k,
    )
    return {
        "project_type": prediction.label,
        "project_title": prediction.title(body.lang),
        **result.to_dict(),
    }
