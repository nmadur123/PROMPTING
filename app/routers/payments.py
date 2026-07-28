"""To'lov endpointlari — provayder `.env` orqali tanlanadi."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.schemas.blueprint import PaymentCreateRequest
from app.services import payment_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/payments", tags=["Payments"])


@router.get("/status", summary="Qaysi provayder faol")
async def payment_status() -> dict:
    """Faol provayder va u haqiqiy pul o'tkazadimi.

    `live` — eng muhim maydon: `false` bo'lsa hech qanday pul harakati yo'q.
    Sozlanmagan Payme/Click mock'ga tushadi, shuning uchun `provider` ning
    o'zi yetarli signal emas.
    """
    settings = get_settings()
    provider = payment_service.get_provider()
    requested = (settings.payment_provider or "mock").lower()
    live = provider.name != "mock"

    if live:
        note = f"{provider.name} sozlangan — to'lovlar haqiqiy."
    elif requested == "mock":
        note = "Mock rejim: haqiqiy pul o'tkazilmaydi (.env da PAYMENT_PROVIDER=mock)."
    else:
        note = (
            f"'{requested}' tanlangan, lekin kalitlari to'ldirilmagan — mock rejimga "
            "tushdi. .env dagi merchant kalitlarini kiriting."
        )

    return {"provider": provider.name, "requested": requested, "live": live, "note": note}


@router.post("/create", summary="To'lov yaratish")
async def create_payment(body: PaymentCreateRequest) -> dict:
    provider = payment_service.get_provider(body.provider)
    try:
        intent = provider.create_intent(body.order_id, body.amount_uzs, body.return_url)
    except payment_service.PaymentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return intent.to_dict()


@router.post("/webhook/{provider_name}", summary="To'lov webhook'i")
async def payment_webhook(provider_name: str, request: Request) -> dict:
    """Provayderdan keladigan bildirishnoma.

    Imzo tekshirilmasa 403 qaytariladi — imzosiz webhook'ga ishonish
    to'lovni soxtalashtirishga to'g'ridan-to'g'ri yo'l ochadi.
    """
    provider = payment_service.get_provider(provider_name)
    try:
        body = await request.json()
    except ValueError:
        body = {}

    headers = {k.lower(): v for k, v in request.headers.items()}
    if not provider.verify_webhook(headers, body):
        logger.warning("To'lov webhook imzosi noto'g'ri: provider=%s", provider_name)
        raise HTTPException(status_code=403, detail="Imzo noto'g'ri")

    # Bu yerda haqiqiy mahsulotda buyurtma holati yangilanadi.
    # Idempotentlik: bir xil tranzaksiya id ikki marta hisoblanmasligi shart.
    logger.info("To'lov webhook qabul qilindi: provider=%s body=%s", provider_name, body)
    return {"ok": True}
