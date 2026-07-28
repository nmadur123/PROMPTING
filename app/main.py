"""sys42 — startup g'oyasidan tayyor texnik topshiriq va promptgacha."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.ml.engine import engine
from app.routers import admin_billing, admin_stats, auth, blueprint, content, estimates, models, payments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


def _check_production_config() -> None:
    """Xavfli standart sozlamalar bilan ishga tushishni to'sadi.

    `secret_key` standart qiymatda qolsa, JWT ommaga ma'lum kalit bilan
    imzolanadi — istalgan odam o'ziga superadmin token yasay oladi. Serverga
    `.env` siz deploy qilinsa bu jimgina sodir bo'ladi, shuning uchun ilova
    umuman ko'tarilmasligi kerak.

    Lokal ishlab chiqishga xalaqit bermaydi: `.env` da kalit bor.
    """
    problems: list[str] = []

    if settings.secret_key in ("", "change-me"):
        problems.append(
            "SECRET_KEY standart qiymatda. Yangi kalit yarating: "
            "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )
    elif len(settings.secret_key) < 32:
        logger.warning(
            "SECRET_KEY qisqa (%d belgi) — kamida 32 belgi tavsiya etiladi",
            len(settings.secret_key),
        )

    if "*" in settings.cors_list:
        problems.append("CORS_ORIGINS da `*` bor — allow_credentials bilan birga xavfli.")

    if problems:
        bullets = "\n  - ".join(problems)
        raise RuntimeError(
            f"Ishga tushirish to'xtatildi — sozlamalarni to'g'rilang:\n  - {bullets}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_production_config()
    await init_db()
    # ML modelini oldindan yuklaymiz — birinchi foydalanuvchi kutmasin.
    try:
        engine.warmup()
        logger.info("ML yadro tayyor: %s", engine.stats())
    except Exception:  # noqa: BLE001 — ML tayyor bo'lmasa ham API ko'tarilsin
        logger.exception("ML yadro yuklanmadi — /api/analyze ishlamaydi")
    yield


app = FastAPI(
    title="sys42 API",
    description=(
        "Startup g'oyasini tahlil qilib, stack / server / UI-UX / domen / to'lov "
        "tavsiyalarini beradi va tanlangan AI model uchun tayyor prompt yozadi."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin_stats.router)
app.include_router(estimates.router)
app.include_router(content.router)
app.include_router(blueprint.router)
app.include_router(models.router)
app.include_router(admin_billing.router)
app.include_router(payments.router)


@app.get("/api/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok", "service": "prompting"}
