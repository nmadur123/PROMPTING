"""Linkup orqali jonli web-tadqiqot — raqobatchilarni internetdan topadi.

Nega kerak. Lokal dataset (`competitors.py`) faqat ikkita portfeldan yig'ilgan:
thepitch.uz va uzcombinator.uz. Ular akseleratorga kirgan startuplar, ya'ni
bozorning kichik qismi. Sinovda "restoran uchun stol band qilish" so'roviga
Linkup NoAsk va Stolik ni topdi — ikkalasi ham haqiqiy, ishlayotgan mahsulot,
lekin portfel bazasida umuman yo'q.

Ikkisi bir-birini almashtirmaydi, to'ldiradi:
  lokal  — tuzilgan, tez, bepul, aniq (nomi/havolasi/investitsiyasi bilan);
  Linkup — keng va jonli, lekin so'rov krediti sarflaydi.

DIQQAT — LLM haqida. `outputType` ikki xil bo'ladi:
  "searchResults" — xom natijalar, hech qanday model ishlatilmaydi;
  "sourcedAnswer" — Linkup TOMONIDA model javobni yig'ib beradi.
Ikkinchisi ancha foydali (bir necha manbani bir joyga yig'adi), lekin u
model natijasi va shunday deb belgilanadi. Tanlov sozlamada.

Kalit yo'q bo'lsa xizmat butunlay o'chadi va generatsiya o'z yo'lida davom
etadi — jonli tadqiqot qulaylik, majburiy qadam emas.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from app.config import DATA_DIR, get_settings

logger = logging.getLogger(__name__)

_ENDPOINT = "https://api.linkup.so/v1/search"
_TIMEOUT = httpx.Timeout(90.0, connect=20.0)

# Kesh diskda. Sabab: har bir so'rov kredit sarflaydi, bir xil g'oya esa
# qayta-qayta yuboriladi (foydalanuvchi tahrirlaydi, qayta generatsiya
# qiladi). Keshsiz bitta g'oya uchun o'nlab so'rov ketardi.
_CACHE_DIR = DATA_DIR / "research_cache"
_CACHE_TTL_SECONDS = 14 * 24 * 3600  # startup bozori bir kunda o'zgarmaydi

# Manbalardan shuncha tasi olinadi. 20 ta keladi, lekin oxirgilari ko'pincha
# umumiy maqolalar bo'ladi va topshiriqni shovqin bilan to'ldiradi.
_MAX_SOURCES = 6


@dataclass
class ResearchSource:
    name: str
    url: str
    snippet: str = ""


@dataclass
class ResearchResult:
    query: str = ""
    answer: str = ""
    sources: list[ResearchSource] = field(default_factory=list)
    # `True` bo'lsa `answer` Linkup tomonidagi model tomonidan yozilgan.
    answer_is_model_generated: bool = False
    from_cache: bool = False
    ok: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "answer_is_model_generated": self.answer_is_model_generated,
            "sources": [
                {"name": s.name, "url": s.url, "snippet": s.snippet} for s in self.sources
            ],
            "from_cache": self.from_cache,
            "ok": self.ok,
            "error": self.error,
        }


# --------------------------------------------------------------------------- #
# Kesh
# --------------------------------------------------------------------------- #


def _cache_path(query: str, output_type: str, depth: str) -> "object":
    key = hashlib.sha256(f"{query}|{output_type}|{depth}".encode()).hexdigest()[:32]
    return _CACHE_DIR / f"{key}.json"


def _cache_read(path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - float(payload.get("stored_at", 0)) > _CACHE_TTL_SECONDS:
        return None
    return payload.get("data")


def _cache_write(path, data: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"stored_at": time.time(), "data": data}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        # Kesh yozilmasa ishlash davom etadi, faqat kredit ko'proq ketadi.
        logger.warning("Tadqiqot keshini yozib bo'lmadi: %s", exc)


# --------------------------------------------------------------------------- #
# So'rov
# --------------------------------------------------------------------------- #


def enabled() -> bool:
    settings = get_settings()
    return bool(settings.linkup_enabled and settings.linkup_api_key)


def search(query: str, *, depth: Optional[str] = None, output_type: Optional[str] = None) -> ResearchResult:
    """Linkup'ga bitta so'rov yuboradi (yoki keshdan oladi)."""
    settings = get_settings()
    if not enabled():
        return ResearchResult(query=query, error="Linkup o'chirilgan yoki kalit yo'q")

    depth = depth or settings.linkup_depth
    output_type = output_type or settings.linkup_output_type
    path = _cache_path(query, output_type, depth)

    cached = _cache_read(path)
    if cached is not None:
        result = _parse(cached, query, output_type)
        result.from_cache = True
        return result

    body = {
        "q": query,
        "depth": depth,
        "outputType": output_type,
        "includeImages": False,
    }
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                _ENDPOINT,
                headers={
                    "Authorization": f"Bearer {settings.linkup_api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
    except httpx.HTTPError as exc:
        return ResearchResult(query=query, error=f"Linkup'ga ulanib bo'lmadi: {exc}")

    if resp.status_code == 429:
        return ResearchResult(query=query, error="Linkup: so'rov limiti tugadi")
    if resp.status_code != 200:
        return ResearchResult(
            query=query, error=f"Linkup xatosi {resp.status_code}: {resp.text[:200]}"
        )

    try:
        data = resp.json()
    except ValueError:
        return ResearchResult(query=query, error="Linkup javobi JSON emas")

    _cache_write(path, data)
    return _parse(data, query, output_type)


def _parse(data: dict, query: str, output_type: str) -> ResearchResult:
    raw_sources = data.get("sources") or data.get("results") or []
    sources = [
        ResearchSource(
            name=str(s.get("name") or s.get("title") or "")[:200],
            url=str(s.get("url") or ""),
            snippet=" ".join(str(s.get("snippet") or "").split())[:300],
        )
        for s in raw_sources[:_MAX_SOURCES]
        if s.get("url")
    ]
    return ResearchResult(
        query=query,
        answer=str(data.get("answer") or "").strip(),
        sources=sources,
        answer_is_model_generated=(output_type == "sourcedAnswer"),
        ok=True,
    )


# --------------------------------------------------------------------------- #
# Raqobat tadqiqoti
# --------------------------------------------------------------------------- #

# So'rov ML aniqlagan turdan qurilади, foydalanuvchining xom matnidan emas.
#
# Nega: xom g'oya bir necha jumla bo'ladi va ichida qidiruv uchun keraksiz
# tafsilot ko'p ("usta tasdiqlaydi, SMS eslatma keladi"). Tur nomi esa aynan
# bozor nomi — qidiruv tizimi shu bilan yaxshi ishlaydi.
_TYPE_QUERY: dict[str, str] = {
    "marketplace": "onlayn bozor va agregator platformasi",
    "ecommerce": "onlayn do'kon va internet-savdo",
    "saas_dashboard": "biznes uchun CRM va boshqaruv paneli",
    "booking_service": "onlayn navbat va joy band qilish tizimi",
    "delivery_logistics": "yetkazib berish va logistika platformasi",
    "fintech": "moliyaviy texnologiyalar va to'lov xizmati",
    "edtech": "onlayn ta'lim platformasi",
    "healthtech": "tibbiyot va klinika uchun raqamli tizim",
    "social_community": "ijtimoiy tarmoq va jamoa platformasi",
    "content_media": "media va kontent platformasi",
    "ai_tool": "sun'iy intellekt asosidagi xizmat",
    "devtool_api": "dasturchilar uchun API xizmati",
    "mobile_app": "mobil ilova",
    "game": "onlayn o'yin",
}


def build_query(description: str, project_type: str, region: str = "uz") -> str:
    """Qidiruv so'rovini yig'adi."""
    market = _TYPE_QUERY.get(project_type, "startup")
    place = "O'zbekistonda" if region == "uz" else ""
    # Foydalanuvchi matnidan qisqa bo'lak ham qo'shiladi — tur umumiy, g'oya
    # esa aniq nishani ko'rsatadi.
    hint = " ".join(description.split()[:12])
    return f"{place} {market} startuplari va raqobatchilari: {hint}".strip()


def research_competitors(
    description: str, project_type: str = "", region: str = "uz"
) -> ResearchResult:
    """Jonli raqobat tadqiqoti. Kalit yo'q bo'lsa bo'sh natija."""
    if not enabled():
        return ResearchResult(error="Linkup o'chirilgan")
    return search(build_query(description, project_type, region))
