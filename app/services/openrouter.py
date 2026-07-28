"""OpenRouter mijozi: model katalogi + chat completion.

Katalog OpenRouter'dan dinamik olinadi (kalit talab qilmaydi) va 30 daqiqa
keshlanadi. Har bir model "oila"ga (claude / openai / google / meta / ...)
ajratiladi — prompt uslubi shu oilaga qarab tanlanadi.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import AsyncIterator, Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 1800
_TIMEOUT = httpx.Timeout(25.0, connect=5.0)

# Maksimal darajada token tejash uchun max_tokens 1500 ga tushirildi.
_DEFAULT_MAX_TOKENS = 1500
_MIN_USEFUL_TOKENS = 600

# OpenRouter 402 javobida ruxsat etilgan chegarani matn ichida qaytaradi:
# "You requested up to 8000 tokens, but can only afford 4000".
_AFFORD_RE = re.compile(r"can only afford (\d+)")


def _affordable_tokens(payload: str) -> Optional[int]:
    match = _AFFORD_RE.search(payload or "")
    return int(match.group(1)) if match else None


class OpenRouterError(RuntimeError):
    """OpenRouter so'rovi muvaffaqiyatsiz tugadi."""


@dataclass
class ModelInfo:
    id: str
    name: str
    family: str
    context_length: int
    prompt_usd_per_1m: Optional[float]
    completion_usd_per_1m: Optional[float]
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


# Model id prefiksidan oilaga xaritalash. Prompt shabloni shu bo'yicha tanlanadi.
_FAMILY_PREFIX = {
    "anthropic/": "claude",
    "openai/": "openai",
    "google/": "google",
    "meta-llama/": "meta",
    "mistralai/": "mistral",
    "deepseek/": "deepseek",
    "qwen/": "qwen",
    "x-ai/": "xai",
    "cohere/": "cohere",
}


def family_of(model_id: str) -> str:
    for prefix, family in _FAMILY_PREFIX.items():
        if model_id.startswith(prefix):
            return family
    return "other"


def _price(value: Optional[str]) -> Optional[float]:
    """OpenRouter narxni token boshiga string qaytaradi — 1M tokenga o'giramiz."""
    try:
        return round(float(value) * 1_000_000, 4) if value is not None else None
    except (TypeError, ValueError):
        return None


class OpenRouterClient:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._models_cache: list[ModelInfo] = []
        self._cached_at: float = 0.0

    # ----------------------------------------------------------------- katalog

    async def list_models(self, force: bool = False) -> list[ModelInfo]:
        now = time.monotonic()
        if not force and self._models_cache and (now - self._cached_at) < _CACHE_TTL_SECONDS:
            return self._models_cache

        url = f"{self._settings.openrouter_base_url}/models"
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            if self._models_cache:
                logger.warning("Model katalogi yangilanmadi, eski kesh ishlatilyapti: %s", exc)
                return self._models_cache
            raise OpenRouterError(f"Model ro'yxatini olib bo'lmadi: {exc}") from exc

        models: list[ModelInfo] = []
        for item in payload.get("data", []):
            model_id = item.get("id", "")
            if not model_id:
                continue
            pricing = item.get("pricing") or {}
            models.append(ModelInfo(
                id=model_id,
                name=item.get("name") or model_id,
                family=family_of(model_id),
                context_length=int(item.get("context_length") or 0),
                prompt_usd_per_1m=_price(pricing.get("prompt")),
                completion_usd_per_1m=_price(pricing.get("completion")),
                description=(item.get("description") or "")[:400],
            ))

        models.sort(key=lambda m: (m.family != "claude", m.family, m.name))
        self._models_cache = models
        self._cached_at = now
        return models

    async def popular_models(self, limit: int = 12) -> list[ModelInfo]:
        """Startup uchun eng ko'p ishlatiladigan modellar — tanlash osonlashsin."""
        preferred = [
            "anthropic/claude-opus-5",
            "anthropic/claude-sonnet-5",
            "anthropic/claude-fable-5",
            "anthropic/claude-opus-4.8",
            "openai/gpt-5",
            "google/gemini-2.5-pro",
            "deepseek/deepseek-chat",
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwen-2.5-72b-instruct",
            "mistralai/mistral-large",
        ]
        models = await self.list_models()
        by_id = {m.id: m for m in models}
        out = [by_id[i] for i in preferred if i in by_id]
        for m in models:
            if len(out) >= limit:
                break
            if m not in out and m.family in ("claude", "openai", "google"):
                out.append(m)
        return out[:limit]

    # -------------------------------------------------------------- generatsiya

    def _headers(self) -> dict[str, str]:
        key = self._settings.openrouter_api_key
        if not key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY sozlanmagan. backend/.env faylini to'ldiring."
            )
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self._settings.openrouter_app_url,
            "X-Title": self._settings.openrouter_app_title,
        }

    async def complete(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.4,
        min_useful: int | None = None,
    ) -> str:
        """Bitta javob qaytaradi (oqimsiz).

        Hisobda kredit yetmasa OpenRouter 402 bilan "can only afford N tokens"
        deb javob beradi. Bunday holatda butunlay tushib qolmasdan, ruxsat
        etilgan chegara bilan bir marta qayta urinamiz — qisqaroq prompt
        umuman promptsiz qolishdan yaxshiroq.
        """
        url = f"{self._settings.openrouter_base_url}/chat/completions"
        model_id = model or self._settings.openrouter_writer_model

        async def call(limit: int) -> httpx.Response:
            body = {
                "model": model_id,
                "messages": messages,
                "max_tokens": limit,
                "temperature": temperature,
            }
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                return await client.post(url, headers=self._headers(), json=body)

        try:
            resp = await call(max_tokens)
            if resp.status_code == 402:
                # Chegara chaqiruvga qarab: qisqa savollar 400 tokenga sig'adi,
                # 14 000 belgilik promptni qayta yozish esa sig'maydi. Yagona
                # global chegara ikkalasiga ham to'g'ri kelmaydi.
                floor = _MIN_USEFUL_TOKENS if min_useful is None else min_useful
                affordable = _affordable_tokens(resp.text)
                if affordable and affordable >= floor:
                    logger.warning(
                        "OpenRouter krediti cheklangan: %s -> %s token bilan qayta urinilyapti",
                        max_tokens, affordable,
                    )
                    resp = await call(affordable)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"OpenRouter'ga ulanib bo'lmadi: {exc}") from exc

        if resp.status_code != 200:
            raise OpenRouterError(f"OpenRouter xatosi {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise OpenRouterError("OpenRouter bo'sh javob qaytardi.")
        return choices[0].get("message", {}).get("content", "")

    async def stream(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        max_tokens: int = 8000,
        temperature: float = 0.4,
    ) -> AsyncIterator[str]:
        """Javobni bo'lak-bo'lak qaytaradi — UI'da darhol ko'rsatish uchun."""
        body = {
            "model": model or self._settings.openrouter_writer_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        url = f"{self._settings.openrouter_base_url}/chat/completions"
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            async with client.stream("POST", url, headers=self._headers(), json=body) as resp:
                if resp.status_code != 200:
                    detail = (await resp.aread()).decode("utf-8", "replace")[:400]
                    raise OpenRouterError(f"OpenRouter xatosi {resp.status_code}: {detail}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        import json

                        delta = json.loads(chunk)["choices"][0]["delta"]
                    except (KeyError, IndexError, ValueError):
                        continue
                    content = delta.get("content")
                    if content:
                        yield content


client = OpenRouterClient()
