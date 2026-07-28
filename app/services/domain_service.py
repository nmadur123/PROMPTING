"""Domen tanlash: nom taklif qilish + bandligini tekshirish + narx.

Bandlikni RDAP orqali tekshiramiz — bu ochiq standart, kalit talab qilmaydi.
`rdap.org` bootstrap servisi so'rovni kerakli registry'ga yo'naltiradi.

Muhim: RDAP hamma TLD uchun mavjud emas (masalan `.uz` da yo'q). Bunday
holatda "noma'lum" qaytariladi — "bo'sh" deb yolg'on aytilmaydi.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, asdict
from typing import Literal, Optional

import httpx

RDAP_BOOTSTRAP = "https://rdap.org/domain/{domain}"
_TIMEOUT = httpx.Timeout(8.0, connect=4.0)

Availability = Literal["available", "taken", "unknown"]

# Ma'lumotnoma narxlar (USD/yil, birinchi yil / uzaytirish). 2026-01 holati.
TLD_INFO: dict[str, dict] = {
    ".com":   {"first_year": 11.0, "renewal": 15.0, "note": "Eng ishonchli, xalqaro. Birinchi tanlov."},
    ".uz":    {"first_year": 12.0, "renewal": 12.0, "note": "O'zbekiston. Mahalliy ishonch yuqori; ro'yxatdan o'tish hujjat talab qiladi.", "rdap": False},
    ".io":    {"first_year": 35.0, "renewal": 45.0, "note": "Texnologik startuplar orasida ommabop, lekin qimmat."},
    ".ai":    {"first_year": 70.0, "renewal": 90.0, "note": "AI mahsulot uchun aniq signal, narxi baland."},
    ".app":   {"first_year": 14.0, "renewal": 18.0, "note": "HTTPS majburiy — mobil/web ilova uchun mos."},
    ".dev":   {"first_year": 13.0, "renewal": 16.0, "note": "Dasturchilarga qaratilgan mahsulot uchun."},
    ".co":    {"first_year": 12.0, "renewal": 30.0, "note": "`.com` band bo'lsa yaxshi zaxira."},
    ".shop":  {"first_year": 3.0,  "renewal": 35.0, "note": "E-commerce uchun; uzaytirish narxiga e'tibor bering."},
    ".store": {"first_year": 4.0,  "renewal": 55.0, "note": "Birinchi yil arzon, keyin qimmat."},
    ".net":   {"first_year": 13.0, "renewal": 17.0, "note": "Klassik zaxira variant."},
    ".org":   {"first_year": 12.0, "renewal": 15.0, "note": "Nodavlat/jamoat loyihalari uchun."},
    ".tech":  {"first_year": 5.0,  "renewal": 50.0, "note": "Birinchi yil arzon, uzaytirish qimmat."},
}

# Loyiha turiga qarab qaysi TLD birinchi tavsiya qilinadi.
_TLD_PRIORITY: dict[str, list[str]] = {
    "ai_tool":            [".ai", ".com", ".app", ".io"],
    "devtool_api":        [".dev", ".io", ".com", ".app"],
    "ecommerce":          [".uz", ".com", ".shop", ".store"],
    "marketplace":        [".uz", ".com", ".co", ".net"],
    "mobile_app":         [".app", ".com", ".uz", ".io"],
    "fintech":            [".com", ".uz", ".co", ".net"],
    "healthtech":         [".uz", ".com", ".org", ".co"],
    "edtech":             [".uz", ".com", ".org", ".app"],
    "content_media":      [".uz", ".com", ".org", ".net"],
    "social_community":   [".com", ".app", ".co", ".uz"],
    "booking_service":    [".uz", ".com", ".app", ".co"],
    "delivery_logistics": [".uz", ".com", ".app", ".co"],
    "saas_dashboard":     [".com", ".io", ".app", ".co"],
    "game":               [".com", ".io", ".app", ".co"],
}
_DEFAULT_TLDS = [".com", ".uz", ".app", ".co"]

# Nomdan tashlab yuboriladigan umumiy so'zlar (uz/ru/en).
_STOPWORDS = {
    # umumiy / grammatik
    "uchun", "va", "bilan", "qilmoqchiman", "kerak", "boladi", "bor", "yangi", "men",
    "hozir", "keyin", "avval", "yoki", "ham", "hech", "qanday", "qancha", "birinchi",
    "kichik", "katta", "oz", "ozi", "shu", "bunda", "hamda", "lekin",
    "startup", "loyiha", "platforma", "ilova", "sayt", "tizim", "xizmat", "servis",
    "versiya", "foydalanuvchi", "mijoz", "narsa", "yilda", "kutyapman",
    # brend sifatida ma'nosiz umumiy so'zlar
    "onlayn", "online", "guruh", "orqali", "tartib", "tartibi", "ishlash",
    "hamda", "mumkin", "boshqa", "barcha", "qiladi", "boradi", "beradi",
    "the", "and", "for", "with", "app", "platform", "service", "system", "new", "want",
    "users", "user", "client", "clients", "first", "then", "site", "website", "build",
    "и", "для", "с", "приложение", "платформа", "сервис", "система", "хочу", "новый",
    "пользователь", "пользователи", "клиент", "клиенты", "сайт", "сначала", "потом",
}

_SUFFIXES = ["hub", "ly", "io", "go", "up", "lab", "kit", "box", "one", "now", "pro"]
_PREFIXES = ["get", "try", "my", "the", "go"]


@dataclass
class DomainCandidate:
    domain: str
    tld: str
    availability: Availability
    first_year_usd: Optional[float]
    renewal_usd: Optional[float]
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


# O'zbek tilidagi kelishik/egalik qo'shimchalari. Bularsiz "samarqanddagi",
# "sartaroshxonalar" kabi grammatik shakllar domen nomiga aylanib qolardi.
_SUFFIX_STRIP = (
    "lardagi", "lardan", "larning", "lariga", "larida", "lari", "lar",
    "dagi", "ning", "dan", "ga", "da", "ni", "im", "ing", "si", "i",
)


def _stem(word: str) -> str:
    """So'zdan qo'shimchani olib tashlaydi (eng uzun moslikdan boshlab)."""
    for suffix in _SUFFIX_STRIP:
        if len(word) - len(suffix) >= 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _slugify(text: str) -> list[str]:
    """Matndan domen uchun yaroqli o'zaklarni ajratadi.

    Domen brend nomi bo'ladi, shuning uchun grammatik shakl emas, o'zak kerak:
    "samarqanddagi" -> "samarqand", "sartaroshxonalar" -> "sartaroshxona".
    Juda uzun o'zaklar ham tashlanadi — ular domen sifatida yaroqsiz.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (text or "").lower())
    seen: set[str] = set()
    words: list[str] = []
    for raw in cleaned.split():
        if len(raw) < 4 or raw in _STOPWORDS:
            continue
        stem = _stem(raw)
        if len(stem) < 4 or len(stem) > 12 or stem in _STOPWORDS or stem in seen:
            continue
        seen.add(stem)
        words.append(stem)
    # Qisqaroq o'zak yaxshiroq domen bo'ladi, lekin matndagi tartib ham muhim —
    # birinchi jumladagi so'zlar odatda mahsulotning o'zagi.
    head, tail = words[:8], words[8:]
    head.sort(key=len)
    return (head + tail)[:6]


def suggest_names(description: str, brand_hint: str = "", limit: int = 8) -> list[str]:
    """Tavsifdan domen nomlari yasaydi (registratsiyasiz, sof matn ishlovi)."""
    names: list[str] = []

    def add(candidate: str) -> None:
        candidate = re.sub(r"[^a-z0-9-]", "", candidate.lower())
        if 3 <= len(candidate) <= 20 and candidate not in names:
            names.append(candidate)

    if brand_hint:
        add(brand_hint)

    words = _slugify(description)
    for w in words[:3]:
        add(w)
    # Ikki so'zni birlashtirish — eng tabiiy natija shu.
    for i in range(min(3, len(words))):
        for j in range(i + 1, min(4, len(words))):
            add(words[i] + words[j])
    base = brand_hint or (words[0] if words else "startup")
    for suffix in _SUFFIXES:
        add(base + suffix)
    for prefix in _PREFIXES:
        add(prefix + base)

    return names[:limit]


async def check_availability(domain: str, client: httpx.AsyncClient) -> Availability:
    """RDAP orqali domen bandligini tekshiradi.

    404 — ro'yxatda yo'q (bo'sh). 200 — band. Qolgan hamma holat "noma'lum":
    tarmoq xatosi yoki RDAP qo'llab-quvvatlanmasligini "bo'sh" deb ko'rsatish
    foydalanuvchini noto'g'ri qarorga olib keladi.
    """
    tld = "." + domain.rsplit(".", 1)[-1]
    if TLD_INFO.get(tld, {}).get("rdap") is False:
        return "unknown"
    try:
        resp = await client.get(RDAP_BOOTSTRAP.format(domain=domain), follow_redirects=True)
    except (httpx.HTTPError, asyncio.TimeoutError):
        return "unknown"
    if resp.status_code == 404:
        return "available"
    if resp.status_code == 200:
        return "taken"
    return "unknown"


async def find_domains(
    description: str,
    project_type: str,
    brand_hint: str = "",
    max_checks: int = 18,
) -> list[DomainCandidate]:
    """Nom takliflarini yasab, bandligini parallel tekshiradi."""
    names = suggest_names(description, brand_hint)
    tlds = _TLD_PRIORITY.get(project_type, _DEFAULT_TLDS)

    pairs: list[tuple[str, str]] = []
    for name in names:
        for tld in tlds:
            pairs.append((name, tld))
    pairs = pairs[:max_checks]

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        results = await asyncio.gather(
            *(check_availability(f"{n}{t}", client) for n, t in pairs),
            return_exceptions=True,
        )

    candidates: list[DomainCandidate] = []
    for (name, tld), status in zip(pairs, results):
        info = TLD_INFO.get(tld, {})
        candidates.append(DomainCandidate(
            domain=f"{name}{tld}",
            tld=tld,
            availability=status if isinstance(status, str) else "unknown",
            first_year_usd=info.get("first_year"),
            renewal_usd=info.get("renewal"),
            note=info.get("note", ""),
        ))

    # Bo'shlari oldinda, keyin noma'lum, oxirida band. Ichida narx bo'yicha.
    rank = {"available": 0, "unknown": 1, "taken": 2}
    candidates.sort(key=lambda c: (rank[c.availability], c.first_year_usd or 999))
    return candidates


def registrar_advice(region: str = "global") -> list[dict]:
    """Qayerdan sotib olish — narx va qulaylik bo'yicha."""
    base = [
        {"name": "Cloudflare Registrar", "why": "Ustama narxsiz sotadi (self-cost) va DNS/CDN bir joyda. Eng arzon uzaytirish.", "caveat": "`.uz` ni sotmaydi."},
        {"name": "Namecheap", "why": "Arzon birinchi yil, WHOIS maxfiyligi bepul, panel qulay.", "caveat": "Uzaytirish narxi Cloudflare'dan yuqori."},
        {"name": "Porkbun", "why": "Narxi past, `.dev`/`.app`/`.io` da yaxshi taklif.", "caveat": "Qo'llab-quvvatlash faqat inglizchada."},
    ]
    if region == "uz":
        base.insert(0, {
            "name": "ahost.uz / uzinfocom (.uz uchun)",
            "why": "`.uz` domenini faqat mahalliy registrator beradi; to'lov UZS'da, hujjat bilan.",
            "caveat": "Jismoniy shaxs uchun pasport, yuridik shaxs uchun guvohnoma talab qilinadi.",
        })
    return base
