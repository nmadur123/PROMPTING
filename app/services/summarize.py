"""Tahlildan tarix uchun qisqa yozuv yasaydi.

Oylik summa frontenddagi varaq bilan **bir xil** qoida bo'yicha hisoblanadi:
tavsiya etilgan konfiguratsiyaning serveri + qo'shimchalari + eng arzon bo'sh
domen (yillikdan oyliкka bo'lingan). Aks holda tarixda ko'rsatilgan raqam
foydalanuvchi o'sha paytda ko'rgan raqamdan farq qilardi.
"""

from __future__ import annotations

import re


def monthly_total(analysis: dict) -> float:
    options = analysis.get("server_options") or []
    if not options:
        return 0.0

    option = next((o for o in options if o.get("tier") == "recommended"), options[0])
    total = float(option.get("plan", {}).get("usd_month", 0) or 0)
    for addon in option.get("addons") or []:
        total += float(addon.get("usd_month", 0) or 0)

    free = [
        d for d in (analysis.get("domains") or [])
        if d.get("availability") == "available" and d.get("first_year_usd")
    ]
    if free:
        cheapest = min(free, key=lambda d: d["first_year_usd"])
        total += float(cheapest["first_year_usd"]) / 12

    return round(total, 2)


def stack_chips(analysis: dict, limit: int = 4) -> list[str]:
    """Stack tanlovlarining qisqa nomlari — tarix qatoridagi teglar uchun."""
    chips: list[str] = []
    for choice in analysis.get("stack") or []:
        pick = str(choice.get("pick", ""))
        # "Next.js 15 (App Router) + Tailwind" -> "Next.js 15"
        short = re.split(r"[(+·/]", pick)[0].strip()
        if short and short not in chips:
            chips.append(short)
        if len(chips) >= limit:
            break
    return chips


def title_from(description: str, limit: int = 70) -> str:
    """Tarix ro'yxatida ko'rinadigan sarlavha — matnning birinchi jumlasi.

    Foydalanuvchi sarlavha yozmaydi, shuning uchun uni matndan olamiz. Butun
    matnni ko'rsatish ro'yxatni o'qib bo'lmas qilardi.
    """
    text = " ".join((description or "").split())
    if not text:
        return "—"
    first = re.split(r"(?<=[.!?])\s", text)[0]
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"
