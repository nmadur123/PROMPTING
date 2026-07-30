"""O'zbekistondagi startup portfelilarini yig'ib, raqobat datasetini yasaydi.

Manbalar (ikkalasi ham ochiq, ro'yxatdan o'tish talab qilmaydi):
  - uzcombinator.uz/portfolio — HTML kartochkalar
  - thepitch.uz               — Vue SPA; ma'lumot JS bundle ichida

Nega ikki xil usul: uzcombinator sahifani serverda renderlaydi, thepitch esa
brauzerda. Ikkinchisida serverga so'rov yuborib HTML olish foydasiz — sahifa
bo'sh keladi, ma'lumot esa bundle ichida qattiq yozilgan. Shuning uchun bundle
o'qiladi. Bundle nomida hash bor (`index-<hash>.js`) va u har deploydan keyin
o'zgaradi, shuning uchun nom bosh sahifadan o'qib olinadi — qattiq yozilsa
skript birinchi deploydan keyin jimgina bo'sh natija qaytarardi.

Natija: data/competitors/competitors.json

Ishga tushirish:
    python -m scripts.fetch_competitors
    python -m scripts.fetch_competitors --dry-run    # faylga yozmaydi
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

from app.config import DATA_DIR

OUT_PATH = DATA_DIR / "competitors" / "competitors.json"

# O'zimizni tanitamiz — anonim so'rov bloklanishi mumkin va bu to'g'ri ham.
_UA = "sys42-competitor-index/1.0 (+https://sys42.xyz)"
_HEADERS = {
    # Ba'zi hostinglar tanish bo'lmagan UA ni to'sadi; brauzer satri bilan
    # birga o'z nomimizni ham qo'shamiz.
    "User-Agent": f"Mozilla/5.0 (compatible; {_UA})",
    "Accept-Language": "uz,ru;q=0.9,en;q=0.8",
}
_TIMEOUT = httpx.Timeout(60.0, connect=20.0)


@dataclass
class Competitor:
    name: str
    slug: str
    description: str
    source: str
    url: str = ""
    profile_url: str = ""
    # thepitch investitsiya summasini ko'rsatadi — raqobat tahlilida bu eng
    # qimmatli maydon: raqobatchi pul ko'targanini bilish strategiyani
    # o'zgartiradi.
    investment: str = ""
    season: str = ""


# --------------------------------------------------------------------------- #
# uzcombinator.uz — server tomonda renderlangan HTML
# --------------------------------------------------------------------------- #

_UZC_URL = "https://uzcombinator.uz/portfolio"

# Kartochka `<h3>` bo'yicha bo'linadi, bitta katta naqsh bilan emas.
#
# Nega: sahifadagi haqiqiy tartib `<h3>nom</h3> ... <p>tavsif</p> ... href`.
# Ilgari naqsh href'dan boshlanardi va u kartochka OXIRIDA turgani uchun har
# bir havola KEYINGI kartochkaning nomiga ulanib qolardi. Nom va tavsif
# to'g'ri juftlashgani uchun natija ishonchli ko'rinardi — faqat har bir
# havola boshqa startupga olib borardi.
_UZC_NAME = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)
_UZC_DESC = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_UZC_SLUG = re.compile(r'href="/portfolio/([a-z0-9.-]+)"')
_UZC_SEASON = re.compile(r">(\d+)-mavsum<")

# Kartochka ichidagi maydonlar shu oynadan tashqariga chiqmaydi. Chegara
# kerak: chegarasiz oxirgi kartochka butun sahifa oxirigacha "yoyilib"
# ketadi va tasodifiy matnni tavsif deb oladi.
_UZC_WINDOW = 1200
_UZC_LOOKBACK = 300


def _clean(markup: str) -> str:
    """HTML teglari va belgilarini tozalaydi.

    `html.unescape` qo'lda yozilgan jadvaldan afzal: u raqamli belgilarni ham
    (`&#x27;`, `&#039;`) ochadi. Qo'lda jadval bilan aynan shular o'tib
    ketardi va tavsif ichida `&#x27;` ko'rinib turardi.
    """
    text = re.sub(r"<[^>]+>", " ", markup)
    return " ".join(html.unescape(text).split()).strip()


def fetch_uzcombinator(client: httpx.Client) -> list[Competitor]:
    html = client.get(_UZC_URL).raise_for_status().text

    out: list[Competitor] = []
    seen: set[str] = set()
    for name_match in _UZC_NAME.finditer(html):
        name = _clean(name_match.group(1))
        if not name:
            continue

        window = html[name_match.end():name_match.end() + _UZC_WINDOW]
        desc_match = _UZC_DESC.search(window)
        slug_match = _UZC_SLUG.search(window)
        if not desc_match or not slug_match:
            # Sahifada kartochka bo'lmagan `<h3>` ham bor (sarlavhalar) —
            # ularda tavsif yoki havola bo'lmaydi, o'tkazib yuboriladi.
            continue

        slug = slug_match.group(1)
        if slug in seen:
            continue
        seen.add(slug)

        # Mavsum belgisi nomdan OLDIN turadi.
        back = html[max(0, name_match.start() - _UZC_LOOKBACK):name_match.start()]
        season = _UZC_SEASON.findall(back)

        out.append(Competitor(
            name=name,
            slug=slug,
            description=_clean(desc_match.group(1)),
            source="uzcombinator",
            profile_url=f"https://uzcombinator.uz/portfolio/{slug}",
            season=(season[-1] if season else ""),
        ))
    return out


# --------------------------------------------------------------------------- #
# thepitch.uz — Vue SPA, ma'lumot bundle ichida
# --------------------------------------------------------------------------- #

_TP_ROOT = "https://thepitch.uz/"
_TP_BUNDLE = re.compile(r'src="(/assets/index-[^"]+\.js)"')

# Bundle ichidagi obyekt: slug: {shortDescription, ..., officialWebsite, ...}
# Maydonlar tartibi minifikatsiyada o'zgarishi mumkin, shuning uchun har biri
# alohida qidiriladi, bir butun naqsh bilan emas.
_TP_ENTRY = re.compile(
    r'["\']?([a-zA-Z0-9_-]{2,})["\']?\s*:\s*\{([^{}]*shortDescription[^{}]*)\}'
)
_TP_FIELD = {
    "description": re.compile(r'shortDescription\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "investment": re.compile(r'totalInvestment\s*:\s*"((?:[^"\\]|\\.)*)"'),
    "url": re.compile(r'officialWebsite\s*:\s*"((?:[^"\\]|\\.)*)"'),
}


def _unescape_js(text: str) -> str:
    """JS satridagi \\uXXXX va \\" kabi belgilarni ochadi."""
    try:
        return json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text.replace('\\"', '"').replace("\\\\", "\\")


def _title_from_slug(slug: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[-_]", slug) if part)


def fetch_thepitch(client: httpx.Client) -> list[Competitor]:
    root = client.get(_TP_ROOT).raise_for_status().text
    match = _TP_BUNDLE.search(root)
    if not match:
        print("  DIQQAT: thepitch.uz bosh sahifasida JS bundle topilmadi — "
              "sayt tuzilishi o'zgargan bo'lishi mumkin", file=sys.stderr)
        return []

    bundle_url = f"https://thepitch.uz{match.group(1)}"
    bundle = client.get(bundle_url).raise_for_status().text

    out: list[Competitor] = []
    seen: set[str] = set()
    for m in _TP_ENTRY.finditer(bundle):
        slug, body = m.group(1), m.group(2)
        if slug in seen:
            continue

        desc_m = _TP_FIELD["description"].search(body)
        if not desc_m:
            continue
        description = _unescape_js(desc_m.group(1)).strip()
        if len(description) < 15:
            continue

        seen.add(slug)
        url_m = _TP_FIELD["url"].search(body)
        inv_m = _TP_FIELD["investment"].search(body)
        out.append(Competitor(
            name=_title_from_slug(slug),
            slug=slug,
            description=description,
            source="thepitch",
            url=_unescape_js(url_m.group(1)) if url_m else "",
            investment=_unescape_js(inv_m.group(1)) if inv_m else "",
        ))
    return out


# --------------------------------------------------------------------------- #


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="faylga yozmaydi")
    args = parser.parse_args()

    sources = (
        ("uzcombinator.uz", fetch_uzcombinator),
        ("thepitch.uz", fetch_thepitch),
    )

    rows: list[Competitor] = []
    with httpx.Client(timeout=_TIMEOUT, headers=_HEADERS, follow_redirects=True) as client:
        for label, fetch in sources:
            try:
                found = fetch(client)
            except httpx.HTTPError as exc:
                # Bitta manba yiqilsa ikkinchisi baribir yig'ilsin — natija
                # kamroq bo'ladi, lekin bor.
                print(f"  {label}: XATO — {exc}", file=sys.stderr)
                continue
            print(f"  {label}: {len(found)} ta")
            rows.extend(found)

    if not rows:
        print("Hech narsa topilmadi — fayl o'zgartirilmadi.", file=sys.stderr)
        return 1

    with_url = sum(1 for r in rows if r.url or r.profile_url)
    with_investment = sum(1 for r in rows if r.investment)
    print(f"\nJami: {len(rows)} ta startup")
    print(f"  havolasi bor      : {with_url}")
    print(f"  investitsiyasi bor: {with_investment}")

    if args.dry_run:
        print("\n--dry-run: fayl yozilmadi. Namuna:")
        for r in rows[:5]:
            print(f"  [{r.source}] {r.name}: {r.description[:70]}")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"count": len(rows), "items": [asdict(r) for r in rows]}
    OUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nYozildi: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
