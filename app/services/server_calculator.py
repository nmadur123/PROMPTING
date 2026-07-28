"""Server sizing kalkulyatori.

Kutilayotgan foydalanuvchi soni va loyiha turidan kelib chiqib peak RPS,
CPU/RAM, disk, trafik va DB hajmini hisoblaydi, so'ng katalogdan mos
tariflarni tanlaydi.

Narxlar — 2026-yil boshidagi ochiq e'lon qilingan ro'yxat narxlari, taxminiy
mo'ljal uchun. Sotib olishdan oldin provayder saytida tekshirilishi kerak;
`PRICES_UPDATED` shu sababli javobga qo'shib yuboriladi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Literal, Optional

from app.copy import tr

PRICES_UPDATED = "2026-01"

Region = Literal["uz", "eu", "us", "global"]


# --------------------------------------------------------------------------- #
# Loyiha turi profili — trafik xarakteri turga qarab keskin farq qiladi
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrafficProfile:
    """Bitta loyiha turi uchun trafik xarakteristikasi."""

    dau_ratio: float          # oylik faoldan kunlik faolga nisbat
    sessions_per_day: float   # bitta DAU kuniga necha marta kiradi
    requests_per_session: float
    peak_factor: float        # sutkalik o'rtachadan cho'qqigacha koeffitsient
    avg_response_kb: float    # o'rtacha javob hajmi (API + sahifa)
    db_kb_per_user: float     # bitta foydalanuvchiga to'g'ri keladigan DB hajmi
    storage_mb_per_user: float  # media/fayl


_PROFILES: dict[str, TrafficProfile] = {
    "marketplace":        TrafficProfile(0.18, 1.8, 28, 4.0, 45, 90, 3.0),
    "ecommerce":          TrafficProfile(0.15, 1.6, 24, 5.0, 55, 70, 4.0),
    "saas_dashboard":     TrafficProfile(0.45, 3.2, 60, 2.8, 35, 220, 1.5),
    "booking_service":    TrafficProfile(0.12, 1.4, 18, 4.5, 25, 60, 0.5),
    "delivery_logistics": TrafficProfile(0.35, 3.5, 90, 3.5, 18, 120, 1.0),
    "fintech":            TrafficProfile(0.30, 2.6, 30, 4.0, 20, 150, 0.4),
    "edtech":             TrafficProfile(0.25, 2.0, 35, 3.2, 60, 110, 12.0),
    "healthtech":         TrafficProfile(0.20, 1.6, 25, 3.0, 30, 180, 6.0),
    "social_community":   TrafficProfile(0.40, 4.5, 70, 5.0, 40, 140, 8.0),
    "content_media":      TrafficProfile(0.22, 1.5, 16, 6.0, 80, 50, 25.0),
    "ai_tool":            TrafficProfile(0.30, 2.2, 20, 3.5, 25, 120, 2.0),
    "devtool_api":        TrafficProfile(0.50, 6.0, 120, 3.0, 12, 80, 0.5),
    "mobile_app":         TrafficProfile(0.35, 3.0, 45, 3.5, 20, 100, 3.0),
    "game":               TrafficProfile(0.35, 3.5, 150, 4.5, 10, 90, 2.0),
}

_DEFAULT_PROFILE = TrafficProfile(0.25, 2.0, 30, 4.0, 35, 100, 3.0)

# Bitta CPU yadrosi ko'tara oladigan RPS.
#
# Bu butun hisobning eng ta'sirchan farazi, shuning uchun ehtiyotkor olingan.
# Sof JSON qaytaradigan async endpoint yadroda 1000+ rps beradi, lekin real
# endpoint 2-5 ta DB so'rovi qiladi, ORM orqali obyektga o'giradi va JSON'ga
# serializatsiya qiladi — amalda 80-200 rps/yadro chiqadi. Kam baholab
# server tanlashdan ko'ra, ozgina ortiqcha zaxira olish arzonroq.
_RPS_PER_CORE_BASE = 130.0


@dataclass
class Requirements:
    """Hisoblangan resurs talabi."""

    dau: int
    requests_per_day: int
    avg_rps: float
    peak_rps: float
    vcpu: int
    ram_gb: int
    disk_gb: int
    bandwidth_gb_month: int
    db_size_gb: float
    concurrent_connections: int
    needs_managed_db: bool
    needs_cdn: bool
    needs_cache: bool
    needs_object_storage: bool
    needs_queue: bool
    notes: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_requirements(
    project_type: str,
    monthly_users: int,
    signals: Optional[dict[str, bool]] = None,
    lang: str = "uz",
) -> Requirements:
    """Kutilayotgan foydalanuvchidan resurs talabini hisoblaydi."""
    signals = signals or {}
    profile = _PROFILES.get(project_type, _DEFAULT_PROFILE)
    monthly_users = max(1, int(monthly_users))
    notes: list[str] = []

    dau = max(1, int(monthly_users * profile.dau_ratio))
    requests_per_day = int(dau * profile.sessions_per_day * profile.requests_per_session)
    avg_rps = requests_per_day / 86_400
    peak_rps = avg_rps * profile.peak_factor

    # Realtime ulanishlar CPU'ni emas, RAM va fayl deskriptorlarini yeydi.
    concurrent = int(dau * 0.06) if signals.get("realtime") else int(dau * 0.02)
    concurrent = max(concurrent, 10)

    rps_per_core = _RPS_PER_CORE_BASE
    if signals.get("heavy_compute"):
        rps_per_core *= 0.25
        notes.append(tr("srv.heavy", lang))
    if signals.get("ai"):
        rps_per_core *= 0.75
        notes.append(tr("srv.ai", lang))
    if signals.get("search_heavy"):
        rps_per_core *= 0.8

    vcpu = max(1, math.ceil(peak_rps / rps_per_core))

    # RAM: baza + yadro boshiga + realtime ulanishlar + kesh
    ram = 1.0 + vcpu * 0.75 + concurrent * 0.0004
    if signals.get("realtime"):
        ram += 1.0
    if signals.get("ai"):
        ram += 1.0
    ram_gb = max(2, math.ceil(ram))

    db_size_gb = (monthly_users * profile.db_kb_per_user) / 1_048_576
    storage_gb = (monthly_users * profile.storage_mb_per_user) / 1024
    if signals.get("media_heavy"):
        storage_gb *= 2.5

    # Diskda: OS + kod + loglar + DB (agar bir serverda) + media (agar S3 yo'q)
    needs_object_storage = storage_gb > 25 or signals.get("media_heavy", False)
    disk_gb = 20 + math.ceil(db_size_gb * 2.5)
    if not needs_object_storage:
        disk_gb += math.ceil(storage_gb)
    disk_gb = max(20, disk_gb)

    bandwidth_gb = (requests_per_day * 30 * profile.avg_response_kb) / 1_048_576
    if signals.get("media_heavy"):
        bandwidth_gb *= 3

    needs_managed_db = db_size_gb > 20 or monthly_users > 50_000
    needs_cdn = bandwidth_gb > 200 or signals.get("media_heavy", False) or signals.get("seo", False)
    needs_cache = peak_rps > 40 or monthly_users > 20_000
    needs_queue = signals.get("ai", False) or signals.get("heavy_compute", False) or monthly_users > 30_000

    notes.append(tr("srv.assume", lang, rps=int(rps_per_core)))
    if needs_managed_db:
        notes.append(tr("srv.managed_db", lang))
    if needs_cdn:
        notes.append(tr("srv.cdn", lang))
    if needs_cache:
        notes.append(tr("srv.cache", lang))

    return Requirements(
        dau=dau,
        requests_per_day=requests_per_day,
        avg_rps=round(avg_rps, 2),
        peak_rps=round(peak_rps, 2),
        vcpu=vcpu,
        ram_gb=ram_gb,
        disk_gb=int(disk_gb),
        bandwidth_gb_month=int(math.ceil(bandwidth_gb)),
        db_size_gb=round(db_size_gb, 2),
        concurrent_connections=concurrent,
        needs_managed_db=needs_managed_db,
        needs_cdn=needs_cdn,
        needs_cache=needs_cache,
        needs_object_storage=needs_object_storage,
        needs_queue=needs_queue,
        notes=notes,
    )


# --------------------------------------------------------------------------- #
# Provayder katalogi
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Plan:
    provider: str
    name: str
    vcpu: int
    ram_gb: int
    disk_gb: int
    bandwidth_tb: float          # oyiga kiritilgan trafik (TB); 0 = hisobga olinmaydi
    usd_month: float
    regions: tuple[str, ...]
    kind: Literal["vps", "paas", "serverless"]
    note: str = ""


# Ma'lumotnoma narxlar — PRICES_UPDATED holatiga ko'ra.
PLANS: list[Plan] = [
    # --- Hetzner Cloud: EU'da eng arzon narx/quvvat nisbati ---
    Plan("Hetzner", "CX22", 2, 4, 40, 20, 4.5, ("eu",), "vps", "EU uchun eng yaxshi narx/quvvat"),
    Plan("Hetzner", "CX32", 4, 8, 80, 20, 7.5, ("eu",), "vps"),
    Plan("Hetzner", "CX42", 8, 16, 160, 20, 17.0, ("eu",), "vps"),
    Plan("Hetzner", "CCX23", 4, 16, 160, 20, 26.0, ("eu", "us"), "vps", "Ajratilgan (dedicated) vCPU"),
    Plan("Hetzner", "CCX33", 8, 32, 240, 30, 51.0, ("eu", "us"), "vps", "Ajratilgan vCPU"),
    Plan("Hetzner", "CCX43", 16, 64, 360, 30, 101.0, ("eu", "us"), "vps", "Ajratilgan vCPU"),

    # --- Contabo: RAM ko'p, narx past, lekin I/O sekinroq ---
    Plan("Contabo", "Cloud VPS 10", 3, 8, 75, 32, 5.5, ("eu", "us"), "vps", "Arzon, lekin disk I/O sekinroq"),
    Plan("Contabo", "Cloud VPS 20", 6, 12, 100, 32, 11.0, ("eu", "us"), "vps"),
    Plan("Contabo", "Cloud VPS 30", 8, 24, 200, 32, 19.0, ("eu", "us"), "vps"),

    # --- DigitalOcean: hujjati yaxshi, managed servislari kuchli ---
    Plan("DigitalOcean", "Basic 2GB", 1, 2, 50, 2, 12.0, ("eu", "us", "global"), "vps"),
    Plan("DigitalOcean", "Basic 4GB", 2, 4, 80, 4, 24.0, ("eu", "us", "global"), "vps"),
    Plan("DigitalOcean", "General 8GB", 2, 8, 100, 4, 63.0, ("eu", "us", "global"), "vps"),
    Plan("DigitalOcean", "General 16GB", 4, 16, 200, 5, 126.0, ("eu", "us", "global"), "vps"),

    # --- Vultr ---
    Plan("Vultr", "HF 2GB", 1, 2, 64, 2, 12.0, ("eu", "us", "global"), "vps", "Yuqori chastotali CPU"),
    Plan("Vultr", "HF 4GB", 2, 4, 128, 3, 24.0, ("eu", "us", "global"), "vps"),
    Plan("Vultr", "HF 8GB", 3, 8, 256, 4, 48.0, ("eu", "us", "global"), "vps"),

    # --- O'zbekiston: mahalliy kechikish (latency) va ma'lumot rezidentligi ---
    Plan("PS.UZ", "VPS Start", 2, 4, 60, 0, 12.0, ("uz",), "vps", "Toshkentda ~5-15 ms, UZS to'lov"),
    Plan("PS.UZ", "VPS Business", 4, 8, 120, 0, 24.0, ("uz",), "vps", "Mahalliy ma'lumot rezidentligi"),
    Plan("UZCLOUD", "Cloud M", 4, 8, 100, 0, 28.0, ("uz",), "vps", "Davlat bulut infratuzilmasi"),
    Plan("UZCLOUD", "Cloud L", 8, 16, 200, 0, 55.0, ("uz",), "vps"),

    # --- PaaS: DevOps'siz ishga tushirish ---
    Plan("Railway", "Hobby", 2, 2, 20, 0, 5.0, ("us", "eu", "global"), "paas", "Deploy juda oson, usage-based"),
    Plan("Railway", "Pro", 8, 8, 50, 0, 20.0, ("us", "eu", "global"), "paas", "Usage-based, bazadan yuqori"),
    Plan("Render", "Standard", 1, 2, 20, 0, 25.0, ("us", "eu"), "paas", "Managed Postgres bilan yaxshi ishlaydi"),
    Plan("Render", "Pro", 4, 8, 50, 0, 85.0, ("us", "eu"), "paas"),
    Plan("Fly.io", "shared-2x", 2, 4, 40, 0, 15.0, ("global",), "paas", "Bir nechta regionga yoyish oson"),
]

# Qo'shimcha servislar — taxminiy oylik narx.
ADDONS = {
    "managed_postgres_small": {"name": "Managed PostgreSQL (2 vCPU / 4 GB)", "usd_month": 25.0},
    "managed_postgres_mid": {"name": "Managed PostgreSQL (4 vCPU / 16 GB)", "usd_month": 90.0},
    "redis": {"name": "Managed Redis (1 GB)", "usd_month": 12.0},
    "object_storage": {"name": "Object storage (S3-mos, 250 GB)", "usd_month": 6.0},
    "cdn": {"name": "CDN (Cloudflare Pro yoki bulut CDN)", "usd_month": 20.0},
    "backup": {"name": "Avtomatik backup (snapshot)", "usd_month": 5.0},
    "monitoring": {"name": "Monitoring + log (Grafana Cloud free-dan yuqori)", "usd_month": 10.0},
}


@dataclass
class ServerOption:
    tier: Literal["budget", "recommended", "scale"]
    plan: dict
    fits: bool
    headroom_pct: int
    monthly_total_usd: float
    addons: list[dict]
    reasoning: str


def _plan_fits(plan: Plan, req: Requirements, headroom: float) -> bool:
    return (
        plan.vcpu >= math.ceil(req.vcpu * headroom)
        and plan.ram_gb >= math.ceil(req.ram_gb * headroom)
        and plan.disk_gb >= req.disk_gb
    )


def _addons_for(req: Requirements) -> list[dict]:
    picked: list[dict] = []
    if req.needs_managed_db:
        key = "managed_postgres_mid" if req.db_size_gb > 60 else "managed_postgres_small"
        picked.append(ADDONS[key])
    if req.needs_cache:
        picked.append(ADDONS["redis"])
    if req.needs_object_storage:
        picked.append(ADDONS["object_storage"])
    if req.needs_cdn:
        picked.append(ADDONS["cdn"])
    picked.append(ADDONS["backup"])
    if req.peak_rps > 20:
        picked.append(ADDONS["monitoring"])
    return picked


def recommend_servers(
    req: Requirements, region: Region = "global", lang: str = "uz"
) -> list[ServerOption]:
    """Talabga mos 3 ta variant qaytaradi: arzon, tavsiya etilgan, o'sish uchun."""
    candidates = [p for p in PLANS if region == "global" or region in p.regions or "global" in p.regions]
    if not candidates:
        candidates = list(PLANS)

    addons = _addons_for(req)
    addons_cost = sum(a["usd_month"] for a in addons)

    def build(plan: Plan, tier: str, reasoning: str) -> ServerOption:
        headroom = min(
            (plan.vcpu / max(req.vcpu, 1)),
            (plan.ram_gb / max(req.ram_gb, 1)),
        )
        return ServerOption(
            tier=tier,  # type: ignore[arg-type]
            plan={**asdict(plan), "regions": list(plan.regions)},
            fits=_plan_fits(plan, req, 1.0),
            headroom_pct=int((headroom - 1) * 100),
            monthly_total_usd=round(plan.usd_month + addons_cost, 2),
            addons=addons,
            reasoning=reasoning,
        )

    def order(plan: Plan) -> tuple:
        """Narx bo'yicha, lekin bir xil narxda mahalliy provayder ustun.

        Hudud tanlangan bo'lsa (masalan `uz`), o'sha hududdagi provayder
        kechikish va to'lov qulayligi bo'yicha yutadi — narx tengda uni oldinga
        chiqaramiz.
        """
        local = 0 if (region != "global" and region in plan.regions) else 1
        return (plan.usd_month, local, plan.provider)

    # Arzon: talabni aynan qoplaydi (zaxirasiz)
    budget_pool = sorted([p for p in candidates if _plan_fits(p, req, 1.0)], key=order)
    # Tavsiya etilgan: 1.6x zaxira — cho'qqi va o'sish uchun
    rec_pool = sorted([p for p in candidates if _plan_fits(p, req, 1.6)], key=order)
    # O'sish: 3x zaxira
    scale_pool = sorted([p for p in candidates if _plan_fits(p, req, 3.0)], key=order)

    biggest = max(candidates, key=lambda p: (p.vcpu, p.ram_gb))
    options: list[ServerOption] = []

    # Arzon tarif ba'zan tavsiya darajasidagi zaxirani ham beradi (kichik
    # loyihalarda tez-tez). Bunday holatda uni "arzon" deb ko'rsatish
    # chalg'itadi — u aslida tavsiya etiladigan variant.
    same_pick = bool(budget_pool and rec_pool and budget_pool[0] is rec_pool[0])

    if budget_pool and not same_pick:
        options.append(build(budget_pool[0], "budget",
                             tr("srv.tier.budget", lang, vcpu=req.vcpu, ram=req.ram_gb)))
    elif same_pick:
        options.append(build(budget_pool[0], "recommended",
                             tr("srv.tier.same", lang, vcpu=req.vcpu, ram=req.ram_gb)))

    def already_picked(plan: Plan) -> bool:
        # Provayder + tarif nomi birgalikda taqqoslanadi: turli provayderlarda
        # bir xil nomli tarif bo'lishi mumkin ("Pro", "Standard").
        return any((o.plan["provider"], o.plan["name"]) == (plan.provider, plan.name) for o in options)

    if rec_pool and not already_picked(rec_pool[0]):
        options.append(build(rec_pool[0], "recommended", tr("srv.tier.recommended", lang)))
    if scale_pool and not already_picked(scale_pool[0]):
        options.append(build(scale_pool[0], "scale", tr("srv.tier.scale", lang)))

    if not options:
        options.append(build(biggest, "scale", tr("srv.tier.horizontal", lang)))
    return options


def scaling_advice(req: Requirements, project_type: str, lang: str = "uz") -> list[str]:
    """Nagruzka oshganda nima qilish kerakligi — bosqichma-bosqich."""
    tips = [tr(f"scale.{n}", lang) for n in (1, 2, 3, 4)]
    if req.needs_cdn:
        tips.append(tr("scale.cdn", lang))
    if req.concurrent_connections > 500:
        tips.append(tr("scale.ws", lang, n=req.concurrent_connections))
    if project_type in ("delivery_logistics", "game", "social_community"):
        tips.append(tr("scale.spiky", lang))
    if req.db_size_gb > 50:
        tips.append(tr("scale.db", lang, gb=f"{req.db_size_gb:.0f}"))
    return tips
