"""Loyiha turi va texnik signallarga qarab stack tavsiya qiladi.

Har bir tanlov uchun sabab ham qaytariladi — foydalanuvchi "nega aynan shu"
degan savolga javob olsin va keraksiz texnologiyani olib tashlay olsin.

Matnlar `app.copy` katalogidan olinadi, shuning uchun javob so'ralgan tilda
keladi. Mahsulot nomlari (Next.js, PostgreSQL) tarjima qilinmaydi.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from app.copy import tr


@dataclass
class Choice:
    category: str
    pick: str
    why: str
    alternatives: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


# Loyiha turi -> (pick, why kaliti, alternativalar). Pick va alternativalar —
# mahsulot nomlari, ular hech qaysi tilda o'zgarmaydi.
_FRONTEND: dict[str, tuple[str, str, list[str]]] = {
    "ecommerce": ("Next.js 15 (App Router) + Tailwind CSS", "why.fe.ecommerce",
                  ["Astro", "Nuxt 3"]),
    "content_media": ("Astro + Tailwind CSS", "why.fe.content",
                      ["Next.js 15", "Hugo"]),
    "saas_dashboard": ("Next.js 15 (App Router) + Tailwind + shadcn/ui", "why.fe.dashboard",
                       ["Vite + React", "SvelteKit"]),
    "devtool_api": ("Astro (docs) + Next.js (dashboard)", "why.fe.devtool",
                    ["Docusaurus", "Mintlify"]),
    "mobile_app": ("React Native (Expo)", "why.fe.mobile",
                   ["Flutter", "Native (Swift / Kotlin)"]),
    "game": ("Phaser 3 (2D) / Three.js (3D) + Vite", "why.fe.game",
             ["Godot (web export)", "Unity WebGL"]),
}
_FRONTEND_DEFAULT = ("Next.js 15 (App Router) + Tailwind CSS", "why.fe.default",
                     ["Nuxt 3", "SvelteKit", "Remix"])


def _frontend(project_type: str, lang: str) -> Choice:
    pick, why_key, alts = _FRONTEND.get(project_type, _FRONTEND_DEFAULT)
    return Choice(tr("cat.frontend", lang), pick, tr(why_key, lang), alts)


def _backend(project_type: str, signals: dict, monthly_users: int, lang: str) -> Choice:
    cat = tr("cat.backend", lang)
    if signals.get("ai"):
        return Choice(cat, "Python 3.12 + FastAPI (async)", tr("why.be.ai", lang),
                      ["Node.js + Fastify", "Go + Fiber"])
    if project_type == "devtool_api" or monthly_users > 200_000:
        return Choice(cat, "Go + Fiber (yoki Chi)", tr("why.be.go", lang),
                      ["Rust + Axum", "Node.js + Fastify"])
    if project_type in ("delivery_logistics", "social_community", "game") or signals.get("realtime"):
        return Choice(cat, "Node.js 22 + Fastify + Socket.IO", tr("why.be.node", lang),
                      ["Python + FastAPI + WebSocket", "Elixir + Phoenix"])
    return Choice(cat, "Python 3.12 + FastAPI + SQLAlchemy 2 (async)", tr("why.be.default", lang),
                  ["Node.js + NestJS", "Laravel"])


def _database(project_type: str, signals: dict, db_size_gb: float, lang: str) -> Choice:
    cat = tr("cat.database", lang)
    if project_type == "fintech" or signals.get("compliance"):
        return Choice(cat, "PostgreSQL 17 (managed, PITR)", tr("why.db.fintech", lang),
                      ["CockroachDB"])
    if signals.get("geo"):
        return Choice(cat, "PostgreSQL 17 + PostGIS", tr("why.db.geo", lang), ["MongoDB"])
    suffix = " (managed)" if db_size_gb > 20 else ""
    return Choice(cat, f"PostgreSQL 17{suffix}", tr("why.db.default", lang),
                  ["MySQL 8", "SQLite (MVP)"])


def _auth(project_type: str, signals: dict, lang: str) -> Choice:
    cat = tr("cat.auth", lang)
    if signals.get("multi_tenant"):
        return Choice(cat, "JWT + rol va tenant tekshiruvi", tr("why.auth.tenant", lang),
                      ["Clerk", "Auth0", "Supabase Auth"])
    if project_type in ("social_community", "mobile_app", "game"):
        return Choice(cat, "SMS OTP + ijtimoiy tarmoq orqali kirish", tr("why.auth.phone", lang),
                      ["Firebase Auth", "Clerk"])
    return Choice(cat, "JWT (access + refresh) + bcrypt", tr("why.auth.default", lang),
                  ["Clerk", "Supabase Auth", "Auth.js"])


def recommend_stack(
    project_type: str,
    signals: dict,
    monthly_users: int,
    db_size_gb: float = 1.0,
    region: str = "global",
    lang: str = "uz",
) -> list[Choice]:
    """To'liq stack tavsiyasi — har bir qatlam uchun bitta tanlov va sabab."""
    choices = [
        _frontend(project_type, lang),
        _backend(project_type, signals, monthly_users, lang),
        _database(project_type, signals, db_size_gb, lang),
        _auth(project_type, signals, lang),
    ]

    if signals.get("realtime"):
        choices.append(Choice(tr("cat.realtime", lang), "Socket.IO + Redis Pub/Sub",
                              tr("why.realtime", lang), ["Centrifugo", "Pusher / Ably"]))

    if monthly_users > 20_000 or signals.get("search_heavy"):
        choices.append(Choice(tr("cat.cache", lang), "Redis 7", tr("why.cache", lang),
                              ["Valkey", "Memcached"]))

    if signals.get("search_heavy"):
        choices.append(Choice(tr("cat.search", lang),
                              "PostgreSQL full-text → Meilisearch", tr("why.search", lang),
                              ["Typesense", "Elasticsearch"]))

    if signals.get("ai") or project_type == "ai_tool":
        choices.append(Choice(tr("cat.ai", lang), "OpenRouter + o'z RAG indeksingiz",
                              tr("why.ai", lang), ["Anthropic / OpenAI API", "Ollama"]))
        choices.append(Choice(tr("cat.queue", lang), "Redis + RQ (Python) / BullMQ (Node)",
                              tr("why.queue", lang), ["Celery", "Temporal"]))

    if signals.get("media_heavy") or project_type in ("content_media", "social_community", "edtech"):
        choices.append(Choice(tr("cat.storage", lang), "Cloudflare R2 / Backblaze B2 + CDN",
                              tr("why.storage", lang), ["AWS S3 + CloudFront", "DigitalOcean Spaces"]))

    if signals.get("payments") or project_type in ("ecommerce", "marketplace", "fintech", "booking_service"):
        if region == "uz":
            choices.append(Choice(tr("cat.payment", lang), "Payme + Click",
                                  tr("why.pay.uz", lang), ["Uzum Bank / Paynet", "Stripe Connect"]))
        else:
            choices.append(Choice(tr("cat.payment", lang), "Stripe + Payme/Click",
                                  tr("why.pay.intl", lang), ["Paddle", "Lemon Squeezy"]))

    choices.append(Choice(tr("cat.monitoring", lang), "Sentry + Uptime Kuma",
                          tr("why.monitoring", lang), ["Grafana Cloud", "BetterStack"]))
    choices.append(Choice(tr("cat.cicd", lang), "GitHub Actions + Docker",
                          tr("why.cicd", lang), ["Coolify", "Kamal", "Kubernetes"]))
    return choices


def anti_recommendations(
    project_type: str, signals: dict, monthly_users: int, lang: str = "uz"
) -> list[str]:
    """Nima QILMASLIK kerak — startup shu xatolarda vaqt va pul yo'qotadi."""
    out = [tr("anti.k8s", lang), tr("anti.micro", lang)]
    if monthly_users < 50_000:
        out.append(tr("anti.serverless", lang))
    if not signals.get("search_heavy"):
        out.append(tr("anti.elastic", lang))
    if not signals.get("realtime"):
        out.append(tr("anti.ws", lang))
    if project_type != "ai_tool" and not signals.get("ai"):
        out.append(tr("anti.ai", lang))
    out.append(tr("anti.mongo", lang))
    return out
