"""Tayyor "mega-prompt" quruvchi.

Ikki bosqichli:
  1) Deterministik skelet — ML tahlili (tur, stack, server, UI/UX, domen,
     to'lov) asosida XML bo'limlarga solingan to'liq texnik topshiriq.
     Bu OpenRouter'siz ham ishlaydi.
  2) RAG bilan boyitish — foydalanuvchi tanlagan model uchun aynan yozilgan
     prompt-engineering qoidalari korpusdan topiladi va yozuvchi modelga
     "shu qoidalarga muvofiq qayta yoz" deb beriladi.

Nega shunday: skelet faktlarni kafolatlaydi (model o'ylab topmaydi), RAG esa
uslubni tanlangan modelga moslaydi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import get_settings
from app.ml.engine import RetrievedChunk, engine
from app.services import llm, openrouter, project_playbooks

logger = logging.getLogger(__name__)

# Yakuniy promptni yozish uchun chiqish chegarasi.
#
# Ilgari bu yerda `complete()` ning umumiy standarti (1500 token ≈ 6000 belgi)
# ishlatilardi va aynan shu sabab prompt qanchalik boy tayyorlangan bo'lmasin,
# yozuvchi model uni o'rtasidan kesib tashlardi. Skeletning o'zi hozir 12 000
# belgidan oshadi, ya'ni 1500 token unga yetmaydi.
_PROMPT_MAX_TOKENS = 12_000

# Qayta yozilgan prompt skeletdan shuncha marta qisqa bo'lsa, u qisqartirib
# yuborilgan deb hisoblanadi va skeletning o'zi qaytariladi. Model ba'zan
# "optimallashtiraman" deb bo'limlarni tashlab ketadi — bunday javob
# deterministik skeletdan yomonroq.
_MIN_LENGTH_RATIO = 0.55

# OpenRouter model id -> korpusdagi model slug (RAG'da ustuvorlik uchun).
_MODEL_DOC_SLUG: dict[str, str] = {
    "anthropic/claude-opus-5": "opus-5",
    "anthropic/claude-opus-4.8": "opus-4-8",
    "anthropic/claude-opus-4-8": "opus-4-8",
    "anthropic/claude-sonnet-5": "sonnet-5",
    "anthropic/claude-fable-5": "fable-5",
}

# Oilaga qarab prompt uslubi — korpusdagi asosiy tavsiyalarning qisqartmasi.
_FAMILY_STYLE: dict[str, dict] = {
    "claude": {
        "structure": "XML teglar bilan bo'limlarga ajratilgan (<project>, <stack>, <constraints>, ...)",
        "rules": [
            "Aniq va to'g'ridan-to'g'ri yozing: modelga nima kerakligini ayting, nima kerak emasligini emas.",
            "Kontekst bering — vazifa nima uchun bajarilayotganini aytsangiz natija yaxshilanadi.",
            "Rol bering (system prompt): kim bo'lib ishlashi kerakligini belgilang.",
            "Misol keltiring — kutilgan chiqish formatining 1-2 ta namunasi.",
            "Chiqish formatini aniq belgilang (fayl tuzilishi, kod uslubi, izoh darajasi).",
            "Vazifa hajmini aniq chegaralang — kengaytirib yuborishga yo'l qo'ymang.",
            "\"Tekshir\", \"qayta tekshir\" kabi ko'rsatmalarni QO'SHMANG — zamonaviy modellar buni o'zi qiladi, "
            "qo'shimcha ko'rsatma ortiqcha token va ortiqcha tekshiruvga olib keladi.",
        ],
    },
    "openai": {
        "structure": "Markdown sarlavhalar bilan bo'limlarga ajratilgan, har bo'lim aniq nomlangan",
        "rules": [
            "Kerakli NATIJANI tasvirlang — jarayonni emas, oxirgi mahsulotni.",
            "Foydali kontekst qo'shing: kim uchun, qanday sharoitda, qanday cheklovlar bilan.",
            "Chegaralarni belgilang: nimaga tegmaslik kerak, nima o'zgarmasligi shart.",
            "Natijani darhol ishlatsa bo'ladigan ko'rinishda so'rang (to'liq fayl, ishlaydigan kod).",
            "Formatni aniq ayting: fayl nomlari, papka tuzilishi, kod tili.",
        ],
    },
    "google": {
        "structure": "Markdown sarlavhalar + aniq ro'yxatlar",
        "rules": [
            "Vazifani bosqichlarga bo'ling va har bosqich natijasini aniq ayting.",
            "Kontekstni oldinga qo'ying, ko'rsatmani oxiriga.",
            "Chiqish formatini namuna bilan ko'rsating.",
        ],
    },
}

_DEFAULT_STYLE = {
    "structure": "Markdown sarlavhalar bilan bo'limlarga ajratilgan",
    "rules": [
        "Vazifani aniq va to'liq bayon qiling.",
        "Texnik cheklovlarni ro'yxat qilib bering.",
        "Kutilgan natija formatini aniq ayting.",
    ],
}


@dataclass
class BuildContext:
    """Promptga kiradigan barcha faktlar."""

    description: str
    project_type: str
    project_title: str
    confidence: float
    signals: dict
    monthly_users: int
    stack: list[dict]
    anti_patterns: list[str]
    requirements: dict
    server_options: list[dict]
    scaling: list[str]
    uiux: dict
    domains: list[dict] = field(default_factory=list)
    payment: Optional[dict] = None
    target_model: str = "anthropic/claude-opus-5"
    target_model_name: str = "Claude Opus 5"
    lang: str = "uz"
    # Foydalanuvchi aniqlashtiruvchi savollarga bergan javoblar:
    # [{"question": ..., "answer": ...}]. Bo'sh bo'lishi mumkin.
    clarifications: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# RAG — tanlangan model uchun prompt qoidalarini topish
# --------------------------------------------------------------------------- #

# Har bir so'rov promptning boshqa jihatini qidiradi — bittasi hammasini topmaydi.
#
# So'rovlar oilaga qarab boshqacha: Claude hujjatlari "XML tags", "verbosity"
# atamalarini ishlatadi, OpenAI hujjatlari esa "describe the result",
# "set boundaries". Umumiy so'rov ro'yxati bilan noto'g'ri oila hujjati
# tortilib kelardi.
_QUERIES_BY_FAMILY: dict[str, list[str]] = {
    "claude": [
        "structure prompts with XML tags and clear sections",
        "be clear and direct give context and role",
        "control the format of responses and output style",
        "task scope constrain what to build avoid expanding",
        "agentic coding complete task specification multi file feature",
        "frontend design and UI implementation defaults",
        "response length verbosity conciseness",
        "use examples effectively few shot",
    ],
    "openai": [
        "describe the result you need not the process",
        "add useful context who it is for and constraints",
        "set boundaries that prevent real problems",
        "make the result ready to use finished files",
        "prompting codex explain a codebase and fix a bug",
        "prototype from a screenshot and iterate on UI",
        "improve the result with follow up messages",
        "put the pieces together full prompt example",
    ],
}

_GENERIC_QUERIES = [
    "write a clear complete task specification",
    "describe the expected output format and file structure",
    "give context constraints and success criteria",
    "build a full production ready application",
]

# Yakuniy to'plamda tanlangan oiladan kamida shuncha bo'lak bo'lishi kafolatlanadi.
_FAMILY_QUOTA = 4


def retrieve_guidance(ctx: BuildContext, per_query: int = 1, total: int = 4) -> list[RetrievedChunk]:
    """Korpusdan tanlangan modelga mos prompt qoidalarini yig'adi."""
    family = openrouter.family_of(ctx.target_model)
    doc_model = _MODEL_DOC_SLUG.get(ctx.target_model)
    queries = _QUERIES_BY_FAMILY.get(family, _GENERIC_QUERIES)

    seen: set[str] = set()
    quota: list[RetrievedChunk] = []
    picked: list[RetrievedChunk] = []

    def collect(target: list[RetrievedChunk], hits: list[RetrievedChunk]) -> None:
        for hit in hits:
            if hit.chunk.chunk_id in seen:
                continue
            seen.add(hit.chunk.chunk_id)
            target.append(hit)

    # 1) Kvota — faqat tanlangan oila ichidan.
    if family in _QUERIES_BY_FAMILY:
        for query in queries:
            if len(quota) >= 2:
                break
            collect(quota, engine.retriever.search(
                query, top_k=per_query, family=family, model=doc_model, family_only=True
            ))

    # 2) Qolgan joylar — butun korpusdan, oila/model bonusi bilan.
    project_query = f"{ctx.project_type} {ctx.description[:100]}"
    for query in [*queries[:2], project_query]:
        collect(picked, engine.retriever.search(
            query, top_k=per_query, family=family, model=doc_model
        ))

    quota.sort(key=lambda h: h.score, reverse=True)
    picked.sort(key=lambda h: h.score, reverse=True)
    selected = (quota[:2] + picked)[:total]
    selected.sort(key=lambda h: h.score, reverse=True)
    return selected


def format_guidance(hits: list[RetrievedChunk], max_chars: int = 2500) -> str:
    """Topilgan bo'laklarni manbasi bilan matnga aylantiradi."""
    out: list[str] = []
    used = 0
    for hit in hits:
        c = hit.chunk
        block = f"### {c.doc_title} — {c.section}\n(manba: {c.source})\n\n{c.text.strip()}\n"
        if used + len(block) > max_chars:
            break
        out.append(block)
        used += len(block)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Deterministik skelet
# --------------------------------------------------------------------------- #


# Skelet yorliqlari. Alohida jadval, chunki bu matn foydalanuvchi ko'radigan
# yakuniy mahsulot — interfeys emas, promptning o'zi. Foydalanuvchi ingliz
# tilini tanlab, o'zbekcha prompt olishi — mahsulotning asosiy va'dasi buzilishi.
_LABELS: dict[str, dict[str, str]] = {
    "uz": {
        "reason": "Sabab",
        "mau": "Kutilayotgan oylik faol foydalanuvchi",
        "dau": "Kunlik faol (hisoblangan)",
        "rpd": "Kunlik so'rov",
        "rps": "O'rtacha RPS: {avg}, cho'qqi RPS: {peak}",
        "min_res": "Minimal resurs: {vcpu} vCPU, {ram} GB RAM, {disk} GB disk",
        "traffic": "Oylik trafik",
        "db": "DB hajmi (prognoz)",
        "per_month": "oy",
        "with_addons": "qo'shimchalar bilan jami",
        "render": "Render strategiyasi",
        "budget": "O'lchanadigan maqsadlar",
        "no_domains": "(domen tekshirilmagan)",
        "available": "bo'sh",
        "taken": "band",
        "unknown": "noma'lum",
        "per_year": "yil",
        "no_price": "narx noma'lum",
        "no_answers": "qo'shimcha savollarga javob berilmadi",
        "hot_paths": "Yuk ortganda birinchi sinadigan joylar",
    },
    "ru": {
        "reason": "Причина",
        "mau": "Ожидаемые активные пользователи в месяц",
        "dau": "Активных в день (расчёт)",
        "rpd": "Запросов в день",
        "rps": "Средний RPS: {avg}, пиковый RPS: {peak}",
        "min_res": "Минимальные ресурсы: {vcpu} vCPU, {ram} ГБ RAM, {disk} ГБ диск",
        "traffic": "Трафик в месяц",
        "db": "Размер БД (прогноз)",
        "per_month": "мес",
        "with_addons": "с дополнениями итого",
        "render": "Стратегия рендеринга",
        "budget": "Измеримые цели",
        "no_domains": "(домен не проверялся)",
        "available": "свободен",
        "taken": "занят",
        "unknown": "неизвестно",
        "per_year": "год",
        "no_price": "цена неизвестна",
        "no_answers": "на дополнительные вопросы не ответили",
        "hot_paths": "Что первым не выдержит роста нагрузки",
    },
    "en": {
        "reason": "Why",
        "mau": "Expected monthly active users",
        "dau": "Daily active (derived)",
        "rpd": "Requests per day",
        "rps": "Average RPS: {avg}, peak RPS: {peak}",
        "min_res": "Minimum resources: {vcpu} vCPU, {ram} GB RAM, {disk} GB disk",
        "traffic": "Monthly bandwidth",
        "db": "Database size (projected)",
        "per_month": "mo",
        "with_addons": "with add-ons, total",
        "render": "Render strategy",
        "budget": "Measurable targets",
        "no_domains": "(domains not checked)",
        "available": "available",
        "taken": "taken",
        "unknown": "unknown",
        "per_year": "yr",
        "no_price": "price unknown",
        "no_answers": "the follow-up questions were not answered",
        "hot_paths": "What breaks first as load grows",
    },
}


def _labels(lang: str) -> dict[str, str]:
    return _LABELS.get(lang) or _LABELS["uz"]


def _fmt_money(value: float) -> str:
    return f"${value:,.0f}".replace(",", " ")


def _stack_block(stack: list[dict], lang: str) -> str:
    L = _labels(lang)
    lines = []
    for c in stack:
        lines.append(f"- **{c['category']}**: {c['pick']}")
        lines.append(f"  - {L['reason']}: {c['why']}")
    return "\n".join(lines)


def _server_block(ctx: BuildContext) -> str:
    L = _labels(ctx.lang)
    req = ctx.requirements
    lines = [
        f"- {L['mau']}: {ctx.monthly_users:,}".replace(",", " "),
        f"- {L['dau']}: {req['dau']:,}".replace(",", " "),
        f"- {L['rpd']}: {req['requests_per_day']:,}".replace(",", " "),
        "- " + L["rps"].format(avg=req["avg_rps"], peak=req["peak_rps"]),
        "- " + L["min_res"].format(vcpu=req["vcpu"], ram=req["ram_gb"], disk=req["disk_gb"]),
        f"- {L['traffic']}: ~{req['bandwidth_gb_month']} GB",
        f"- {L['db']}: ~{req['db_size_gb']} GB",
    ]
    for opt in ctx.server_options:
        plan = opt["plan"]
        lines.append(
            f"- **{opt['tier']}**: {plan['provider']} {plan['name']} "
            f"({plan['vcpu']} vCPU / {plan['ram_gb']} GB / {plan['disk_gb']} GB) — "
            f"{_fmt_money(plan['usd_month'])}/{L['per_month']}, {L['with_addons']} "
            f"{_fmt_money(opt['monthly_total_usd'])}/{L['per_month']}"
        )
    return "\n".join(lines)


def _uiux_block(uiux: dict, lang: str) -> str:
    L = _labels(lang)
    lines = [f"- {L['render']}: {uiux['render_strategy']['rule']}"]
    for tip in uiux["tips"]:
        lines.append(f"- {tip['area']}: {tip['rule']}")
    lines.append(f"- {L['budget']}: " + ", ".join(f"{k} {v}" for k, v in uiux["performance_budget"].items()))
    return "\n".join(lines)


def _bullets(items: list[str]) -> str:
    """Ro'yxatni bir xil ko'rinishdagi bulletlarga aylantiradi."""
    return "\n".join(f"- {x}" for x in items) if items else "- —"


def _clarifications_block(ctx: BuildContext) -> str:
    """Foydalanuvchining savollarga bergan javoblari.

    Bu blok promptdagi eng qimmatli ma'lumot: qolgan hamma narsa ML taxmini,
    bu esa muallifning o'z og'zidan chiqqan qaror. Shuning uchun u shablonda
    stack va infratuzilmadan OLDIN turadi — model uni birinchi o'qiydi.
    """
    L = _labels(ctx.lang)
    if not ctx.clarifications:
        return f"- {L['no_answers']}"
    lines = []
    for item in ctx.clarifications:
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not a:
            continue
        lines.append(f"- **{q}**")
        lines.append(f"  {a}")
    return "\n".join(lines) if lines else f"- {L['no_answers']}"


def _load_profile_block(ctx: BuildContext, hot_paths: list[str]) -> str:
    """Yuk xarakteristikasi va undan kelib chiqadigan aniq muhandislik talablari.

    Faqat "peak RPS = 12" deb yozish yetarli emas — model bu raqamdan nima
    qilish kerakligini o'zi taxmin qiladi. Shuning uchun raqamdan kelib
    chiqadigan qarorlar (kesh, navbat, CDN, pul (pool) hajmi) shu yerda
    ochiq aytiladi.
    """
    req = ctx.requirements
    lines: list[str] = [
        f"- Peak load to design for: {req['peak_rps']} req/s "
        f"({req['requests_per_day']:,} requests/day, {req['dau']:,} daily active users)".replace(",", " "),
        f"- Concurrent connections to expect: ~{req['concurrent_connections']}",
        f"- Projected database size: ~{req['db_size_gb']} GB; monthly bandwidth ~{req['bandwidth_gb_month']} GB",
    ]

    # Hisoblangan bayroqlardan kelib chiqadigan aniq talablar
    if req.get("needs_cache"):
        lines.append(
            "- Redis cache is REQUIRED at this load. Cache the read-heavy endpoints listed below, "
            "with explicit TTLs and explicit invalidation on write. State the cache key format for each."
        )
    if req.get("needs_queue"):
        lines.append(
            "- A background job queue is REQUIRED. Anything slower than ~200ms that the user does not "
            "need to wait for (email, export, image/video processing, third-party calls, notifications) "
            "runs as a job with retry and a dead-letter path — never inside the request."
        )
    if req.get("needs_cdn"):
        lines.append(
            "- A CDN is REQUIRED for static assets and media. The application server must never be the "
            "one serving images, video or bundles at this volume."
        )
    if req.get("needs_object_storage"):
        lines.append(
            "- Object storage (S3-compatible) is REQUIRED for user uploads. Never store uploads on the "
            "application server's local disk — it does not survive a redeploy and does not scale horizontally."
        )
    if req.get("needs_managed_db"):
        lines.append(
            "- A managed database is REQUIRED (automatic backups, point-in-time recovery, patching). "
            "Include the backup and restore procedure in the README."
        )

    lines.append(
        f"- Database connection pool: size it for the worker count, not for the user count. "
        f"With {req['vcpu']} vCPU plan for roughly {max(4, req['vcpu'] * 4)} pooled connections total, "
        f"and state the value explicitly in the configuration."
    )
    lines.append(
        "- Every list endpoint is paginated (cursor-based where the data shifts). No endpoint may return "
        "an unbounded result set."
    )
    lines.append(
        "- Every foreign key and every column used in a WHERE, ORDER BY or JOIN gets an index. "
        "For each hot query below, state the exact composite index that serves it."
    )
    lines.append(
        "- Rate limiting on authentication, write endpoints and any expensive operation, enforced in shared "
        "storage (Redis) rather than in process memory, so it still holds with several instances running."
    )

    if hot_paths:
        L = _labels(ctx.lang)
        lines.append(f"- {L['hot_paths']}:")
        lines.extend(f"  - {h}" for h in hot_paths)

    return "\n".join(lines)


def _domain_block(domains: list[dict], lang: str) -> str:
    L = _labels(lang)
    if not domains:
        return f"- {L['no_domains']}"
    lines = []
    label = {"available": L["available"], "taken": L["taken"], "unknown": L["unknown"]}
    for d in domains[:6]:
        price = (
            f"{_fmt_money(d['first_year_usd'])}/{L['per_year']}"
            if d.get("first_year_usd") else L["no_price"]
        )
        lines.append(f"- {d['domain']} — {label.get(d['availability'], d['availability'])}, {price}")
    return "\n".join(lines)


# Skeletning o'zi — har til uchun to'liq shablon.
#
# Ataylab bo'laklarga bo'linmagan. Bu matnni odam boshidan oxirigacha o'qiydi va
# nusxa olib ishlatadi; o'ttizta kalitdan yig'ilgan hujjatni na tarjimon, na
# muallif bir butun holda ko'ra oladi — natijada uzilib qolgan jumlalar chiqadi.
_SKELETON: dict[str, str] = {}

_SKELETON["uz"] = """<role>
Siz tajribali full-stack muhandis va mahsulot arxitektorisiz. Quyidagi startup
uchun ishlab chiqarishga tayyor (production-ready) mahsulot quryapsiz. Bu
o'quv loyihasi yoki demo emas — kod real foydalanuvchilar va real pul bilan
ishlaydi.
</role>

<project>
Startup g'oyasi (muallif so'zlari bilan):
{description}

Aniqlangan tur: {project_title}
Texnik signallar: {signals}
Kutilayotgan hajm: oyiga ~{monthly_users} faol foydalanuvchi
</project>

<founder_answers>
Muallif aniqlashtiruvchi savollarga quyidagicha javob berdi. BU BO'LIM ENG
MUHIM: qolgan hamma narsa avtomatik tahlil natijasi, bu esa muallifning o'z
qarori. Ular bilan boshqa bo'lim ziddiyatga tushsa, shu yerdagi javob ustun
turadi.

{clarifications}
</founder_answers>

<domain_model>
Shu turdagi mahsulot uchun ma'lumotlar modelining o'zagi quyidagicha bo'ladi.
Bu ro'yxatni asos qilib oling, muallif javoblariga qarab kengaytiring va
har bir obyekt uchun to'liq sxema yozing (maydonlar, tiplar, majburiylik,
noyoblik, tashqi kalitlar, indekslar):

{entities}

Har bir jadval uchun quyidagilarni aniq ko'rsating:
- birlamchi kalit turi (UUID yoki bigint — tanlovni asoslang);
- yaratilgan va o'zgartirilgan vaqt maydonlari;
- yumshoq o'chirish (soft delete) kerakmi yoki yo'qmi va nega;
- qaysi maydonlar bo'yicha indeks va nega aynan shu tartibda (kompozit
  indekslarda ustunlar ketma-ketligi muhim).
</domain_model>

<core_flows>
Quyidagi stsenariylar mahsulotning o'zagi. Ularning har biri uchidan uchigacha
ishlashi shart — birortasi ham chala qolmasin:

{flows}

Har bir stsenariy uchun xatolik yo'lini ham yozing: tashqi xizmat javob
bermasa, to'lov rad etilsa, foydalanuvchi yarim yo'lda chiqib ketsa nima
bo'ladi.
</core_flows>

<required_features>
Bu turdagi mahsulot uchun quyidagilar majburiy. Bularsiz mahsulot ishlamaydi
yoki huquqiy/moliyaviy muammo tug'diradi:

{must_have}
</required_features>

<tech_stack>
Quyidagi stack tanlangan. Boshqa texnologiyaga o'tmang — sabablari yozilgan:

{stack}
</tech_stack>

<infrastructure>
Hisoblangan yuk va tanlangan server:

{server}

Masshtablash rejasi:
{scaling}
</infrastructure>

<load_and_performance>
Bu bo'lim shu mahsulot turining trafik xarakteristikasidan kelib chiqadi.
Kodni aynan shu yuk uchun yozing — "keyin optimallashtiramiz" deb qoldirmang,
chunki sanab o'tilgan joylar birinchi kunlardanoq yuklanadi:

{load_profile}

Qo'shimcha talablar:
- Har bir og'ir so'rov uchun `EXPLAIN` natijasini nazarda tuting: to'liq jadval
  skani (sequential scan) qolmasin.
- N+1 so'rovlarni oldindan yo'q qiling — bog'liq ma'lumotlarni bitta so'rovda
  yuklang.
- Statik va media fayllar ilova serveridan uzatilmasin.
- Sekin tashqi chaqiruvlar uchun taymaut va qayta urinish (backoff bilan)
  belgilang; javob kelmasa mahsulot butunlay to'xtab qolmasin.
</load_and_performance>

<security>
Xavfsizlik talablari — bularni keyinga qoldirib bo'lmaydi:
- Har bir endpoint autentifikatsiyani VA huquqni tekshiradi. "Bu sahifa
  interfeysda ko'rinmaydi" — himoya emas.
- Har bir so'rov tanasi va parametri sxema orqali validatsiya qilinadi
  (Pydantic/Zod kabi). Validatsiyadan o'tmagan ma'lumot bazaga yetib bormaydi.
- SQL faqat parametrlangan so'rovlar yoki ORM orqali. Satr birlashtirish yo'q.
- Parollar zamonaviy algoritm bilan xeshlanadi (bcrypt/argon2), hech qachon
  ochiq saqlanmaydi va loglarga tushmaydi.
- Sirlar (API kalit, parol, token) faqat muhit o'zgaruvchilarida. Kodda,
  git tarixida va frontend bundle'ida sir bo'lmaydi.
- CORS aniq ro'yxat bilan cheklanadi, `*` emas.
- Fayl yuklashda: tur va hajm tekshiriladi, nom tozalanadi, fayl ilova
  ildizidan tashqarida saqlanadi.
- Kirish, parol tiklash va boshqa nozik amallar uchun urinishlar soni
  cheklanadi.
- Xato xabarlari foydalanuvchiga ichki tafsilotni (stack trace, SQL, fayl
  yo'li) ko'rsatmaydi — ular faqat logga yoziladi.
</security>

<ui_ux>
Interfeys quyidagi qoidalarga qat'iy rioya qilsin — maqsad: server va brauzerga
ortiqcha yuk tushmasin.

{uiux}

Bundan tashqari:
- Har bir yuklanish holati (loading), bo'sh holat (empty) va xato holati
  (error) uchun alohida ko'rinish bo'lsin — faqat "muvaffaqiyatli" holatni
  chizib qo'ymang.
- Barcha shakllarda validatsiya xabari maydon yonida chiqadi, faqat yuqorida
  emas.
- Klaviatura bilan to'liq ishlash mumkin, fokus ko'rinadi.
- Matn kontrasti kamida 4.5:1.
</ui_ux>

<domain_and_payment>
Domen variantlari:
{domains}

To'lov tizimi: {payment}
</domain_and_payment>

<constraints>
Quyidagilarni QILMANG. Birinchi ro'yxat — umumiy, ikkinchisi — aynan shu
turdagi mahsulotlarda tez-tez uchraydigan xatolar:

{anti_patterns}

{pitfalls}
</constraints>

<deliverables>
Quyidagilarni to'liq, ishlaydigan holda yozing — stub, TODO yoki "keyin
qo'shiladi" qoldirmang:

1. Papka tuzilishi (butun loyiha uchun, fayl nomlari bilan).
2. Ma'lumotlar bazasi sxemasi: jadvallar, maydonlar, tiplar, cheklovlar,
   indekslar, bog'lanishlar va migratsiya fayli.
3. Backend: barcha endpoint'lar (metod, yo'l, so'rov/javob sxemasi, status
   kodlari), autentifikatsiya, huquq tekshiruvi, xatolar bilan ishlashning
   yagona formati.
4. Frontend: barcha sahifalar va asosiy komponentlar, holat boshqaruvi,
   ma'lumot olish qatlami, yuklanish/bo'sh/xato holatlari.
5. Fon vazifalari: navbat, ishchi (worker) jarayoni, qayta urinish qoidasi.
6. Docker Compose + `.env.example` + deploy bosqichlari.
7. Testlar (quyidagi bo'limga qarang).
8. README: lokal ishga tushirish, muhit o'zgaruvchilari ro'yxati, deploy va
   zaxira nusxa (backup) yo'riqnomasi.
</deliverables>

<testing>
Testlar quyidagilarni qamrab olsin — 100% qoplash kerak emas, lekin bu
nuqtalar tekshirilmasa mahsulot ishonchsiz:
- Har bir asosiy stsenariy uchun uchidan uchigacha (end-to-end) bitta test.
- Huquq tekshiruvi: begona foydalanuvchi ma'lumotiga kirishga urinish 403
  qaytarishini tasdiqlovchi test.
- Poyga holati (race condition) bo'lishi mumkin bo'lgan joylar uchun
  parallel so'rovli test (masalan bitta oxirgi tovarni ikki xaridor olishi).
- Pul yoki hisob bilan bog'liq har bir amal uchun aniq kutilgan natijali test.
- Validatsiya: noto'g'ri kirish ma'lumoti 4xx qaytaradi, 500 emas.
</testing>

<observability>
Ishlab chiqarishda nima bo'layotganini ko'rish uchun:
- Strukturaviy loglar (JSON), har bir yozuvda so'rov identifikatori bo'lsin.
- Sog'liq tekshiruvi endpoint'i (`/health`) — baza va keshga ulanishni ham
  tekshiradi.
- Xatoliklarni kuzatuvchi vosita (Sentry va h.k.) uchun ulanish nuqtasi.
- Asosiy metrikalar: so'rov soni, javob vaqti (p95), xatolik ulushi, navbat
  uzunligi.
</observability>

<definition_of_done>
- `docker compose up` dan keyin loyiha lokal ishlaydi.
- Ro'yxatdan o'tish → asosiy amal → natija ko'rish stsenariysi uzilishsiz ishlaydi.
- Har bir endpoint huquqni tekshiradi; boshqa foydalanuvchi ma'lumotiga kirib bo'lmaydi.
- Yuqoridagi yuk ko'rsatkichlarida sekinlashuv yo'q: hot so'rovlar indeksdan
  foydalanadi, kesh va navbat o'z o'rnida ishlaydi.
- UI yuqoridagi o'lchanadigan maqsadlarga javob beradi.
- Hech qanday sir (kalit, parol) kodda yozilmagan — hammasi `.env` da.
- Testlar o'tadi.
</definition_of_done>

<output_format>
Avval qisqa arxitektura izohi (10 qatordan oshmasin), keyin fayllar bo'yicha
to'liq kod. Har bir fayl `// path/to/file` sarlavhasi bilan boshlansin.
Tushuntirishni kodga izoh sifatida emas, kod bloklari orasida bering.
Kod ko'p bo'lsa, bo'limlarga bo'lib, tugallangan holda yozing — yarim
qoldirilgan fayldan ko'ra kamroq fayl, lekin to'liq bo'lgani yaxshi.
</output_format>"""

_SKELETON["ru"] = """<role>
Вы — опытный full-stack инженер и продуктовый архитектор. Вы собираете
production-ready продукт для стартапа, описанного ниже. Это не учебный проект
и не демо — код будет работать с реальными пользователями и реальными деньгами.
</role>

<project>
Идея стартапа (словами автора):
{description}

Определённый тип: {project_title}
Технические сигналы: {signals}
Ожидаемый объём: ~{monthly_users} активных пользователей в месяц
</project>

<founder_answers>
Автор ответил на уточняющие вопросы так. ЭТОТ РАЗДЕЛ ВАЖНЕЕ ОСТАЛЬНЫХ: всё
прочее — результат автоматического анализа, а здесь решение самого автора.
Если другой раздел противоречит этим ответам, приоритет у ответов.

{clarifications}
</founder_answers>

<domain_model>
Ядро модели данных для продукта такого типа выглядит так. Возьмите этот
список за основу, расширьте его по ответам автора и напишите полную схему
для каждой сущности (поля, типы, обязательность, уникальность, внешние
ключи, индексы):

{entities}

Для каждой таблицы явно укажите:
- тип первичного ключа (UUID или bigint — обоснуйте выбор);
- поля времени создания и изменения;
- нужно ли мягкое удаление (soft delete) и почему;
- по каким полям индексы и почему именно в таком порядке (в составных
  индексах порядок столбцов решает).
</domain_model>

<core_flows>
Следующие сценарии — ядро продукта. Каждый из них должен работать от начала
до конца, ни один не остаётся наполовину:

{flows}

Для каждого сценария опишите и путь ошибки: что происходит, если внешний
сервис не ответил, платёж отклонён или пользователь ушёл на полпути.
</core_flows>

<required_features>
Для продукта такого типа перечисленное обязательно. Без этого продукт не
работает либо создаёт юридические и финансовые проблемы:

{must_have}
</required_features>

<tech_stack>
Выбран следующий стек. Не переходите на другие технологии — причины указаны:

{stack}
</tech_stack>

<infrastructure>
Расчётная нагрузка и выбранный сервер:

{server}

План масштабирования:
{scaling}
</infrastructure>

<load_and_performance>
Этот раздел выведен из характера трафика именно этого типа продукта. Пишите
код сразу под такую нагрузку — не откладывайте на «потом оптимизируем»,
потому что перечисленные места нагружаются с первых дней:

{load_profile}

Дополнительно:
- Для каждого тяжёлого запроса держите в уме результат `EXPLAIN`: полного
  сканирования таблицы (sequential scan) быть не должно.
- Заранее устраните N+1: связанные данные загружаются одним запросом.
- Статика и медиа не отдаются с сервера приложения.
- Для медленных внешних вызовов задайте таймаут и повтор с экспоненциальной
  задержкой; отсутствие ответа не должно останавливать весь продукт.
</load_and_performance>

<security>
Требования безопасности — их нельзя отложить:
- Каждый эндпоинт проверяет аутентификацию И права. «Этой страницы нет в
  интерфейсе» — не защита.
- Каждое тело запроса и каждый параметр валидируются по схеме (Pydantic/Zod).
  Непровалидированные данные до базы не доходят.
- SQL только через параметризованные запросы или ORM. Никакой конкатенации строк.
- Пароли хешируются современным алгоритмом (bcrypt/argon2), никогда не
  хранятся в открытом виде и не попадают в логи.
- Секреты (API-ключи, пароли, токены) только в переменных окружения. Ни в
  коде, ни в истории git, ни во frontend-бандле.
- CORS ограничен явным списком, а не `*`.
- Загрузка файлов: проверка типа и размера, очистка имени, хранение вне
  корня приложения.
- Ограничение числа попыток для входа, восстановления пароля и других
  чувствительных операций.
- Сообщения об ошибках не показывают пользователю внутренние детали
  (stack trace, SQL, пути к файлам) — они идут только в лог.
</security>

<ui_ux>
Интерфейс должен строго следовать правилам ниже. Цель — не создавать лишней
нагрузки ни на сервер, ни на браузер.

{uiux}

Кроме того:
- Для каждого экрана предусмотрите состояние загрузки, пустое состояние и
  состояние ошибки — не рисуйте только «успешный» вариант.
- Во всех формах сообщение валидации появляется рядом с полем, а не только
  сверху.
- Полная работа с клавиатуры, фокус видим.
- Контраст текста не ниже 4.5:1.
</ui_ux>

<domain_and_payment>
Варианты домена:
{domains}

Платёжная система: {payment}
</domain_and_payment>

<constraints>
НЕ делайте следующего. Первый список — общий, второй — ошибки, характерные
именно для этого типа продукта:

{anti_patterns}

{pitfalls}
</constraints>

<deliverables>
Напишите всё перечисленное полностью и в рабочем виде — без заглушек, TODO
и «добавим позже»:

1. Структура папок (для всего проекта, с именами файлов).
2. Схема базы данных: таблицы, поля, типы, ограничения, индексы, связи и
   файл миграции.
3. Бэкенд: все эндпоинты (метод, путь, схема запроса/ответа, коды статусов),
   аутентификация, проверка прав, единый формат обработки ошибок.
4. Фронтенд: все страницы и основные компоненты, управление состоянием, слой
   получения данных, состояния загрузки/пустоты/ошибки.
5. Фоновые задачи: очередь, воркер, правила повторов.
6. Docker Compose + `.env.example` + шаги деплоя.
7. Тесты (см. раздел ниже).
8. README: локальный запуск, список переменных окружения, инструкция по
   деплою и резервному копированию.
</deliverables>

<testing>
Тесты должны покрывать следующее — стопроцентное покрытие не нужно, но без
этих точек продукту нельзя доверять:
- По одному сквозному (end-to-end) тесту на каждый основной сценарий.
- Проверка прав: тест, подтверждающий, что попытка получить чужие данные
  возвращает 403.
- Тест с параллельными запросами для мест с возможной гонкой (например, два
  покупателя на последний товар).
- Для каждой операции с деньгами или балансом — тест с точным ожидаемым
  результатом.
- Валидация: некорректный ввод возвращает 4xx, а не 500.
</testing>

<observability>
Чтобы видеть, что происходит в продакшене:
- Структурированные логи (JSON), в каждой записи идентификатор запроса.
- Эндпоинт проверки состояния (`/health`), проверяющий и базу, и кеш.
- Точка подключения трекера ошибок (Sentry и т.п.).
- Ключевые метрики: число запросов, время ответа (p95), доля ошибок, длина
  очереди.
</observability>

<definition_of_done>
- После `docker compose up` проект работает локально.
- Сценарий регистрация → основное действие → просмотр результата проходит без сбоев.
- Каждый эндпоинт проверяет права; доступ к чужим данным невозможен.
- На указанной выше нагрузке нет замедления: горячие запросы идут по индексам,
  кеш и очередь работают на своих местах.
- UI укладывается в измеримые цели, указанные выше.
- Ни один секрет (ключ, пароль) не записан в коде — всё в `.env`.
- Тесты проходят.
</definition_of_done>

<output_format>
Сначала краткое описание архитектуры (не более 10 строк), затем полный код по
файлам. Каждый файл начинается с заголовка `// path/to/file`. Пояснения давайте
между блоками кода, а не комментариями внутри кода. Если кода много, разбейте
на разделы, но пишите завершённо — лучше меньше файлов, но целиком, чем
брошенный на середине файл.
</output_format>"""

_SKELETON["en"] = """<role>
You are an experienced full-stack engineer and product architect. You are
building a production-ready product for the startup described below. This is
not a tutorial project or a demo — the code will run with real users and real
money.
</role>

<project>
The startup idea, in the founder's own words:
{description}

Detected type: {project_title}
Technical signals: {signals}
Expected scale: ~{monthly_users} monthly active users
</project>

<founder_answers>
The founder answered the follow-up questions as follows. THIS SECTION OUTRANKS
THE OTHERS: everything else is the output of automated analysis, while this is
the founder's own decision. Where another section conflicts with these answers,
these answers win.

{clarifications}
</founder_answers>

<domain_model>
The core of the data model for this product type looks like this. Take it as
the baseline, extend it according to the founder's answers, and write the full
schema for every entity (fields, types, nullability, uniqueness, foreign keys,
indexes):

{entities}

For every table, state explicitly:
- the primary key type (UUID or bigint — justify the choice);
- created/updated timestamp columns;
- whether soft delete is needed and why;
- which columns are indexed and why in that specific order (in composite
  indexes, column order decides whether the index is used at all).
</domain_model>

<core_flows>
The following scenarios are the core of the product. Each one must work end to
end; none may be left half-finished:

{flows}

For every scenario, describe the failure path as well: what happens when an
external service does not respond, a payment is declined, or the user abandons
the flow halfway.
</core_flows>

<required_features>
The following are mandatory for this product type. Without them the product
either does not work or creates legal and financial problems:

{must_have}
</required_features>

<tech_stack>
This stack has been chosen. Do not substitute other technologies — the reasons
are given:

{stack}
</tech_stack>

<infrastructure>
Calculated load and the selected server:

{server}

Scaling plan:
{scaling}
</infrastructure>

<load_and_performance>
This section is derived from the traffic characteristics of this specific
product type. Write the code for this load from the start — do not defer it as
"we will optimise later", because the places listed here are under pressure
from the first days:

{load_profile}

In addition:
- Keep the `EXPLAIN` output in mind for every heavy query: no sequential scans
  on large tables.
- Eliminate N+1 queries up front — load related data in a single query.
- Static assets and media are never served by the application server.
- Set timeouts and retry-with-backoff for slow external calls; a missing
  response must not take the whole product down.
</load_and_performance>

<security>
Security requirements — none of these can be deferred:
- Every endpoint checks authentication AND authorisation. "The page is not
  linked in the UI" is not protection.
- Every request body and parameter is validated against a schema (Pydantic/Zod
  or equivalent). Unvalidated data never reaches the database.
- SQL only through parameterised queries or an ORM. No string concatenation.
- Passwords hashed with a modern algorithm (bcrypt/argon2), never stored in
  plain text and never written to logs.
- Secrets (API keys, passwords, tokens) only in environment variables. Not in
  the code, not in git history, not in the frontend bundle.
- CORS restricted to an explicit allowlist, not `*`.
- File uploads: validate type and size, sanitise the filename, store outside
  the application root.
- Rate limiting on sign-in, password reset and other sensitive operations.
- Error messages never expose internals (stack trace, SQL, file paths) to the
  user — those go to the log only.
</security>

<ui_ux>
The interface must follow the rules below exactly. The goal is to put no
unnecessary load on either the server or the browser.

{uiux}

In addition:
- Every screen needs a loading state, an empty state and an error state — do
  not design only the successful path.
- In every form, the validation message appears next to the field, not only at
  the top.
- Full keyboard operation with a visible focus indicator.
- Text contrast at least 4.5:1.
</ui_ux>

<domain_and_payment>
Domain options:
{domains}

Payment system: {payment}
</domain_and_payment>

<constraints>
Do NOT do any of the following. The first list is general; the second is the
set of mistakes that specifically recur in this product type:

{anti_patterns}

{pitfalls}
</constraints>

<deliverables>
Write all of the following in full and working order — no stubs, no TODOs, no
"to be added later":

1. Folder structure (for the whole project, with file names).
2. Database schema: tables, fields, types, constraints, indexes, relations and
   the migration file.
3. Backend: every endpoint (method, path, request/response schema, status
   codes), authentication, authorisation checks, one consistent error format.
4. Frontend: every page and the main components, state management, the data
   fetching layer, loading/empty/error states.
5. Background work: the queue, the worker process, the retry policy.
6. Docker Compose + `.env.example` + deployment steps.
7. Tests (see the section below).
8. README: how to run locally, the full list of environment variables, how to
   deploy and how to back up and restore.
</deliverables>

<testing>
Tests must cover the following — full coverage is not required, but without
these points the product cannot be trusted:
- One end-to-end test per core scenario.
- An authorisation test proving that reaching another user's data returns 403.
- A concurrent-request test for every place where a race is possible (for
  example two buyers competing for the last item in stock).
- For every money or balance operation, a test with an exact expected result.
- Validation: malformed input returns 4xx, not 500.
</testing>

<observability>
So that production behaviour is visible:
- Structured logs (JSON) with a request id on every entry.
- A health endpoint (`/health`) that also checks the database and the cache.
- An integration point for an error tracker (Sentry or equivalent).
- Key metrics: request count, response time (p95), error rate, queue depth.
</observability>

<definition_of_done>
- The project runs locally after `docker compose up`.
- Sign up → main action → see the result runs end to end without a break.
- Every endpoint checks authorisation; no user can reach another user's data.
- At the load stated above there is no degradation: hot queries use indexes,
  the cache and the queue do their jobs.
- The UI meets the measurable targets stated above.
- No secret (key, password) is written in the code — all of them live in `.env`.
- The tests pass.
</definition_of_done>

<output_format>
First a short architecture note (10 lines at most), then the full code file by
file. Start each file with a `// path/to/file` header. Put explanations between
code blocks rather than as comments inside the code. If there is a lot of code,
split it into sections but keep every file complete — fewer whole files beat a
file abandoned halfway.
</output_format>"""


# Shablonlarda yo'q, chunki bular qiymatning o'rniga tushadigan qisqa iboralar.
_NO_SIGNALS = {
    "uz": "maxsus signal aniqlanmadi",
    "ru": "особых сигналов не обнаружено",
    "en": "no special signals detected",
}
_NO_PAYMENT = {
    "uz": "To'lov talab qilinmaydi",
    "ru": "Оплата не требуется",
    "en": "No payment required",
}


def build_skeleton(ctx: BuildContext) -> str:
    """Faktlarga asoslangan to'liq topshiriq. OpenRouter ishlamasa ham shu beriladi.

    Til `ctx.lang` bo'yicha tanlanadi. Bu shunchaki qulaylik emas: prompt —
    mahsulotning o'zi, va ingliz tilini tanlagan odamga o'zbekcha prompt berish
    uni butunlay yaroqsiz qiladi.
    """
    lang = ctx.lang if ctx.lang in _SKELETON else "uz"
    signals_on = [k for k, v in ctx.signals.items() if v]

    # Turga xos bilim — promptni "hamma uchun bir xil shablon" bo'lishdan
    # qutqaradigan qism. Noma'lum tur uchun umumiy yo'riqnoma qaytadi.
    pb = project_playbooks.get(ctx.project_type)

    return _SKELETON[lang].format(
        description=ctx.description.strip(),
        project_title=ctx.project_title,
        signals=", ".join(signals_on) if signals_on else _NO_SIGNALS[lang],
        monthly_users=f"{ctx.monthly_users:,}".replace(",", " "),
        clarifications=_clarifications_block(ctx),
        entities=_bullets(pb.entities),
        flows=_bullets(pb.flows),
        must_have=_bullets(pb.must_have),
        stack=_stack_block(ctx.stack, lang),
        server=_server_block(ctx),
        scaling="\n".join("- " + s for s in ctx.scaling),
        load_profile=_load_profile_block(ctx, pb.hot_paths),
        uiux=_uiux_block(ctx.uiux, lang),
        domains=_domain_block(ctx.domains, lang),
        payment=(
            f"{ctx.payment['provider']} — {ctx.payment['why']}"
            if ctx.payment else _NO_PAYMENT[lang]
        ),
        anti_patterns="\n".join("- " + a for a in ctx.anti_patterns),
        pitfalls=_bullets(pb.pitfalls),
    )


# --------------------------------------------------------------------------- #
# Yozuvchi model uchun meta-prompt
# --------------------------------------------------------------------------- #


# Yakuniy promptning tili. Texnik atamalar hamma variantda inglizcha qoladi —
# "ma'lumotlar bazasi migratsiyasi" deb yozilgan prompt modelni chalg'itadi.
_OUTPUT_LANG = {
    "uz": "Prompt o'zbek tilida bo'lsin, texnik atamalar inglizcha qolsin.",
    "ru": "Промпт должен быть на русском языке, технические термины оставьте на английском.",
    "en": "Write the prompt in English.",
}


def build_meta_prompt(ctx: BuildContext, guidance: str, skeleton: str) -> list[dict]:
    """Yozuvchi modelga: "shu faktlar + shu qoidalar asosida final promptni yoz"."""
    family = openrouter.family_of(ctx.target_model)
    style = _FAMILY_STYLE.get(family, _DEFAULT_STYLE)
    rules = "\n".join(f"- {r}" for r in style["rules"])

    system = (
        "Siz prompt-engineering bo'yicha mutaxassissiz. Sizning vazifangiz — "
        "boshqa AI modelga beriladigan YAKUNIY PROMPTni yozish. Siz mahsulotni "
        "o'zingiz qurmaysiz va kod yozmaysiz; siz faqat promptni yozasiz.\n\n"
        "Qat'iy qoidalar:\n"
        "1. Berilgan faktlarni (stack, server, narx, domen, UI/UX raqamlari) "
        "AYNAN saqlang. Hech narsani o'ylab topmang va o'zgartirmang.\n"
        "2. Natija — foydalanuvchi nusxa olib, to'g'ridan-to'g'ri modelga "
        "yuboradigan tayyor prompt bo'lsin.\n"
        "3. Prompt haqida izoh bermang, muqaddima yozmang. Faqat promptning "
        "o'zini qaytaring.\n"
        f"4. {_OUTPUT_LANG.get(ctx.lang, _OUTPUT_LANG['uz'])}"
    )

    user = f"""Yakuniy prompt **{ctx.target_model_name}** (`{ctx.target_model}`) modeliga beriladi.

<target_model_guidance>
Quyida aynan shu model oilasi uchun rasmiy hujjatlardan olingan prompt-engineering
ko'rsatmalari. Yakuniy promptni SHU ko'rsatmalarga muvofiq tuzing.

Tuzilish: {style['structure']}

Asosiy qoidalar:
{rules}

Rasmiy hujjatlardan tegishli parchalar:
{guidance}
</target_model_guidance>

<facts_and_draft>
Quyida ML tahlili natijasida tayyorlangan qoralama prompt. Undagi barcha
texnik faktlar to'g'ri va o'zgartirilmasligi kerak:

{skeleton}
</facts_and_draft>

<your_task>
Yuqoridagi qoralamani {ctx.target_model_name} uchun optimallashtirilgan
YAKUNIY promptga aylantiring:

- Tuzilishni target model uslubiga moslang.
- QORALAMADAGI BARCHA BO'LIMLAR yakuniy promptda ham qolsin. Bo'limni
  tashlab ketish, bir necha bo'limni bittaga qo'shib yuborish yoki ro'yxatni
  qisqartirish MUMKIN EMAS — har bir band ataylab qo'yilgan.
- Faktlarni (raqamlar, narxlar, texnologiya nomlari, domenlar) aynan saqlang.
- Muallif javoblari bo'limi eng ustun: agar boshqa bo'lim unga zid kelsa,
  muallif javobiga moslang.
- Noaniq joylarni aniq, o'lchanadigan talabga aylantiring.
- Umumiy jumlalarni ("yaxshi kod yozing", "xavfsiz bo'lsin") aniq, tekshirsa
  bo'ladigan talabga almashtiring.
- Yakuniy prompt qoralamadan QISQA bo'lmasligi kerak. Maqsad — ixchamlashtirish
  emas, aniqlashtirish va target model uslubiga moslash.
- Prompt oxirida mahsulot tayyor deb hisoblanishi uchun bajarilishi shart
  bo'lgan aniq shartlar ro'yxati bo'lsin.

Faqat yakuniy promptni qaytaring.
</your_task>"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Yuqori darajali funksiya
# --------------------------------------------------------------------------- #


@dataclass
class GeneratedPrompt:
    prompt: str
    skeleton: str
    used_llm: bool
    target_model: str
    sources: list[dict]
    warning: Optional[str] = None


# Model ishlamaganda foydalanuvchi ko'radigan yagona jumla. Texnik sabab
# jurnalda qoladi: xato matnini ekranga chiqarish odamga hech narsa bermaydi,
# faqat mahsulot buzilgandek ko'rsatadi.
# Ilgari bu xabar "prompt shablondan yig'ildi, u ham to'liq va ishlatishga
# tayyor" derdi. Bu chalg'itardi: foydalanuvchi AI qayta yozgan variantni
# olyapman deb o'ylab, aslida shablonni olardi va nega sifat pastligini
# tushunmasdi. Endi sabab ochiq aytiladi — shablon baribir beriladi, lekin
# uni to'liq qiymatli natija deb ko'rsatmaymiz.
_FALLBACK_NOTE = {
    "uz": "AI xizmati vaqtinchalik ishlamayapti. Shablon asosidagi variant berildi — biroz keyin qayta urinib ko'ring.",
    "ru": "AI-сервис временно не работает. Выдан вариант на основе шаблона — попробуйте позже.",
    "en": "The AI service is temporarily unavailable. A template-based version was returned — please try again later.",
}

# Qayta yozish bo'limlarni yo'qotgan holat — foydalanuvchi to'liq variantni oladi.
_TRUNCATED_NOTE = {
    "uz": "Qayta yozilgan variant to'liq chiqmadi, shuning uchun to'liq shablon versiyasi berildi.",
    "ru": "Переписанный вариант вышел неполным, поэтому выдана полная версия из шаблона.",
    "en": "The rewritten version came out incomplete, so the full template version was returned instead.",
}


async def generate(ctx: BuildContext) -> GeneratedPrompt:
    """Yakuniy promptni qaytaradi.

    Standart holatda topshiriq to'liq shu yerda yig'iladi: ML yadro turni
    tasniflaydi va signallarni ajratadi, qoidalar stack/server/UI-UX/domenni
    tanlaydi, playbook turga xos obyekt va oqimlarni beradi, retriever esa
    korpusdan tegishli manbalarni topadi. Tashqi model chaqirilmaydi.

    `USE_LLM=true` bo'lsa, yig'ilgan matn ustidan LLM qayta yozib chiqadi.
    """
    hits = retrieve_guidance(ctx)
    guidance = format_guidance(hits)
    skeleton = build_skeleton(ctx)
    sources = [
        {
            "doc": h.chunk.doc_title,
            "section": h.chunk.section,
            "url": h.chunk.source,
            "score": round(h.score, 4),
        }
        for h in hits
    ]

    if not get_settings().use_llm:
        # Ogohlantirish YO'Q: bu zaxira yo'l emas, asosiy yo'l. Ilgari shu
        # holatda "AI ishlamayapti" deyilardi va foydalanuvchi to'liq
        # natijani nosozlik deb qabul qilardi.
        return GeneratedPrompt(
            prompt=skeleton, skeleton=skeleton, used_llm=False,
            target_model=ctx.target_model, sources=sources,
        )

    try:
        messages = build_meta_prompt(ctx, guidance, skeleton)
        text, provider = await llm.complete(
            messages,
            max_tokens=_PROMPT_MAX_TOKENS,
            temperature=0.3,
            # Mahsulotning o'zi shu matn — eng kuchli zanjir shu yerga
            # yo'naltiriladi. Savollar (clarifier) esa arzon zanjirda qoladi.
            heavy=True,
        )
        text = (text or "").strip()
        logger.info("Prompt %s orqali qayta yozildi", provider)

        if text:
            # Qisqarib ketganini tekshiramiz. Ikki xil sabab bo'ladi: token
            # chegarasiga urilib yarim yo'lda uzilgan yoki model "ixchamlashtirdim"
            # deb bo'limlarni tashlab ketgan. Ikkalasida ham to'liq skelet afzal.
            if len(text) < len(skeleton) * _MIN_LENGTH_RATIO:
                logger.warning(
                    "Qayta yozilgan prompt juda qisqa (%d < %d belgi) — skelet qaytarilyapti",
                    len(text), len(skeleton),
                )
                return GeneratedPrompt(
                    prompt=skeleton, skeleton=skeleton, used_llm=False,
                    target_model=ctx.target_model, sources=sources,
                    warning=_TRUNCATED_NOTE.get(ctx.lang, _TRUNCATED_NOTE["uz"]),
                )

            return GeneratedPrompt(
                prompt=text, skeleton=skeleton, used_llm=True,
                target_model=ctx.target_model, sources=sources,
            )

        logger.warning("Model bo'sh javob qaytardi: %s", ctx.target_model)
        warning = _FALLBACK_NOTE.get(ctx.lang, _FALLBACK_NOTE["uz"])
    except llm.LLMError as exc:
        # Sabab jurnalga yoziladi, foydalanuvchiga emas. Ilgari bu yerga
        # istisnaning o'zi qo'shilar edi va ekranga OpenRouter'ning butun JSON
        # xatosi chiqardi — odam uchun ma'nosiz, va hamisha o'zbekcha.
        logger.warning("Hech bir provayder ishlamadi (%s): %s", ctx.target_model, exc)
        warning = _FALLBACK_NOTE.get(ctx.lang, _FALLBACK_NOTE["uz"])

    return GeneratedPrompt(
        prompt=skeleton, skeleton=skeleton, used_llm=False,
        target_model=ctx.target_model, sources=sources, warning=warning,
    )
