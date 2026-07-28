"""G'oyaga moslangan aniqlashtiruvchi savollar.

Nega kerak: foydalanuvchi "marketplace qilmoqchiman" deb yozadi va shu bilan
tugaydi. Bu matndan arxitekturaga ta'sir qiladigan o'nlab qaror ko'rinmaydi —
to'lov platformada ushlanadimi, sotuvchilarni kim tasdiqlaydi, yetkazib berish
kimning zimmasida. Javobsiz qolgan har bir savol promptni umumiylashtiradi.

Shuning uchun generatsiyadan oldin model g'oyani o'qib, aynan shu g'oyaga tegishli
4-6 ta savol beradi. Javoblar promptga alohida bo'lim bo'lib kiradi.

Ishonchlilik: LLM javob bermasa yoki JSON buzilsa, statik zaxira savollar
qaytariladi. Bu qadam hech qachon oqimni to'xtatmasligi kerak — savol bermaslik
promptni zaiflashtiradi, lekin butunlay to'sib qo'yish mahsulotni buzadi.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field

from app.services import llm, mobile_platform
from app.services.project_playbooks import get as get_playbook

logger = logging.getLogger(__name__)

# Savollar qisqa — uzun javob kutilmaydi, shuning uchun token chegarasi ham kichik.
_MAX_TOKENS = 1200
_MIN_QUESTIONS = 3
_MAX_QUESTIONS = 6


@dataclass
class Question:
    id: str
    question: str
    hint: str = ""
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClarifyResult:
    project_type: str
    project_title: str
    questions: list[Question]
    generated_by_llm: bool
    warning: str | None = None


# --------------------------------------------------------------------------- #
# Statik zaxira — LLM ishlamasa
# --------------------------------------------------------------------------- #

# Ataylab umumiy: har qanday loyihaga mos keladi va har biri arxitekturaga
# haqiqatan ta'sir qiladi. Turga xos aniqlik LLM javob berganda qo'shiladi.
_FALLBACK: dict[str, list[tuple[str, str]]] = {
    "uz": [
        ("Mahsulotda qanday foydalanuvchi rollari bo'ladi va ular nima qila oladi?",
         "Masalan: xaridor, sotuvchi, moderator, admin"),
        ("Foydalanuvchi kirgach bajaradigan ENG asosiy amal nima?",
         "Bitta jumla bilan — mahsulotning yuragi nima"),
        ("Qaysi tashqi xizmatlar ulanishi shart?",
         "To'lov (Payme/Click), SMS, xarita, Telegram bot, email, 1C va h.k."),
        ("Admin paneli, mobil ilova yoki ko'p tillilik kerakmi?",
         "Birinchi versiyada aynan nima bo'lishi shart"),
        ("Birinchi versiyada nima BO'LMASLIGI kerak?",
         "Keyinga qoldiriladigan narsalar — hajmni chegaralash uchun"),
    ],
    "ru": [
        ("Какие роли пользователей будут в продукте и что каждая может делать?",
         "Например: покупатель, продавец, модератор, админ"),
        ("Какое САМОЕ главное действие выполняет пользователь после входа?",
         "Одним предложением — в чём суть продукта"),
        ("Какие внешние сервисы обязательно нужно подключить?",
         "Оплата, SMS, карты, Telegram-бот, email, 1С и т.д."),
        ("Нужны ли админ-панель, мобильное приложение или мультиязычность?",
         "Что именно должно быть в первой версии"),
        ("Чего НЕ должно быть в первой версии?",
         "Что откладывается — чтобы ограничить объём"),
    ],
    "en": [
        ("Which user roles will the product have, and what can each of them do?",
         "For example: buyer, seller, moderator, admin"),
        ("What is the single MOST important action a user performs after signing in?",
         "One sentence — the core of the product"),
        ("Which external services must be integrated?",
         "Payments, SMS, maps, Telegram bot, email, accounting systems, etc."),
        ("Do you need an admin panel, a mobile app, or multiple languages?",
         "What exactly must be in the first version"),
        ("What must NOT be in the first version?",
         "What is deliberately deferred — to bound the scope"),
    ],
}

_NO_LLM_WARNING = {
    "uz": "Savollar umumiy ro'yxatdan olindi (AI javob bermadi).",
    "ru": "Вопросы взяты из общего списка (AI не ответил).",
    "en": "Questions came from the generic list (the AI did not respond).",
}


def _fallback_questions(lang: str) -> list[Question]:
    rows = _FALLBACK.get(lang, _FALLBACK["uz"])
    return [Question(id=f"q{i + 1}", question=q, hint=h) for i, (q, h) in enumerate(rows)]


# --------------------------------------------------------------------------- #
# LLM so'rovi
# --------------------------------------------------------------------------- #

_LANG_NAME = {"uz": "O'zbek", "ru": "Русский", "en": "English"}

_SYSTEM = """You interview a founder about the product they want built, so that a
technical specification can be written afterwards.

Your job: read the idea and ask the few questions whose answers would MOST change
the architecture, the data model, or the scope of the first version.

Rules:
- Ask exactly {n} questions.
- Each question must be specific to THIS idea. Never ask something you could ask
  about any product ("who is your target audience", "what is your budget").
- Ask about decisions, not preferences: who does what, where the money is held,
  what happens on failure, which side owns a process, what is out of scope.
- Do not ask anything already stated in the idea.
- Do not ask about technology choices — the stack is decided separately.
- One sentence per question. A founder without technical background must
  understand it.
- Write the questions in {language}.

Return ONLY valid JSON, no prose and no markdown fences:
{{"questions": [{{"q": "...", "hint": "...", "examples": ["...", "..."]}}]}}

"hint" is a short clue about what kind of answer is expected.
"examples" holds 2-3 short possible answers the founder could pick from."""

_USER = """Product type detected by the classifier: {project_title} ({project_type})

Founder's own words:
\"\"\"
{description}
\"\"\"

Technical areas that usually decide the architecture for this product type:
{must_have}

Ask the {n} questions."""


def _extract_json(raw: str) -> dict | None:
    """Modelning javobidan JSON obyektni ajratadi.

    Modellar ba'zan JSON'ni ```json bloklariga o'raydi yoki oldiga bir jumla
    qo'shadi. Shuning uchun to'g'ridan-to'g'ri parse qilish yetarli emas.
    """
    raw = raw.strip()
    if not raw:
        return None

    # ```json ... ``` bo'lsa ichini olamiz
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()

    try:
        return json.loads(raw)
    except ValueError:
        pass

    # Oxirgi chora: birinchi { dan oxirgi } gacha
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except ValueError:
            return None
    return None


def _salvage(raw: str) -> list[dict]:
    """Kesilgan JSON'dan to'liq savol obyektlarini ajratib oladi.

    Token chegarasi past bo'lganda model JSON'ni oxirigacha yozib ulgurmaydi va
    `json.loads` butun javobni tashlab yuboradi. Holbuki birinchi 2-3 ta savol
    to'liq yozilgan bo'ladi — ularni yo'qotish o'rniga qutqaramiz.
    """
    out: list[dict] = []
    for m in re.finditer(r'\{[^{}]*"q"\s*:\s*"(?:[^"\]|\.)*"[^{}]*\}', raw, re.S):
        try:
            out.append(json.loads(m.group(0)))
        except ValueError:
            continue
    return out


def _parse(raw: str) -> list[Question]:
    data = _extract_json(raw)
    items = data.get("questions") if isinstance(data, dict) else None

    if not isinstance(items, list):
        items = _salvage(raw)
    if not items:
        return []

    out: list[Question] = []
    for i, item in enumerate(items[:_MAX_QUESTIONS]):
        if not isinstance(item, dict):
            continue
        text = str(item.get("q") or item.get("question") or "").strip()
        if not text:
            continue
        examples = item.get("examples")
        out.append(Question(
            id=f"q{i + 1}",
            question=text,
            hint=str(item.get("hint") or "").strip(),
            examples=[str(e).strip() for e in examples][:3] if isinstance(examples, list) else [],
        ))
    return out


def _with_platform_question(
    questions: list[Question], project_type: str, description: str, lang: str
) -> list[Question]:
    """Mobil loyihaga platforma savolini birinchi bo'lib qo'shadi.

    Bu savol ataylab LLM'ga qoldirilmagan. Birinchidan, tizim ko'rsatmasida
    modelga "texnologiya tanlovi haqida so'rama" deyilgan. Ikkinchidan, javob
    stackni to'g'ridan-to'g'ri o'zgartiradi (Android -> Kotlin, iOS -> Swift,
    ikkalasi -> Flutter), shuning uchun model ishlamay qolganda ham berilishi
    shart — aks holda tavsiya taxminga asoslanadi.

    Foydalanuvchi platformani g'oyasida allaqachon aytgan bo'lsa, savol
    qo'shilmaydi: aytilgan narsani qayta so'rash vaqtni oladi.
    """
    if not mobile_platform.is_mobile_product(description, project_type):
        return questions
    if mobile_platform.detect(description) != "unknown":
        return questions

    text, hint, examples = mobile_platform.QUESTION.get(
        lang, mobile_platform.QUESTION["uz"]
    )
    head = Question(id="platform", question=text, hint=hint, examples=list(examples))
    # Umumiy soni chegaradan oshmasin — oxirgi savol o'rnini bo'shatadi.
    return [head, *questions][:_MAX_QUESTIONS]


async def ask(
    description: str,
    project_type: str,
    project_title: str,
    lang: str = "uz",
    count: int = 5,
) -> ClarifyResult:
    """G'oyaga moslangan savollarni qaytaradi; LLM ishlamasa — statik ro'yxat."""
    count = max(_MIN_QUESTIONS, min(count, _MAX_QUESTIONS))
    playbook = get_playbook(project_type)

    messages = [
        {
            "role": "system",
            "content": _SYSTEM.format(n=count, language=_LANG_NAME.get(lang, "O'zbek")),
        },
        {
            "role": "user",
            "content": _USER.format(
                project_title=project_title,
                project_type=project_type,
                description=description.strip()[:4000],
                must_have="\n".join(f"- {m}" for m in playbook.must_have[:5]),
                n=count,
            ),
        },
    ]

    try:
        raw, provider = await llm.complete(
            messages,
            max_tokens=_MAX_TOKENS,
            # 5 ta qisqa savol ~400 tokenga sig'adi — balans oz bo'lsa ham
            # savollarni umumiy zaxiraga tashlab yubormaymiz.
            min_useful=350,
            # Savollar barqaror bo'lsin — bir xil g'oyaga har safar butunlay
            # boshqa savollar berilishi foydalanuvchini chalg'itadi.
            temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 — bu qadam hech qachon oqimni to'xtatmasin
        logger.warning("Savollarni yaratib bo'lmadi: %s", exc)
        return ClarifyResult(
            project_type=project_type,
            project_title=project_title,
            questions=_with_platform_question(
                _fallback_questions(lang), project_type, description, lang
            ),
            generated_by_llm=False,
            warning=_NO_LLM_WARNING.get(lang, _NO_LLM_WARNING["uz"]),
        )

    questions = _parse(raw)
    if len(questions) < _MIN_QUESTIONS:
        logger.warning("Model %d ta savol qaytardi — zaxira ro'yxat ishlatilyapti", len(questions))
        return ClarifyResult(
            project_type=project_type,
            project_title=project_title,
            questions=_with_platform_question(
                _fallback_questions(lang), project_type, description, lang
            ),
            generated_by_llm=False,
            warning=_NO_LLM_WARNING.get(lang, _NO_LLM_WARNING["uz"]),
        )

    return ClarifyResult(
        project_type=project_type,
        project_title=project_title,
        questions=_with_platform_question(questions, project_type, description, lang),
        generated_by_llm=True,
    )
