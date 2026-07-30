"""Raqobat tahlili — g'oyaga o'xshash o'zbek startuplarini topadi va ustunlik yo'lini aytadi.

Dataset `scripts/fetch_competitors.py` bilan yig'iladi (uzcombinator.uz va
thepitch.uz portfelilari). Moslashtirish TF-IDF + cosine similarity bilan —
LLM chaqirilmaydi.

Nega char n-gram ham ishlatiladi: tavsiflar aralash tilda yozilgan (bir
startup inglizcha, ikkinchisi o'zbekcha), foydalanuvchi ham aralash yozadi.
Faqat word n-gram bo'lsa "yetkazib berish" va "delivery" umuman uchrashmaydi.

Tahlil qismi qoidalarga asoslangan. Sabab oddiy: "qanday ustunlik qilish
mumkin" degan savolga javob raqobatchining tavsifidan emas, LOYIHA TURI va
aniqlangan signallardan kelib chiqadi — bular allaqachon ML tomonidan
hisoblangan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from app.config import DATA_DIR
from app.ml.engine import normalize

logger = logging.getLogger(__name__)

_DATASET = DATA_DIR / "competitors" / "competitors.json"

# --------------------------------------------------------------------------- #
# Til ko'prigi
# --------------------------------------------------------------------------- #
#
# Muammo: dataset ikki tilda yozilgan. uzcombinator startuplari inglizcha
# ("Restaurant reservation and CRM platform"), thepitch esa o'zbekcha
# ("Restoranlarda jarayonlarni avtomatlashtirish"). Foydalanuvchi o'zbekcha
# yozadi, TF-IDF esa so'zma-so'z ishlaydi — natijada datasetning yarmi
# umuman topilmay qolardi. REZVO (restoran band qilish tizimi) aynan mos
# g'oyaga chiqmagani shundan edi.
#
# Yechim: har bir guruh uchun BITTA kanonik token qo'shiladi — indekslashda
# ham, so'rovda ham. Mashina tarjimasi kerak emas: bu yerda faqat soha
# atamalari muhim, ular esa cheklangan va o'zgarmaydi.
#
# Nega bitta token, guruhning hammasi emas: avval guruhdagi barcha shakllar
# qo'shilardi va bu o'xshashlikni sun'iy ko'tarardi. Bitta umumiy tushuncha
# uchrashi kifoya edi — ikkalasiga ham 6-10 ta bir xil so'z tushib, cosine
# similarity keskin oshardi. Natijada "ot boqish uchun ilova" ovqat so'zi
# tufayli restoran startupiga 0.48 ball bilan "bevosita raqobatchi" bo'lib
# chiqdi. Bitta token esa bitta umumiy xususiyat beradi — mutanosib.
#
# Har bir guruh — bir ma'noning turli tildagi shakllari.
_TERM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("restoran", "restaurant", "kafe", "cafe", "ресторан", "horeca"),
    ("band qilish", "bron", "booking", "reservation", "reserve", "бронирование", "navbat", "appointment", "schedule"),
    ("yetkazib berish", "yetkazish", "delivery", "kuryer", "courier", "доставка", "logistika", "logistics", "freight", "yuk"),
    ("taksi", "taxi", "ride", "haydovchi", "driver", "водитель"),
    ("to'lov", "tolov", "payment", "pay", "оплата", "karta", "card", "billing", "checkout"),
    ("kredit", "loan", "lending", "credit", "кредит", "mikrokredit", "microloan", "qarz"),
    ("hamyon", "wallet", "кошелек", "transfer", "otkazma", "перевод"),
    ("moliya", "finance", "financial", "fintech", "финанс", "byudjet", "budget", "bank", "banking"),
    ("dokon", "do'kon", "shop", "store", "magazin", "магазин", "savdo", "retail", "ecommerce", "sotuv"),
    ("bozor", "marketplace", "market", "маркетплейс", "agregator", "aggregator", "birja"),
    ("kurs", "course", "lesson", "dars", "urok", "урок", "oquv", "o'quv", "education", "learning", "edtech", "til organish", "language learning"),
    ("shifokor", "doctor", "врач", "bemor", "patient", "пациент", "klinika", "clinic", "tibbiy", "medical", "health", "salomatlik"),
    ("obuna", "subscription", "подписка", "tarif", "plan", "pricing"),
    ("hisobot", "report", "analytics", "отчет", "dashboard", "panel", "boshqaruv"),
    ("mijoz", "client", "customer", "клиент", "crm"),
    ("xarita", "map", "geolokatsiya", "geolocation", "карта", "manzil", "address", "location"),
    ("sun'iy intellekt", "suniy intellekt", "ai", "artificial intelligence", "искусственный интеллект", "model", "gpt", "agent"),
    ("chat", "messaging", "xabar", "чат", "telegram", "bot"),
    ("video", "media", "kontent", "content", "контент", "podkast", "podcast"),
    ("oyin", "o'yin", "game", "gaming", "игра", "player", "oyinchi"),
    ("ijara", "rent", "rental", "аренда", "arenda"),
    ("ish", "job", "vacancy", "vakansiya", "recruiting", "hr", "xodim", "employee", "staff"),
    ("ombor", "inventory", "warehouse", "склад", "qoldiq", "stock"),
    ("buyurtma", "order", "заказ"),
    ("reyting", "rating", "review", "sharh", "отзыв", "feedback"),
    ("ovqat", "food", "meal", "еда", "taom"),
    ("sayohat", "travel", "tourism", "turizm", "путешествие", "mehmonxona", "hotel", "otel"),
    ("qurilish", "construction", "remont", "repair", "usta", "master", "мастер", "xizmat korsatuvchi", "service provider"),
    ("dasturchi", "developer", "api", "sdk", "разработчик", "integratsiya", "integration"),
    ("avtomatlashtirish", "automation", "автоматизация", "avtomatik", "automatic"),
    ("ijtimoiy", "social", "jamoa", "community", "сообщество", "tanishuv", "dating"),
    ("mobil ilova", "mobile app", "ilova", "app", "приложение", "android", "ios"),
    ("avtomobil", "car", "auto", "mashina", "автомобиль", "vehicle"),
    ("yoqilgi", "yoqilg'i", "fuel", "benzin", "топливо", "gaz"),
)


def _expand_terms(text: str) -> str:
    """Matnga topilgan tushunchalar uchun kanonik tokenlar qo'shadi.

    Kirish `normalize()` dan o'tgan bo'lishi kutiladi. Har bir guruhga ko'pi
    bilan bitta token qo'shiladi — shu tushuncha ikki matnda ham borligini
    bildiruvchi yagona umumiy xususiyat.
    """
    hits = [
        f"kncpt{i}"
        for i, group in enumerate(_TERM_GROUPS)
        if any(term in text for term in group)
    ]
    if not hits:
        return text
    # Har bir token ikki marta: bir marta bo'lsa signal char n-gramlar
    # ichida yo'qolib ketadi (boshqa tildagi mos startup 7-o'ringa tushib
    # qolardi), ko'proq bo'lsa esa ballar shishadi va zaif o'xshashlik ham
    # yuqori ko'rinadi. Qiymat reyting sifati bo'yicha o'lchab tanlangan.
    weighted = [h for h in hits for _ in range(_CONCEPT_WEIGHT)]
    return f"{text} {' '.join(weighted)}"

# Absolyut chegara emas, NISBIY: har bir moslik shu so'rovning o'z bazasiga
# solishtiriladi.
#
# Nega absolyut ishlamaydi. char n-gram ikki o'zbek matniga mavzusidan qat'i
# nazar bazaviy o'xshashlik beradi — bu tilning o'zi bergan ball. Natijada
# "kvant kompyuterlar uchun kriogen sovutish tizimi" ham 60 ta o'zbek
# startupiga 0.19-0.23 ball olib, "to'yingan bozor" deb baholanardi.
#
# Yechim: so'rov butun indeks bo'yicha qancha ball olishini o'lchab, medianani
# shu so'rovning "tili" deb qabul qilamiz va faqat undan sezilarli yuqori
# turganlarni qoldiramiz. Median ishlatiladi, o'rtacha emas: bir-ikkita kuchli
# moslik o'rtachani ko'tarib, o'zini o'zi filtrdan o'tkazib yuborardi.
_BASELINE_RATIO_MIN = 2.2      # ro'yxatga tushish uchun
_BASELINE_RATIO_HIGH = 3.5     # "yuqori" daraja uchun
_BASELINE_RATIO_MEDIUM = 2.8   # "o'rtacha" daraja uchun

# Median juda kichik bo'lsa nisbat ma'nosini yo'qotadi (0.001 ga bo'lish
# har qanday ballni yuqori qiladi), shuning uchun pastdan cheklanadi.
_BASELINE_FLOOR = 0.02

# Nisbat qanchalik yuqori bo'lmasin, bundan past ball haqiqiy moslik emas.
_MIN_SIMILARITY = 0.10

# Kanonik token necha marta takrorlanadi — `_expand_terms` da izohlangan.
_CONCEPT_WEIGHT = 2

# O'xshashlik darajasi. ATAYLAB "bevosita raqobatchi" deb atalmaydi.
#
# Sabab: dataset ikki tilda va tavsiflar qisqa, shuning uchun cosine qiymati
# so'rovlar orasida taqqoslanmaydi — bir so'rovda 0.30 kuchli moslik, boshqada
# tasodifiy umumiylik bo'ladi. O'lchov paytida "ot boqish uchun ilova" restoran
# startupiga 0.38 ball olgan (ikkalasida ham o'zbek tili, ovqat va ilova
# tushunchasi bor) — absolyut chegara buni "bevosita raqobatchi" deb e'lon
# qilardi, bu esa xato hukm.
#
# Shuning uchun ball hukm emas, TARTIB uchun ishlatiladi: ro'yxat saralanadi,
# daraja ko'rsatiladi, qarorni foydalanuvchi havolani ochib o'zi qiladi.
_LEVEL_HIGH = 0.28
_LEVEL_MEDIUM = 0.16

# DIQQAT: bu yerda "bozor to'yingan / bo'sh" degan HUKM yo'q va ataylab yo'q.
#
# Uni chiqarishga uch marta urinildi: absolyut ball bo'yicha, yuqori darajali
# mosliklar soni bo'yicha va so'rovning o'z medianasiga nisbat bo'yicha.
# Uchalasi ham bir xil joyda yiqildi — "kvant kompyuterlar uchun kriogen
# sovutish tizimi" (O'zbekistonda raqobatchisi yo'qligi aniq) 60 ta startupga
# 0.19-0.23 ball oldi, ya'ni haqiqiy moslik REZVO (0.24) bilan bir darajada.
#
# Sabab xususiyatlarda: 61 ta qisqa, ikki tilli tavsifda char n-gram har qanday
# o'zbek matniga bazaviy o'xshashlik beradi va bu "zaif mavzuli moslik" dan
# farq qilmaydi. Ya'ni ma'lumot yetarli emas, chegara esa yordam bermaydi.
#
# Shuning uchun ball TARTIB uchun ishlatiladi, hukm uchun emas: ro'yxat
# saralanib beriladi, qarorni asoschi havolani ochib o'zi qiladi. Bu kamroq
# ta'sirli, lekin to'g'ri — noto'g'ri "bozor bo'sh" degan xulosa asoschini
# xato qarorga olib borishi mumkin.


@dataclass
class CompetitorMatch:
    name: str
    description: str
    similarity: float
    source: str
    url: str = ""
    profile_url: str = ""
    investment: str = ""
    # yuqori | o'rtacha | past — hukm emas, tartib ko'rsatkichi.
    level: str = "past"

    @property
    def funded(self) -> bool:
        """Investitsiya olganmi. thepitch datasetida `TAKLIF OLGAN` deb yoziladi."""
        return "OLGAN" in self.investment and "OLMAGAN" not in self.investment


@dataclass
class CompetitorAnalysis:
    matches: list[CompetitorMatch] = field(default_factory=list)

    advantages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dataset_size: int = 0

    def to_dict(self) -> dict:
        return {
            "dataset_size": self.dataset_size,
            "matches": [
                {
                    "name": m.name,
                    "description": m.description,
                    "similarity": round(m.similarity, 3),
                    "source": m.source,
                    "url": m.url,
                    "profile_url": m.profile_url,
                    "investment": m.investment,
                    "level": m.level,
                    "funded": m.funded,
                }
                for m in self.matches
            ],
            "advantages": self.advantages,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# Indeks
# --------------------------------------------------------------------------- #


def _level(ratio: float) -> str:
    """Daraja bazaga nisbatan aniqlanadi, absolyut ball bo'yicha emas."""
    if ratio >= _BASELINE_RATIO_HIGH:
        return "yuqori"
    if ratio >= _BASELINE_RATIO_MEDIUM:
        return "o'rtacha"
    return "past"


def _dedupe(items) -> list[dict]:
    """Bir startupning ikki yozuvini birlashtiradi.

    Datasetda bir xil mahsulot ikki slug bilan uchraydi (`hisobchi` va
    `hisobchi-ai`, `anora` va `venaz`) — ular ikkita alohida raqobatchi bo'lib
    ko'rinsa, to'yinganlik soxta oshib ketadi va foydalanuvchi bozor
    haqiqatdan zichroq deb o'ylaydi.

    Bir xillik tavsif bo'yicha aniqlanadi, nom bo'yicha emas: nom rebrend
    paytida o'zgaradi, tavsif esa nusxa ko'chiriladi.
    """
    seen: dict[str, dict] = {}
    for item in items:
        key = normalize(item["description"])[:120]
        existing = seen.get(key)
        if existing is None:
            seen[key] = item
            continue
        # Ikkisidan ma'lumoti to'liqrogini qoldiramiz.
        if len(str(item.get("investment", ""))) > len(str(existing.get("investment", ""))):
            seen[key] = item
        elif not existing.get("url") and item.get("url"):
            seen[key] = item
    return list(seen.values())


class CompetitorIndex:
    """Raqobatchi tavsiflari ustida TF-IDF qidiruv."""

    def __init__(self, vectorizer: FeatureUnion, matrix, items: list[dict]) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.items = items

    @classmethod
    def build(cls) -> Optional["CompetitorIndex"]:
        """Datasetdan indeks yasaydi. Fayl yo'q bo'lsa `None`.

        Ataylab istisno ko'tarmaydi: dataset bo'lmasa raqobat tahlili
        o'tkazib yuborilishi kerak, butun generatsiya to'xtab qolmasin.
        """
        if not _DATASET.exists():
            logger.info("Raqobat dataseti yo'q (%s) — tahlil o'tkazib yuborildi", _DATASET)
            return None

        try:
            payload = json.loads(_DATASET.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Raqobat datasetini o'qib bo'lmadi: %s", exc)
            return None

        items = _dedupe(i for i in payload.get("items", []) if i.get("description"))
        if not items:
            return None

        # Nom ham indeksga kiradi: foydalanuvchi raqobatchini nomi bilan
        # yozishi mumkin ("Rezvo kabi tizim kerak").
        corpus = [
            _expand_terms(normalize(f"{i.get('name', '')} {i['description']}"))
            for i in items
        ]
        vectorizer = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ])
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer, matrix, items)

    def search(self, description: str, top_k: int = 5) -> list[CompetitorMatch]:
        query = self.vectorizer.transform([_expand_terms(normalize(description))])
        scores = cosine_similarity(query, self.matrix).flatten()

        # Shu so'rovning bazasi — indeks bo'yicha median ball.
        baseline = max(float(np.median(scores)), _BASELINE_FLOOR)

        ranked = sorted(enumerate(scores), key=lambda p: p[1], reverse=True)
        out: list[CompetitorMatch] = []
        for index, score in ranked[:top_k]:
            ratio = float(score) / baseline
            if score < _MIN_SIMILARITY or ratio < _BASELINE_RATIO_MIN:
                break
            item = self.items[index]
            out.append(CompetitorMatch(
                name=item.get("name", ""),
                description=item["description"],
                similarity=float(score),
                source=item.get("source", ""),
                url=item.get("url", ""),
                profile_url=item.get("profile_url", ""),
                investment=item.get("investment", ""),
                level=_level(ratio),
            ))
        return out


@lru_cache(maxsize=1)
def get_index() -> Optional[CompetitorIndex]:
    """Indeks bir marta yig'iladi va xotirada qoladi (62 yozuv — arzon)."""
    return CompetitorIndex.build()


# --------------------------------------------------------------------------- #
# Ustunlik yo'llari
# --------------------------------------------------------------------------- #
#
# Har bir loyiha turi uchun "raqobatchidan qanday ajralish mumkin" degan
# javob. Bu marketing maslahati emas — har biri MAHSULOT qaroriga aylanadi
# va shu sababdan texnik topshiriqqa ham tushadi.

_TYPE_ADVANTAGES: dict[str, list[str]] = {
    "marketplace": [
        "Tovuq-tuxum muammosini bir tomondan yechish: avval taklif tomonini (sotuvchi/usta) to'plang, keyin talabni chaqiring. Raqobatchilar ko'pincha ikkalasini birga o'stirmoqchi bo'lib qoqiladi.",
        "Bitta tor nishda boshlang (masalan faqat santexniklar) va o'sha nishda taklif zichligini raqobatchidan yuqori qiling — keng bozorda ular sizni son bilan bosib ketadi.",
        "Bitim ichidagi ishonchni mahsulotga aylantiring: eskrou, kafolat, nizoni hal qilish. Ko'p agregator faqat e'lon taxtasi bo'lib qoladi.",
    ],
    "ecommerce": [
        "Yetkazib berish muddatini va aniqligini o'zingizga xos ustunlik qiling — katalog hajmi bilan raqobatlashish qimmat va foydasiz.",
        "Ombor qoldig'ini real vaqtda ko'rsating: 'bor/yo'q' aniqligi buyurtma bekor qilinishini keskin kamaytiradi va bu mijozga darhol sezilади.",
    ],
    "saas_dashboard": [
        "Bitta soha uchun chuqur moslashtiring (masalan faqat sartaroshxonalar). Universal CRM'lar bilan funksiya soni bo'yicha raqobatlashib bo'lmaydi, lekin ular sizning sohangizdagi ish oqimini bilmaydi.",
        "Ma'lumot ko'chirishni bir tugmaga aylantiring: mijozning Excel yoki 1C dagi bazasini import qilish — almashishga eng katta to'siq shu.",
    ],
    "booking_service": [
        "Bandlikni to'ldirish vositasiga aylanting, shunchaki kalendarga emas: bo'sh vaqtga chegirma, kutish ro'yxati, kelmay qolganlarni qaytarish.",
        "Eslatma kanalini mahalliy odatga moslang (Telegram + SMS) — email eslatmasi bu yerda ishlamaydi va ko'p tizim shunda yiqiladi.",
    ],
    "delivery_logistics": [
        "Yetkazish vaqtini oldindan aniq aytish va uni bajarish — kuzatuv xaritasidan muhimroq. Xarita hammada bor, aniq muddat kamda.",
        "Kuryer tomonini alohida mahsulot deb qarang: smena, to'lov, marshrut qulayligi. Kuryer ketsa xizmat to'xtaydi.",
    ],
    "fintech": [
        "Litsenziya va integratsiyani vaqtida boshlang — bu texnik emas, muddat masalasi va raqobatchidan orqada qolishning eng ko'p uchraydigan sababi.",
        "Bitta moliyaviy og'riqni to'liq yechib bering (masalan faqat bo'lib to'lash), keyin kengaytiring. Super-app bo'lishga urinish resursni tarqatib yuboradi.",
    ],
    "edtech": [
        "Natijani o'lchab ko'rsating: sertifikat emas, ballning oshgani. Kurs kontenti oson ko'chiriladi, progress tizimi esa yo'q.",
        "O'qituvchi tomonini yengil qiling — dars yuklash, tekshirish, hisobot. Kontent egasi sizda qolsa, o'quvchi ham qoladi.",
    ],
    "healthtech": [
        "Ma'lumot maxfiyligini ko'rinadigan ustunlik qiling: kim qachon nimani ko'rgani jurnalda. Klinika uchun bu sotib olish sababi.",
        "Shifokorning ish vaqtini tejang — yozuvni tez kiritish. Ko'p tizim shifokorga qo'shimcha ish yuklaydi va shu sababdan tashlab ketiladi.",
    ],
    "social_community": [
        "Tor jamoadan boshlang (bitta mahalla, bitta kasb). Umumiy tarmoq bo'lishga urinish bo'sh lenta muammosiga olib keladi.",
        "Moderatsiyani birinchi kundan mahsulotga kiritng — jamoa sifati o'sishdan muhimroq va uni keyin tuzatish qiyin.",
    ],
    "content_media": [
        "Tavsiya sifatini ustunlik qiling: katalog hajmi bilan raqobatlashish qimmat.",
        "Kontent egasiga daromadni ko'rsatib bering — muallif qolsa, auditoriya ham qoladi.",
    ],
    "ai_tool": [
        "Model emas, ish oqimi ustunlik bo'ladi: natijani tahrirlash, taxrirlarni saqlash, qayta ishlatish. Model hammada bir xil.",
        "Noto'g'ri natija bilan ishlash yo'lini mahsulotga kiritng (tekshirish, tuzatish) — foydalanuvchi ishonchi shu joyda yutiladi.",
    ],
    "devtool_api": [
        "Birinchi so'rovgacha ketadigan vaqtni qisqartiring: kalit, misol, sandbox — 5 daqiqada ishlasin. Dasturchi shu bosqichda ketadi.",
        "Xatolarni tushunarli qaytaring va jurnalni ko'rsating — bu yerda hujjat sifatidan ham muhimroq.",
    ],
    "mobile_app": [
        "Oflayn ishlashni jiddiy qabul qiling — mahalliy tarmoq sharoitida bu haqiqiy ustunlik.",
        "Ilova hajmi va ochilish tezligi: og'ir ilova o'chirib tashlanadi va bu raqobatchidan ajralishning arzon yo'li.",
    ],
    "game": [
        "Birinchi sessiyada qiziqarli bo'lishiga hammasini qurbon qiling — o'yinchi ikkinchi marta kirmasa qolgani ahamiyatsiz.",
        "Raqib topishni tez qiling: bo'sh serverda o'yin o'lади. Bot bilan to'ldirish ham yechim.",
    ],
}

_GENERIC_ADVANTAGES = [
    "Raqobatchilar tavsifini o'qib chiqing va ular AYTMAGAN narsani toping — aytilmagan joy ko'pincha qilinmagan joy.",
    "Bitta aniq foydalanuvchi guruhini tanlab, ular uchun eng yaxshi bo'ling. Hammaga mos mahsulot hech kimga to'liq mos kelmaydi.",
]

# Signal bo'yicha ustunlik — ML aniqlagan texnik talab raqobat ustunligiga
# aylanishi mumkin bo'lgan joylar.
_SIGNAL_ADVANTAGES: dict[str, str] = {
    "payments": "To'lovni bir qadamga qisqartiring (saqlangan karta, bir bosishda) — bu konversiyaga bevosita ta'sir qiladi va ko'p raqobatchida uzun.",
    "realtime": "Jonli yangilanishni haqiqatan jonli qiling: kechikish sezilsa foydalanuvchi sahifani qayta yuklay boshlaydi va ishonch yo'qoladi.",
    "geo": "Mahalliy manzil aniqligini yaxshilang — global xarita xizmatlari mahalla va mo'ljal darajasida xato qiladi.",
    "offline": "Internet uzilganda ishlashni ustunlik qiling — mahalliy sharoitda bu kam kimda bor.",
    "multi_tenant": "Tashkilotlar ma'lumotini qat'iy ajratganingizni ko'rsatib bering — korporativ mijoz uchun bu sotib olish sharti.",
    "compliance": "Maxfiylik va jurnalni mahsulot yuzasiga chiqaring — ishonch mavhum tushuncha emas, ko'rinadigan funksiya bo'lsin.",
    "search_heavy": "Qidiruv sifatini o'lchang va yaxshilang (birinchi natijada topilishi) — katalogi katta raqobatchining qidiruvi ko'pincha yomon.",
}


def _advantages(project_type: str, signals: dict[str, bool], matches: list[CompetitorMatch]) -> list[str]:
    out = list(_TYPE_ADVANTAGES.get(project_type) or _GENERIC_ADVANTAGES)

    for name, text in _SIGNAL_ADVANTAGES.items():
        if signals.get(name):
            out.append(text)

    # Investitsiya olgan raqobatchi bor bo'lsa strategiya o'zgaradi.
    if any(m.funded for m in matches):
        out.append(
            "Investitsiya olgan raqobatchi bor — u marketingga ko'p sarflay oladi. "
            "Ochiq bozorda emas, tor nishda yoki xizmat sifatida raqobatlashing."
        )
    return out


def _warnings(matches: list[CompetitorMatch]) -> list[str]:
    out: list[str] = []

    high = [m for m in matches if m.level == "yuqori"]
    if high:
        names = ", ".join(m.name for m in high[:3])
        # "Bevosita raqobatchi" deb qat'iy aytmaymiz — o'lchov buni
        # ko'tarmaydi (izoh `_LEVEL_HIGH` yonida). Foydalanuvchini havolani
        # ochib o'zi tekshirishga yo'naltiramiz.
        out.append(
            f"Tavsifi eng yaqin startuplar: {names}. Havolalarini ochib ko'ring — "
            "agar mahsulot haqiqatan ustma-ust tushsa, nima bilan farq "
            "qilishingizni oldindan yozib qo'ying."
        )

    if not matches:
        # Faqat ro'yxat BUTUNLAY bo'sh bo'lganda aytiladi. Ilgari bu xabar
        # to'yinganlik "bo'sh" bo'lsa chiqardi va ro'yxatda mosliklar
        # turgani holda "topilmadi" deb yozilib qolardi.
        out.append(
            "Bazadagi 60+ startup ichida o'xshashi topilmadi. Ikki xil sabab "
            "bo'lishi mumkin: nishangiz haqiqatan bo'sh, yoki bozor kichik "
            "bo'lgani uchun hech kim kirmagan. Talabni tekshirib oling."
        )

    funded = [m for m in matches if m.funded]
    if funded:
        out.append(
            "Investitsiya olgan: " + ", ".join(f"{m.name} ({m.investment})" for m in funded[:3])
        )
    return out


def analyze(
    description: str,
    project_type: str = "",
    signals: Optional[dict[str, bool]] = None,
    top_k: int = 5,
) -> CompetitorAnalysis:
    """G'oyaga o'xshash startuplarni topadi va ustunlik yo'llarini qaytaradi."""
    index = get_index()
    if index is None:
        return CompetitorAnalysis()

    matches = index.search(description, top_k=top_k)

    return CompetitorAnalysis(
        matches=matches,
        advantages=_advantages(project_type, signals or {}, matches),
        warnings=_warnings(matches),
        dataset_size=len(index.items),
    )
