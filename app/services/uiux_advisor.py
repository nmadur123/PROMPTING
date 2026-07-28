"""UI/UX tavsiyalari — asosiy urg'u: serverga va brauzerga yuk tushmasin.

Har bir tavsiya "nima qilish" + "nega" + "qanday o'lchash" ko'rinishida.
Foydalanuvchi buni to'g'ridan-to'g'ri promptga qo'shib yubora oladi.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from app.copy import tr


@dataclass
class Tip:
    area: str
    rule: str
    why: str
    how: str

    def to_dict(self) -> dict:
        return asdict(self)


# Loyiha turiga qarab render strategiyasi — bu eng katta ta'sir qiladigan tanlov.
_RENDER_STRATEGY: dict[str, tuple[str, str]] = {
    "ecommerce": (
        "Mahsulot va katalog sahifalari — ISR (revalidate 60-300s). Savat va checkout — client.",
        "Katalog har so'rovda DB'ga bormaydi: 10 000 tashrif 1 ta DB so'roviga aylanadi.",
    ),
    "content_media": (
        "Barcha kontent sahifalari — SSG/ISR. Faqat izoh va layk qismi client.",
        "Maqola o'zgarmaydi — uni har safar qayta yasash sof isrof. CDN'dan beriladi.",
    ),
    "marketplace": (
        "E'lon sahifasi — ISR, ro'yxat va filtr — server tomonda sahifalash bilan SSR.",
        "E'lon sahifalari SEO'dan trafik keltiradi; filtrni client'da qilsangiz butun bazani yuborishga majbur bo'lasiz.",
    ),
    "saas_dashboard": (
        "SSR shart emas — SPA (client render) + React Query kesh. Faqat login sahifasi SSR.",
        "Panel Google'ga kerak emas; SSR bu yerda faqat server yukini oshiradi.",
    ),
    "booking_service": (
        "Landing va xizmat sahifalari — SSG. Bo'sh vaqt (slot) jadvali — client, qisqa keshli API.",
        "Slotlar tez o'zgaradi, qolgan hammasi statik — ikkisini aralashtirmaslik kerak.",
    ),
    "delivery_logistics": (
        "Mijoz ilovasi — client render; xarita va joylashuv WebSocket orqali.",
        "Har soniyada joylashuv yangilanadi — HTTP polling serverni behuda yeydi.",
    ),
    "social_community": (
        "Lenta — client render + kursorli (cursor) sahifalash. Profil sahifasi — ISR.",
        "Offset sahifalash katta jadvalda sekinlashadi; kursor doim bir xil tez.",
    ),
    "edtech": (
        "Kurs va dars sahifalari — ISR. Video — HLS orqali CDN'dan.",
        "Videoni o'z serveringizdan bersangiz trafik xarajati hammasidan oshib ketadi.",
    ),
    "ai_tool": (
        "Client render + natijani oqim (stream) bilan bo'lak-bo'lak ko'rsatish.",
        "Foydalanuvchi 30 soniya bo'sh ekranga qaramaydi — birinchi belgi 1 soniyada chiqsin.",
    ),
    "devtool_api": (
        "Hujjatlar — to'liq SSG. Panel — client render.",
        "Hujjat eng ko'p o'qiladigan sahifa; u statik bo'lsa server umuman ishtirok etmaydi.",
    ),
    "fintech": (
        "Barcha pul bilan bog'liq ekran — server render, client'da hech qanday hisob-kitob yo'q.",
        "Summani client hisoblasa, uni o'zgartirish mumkin. Server yagona haqiqat manbai bo'lishi shart.",
    ),
    "healthtech": (
        "Bemor ma'lumoti — faqat autentifikatsiyadan keyin client render, keshsiz.",
        "Tibbiy ma'lumot CDN yoki brauzer keshida qolib ketmasligi kerak.",
    ),
    "mobile_app": (
        "Ro'yxatlarda FlatList + `getItemLayout`, rasm keshi (FastImage).",
        "Mobil qurilmada xotira chegarasi qattiq — virtualizatsiyasiz ilova yopilib ketadi.",
    ),
    "game": (
        "O'yin sikli React'dan tashqarida (canvas), UI qatlami alohida.",
        "React har kadrda qayta render qilsa FPS tushadi — o'yin mantiqi requestAnimationFrame'da bo'lsin.",
    ),
}

_DEFAULT_RENDER = (
    "Ommaga ochiq sahifalar — ISR, foydalanuvchi paneli — client render.",
    "SEO kerak joyda statik, kerak bo'lmagan joyda server yuki nolga tushadi.",
)


def _base_tips() -> list[Tip]:
    return [
        Tip(
            "Rasmlar",
            "Barcha rasmlar AVIF/WebP, `next/image` (yoki `<img loading=\"lazy\" srcset>`) orqali.",
            "Rasm odatda sahifa vaznining 60-70% i. Optimallashtirilmagan 2 MB rasm — "
            "3G'da 8 soniya kutish va bekor ketgan tashrif.",
            "Lighthouse'da LCP < 2.5s; har bir rasm 200 KB dan oshmasin.",
        ),
        Tip(
            "Ro'yxatlar",
            "50+ elementli ro'yxatda virtualizatsiya (`@tanstack/react-virtual`) yoki sahifalash.",
            "1000 ta DOM elementi brauzerni qotiradi — arzon telefonda scroll uzilib qoladi.",
            "DevTools > Performance: scroll paytida FPS 50 dan tushmasin.",
        ),
        Tip(
            "Ma'lumot olish",
            "React Query (TanStack Query) — `staleTime` 30-60s, `refetchOnWindowFocus: false`.",
            "Default sozlamada har fokusda qayta so'rov ketadi. Bu server yukini "
            "sezilarli oshiradi va hech qanday foyda bermaydi.",
            "Network panelida bir xil so'rov takrorlanmasligini tekshiring.",
        ),
        Tip(
            "Qidiruv",
            "Qidiruv maydonida debounce 300-400 ms + kamida 2 belgi.",
            "Debounce'siz har bosilgan harf uchun so'rov ketadi: 10 harfli so'z = "
            "10 ta bekor DB so'rovi.",
            "Bitta qidiruvda 1 ta so'rov ketayotganini Network'da ko'ring.",
        ),
        Tip(
            "Bundle",
            "Boshlang'ich JS < 150 KB (gzip). Og'ir kutubxonani `dynamic(() => import())` bilan yuklang.",
            "Har 100 KB JS arzon telefonda ~1 soniya qo'shadi — bu to'g'ridan-to'g'ri konversiya.",
            "`next build` chiqishida First Load JS ustunini kuzating.",
        ),
        Tip(
            "Shriftlar",
            "1-2 ta shrift, `next/font` bilan o'z serveringizdan, `font-display: swap`.",
            "Tashqi Google Fonts qo'shimcha DNS + TLS qo'l siqishi; matn ko'rinmay turadi (FOIT).",
            "CLS < 0.1 bo'lsin.",
        ),
        Tip(
            "Kutish holati",
            "Spinner emas, skeleton. Yozish amallarida optimistik yangilanish.",
            "Skeleton kutishni qisqaroq his ettiradi; optimistik UI esa tarmoq "
            "kechikishini foydalanuvchidan butunlay yashiradi.",
            "Foydalanuvchi harakatidan keyin 100 ms ichida ekranda o'zgarish bo'lsin.",
        ),
        Tip(
            "Animatsiya",
            "Faqat `transform` va `opacity`. Og'ir animatsiya kutubxonasi shart emas.",
            "`width`/`top` animatsiyasi har kadrda layout qayta hisoblaydi — bu asosiy qotish sababi.",
            "DevTools > Rendering > Paint flashing: animatsiyada ekran yonib turmasin.",
        ),
        Tip(
            "Mobil",
            "Mobile-first yozing, tegish maydoni ≥ 44px, kontrast ≥ 4.5:1.",
            "O'zbekistonda trafikning katta qismi telefondan keladi. Desktop'da qilib "
            "keyin siqish har doim yomon natija beradi.",
            "Chrome DevTools'da o'rta darajali Android + Slow 4G rejimida sinab ko'ring.",
        ),
        Tip(
            "Formalar",
            "Validatsiya client'da darhol, server'da qayta. Xato maydon yonida ko'rsatilsin.",
            "Serverga borib qaytadigan validatsiya har xatoda 300-800 ms kutish demak.",
            "Forma yuborilgach tugma bloklanishi va holat ko'rinishi shart.",
        ),
    ]


def _conditional_tips(signals: dict, project_type: str) -> list[Tip]:
    tips: list[Tip] = []

    if signals.get("realtime"):
        tips.append(Tip(
            "Realtime",
            "Polling emas, WebSocket. Ko'rinmayotgan tab'da ulanishni to'xtating.",
            "5 soniyalik polling 10 000 foydalanuvchida ~2000 RPS beradi — bu bitta "
            "serverni yiqitadi. WebSocket'da esa bu deyarli nol yuk.",
            "`document.visibilityState` bo'yicha ulanishni uzing va tiklang.",
        ))
    if signals.get("media_heavy"):
        tips.append(Tip(
            "Video",
            "Video HLS/DASH bilan moslashuvchan sifatda, faqat CDN orqali. Avtoplay — `muted` va `preload=\"none\"`.",
            "Bitta 100 MB video 1000 marta ko'rilsa — 100 GB trafik. CDN'siz bu "
            "server tarifidan qimmatga tushadi.",
            "Origin serverga video so'rovi umuman tushmasligi kerak.",
        ))
    if signals.get("geo"):
        tips.append(Tip(
            "Xarita",
            "MapLibre GL (bepul) + markerlarni klasterlash. Xaritani `dynamic import` bilan yuklang.",
            "Google Maps SDK ~200 KB va so'rovga pul; klasterlashsiz 500 marker "
            "xaritani qotiradi.",
            "Xarita komponenti boshlang'ich bundle'ga kirmasligini tekshiring.",
        ))
    if signals.get("ai") or project_type == "ai_tool":
        tips.append(Tip(
            "AI javobi",
            "Javobni oqim (streaming) bilan ko'rsating + \"to'xtatish\" tugmasi.",
            "Kutish 20-40 soniya bo'lishi mumkin; oqimsiz foydalanuvchi sahifani yopadi. "
            "To'xtatish tugmasi esa bekor ketadigan token xarajatini kamaytiradi.",
            "Birinchi belgi 1.5 soniyadan kech chiqmasin (TTFT).",
        ))
    if project_type in ("ecommerce", "marketplace"):
        tips.append(Tip(
            "Konversiya",
            "Checkout bitta sahifada, ro'yxatdan o'tmasdan (mehmon sifatida) ham bo'lsin.",
            "Majburiy ro'yxatdan o'tish savatni tashlab ketishning eng katta sababi.",
            "Savatdan to'lovgacha bo'lgan qadamlar soni 3 dan oshmasin.",
        ))
    if signals.get("multi_tenant"):
        tips.append(Tip(
            "Multi-tenant UI",
            "Firma tanlovi URL'da bo'lsin (`/c/[slug]/...`), global holatda emas.",
            "Global holatda saqlansa foydalanuvchi havolani ulashganda noto'g'ri "
            "firma ma'lumotini ko'radi — bu jiddiy xavfsizlik xatosi.",
            "Har bir API so'rovida tenant chegarasi server tomonda tekshirilsin.",
        ))
    if signals.get("offline"):
        tips.append(Tip(
            "Offline",
            "Service worker + IndexedDB, o'zgarishlarni navbatga qo'yib keyin sinxronlash.",
            "Internet uzilganda ilova ishlashda davom etsa, foydalanuvchi ishonchi keskin ortadi.",
            "Aviarejimda asosiy stsenariy ishlashini sinab ko'ring.",
        ))
    if signals.get("compliance"):
        tips.append(Tip(
            "Maxfiylik",
            "Shaxsiy ma'lumotni URL'ga, log'ga va analitikaga yubormang.",
            "Log va analitika ko'pincha uchinchi tomonda saqlanadi — bemor yoki karta "
            "ma'lumoti u yerga tushsa, bu qonun buzilishi.",
            "Log'larni ishga tushirishdan oldin bir marta ko'z bilan tekshiring.",
        ))
    return tips


def performance_budget(monthly_users: int) -> dict:
    """O'lchanadigan maqsadlar — buni to'g'ridan-to'g'ri promptga qo'yish mumkin."""
    return {
        "LCP": "< 2.5 s (4G, o'rta darajali Android)",
        "INP": "< 200 ms",
        "CLS": "< 0.1",
        "initial_js_gzip": "< 150 KB",
        "api_p95": "< 300 ms" if monthly_users < 100_000 else "< 200 ms",
        "lighthouse_mobile": "≥ 90",
        "image_max_size": "200 KB",
    }


# Render strategiyasi katalogda `ui.<tur>.rule` / `ui.<tur>.why` kalitlari
# ostida. Katalogda bo'lmagan tur uchun `ui.default.*` ishlatiladi.
_STRATEGY_KEY = {
    "ecommerce": "ecommerce", "content_media": "content", "marketplace": "marketplace",
    "saas_dashboard": "dashboard", "booking_service": "booking",
    "delivery_logistics": "delivery", "social_community": "social", "edtech": "edtech",
    "ai_tool": "ai", "devtool_api": "devtool", "fintech": "fintech",
    "healthtech": "health", "mobile_app": "mobile", "game": "game",
}


def advise(project_type: str, signals: dict, monthly_users: int, lang: str = "uz") -> dict:
    slug = _STRATEGY_KEY.get(project_type, "default")
    tips = _base_tips() + _conditional_tips(signals, project_type)
    return {
        "render_strategy": {
            "rule": tr(f"ui.{slug}.rule", lang),
            "why": tr(f"ui.{slug}.why", lang),
        },
        # DIQQAT: quyidagi maslahatlar hali faqat o'zbekcha. Ular katalogga
        # ko'chirilmagan — `render_strategy` dan farqli o'laroq, Datum ularni
        # ko'rsatmaydi, sys42 fronti esa ko'rsatadi.
        "tips": [t.to_dict() for t in tips],
        "performance_budget": performance_budget(monthly_users),
    }
