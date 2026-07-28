"""Mobil ilova uchun maqsad platformasini aniqlaydi va stackni shunga bog'laydi.

Nega kerak: ilgari har qanday mobil loyihaga bir xil javob berilardi —
"React Native (Expo)". Holbuki qaror platformaga bog'liq:

* faqat Android  -> Kotlin (Jetpack Compose). Bitta platforma uchun kross-
  platforma qatlamini olib yurish ortiqcha: build sekinlashadi, native API'ga
  har safar ko'prik yozishga to'g'ri keladi.
* faqat iOS      -> Swift (SwiftUI). Xuddi shu sabab.
* ikkalasi       -> Flutter. Ikki jamoa saqlash startup uchun qimmat; bitta
  koddan ikkala do'konga chiqiladi va UI ikkala platformada bir xil bo'ladi.

Platforma matndan topilmasa, `clarifier` foydalanuvchidan aniq so'raydi —
taxmin qilib noto'g'ri stack tavsiya qilishdan ko'ra so'ragan yaxshi.
"""

from __future__ import annotations

import re
from typing import Literal

Platform = Literal["android", "ios", "both", "unknown"]

# Qisqa so'zlar uchun so'z chegarasi shart: "ios" aks holda "studios",
# "radios", "kiosk" ichidan topilib, noto'g'ri xulosa beradi.
_ANDROID_RE = re.compile(
    r"\bandroid\w*\b|\bандроид\w*\b|\bapk\b|play\s*market|play\s*store|google\s*play",
    re.I,
)
_IOS_RE = re.compile(
    r"\bios\b|\biphone\w*\b|\bipad\w*\b|\bайфон\w*\b|\bайос\b|app\s*store",
    re.I,
)
# "Apple Pay" bor deb iOS deb xulosa qilmaymiz — u to'lov, platforma emas.

_BOTH_RE = re.compile(
    r"ikkala\w*|har\s+ikki\w*|\bboth\b|\bоба\b|\bобе\b|обе[йи]х|"
    r"kross[\s-]?platform\w*|cross[\s-]?platform\w*|кросс[\s-]?платформ\w*",
    re.I,
)


# Mahsulot mobilmi degan savol uchun alohida, qat'iyroq ro'yxat.
#
# ML dvigatelidagi "mobile" signali "ilova" so'zini ham o'z ichiga oladi —
# o'zbekchada u shunchaki "dastur" degani, shuning uchun oddiy veb SaaS ham
# mobil deb belgilanadi. Platforma savolini shunga bog'lasak, veb loyiha
# qurayotgan odamdan "Android yoki iOS?" deb so'ralardi. Bu yerda faqat
# mobilni aniq bildiradigan so'zlar turadi.
_MOBILE_RE = re.compile(
    r"\bmobil\w*\b|\bмобильн\w*\b|\bсмартфон\w*\b|\bsmartphone\w*\b|"
    r"\bandroid\w*\b|\bандроид\w*\b|\bios\b|\biphone\w*\b|\bipad\w*\b|\bайфон\w*\b|"
    r"play\s*market|play\s*store|google\s*play|app\s*store|\bapk\b|"
    r"\bflutter\b|react\s*native|\bswiftui\b|jetpack\s*compose",
    re.I,
)


def is_mobile_product(description: str, project_type: str = "") -> bool:
    """Mahsulotda mobil ilova bormi.

    `project_type == "mobile_app"` bo'lsa — albatta. Lekin ko'p hollarda
    klassifikator loyihani sohasi bo'yicha belgilaydi ("kuryerlar uchun mobil
    ilova" -> delivery_logistics), shuning uchun turga tayanish yetarli emas.
    """
    if project_type == "mobile_app":
        return True
    return bool(_MOBILE_RE.search(description or ""))


def detect(*texts: str) -> Platform:
    """Berilgan matnlardan platformani aniqlaydi.

    Bir nechta matn beriladi (g'oya tavsifi + aniqlashtiruvchi javoblar),
    chunki foydalanuvchi platformani ko'pincha savolga javob berganda aytadi.
    """
    blob = "\n".join(t for t in texts if t)
    if not blob.strip():
        return "unknown"

    has_android = bool(_ANDROID_RE.search(blob))
    has_ios = bool(_IOS_RE.search(blob))

    # "ikkalasi ham" deb yozilsa, platformalar nomi tilga olinmagan bo'lishi
    # mumkin — shuning uchun bu tekshiruv birinchi turadi.
    if _BOTH_RE.search(blob) or (has_android and has_ios):
        return "both"
    if has_android:
        return "android"
    if has_ios:
        return "ios"
    return "unknown"


# Platforma -> (tanlov, `copy` kaliti, alternativalar).
# "unknown" ham Flutter'ga tushadi: platforma aytilmagan bo'lsa, mahsulot
# odatda ikkala do'konni ko'zlaydi va Flutter keyin bitta platformaga
# qisqartirilishi mumkin — teskarisi (Kotlin'dan iOS'ga) esa qayta yozish.
STACK: dict[str, tuple[str, str, list[str]]] = {
    "android": ("Kotlin + Jetpack Compose", "why.fe.mobile.android",
                ["Flutter", "Kotlin Multiplatform"]),
    "ios": ("Swift + SwiftUI", "why.fe.mobile.ios",
            ["Flutter", "React Native (Expo)"]),
    "both": ("Flutter 3 (Dart)", "why.fe.mobile.both",
             ["React Native (Expo)", "Kotlin Multiplatform + Compose"]),
    "unknown": ("Flutter 3 (Dart)", "why.fe.mobile.unknown",
                ["React Native (Expo)", "Native (Swift / Kotlin)"]),
}


# Foydalanuvchidan so'raladigan savol. Bu savol LLM'ga qoldirilmaydi: u
# stackni to'g'ridan-to'g'ri o'zgartiradi, shuning uchun model ishlamay
# qolganda ham berilishi shart.
QUESTION: dict[str, tuple[str, str, list[str]]] = {
    "uz": (
        "Ilova qaysi platforma uchun kerak — Android, iOS yoki ikkalasi?",
        "Javobga qarab til tanlanadi: Android — Kotlin, iOS — Swift, ikkalasi — Flutter",
        ["Faqat Android", "Faqat iOS", "Ikkalasi ham"],
    ),
    "ru": (
        "Для какой платформы нужно приложение — Android, iOS или обе?",
        "От ответа зависит язык: Android — Kotlin, iOS — Swift, обе — Flutter",
        ["Только Android", "Только iOS", "Обе"],
    ),
    "en": (
        "Which platform is the app for — Android, iOS, or both?",
        "The answer decides the language: Android — Kotlin, iOS — Swift, both — Flutter",
        ["Android only", "iOS only", "Both"],
    ),
}
