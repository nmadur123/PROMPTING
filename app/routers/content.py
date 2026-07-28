"""Foydalanuvchiga ko'rsatiladigan namuna va yo'riqnoma (uz / ru / en).

Bu birinchi ekranning mazmuni: startupni QANDAY yozish kerakligi tushuntiriladi
va to'ldirilgan namuna beriladi. Yomon kiritish — yomon natija, shuning uchun
bu qadam mahsulot sifatining yarmini hal qiladi.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

router = APIRouter(prefix="/api/content", tags=["Content"])


_GUIDE = {
    "uz": {
        "title": "Startupingizni qanday yozish kerak",
        "subtitle": "Qancha aniq yozsangiz, natija shuncha aniq bo'ladi. 4 ta savolga javob bering — yetarli.",
        "questions": [
            {
                "q": "Kim foydalanadi?",
                "hint": "Auditoriyani aniq ayting: \"Toshkentdagi kichik sartaroshxonalar\" — "
                        "\"hamma\" degandan yaxshiroq.",
                "bad": "Odamlar uchun ilova",
                "good": "Toshkent va Samarqanddagi 2-5 ustali sartaroshxonalar va ularning mijozlari",
            },
            {
                "q": "Qanday muammoni yechadi?",
                "hint": "Hozir bu muammo qanday hal qilinayotganini ayting — shundan farq ko'rinadi.",
                "bad": "Navbatni osonlashtiradi",
                "good": "Hozir navbat Telegramda qo'lda yoziladi, usta vaqtni chalkashtiradi va "
                        "mijoz kelmay qolsa joy bo'sh ketadi",
            },
            {
                "q": "Asosiy amal nima?",
                "hint": "Foydalanuvchi kirib nima qiladi — bitta jumlada, ketma-ketlik bilan.",
                "bad": "Boshqaruv tizimi",
                "good": "Mijoz usta va bo'sh vaqtni tanlaydi → usta tasdiqlaydi → ikkalasiga "
                        "eslatma boradi → xizmatdan keyin mijoz baho qo'yadi",
            },
            {
                "q": "Hajm va cheklovlar?",
                "hint": "Foydalanuvchi soni, to'lov kerakmi, mobil ilova kerakmi, byudjet.",
                "bad": "Katta bo'lishi kerak",
                "good": "Birinchi yil ~5000 oylik foydalanuvchi, Payme orqali oldindan to'lov, "
                        "avval web keyin mobil, server byudjeti oyiga $50 gacha",
            },
        ],
        "example_title": "To'ldirilgan namuna",
        "example": (
            "Toshkent va Samarqanddagi kichik sartaroshxonalar (2-5 usta) uchun onlayn navbat "
            "tizimi qurmoqchiman.\n\n"
            "Hozir navbat Telegram guruhda qo'lda yoziladi: usta vaqtlarni chalkashtiradi, "
            "mijoz kelmay qolsa joy bo'sh ketadi va hech qanday statistika yo'q.\n\n"
            "Ishlash tartibi: mijoz saytdan sartaroshxona va ustani tanlaydi → bo'sh vaqtni "
            "ko'rib bron qiladi → usta tasdiqlaydi → ikkalasiga SMS va Telegram eslatma boradi → "
            "xizmatdan keyin mijoz baho qo'yadi. Usta o'z panelida kunlik jadval, daromad va "
            "doimiy mijozlar ro'yxatini ko'radi.\n\n"
            "Birinchi yilda ~5000 oylik faol foydalanuvchi kutyapman. Payme va Click orqali "
            "oldindan to'lov bo'lsin (kelmay qolishni kamaytirish uchun). Avval web versiya, "
            "keyin mobil ilova. Server byudjeti oyiga $50 gacha."
        ),
        "checklist_title": "Yuborishdan oldin tekshiring",
        "checklist": [
            "Auditoriya aniq aytilganmi (shahar, hajm, kim)?",
            "Muammo hozir qanday hal qilinayotgani yozilganmi?",
            "Asosiy stsenariy ketma-ketlik bilan yozilganmi?",
            "Foydalanuvchi soni va byudjet aytilganmi?",
            "To'lov, mobil ilova, xarita kabi texnik talablar aytilganmi?",
        ],
    },
    "ru": {
        "title": "Как описать свой стартап",
        "subtitle": "Чем конкретнее описание, тем точнее результат. Ответьте на 4 вопроса — этого достаточно.",
        "questions": [
            {
                "q": "Кто будет пользоваться?",
                "hint": "Назовите аудиторию конкретно: «небольшие барбершопы в Ташкенте» лучше, чем «все».",
                "bad": "Приложение для людей",
                "good": "Барбершопы на 2-5 мастеров в Ташкенте и Самарканде и их клиенты",
            },
            {
                "q": "Какую проблему решает?",
                "hint": "Скажите, как эта задача решается сейчас — так будет видна разница.",
                "bad": "Упрощает запись",
                "good": "Сейчас запись ведётся вручную в Telegram, мастер путает время, "
                        "а при неявке слот простаивает",
            },
            {
                "q": "Какое основное действие?",
                "hint": "Что делает пользователь — одной фразой, по шагам.",
                "bad": "Система управления",
                "good": "Клиент выбирает мастера и свободное время → мастер подтверждает → "
                        "обоим приходит напоминание → после услуги клиент ставит оценку",
            },
            {
                "q": "Масштаб и ограничения?",
                "hint": "Число пользователей, нужна ли оплата, нужно ли мобильное приложение, бюджет.",
                "bad": "Должно быть большим",
                "good": "~5000 активных пользователей в месяц в первый год, предоплата через Payme, "
                        "сначала веб, потом мобильное, бюджет на сервер до $50 в месяц",
            },
        ],
        "example_title": "Заполненный пример",
        "example": (
            "Хочу сделать систему онлайн-записи для небольших барбершопов (2-5 мастеров) "
            "в Ташкенте и Самарканде.\n\n"
            "Сейчас запись ведётся вручную в Telegram-группе: мастер путает время, при неявке "
            "клиента слот простаивает, статистики нет вообще.\n\n"
            "Как работает: клиент выбирает барбершоп и мастера на сайте → видит свободное время "
            "и бронирует → мастер подтверждает → обоим приходит SMS и напоминание в Telegram → "
            "после услуги клиент ставит оценку. В своей панели мастер видит расписание на день, "
            "доход и список постоянных клиентов.\n\n"
            "В первый год ожидаю ~5000 активных пользователей в месяц. Нужна предоплата через "
            "Payme и Click, чтобы снизить неявки. Сначала веб-версия, потом мобильное приложение. "
            "Бюджет на сервер — до $50 в месяц."
        ),
        "checklist_title": "Проверьте перед отправкой",
        "checklist": [
            "Указана ли аудитория конкретно (город, размер, кто)?",
            "Написано ли, как проблема решается сейчас?",
            "Описан ли основной сценарий по шагам?",
            "Указаны ли число пользователей и бюджет?",
            "Указаны ли технические требования: оплата, мобильное приложение, карта?",
        ],
    },
    "en": {
        "title": "How to describe your startup",
        "subtitle": "The more specific you are, the more precise the result. Answer 4 questions — that's enough.",
        "questions": [
            {
                "q": "Who will use it?",
                "hint": "Name the audience concretely: \"small barbershops in Tashkent\" beats \"everyone\".",
                "bad": "An app for people",
                "good": "Barbershops with 2-5 barbers in Tashkent and Samarkand, and their clients",
            },
            {
                "q": "What problem does it solve?",
                "hint": "Say how the problem is handled today — that's where the difference shows.",
                "bad": "Makes booking easier",
                "good": "Today booking is handled manually in a Telegram group: barbers mix up times "
                        "and no-shows leave slots empty, with no statistics at all",
            },
            {
                "q": "What is the core action?",
                "hint": "What the user does — one sentence, step by step.",
                "bad": "A management system",
                "good": "Client picks a barber and an open slot → barber confirms → both get a "
                        "reminder → after the service the client leaves a rating",
            },
            {
                "q": "Scale and constraints?",
                "hint": "User count, payments needed, mobile app needed, budget.",
                "bad": "It should be big",
                "good": "~5,000 monthly active users in year one, prepayment via Payme, web first "
                        "then mobile, server budget up to $50/month",
            },
        ],
        "example_title": "A filled-in example",
        "example": (
            "I want to build an online booking system for small barbershops (2-5 barbers) "
            "in Tashkent and Samarkand.\n\n"
            "Today booking happens manually in a Telegram group: barbers mix up time slots, "
            "no-shows leave the chair empty, and there are no statistics at all.\n\n"
            "How it works: the client picks a shop and a barber on the site → sees open slots and "
            "books one → the barber confirms → both receive an SMS and a Telegram reminder → after "
            "the service the client leaves a rating. In their dashboard the barber sees the daily "
            "schedule, revenue, and a list of returning clients.\n\n"
            "In year one I expect ~5,000 monthly active users. I need prepayment through Payme and "
            "Click to cut no-shows. Web first, mobile app later. Server budget up to $50/month."
        ),
        "checklist_title": "Check before submitting",
        "checklist": [
            "Is the audience named concretely (city, size, who)?",
            "Did you write how the problem is solved today?",
            "Is the core scenario written step by step?",
            "Did you state user count and budget?",
            "Did you state technical needs: payments, mobile app, maps?",
        ],
    },
}


@router.get("/guide", summary="Namuna va yo'riqnoma")
async def get_guide(lang: str = Query("uz", pattern="^(uz|ru|en)$")) -> dict:
    return _GUIDE[lang]
