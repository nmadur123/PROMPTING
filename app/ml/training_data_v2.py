"""Uchinchi dataset — chegaralarni o'tkirlashtiradigan misollar.

Nega kerak bo'ldi. `scripts/eval_classifier.py` bazaviy o'lchovi 84.0% chiqdi
va xatolarning ikkita aniq naqshi bor edi:

1. QISQA IBORALAR yiqiladi. "podkast tinglash ilovasi", "shaxmat o'ynash
   platformasi" — uch-to'rt so'zda sinfni ajratadigan signal yetarli emas.
   Real foydalanuvchi esa bir necha jumla yozadi, ya'ni dataset taqsimoti
   haqiqiy kirishga mos kelmasdi.

2. "PLATFORMA" so'zi hamma narsani `marketplace` ga tortadi. O'quv to'plamida
   bu so'z asosan marketplace misollarida uchragani uchun model uni sinf
   belgisi deb o'rgangan: `booking_service`, `edtech`, `fintech`,
   `social_community` — hammasi shu tuzoqqa tushardi (marketplace F1 = 0.64,
   eng past ko'rsatkich).

Shuning uchun bu yerdagi misollar ataylab:
  - uzun va ko'p jumlali (haqiqiy kirishga o'xshash);
  - "platforma", "tizim", "servis" kabi neytral so'zlarni marketplace BO'LMAGAN
    sinflarda ham ishlatadi — spurious korrelyatsiyani sindirish uchun;
  - har bir sinfning O'ZIGA XOS lug'atini olib kiradi (kredit/skoring —
    fintech, vaqt oralig'i/bekor qilish — booking, komissiya/ikki tomon —
    marketplace).

Yangi misol qo'shgandan keyin O'LCHANG:
    python -m scripts.eval_classifier
"""

# (matn, yorliq)
PROJECT_TRAINING_V2: list[tuple[str, str]] = [
    # ------------------------------------------------------------ marketplace
    # Ajratuvchi belgi: IKKI TOMON + vositachilik daromadi (komissiya).
    ("platforma ikki tomonni bogʻlaydi: usta oʻz xizmatini eʼlon qiladi, mijoz tanlaydi va biz har bitimdan komissiya olamiz", "marketplace"),
    ("agregator sayt — koʻp sotuvchi oʻz tovarini qoʻyadi, xaridor taqqoslaydi, pul bizda ushlanadi va yetkazilgach sotuvchiga oʻtadi", "marketplace"),
    ("frilanserlar birjasi: buyurtmachi vazifa eʼlon qiladi, ijrochilar taklif yuboradi, kelishuv boʻlgach mablagʻ eskrouda turadi", "marketplace"),
    ("платформа-посредник: поставщики публикуют предложения, покупатели выбирают, мы удерживаем комиссию с каждой сделки", "marketplace"),
    ("двусторонняя площадка объявлений: продавец размещает лот, покупатель пишет в чат, сделка проходит через нас", "marketplace"),
    ("сервис агрегатор мастеров: специалист заводит профиль с рейтингом и отзывами, клиент оставляет заявку", "marketplace"),
    ("two sided marketplace where independent providers list services, customers book them and the platform takes a commission on each transaction", "marketplace"),
    ("classifieds platform: sellers publish listings, buyers contact them, we monetize with promoted placements and a transaction fee", "marketplace"),

    # -------------------------------------------------------------- ecommerce
    # Ajratuvchi belgi: BITTA sotuvchi, ombor qoldigʻi, savat.
    ("oʻz doʻkonimiz uchun sayt: katalogda mahsulotlar, savat, buyurtma rasmiylashtirish va ombordagi qoldiq avtomatik kamayadi", "ecommerce"),
    ("internet magazin platformasi: mahsulot kartochkasi, oʻlcham va rang variantlari, chegirma kuponi va yetkazib berish narxi hisobi", "ecommerce"),
    ("mebel savdosi uchun onlayn doʻkon, 300 dan ortiq mahsulot, har biriga bir nechta rasm, filtr va taqqoslash", "ecommerce"),
    ("интернет магазин косметики: каталог с фильтрами, корзина, промокоды, остатки на складе и статусы заказа", "ecommerce"),
    ("система для розничного магазина: карточки товаров, скидки, оформление заказа и выгрузка накладных", "ecommerce"),
    ("online store for a single brand with product variants, cart, checkout, inventory tracking and order fulfillment statuses", "ecommerce"),

    # --------------------------------------------------------- saas_dashboard
    # Ajratuvchi belgi: TASHKILOT hisobi, obuna tarifi, rollar, hisobot.
    ("kompaniyalar uchun boshqaruv paneli: har bir tashkilot alohida hisob ochadi, xodimlarga rol beradi va oylik tarif toʻlaydi", "saas_dashboard"),
    ("mijozlar bazasini yuritish tizimi: bitim bosqichlari, vazifalar, eslatmalar va rahbar uchun hisobot grafiklari", "saas_dashboard"),
    ("ichki hisobot paneli: maʼlumot bir nechta manbadan yigʻiladi, jadval va diagrammada koʻrsatiladi, eksport qilinadi", "saas_dashboard"),
    ("saas платформа для бизнеса: организации, роли сотрудников, тарифные планы с ограничениями и биллинг по подписке", "saas_dashboard"),
    ("crm система: воронка сделок, задачи менеджеров, история переписки и отчеты для руководителя", "saas_dashboard"),
    ("b2b dashboard where each organization manages its own workspace, invites team members with roles and pays a monthly subscription", "saas_dashboard"),

    # -------------------------------------------------------- booking_service
    # Ajratuvchi belgi: VAQT ORALIGʻI, jadval, band qilish, bekor qilish siyosati.
    ("platforma orqali mijoz boʻsh vaqt oraligʻini tanlaydi, usta tasdiqlaydi, bir kun oldin eslatma boradi va bekor qilish qoidasi ishlaydi", "booking_service"),
    ("shifoxona uchun navbat tizimi: shifokorning ish jadvali, band qilingan vaqtlar, qayta rejalashtirish va kelmay qolganini belgilash", "booking_service"),
    ("stol band qilish xizmati: zal xaritasi, vaqt oraligʻi, oldindan toʻlov va bekor qilinganda oʻrin boʻshab qolishi", "booking_service"),
    ("сервис записи на приём: расписание специалиста, свободные слоты, подтверждение и напоминание за сутки", "booking_service"),
    ("система бронирования номеров: календарь занятости, типы номеров, предоплата и правила отмены", "booking_service"),
    ("appointment platform where the schedule is generated from recurring availability rules, double booking is prevented and reminders are sent", "booking_service"),

    # ----------------------------------------------------- delivery_logistics
    # Ajratuvchi belgi: KURYER, marshrut, manzil, real vaqtda kuzatuv.
    ("yetkazib berish platformasi: buyurtma kuryerga biriktiriladi, marshrut hisoblanadi va mijoz xaritada kuzatib turadi", "delivery_logistics"),
    ("taksi chaqirish tizimi: yoʻlovchi manzil kiritadi, yaqin haydovchi topiladi, narx masofaga qarab hisoblanadi", "delivery_logistics"),
    ("yuk tashish uchun tizim: yuk egasi eʼlon beradi, haydovchi reys oladi, yoʻl davomida holat yangilanadi", "delivery_logistics"),
    ("платформа доставки еды: заказ передаётся курьеру, строится маршрут, клиент видит машину на карте в реальном времени", "delivery_logistics"),
    ("логистическая система: точки погрузки и разгрузки, оптимизация маршрута и подтверждение доставки фотографией", "delivery_logistics"),
    ("last mile delivery system assigning parcels to couriers, optimizing routes and showing live tracking to the recipient", "delivery_logistics"),

    # ---------------------------------------------------------------- fintech
    # Ajratuvchi belgi: PUL mahsulotning OʻZI — kredit, hisob, tranzaksiya, skoring.
    ("mikrokredit platformasi: ariza, avtomatik skoring, tasdiqlash va toʻlov jadvali boʻyicha qaytarish", "fintech"),
    ("elektron hamyon: kartadan toʻldirish, foydalanuvchilar orasida oʻtkazma, tranzaksiya tarixi va limitlar", "fintech"),
    ("shaxsiy moliya tizimi: kartadagi xarajatlar avtomatik kategoriyalarga ajratiladi, oylik byudjet va oshib ketish ogohlantirishi", "fintech"),
    ("valyuta ayirboshlash xizmati: joriy kurs, komissiya hisobi, buyurtma va hisobga tushirish", "fintech"),
    ("платежный сервис для мерчантов: эквайринг, сверка транзакций, возвраты и выплаты на расчетный счет", "fintech"),
    ("система рассрочки: заявка клиента, скоринг, одобрение лимита и график ежемесячных платежей", "fintech"),
    ("digital wallet with card top up, peer to peer transfers, transaction ledger, KYC verification and spending limits", "fintech"),

    # ----------------------------------------------------------------- edtech
    # Ajratuvchi belgi: KURS, dars, test, oʻquvchi progressi, sertifikat.
    ("oʻquv platformasi: oʻqituvchi dars yuklaydi, oʻquvchi videoni koʻradi, test topshiradi va progress foizda koʻrinadi", "edtech"),
    ("til oʻrganish tizimi: kartochkalar bilan soʻz yodlash, kunlik mashq, takrorlash intervali va daraja testi", "edtech"),
    ("dasturlash oʻrgatuvchi platforma: amaliy topshiriq, kod avtomatik tekshiriladi, xato boʻlsa maslahat beriladi", "edtech"),
    ("oʻquv markazi uchun tizim: guruhlar, davomat, baholar, oylik toʻlov va ota-onaga hisobot", "edtech"),
    ("образовательная платформа: преподаватель загружает уроки и тесты, ученик проходит их, выдается сертификат", "edtech"),
    ("сервис для изучения языков: карточки со словами, интервальное повторение и ежедневная норма", "edtech"),
    ("online course platform with video lessons, quizzes, student progress tracking and completion certificates", "edtech"),

    # ------------------------------------------------------------- healthtech
    # Ajratuvchi belgi: BEMOR, shifokor, tibbiy yozuv, retsept.
    ("klinika uchun tizim: bemor kartasi, tashrif yozuvlari, tashxis, retsept va tahlil natijalari saqlanadi", "healthtech"),
    ("telemeditsina platformasi: bemor shifokor bilan videoaloqa orqali maslahatlashadi, retsept elektron beriladi", "healthtech"),
    ("surunkali kasallikni kuzatish ilovasi: qon bosimi va qand koʻrsatkichi kiritiladi, dori qabuli eslatiladi", "healthtech"),
    ("медицинская система: карта пациента, история приемов, назначения и результаты анализов", "healthtech"),
    ("приложение для врачей: список пациентов, записи о визитах и защищенное хранение медицинских документов", "healthtech"),
    ("patient records platform storing visit notes, diagnoses, prescriptions and lab results with strict access control", "healthtech"),

    # -------------------------------------------------------- social_community
    # Ajratuvchi belgi: PROFIL, lenta, obuna, izoh, moderatsiya.
    ("mahalla platformasi: qoʻshnilar eʼlon qoldiradi, savol beradi, izohlashadi va moderator nomaqbul postni oʻchiradi", "social_community"),
    ("qiziqishlar boʻyicha jamoa: mavzular, muhokamalar, obuna va foydalanuvchi profili", "social_community"),
    ("tanishuv ilovasi: profil, qiziqishlar boʻyicha moslik, oʻzaro yoqtirish va shundan keyin chat ochiladi", "social_community"),
    ("социальная платформа: лента постов, подписки на людей, комментарии, лайки и жалобы на контент", "social_community"),
    ("сообщество по интересам с темами, обсуждениями, репутацией участников и модерацией", "social_community"),
    ("community app with user profiles, a following feed, threaded comments, reactions and moderation tools", "social_community"),

    # ----------------------------------------------------------- content_media
    # Ajratuvchi belgi: NASHR, maqola/video/podkast, reklama yoki obuna daromadi.
    ("onlayn nashr platformasi: muharrir maqola chiqaradi, oʻquvchi oʻqiydi, daromad reklama va obunadan keladi", "content_media"),
    ("video koʻrish xizmati: katalog, obuna, koʻrishni davom ettirish va tavsiyalar", "content_media"),
    ("podkast tinglash ilovasi: epizodlar roʻyxati, yuklab olish, tinglash tarixi va yangi epizod bildirishnomasi", "content_media"),
    ("блог платформа: автор публикует статьи, читатели оформляют платную подписку на автора", "content_media"),
    ("медиа издание: рубрики, лента материалов, рекламные блоки и рассылка дайджеста", "content_media"),
    ("streaming service with a content catalog, subscription tiers, continue watching and personalized recommendations", "content_media"),

    # ---------------------------------------------------------------- ai_tool
    # Ajratuvchi belgi: MODEL/generatsiya mahsulotning oʻzi.
    ("sunʼiy intellekt xizmati: foydalanuvchi matn kiritadi, model rasm generatsiya qiladi va natijani yuklab oladi", "ai_tool"),
    ("hujjatlarni tahlil qiluvchi tizim: fayl yuklanadi, model mazmunini chiqaradi va savolga javob beradi", "ai_tool"),
    ("matnni avtomatik qisqartiruvchi va tarjima qiluvchi platforma, natija sifatini foydalanuvchi baholaydi", "ai_tool"),
    ("сервис генерации текста и изображений по описанию с историей запросов и настройками стиля", "ai_tool"),
    ("ии помощник который разбирает документ и отвечает на вопросы по его содержанию", "ai_tool"),
    ("ai writing assistant that rewrites drafts, adjusts tone and explains the changes it made", "ai_tool"),

    # ------------------------------------------------------------ devtool_api
    # Ajratuvchi belgi: DASTURCHI mijoz — API kalit, soʻrov limiti, SDK.
    ("dasturchilar uchun platforma: API kalit olinadi, soʻrovlar soni hisoblanadi va limitdan oshsa toʻlov qoʻshiladi", "devtool_api"),
    ("api xizmati: hujjatlar, kalit boshqaruvi, webhook va har bir soʻrov jurnalga yoziladi", "devtool_api"),
    ("integratsiya vositasi: sdk kutubxonasi, sandbox muhiti va xatolarni kuzatish paneli", "devtool_api"),
    ("сервис для разработчиков: выдача api ключей, лимиты запросов, тарификация по объему и документация", "devtool_api"),
    ("платформа мониторинга: сбор логов и метрик с приложений клиентов, алерты и дашборд", "devtool_api"),
    ("developer platform issuing api keys, metering requests per plan, exposing webhooks and an sdk with sandbox credentials", "devtool_api"),

    # ------------------------------------------------------------- mobile_app
    # Ajratuvchi belgi: PLATFORMA tanlovi, push, oflayn, store.
    ("android va ios uchun mobil ilova kerak, push bildirishnoma va internetsiz ishlash imkoniyati bilan", "mobile_app"),
    ("faqat telefon uchun ilova: kamera orqali skanerlash, oflayn saqlash va keyin sinxronizatsiya", "mobile_app"),
    ("mobil ilova, app store va play marketga chiqariladi, biometrik kirish va push xabarlar bor", "mobile_app"),
    ("мобильное приложение под android и ios: push уведомления, работа офлайн и синхронизация при появлении сети", "mobile_app"),
    ("нативное приложение для телефона со сканированием камерой и локальным хранилищем", "mobile_app"),
    ("cross platform mobile app for ios and android with push notifications, offline mode and background sync", "mobile_app"),

    # ------------------------------------------------------------------- game
    # Ajratuvchi belgi: OʻYIN mexanikasi — daraja, ochko, raqib, reyting.
    ("koʻp oʻyinchili oʻyin: ikki raqib real vaqtda oʻynaydi, natija reyting jadvaliga yoziladi va yangi raqib avtomatik topiladi", "game"),
    ("brauzer oʻyini: darajalar, toʻplangan ochkolar, kunlik vazifalar va doʻstlar bilan raqobat", "game"),
    ("shaxmat oʻynash xizmati: onlayn raqib topish, vaqt nazorati, partiya tarixi va reyting", "game"),
    ("мобильная игра: уровни, достижения, внутриигровая валюта и покупки", "game"),
    ("многопользовательская игра с подбором соперников, таблицей лидеров и рейтингом", "game"),
    ("realtime multiplayer game with matchmaking, scoring, leaderboards and seasonal ranking", "game"),
]
