"""Loyiha turi klassifikatori uchun qo'lda yozilgan dataset (uz / ru / en).

Foydalanuvchi startup g'oyasini erkin matnda yozadi. Klassifikator uni 14 ta
turdan biriga ajratadi — keyin shu tur bo'yicha stack, server va UI/UX
tavsiyalari beriladi.

Dataset uch tilda: mahalliy foydalanuvchilar o'zbekcha yozadi, lekin texnik
atamalarni inglizcha aralashtiradi ("marketplace qilmoqchiman"). Shuning uchun
klassifikator char n-gram ustida ishlaydi — til aralashsa ham ushlaydi.
"""

# (matn, yorliq)
PROJECT_TRAINING: list[tuple[str, str]] = [
    # ---------------- marketplace ----------------
    ("ikki tomonlama bozor sotuvchi va xaridorni bogʻlaydigan platforma", "marketplace"),
    ("marketplace qilmoqchiman usta va mijozni bogʻlaydi", "marketplace"),
    ("hunarmandlar oʻz mahsulotini qoʻyadi xaridor tanlaydi komissiya olamiz", "marketplace"),
    ("freelance birja frilanser va buyurtmachi uchun", "marketplace"),
    ("kvartira ijaraga berish platformasi egasi eʼlon qoʻyadi", "marketplace"),
    ("ustalar xizmati agregatori santexnik elektrik chaqirish", "marketplace"),
    ("ikkinchi qoʻl mahsulot sotish eʼlonlar sayti", "marketplace"),
    ("платформа маркетплейс продавцы и покупатели комиссия", "marketplace"),
    ("двусторонняя площадка для фрилансеров и заказчиков", "marketplace"),
    ("two sided marketplace connecting vendors and buyers with commission", "marketplace"),
    ("platform where service providers list offers and clients book them", "marketplace"),

    # ---------------- ecommerce ----------------
    ("onlayn doʻkon kiyim sotaman savat va toʻlov kerak", "ecommerce"),
    ("internet magazin mahsulot katalogi savatcha buyurtma", "ecommerce"),
    ("elektronika sotadigan internet doʻkon ochmoqchiman", "ecommerce"),
    ("kosmetika sotish uchun sayt katalog filtrlar chegirma", "ecommerce"),
    ("oziq ovqat doʻkoni onlayn buyurtma yetkazish bilan", "ecommerce"),
    ("brendim uchun onlayn savdo sayti ombor qoldigʻi bilan", "ecommerce"),
    ("интернет магазин корзина оплата доставка каталог товаров", "ecommerce"),
    ("онлайн магазин одежды с фильтрами и скидками", "ecommerce"),
    ("online store with product catalog cart checkout and inventory", "ecommerce"),
    ("d2c brand shop with discounts coupons and order tracking", "ecommerce"),

    # ---------------- saas_dashboard ----------------
    ("kompaniyalar uchun crm tizimi mijozlarni boshqarish", "saas_dashboard"),
    ("b2b saas obuna asosida ishlaydigan admin panel", "saas_dashboard"),
    ("hisobot va analitika dashboard kompaniyalar uchun", "saas_dashboard"),
    ("xodimlarni boshqarish tizimi hr uchun davomat va maosh", "saas_dashboard"),
    ("ombor boshqaruv tizimi qoldiq va hisobot", "saas_dashboard"),
    ("loyihalarni boshqarish task tracker jamoa uchun", "saas_dashboard"),
    ("buxgalteriya uchun saas platforma hisob fakturalar", "saas_dashboard"),
    ("crm система для отдела продаж с воронкой и отчетами", "saas_dashboard"),
    ("b2b saas панель управления подписка тарифы аналитика", "saas_dashboard"),
    ("b2b saas dashboard with subscriptions roles and analytics", "saas_dashboard"),
    ("internal admin panel for managing customers and reports", "saas_dashboard"),

    # ---------------- booking_service ----------------
    ("sartaroshxona uchun navbat olish tizimi", "booking_service"),
    ("shifokorga yozilish bron qilish ilovasi", "booking_service"),
    ("restoranda stol bron qilish tizimi", "booking_service"),
    ("sport zal uchun mashgʻulotga yozilish jadval", "booking_service"),
    ("mehmonxona xona bron qilish kalendar bilan", "booking_service"),
    ("tur paketlarni bron qilish sayti", "booking_service"),
    ("konferens zal ijara vaqt oraligʻini band qilish", "booking_service"),
    ("сервис онлайн записи к врачу расписание слоты", "booking_service"),
    ("бронирование столиков в ресторане календарь", "booking_service"),
    ("appointment booking app with calendar slots and reminders", "booking_service"),
    ("reservation system for salons with time slots", "booking_service"),

    # ---------------- delivery_logistics ----------------
    ("taksi chaqirish ilovasi haydovchi va yoʻlovchi", "delivery_logistics"),
    ("kuryer yetkazib berish xizmati real vaqtda kuzatish", "delivery_logistics"),
    ("ovqat yetkazib berish ilovasi restoran va kuryer", "delivery_logistics"),
    ("yuk tashish logistika platformasi mashina va yuk", "delivery_logistics"),
    ("pochta joʻnatmalarini kuzatish tizimi", "delivery_logistics"),
    ("kuryerlar uchun marshrut optimallashtirish", "delivery_logistics"),
    ("служба доставки еды с отслеживанием курьера на карте", "delivery_logistics"),
    ("приложение такси водитель пассажир геолокация", "delivery_logistics"),
    ("food delivery app with live courier tracking on map", "delivery_logistics"),
    ("ride hailing app matching drivers and riders in realtime", "delivery_logistics"),

    # ---------------- fintech ----------------
    ("elektron hamyon pul oʻtkazish ilovasi", "fintech"),
    ("mikro kredit berish platformasi skoring bilan", "fintech"),
    ("toʻlov tizimi integratsiyasi merchantlar uchun", "fintech"),
    ("xarajatlarni hisoblash shaxsiy byudjet ilovasi", "fintech"),
    ("investitsiya va aksiya sotib olish ilovasi", "fintech"),
    ("qarz va nasiya daftari doʻkonlar uchun", "fintech"),
    ("платежный сервис для мерчантов эквайринг", "fintech"),
    ("приложение для учета личных финансов и бюджета", "fintech"),
    ("digital wallet with p2p transfers and card top up", "fintech"),
    ("lending platform with credit scoring and repayment schedule", "fintech"),

    # ---------------- edtech ----------------
    ("onlayn kurs platformasi video dars va test", "edtech"),
    ("oʻquv markazi uchun platforma oʻquvchi va oʻqituvchi", "edtech"),
    ("til oʻrganish ilovasi soʻz yodlash kartochka", "edtech"),
    ("maktab uchun elektron kundalik baho va davomat", "edtech"),
    ("imtihonga tayyorlanish test bank savollar", "edtech"),
    ("repetitor topish va dars jadvali", "edtech"),
    ("образовательная платформа с курсами и тестами", "edtech"),
    ("приложение для изучения языков карточки слова", "edtech"),
    ("online course platform with video lessons quizzes and certificates", "edtech"),
    ("lms for schools with grades attendance and parent access", "edtech"),

    # ---------------- healthtech ----------------
    ("klinika uchun bemor kartasi va tarix", "healthtech"),
    ("telemeditsina shifokor bilan videoqoʻngʻiroq", "healthtech"),
    ("dori vositalarini eslatuvchi ilova", "healthtech"),
    ("laboratoriya tahlil natijalarini onlayn koʻrish", "healthtech"),
    ("fitnes va salomatlik kuzatuv ilovasi qadam kaloriya", "healthtech"),
    ("tibbiy karta va retsept saqlash tizimi", "healthtech"),
    ("телемедицина видеоконсультация с врачом", "healthtech"),
    ("медицинская карта пациента и история болезни", "healthtech"),
    ("telemedicine app with video consultations and prescriptions", "healthtech"),
    ("patient records ehr system for clinics with lab results", "healthtech"),

    # ---------------- social_community ----------------
    ("ijtimoiy tarmoq post lenta obuna va layk", "social_community"),
    ("qiziqish boʻyicha hamjamiyat forum muhokama", "social_community"),
    ("chat ilovasi guruh va shaxsiy xabar", "social_community"),
    ("tanishuv ilovasi profil va moslik", "social_community"),
    ("mahalla uchun eʼlonlar va muhokama platformasi", "social_community"),
    ("qisqa video ulashish ilovasi lenta bilan", "social_community"),
    ("социальная сеть лента подписки лайки комментарии", "social_community"),
    ("мессенджер групповые чаты и личные сообщения", "social_community"),
    ("social network with feed follows likes and comments", "social_community"),
    ("community forum with threads upvotes and moderation", "social_community"),

    # ---------------- content_media ----------------
    ("yangiliklar sayti maqola va rubrikalar", "content_media"),
    ("blog platformasi muallif va obunachi", "content_media"),
    ("video hosting va striming platformasi", "content_media"),
    ("podkast tinglash ilovasi", "content_media"),
    ("elektron kitob va audio kitob kutubxonasi", "content_media"),
    ("media nashr uchun sayt reklama bilan", "content_media"),
    ("новостной портал со статьями и рубриками", "content_media"),
    ("платформа для блогов с подпиской на авторов", "content_media"),
    ("news portal with articles categories and seo", "content_media"),
    ("video streaming platform with playlists and subscriptions", "content_media"),

    # ---------------- ai_tool ----------------
    ("suniy intellekt yordamida matn yozib beruvchi servis", "ai_tool"),
    ("ai chatbot mijozlar savoliga javob beradi", "ai_tool"),
    ("rasm generatsiya qiluvchi ai ilova", "ai_tool"),
    ("hujjatlarni ai bilan tahlil qiladigan servis", "ai_tool"),
    ("ai yordamchi hisobot va tahlil tayyorlaydi", "ai_tool"),
    ("nutqni matnga oʻgiruvchi ai xizmat", "ai_tool"),
    ("ai asosida rezyume yozib beruvchi platforma", "ai_tool"),
    ("ии сервис генерации текста и изображений", "ai_tool"),
    ("чат бот на базе ии для поддержки клиентов", "ai_tool"),
    ("ai powered writing assistant with rag over user documents", "ai_tool"),
    ("llm chatbot saas answering customer questions from knowledge base", "ai_tool"),

    # ---------------- devtool_api ----------------
    ("dasturchilar uchun api xizmat va sdk", "devtool_api"),
    ("ci cd va deploy avtomatlashtirish vositasi", "devtool_api"),
    ("monitoring va log yigʻish tizimi", "devtool_api"),
    ("sms va email yuborish uchun api", "devtool_api"),
    ("webhook va integratsiya platformasi", "devtool_api"),
    ("api gateway va rate limit xizmati", "devtool_api"),
    ("api сервис для разработчиков с sdk и документацией", "devtool_api"),
    ("инструмент мониторинга логов и метрик", "devtool_api"),
    ("developer api service with sdk keys and usage billing", "devtool_api"),
    ("observability tool collecting logs metrics and traces", "devtool_api"),

    # ---------------- mobile_app ----------------
    ("android va ios uchun mobil ilova qilmoqchiman", "mobile_app"),
    ("faqat telefon uchun ilova offline ishlashi kerak", "mobile_app"),
    ("mobil ilova push bildirishnoma bilan", "mobile_app"),
    ("flutter da mobil ilova yozmoqchiman", "mobile_app"),
    ("react native ilova ios android bitta koddan", "mobile_app"),
    ("telefon ilovasi kamera va galereya bilan ishlaydi", "mobile_app"),
    ("мобильное приложение для android и ios с push", "mobile_app"),
    ("нативное мобильное приложение работающее оффлайн", "mobile_app"),
    ("cross platform mobile app for ios and android with offline mode", "mobile_app"),
    ("native mobile app using camera gps and push notifications", "mobile_app"),

    # ---------------- game ----------------
    ("brauzerda oʻynaladigan oddiy oʻyin", "game"),
    ("koʻp oʻyinchili onlayn oʻyin reyting bilan", "game"),
    ("mobil oʻyin daraja va ochko tizimi", "game"),
    ("viktorina oʻyin savol javob musobaqa", "game"),
    ("shaxmat oʻynash platformasi onlayn raqib", "game"),
    ("oʻyin ichida xarid va donat tizimi", "game"),
    ("многопользовательская онлайн игра с рейтингом", "game"),
    ("мобильная игра с уровнями и внутриигровыми покупками", "game"),
    ("multiplayer browser game with leaderboard and matchmaking", "game"),
    ("mobile game with levels achievements and in app purchases", "game"),
]


# Uzun, realistik misollar alohida faylda — ular real foydalanuvchi yozuviga
# yaqin va chalkashadigan sinflarni ajratadi. Ikkalasi birga o'qitiladi.
from app.ml.training_data_extra import PROJECT_TRAINING_EXTRA  # noqa: E402

PROJECT_TRAINING = PROJECT_TRAINING + PROJECT_TRAINING_EXTRA


# Yorliq -> odam oʻqiy oladigan nom (uz / ru / en).
PROJECT_LABELS: dict[str, dict[str, str]] = {
    "marketplace":        {"uz": "Marketplace (ikki tomonlama bozor)", "ru": "Маркетплейс", "en": "Marketplace"},
    "ecommerce":          {"uz": "Onlayn do'kon (e-commerce)", "ru": "Интернет-магазин", "en": "E-commerce"},
    "saas_dashboard":     {"uz": "B2B SaaS / boshqaruv paneli", "ru": "B2B SaaS / панель", "en": "B2B SaaS / dashboard"},
    "booking_service":    {"uz": "Bron va navbat xizmati", "ru": "Бронирование и запись", "en": "Booking & scheduling"},
    "delivery_logistics": {"uz": "Yetkazib berish / logistika", "ru": "Доставка и логистика", "en": "Delivery & logistics"},
    "fintech":            {"uz": "Fintech / to'lov", "ru": "Финтех / платежи", "en": "Fintech / payments"},
    "edtech":             {"uz": "Ta'lim platformasi (EdTech)", "ru": "Образование (EdTech)", "en": "EdTech"},
    "healthtech":         {"uz": "Tibbiyot platformasi", "ru": "Медицина (HealthTech)", "en": "HealthTech"},
    "social_community":   {"uz": "Ijtimoiy tarmoq / hamjamiyat", "ru": "Соцсеть / сообщество", "en": "Social / community"},
    "content_media":      {"uz": "Kontent va media", "ru": "Контент и медиа", "en": "Content & media"},
    "ai_tool":            {"uz": "AI mahsulot", "ru": "AI-продукт", "en": "AI product"},
    "devtool_api":        {"uz": "Dasturchilar uchun API / vosita", "ru": "API / инструменты для разработчиков", "en": "Developer API / tooling"},
    "mobile_app":         {"uz": "Mobil ilova", "ru": "Мобильное приложение", "en": "Mobile app"},
    "game":               {"uz": "O'yin", "ru": "Игра", "en": "Game"},
}
