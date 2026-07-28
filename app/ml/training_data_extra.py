"""Qo'shimcha dataset — real foydalanuvchi yozuviga o'xshash uzun misollar.

Nega kerak: asosiy dataset qisqa iboralardan iborat ("onlayn dokon"), real
foydalanuvchi esa bir necha jumlali paragraf yozadi. Bu taqsimot farqi
klassifikator aniqligini pasaytirardi — bu yerda har bir sinf uchun uzun,
realistik misollar berilgan.

Shuningdek chalkashib ketadigan juftliklar (content_media ↔ edtech,
ecommerce ↔ content_media, fintech ↔ boshqalar) uchun ajratuvchi
so'z boyligi qo'shilgan.
"""

PROJECT_TRAINING_EXTRA: list[tuple[str, str]] = [
    # ---------------- marketplace: komissiya, ikki tomon, vositachi ----------
    ("platformada usta oz xizmatini elon qiladi mijoz tanlaydi biz har bitimdan komissiya olamiz", "marketplace"),
    ("sotuvchilar ozi royxatdan otadi mahsulot qoyadi biz faqat vositachimiz ombor tutmaymiz", "marketplace"),
    ("ikki tomonlama platforma: bir tomonda xizmat korsatuvchilar boshqa tomonda buyurtmachilar reyting bilan", "marketplace"),
    ("dehqonlar mahsulotini togridan togri restoranlarga sotadigan platforma vositachisiz", "marketplace"),
    ("avtomobil ijarasi platformasi mashina egasi qoyadi ijarachi oladi biz sugurta va tolovni boshqaramiz", "marketplace"),
    ("agregator: barcha tur firmalarning paketlari bir joyda mijoz taqqoslab tanlaydi", "marketplace"),
    ("хочу площадку где мастера размещают услуги а клиенты выбирают мы берем комиссию с каждой сделки", "marketplace"),
    ("платформа агрегатор объединяет много поставщиков в одном каталоге с рейтингом и отзывами", "marketplace"),
    ("marketplace where independent sellers list their own inventory and we take a cut of each transaction", "marketplace"),
    ("aggregator platform bringing many providers into one searchable catalog with ratings", "marketplace"),

    # ---------------- ecommerce: bitta brend, oz ombori, savat ---------------
    ("ozimning kiyim brendim bor onlayn sotmoqchiman ombor qoldigi savat va yetkazish bilan", "ecommerce"),
    ("bitta dokon uchun sayt: mahsulot kartochkasi olcham rang tanlash savatga qoshish va tolov", "ecommerce"),
    ("mebel sotamiz katalogda 300 ta mahsulot bor har biriga rasm va tavsif kerak chegirma tizimi ham", "ecommerce"),
    ("elektronika dokoni uchun internet magazin: kafolat muddati taqqoslash va qismlarga bolib tolash", "ecommerce"),
    ("kosmetika brendi uchun onlayn savdo promokod sovga qadoq va takroriy buyurtma tugmasi bilan", "ecommerce"),
    ("оптовый магазин: свой склад свой каталог корзина оформление заказа и отслеживание доставки", "ecommerce"),
    ("интернет магазин одного бренда с остатками на складе промокодами и историей заказов", "ecommerce"),
    ("single brand online store: our own inventory, cart, checkout, order tracking and returns", "ecommerce"),
    ("d2c store where we sell our own products with stock levels discount codes and shipping", "ecommerce"),

    # ---------------- content_media: maqola, tomosha, obuna, reklama --------
    ("yangiliklar portali: kuniga 20 ta maqola chiqadi rubrikalar tahririyat va reklama bannerlari", "content_media"),
    ("ozbek tilidagi maqolalar sayti oquvchi bepul oqiydi daromad reklamadan keladi seo juda muhim", "content_media"),
    ("film va serial tomosha qilish platformasi obuna asosida katalog va davom ettirish tugmasi", "content_media"),
    ("podkast platformasi: mualliflar epizod yuklaydi tinglovchilar obuna boladi va tinglaydi", "content_media"),
    ("blog platformasi mualliflar maqola yozadi oquvchilar obuna boladi pullik obuna ham bor", "content_media"),
    ("audio kitob va elektron kitob kutubxonasi oylik obuna bilan cheksiz oqish", "content_media"),
    ("медиа портал: редакция публикует статьи читатели читают бесплатно доход от рекламы", "content_media"),
    ("платформа для просмотра видео по подписке каталог фильмов и продолжить просмотр", "content_media"),
    ("news and media site where an editorial team publishes articles and revenue comes from ads and seo traffic", "content_media"),
    ("subscription video streaming catalog where viewers watch and continue where they left off", "content_media"),

    # ---------------- edtech: oqituvchi, oquvchi, baho, sertifikat ----------
    ("onlayn kurs platformasi: oqituvchi video dars yuklaydi oquvchi korib test topshiradi sertifikat oladi", "edtech"),
    ("oquv markazi uchun tizim: guruhlar davomat baholar tolov va ota onaga hisobot", "edtech"),
    ("dasturlash orgatuvchi platforma amaliy topshiriq avtomatik tekshiruv va progress kuzatuvi", "edtech"),
    ("til kurslari uchun ilova: darslar leksika mashqlari kunlik seriya va daraja testi", "edtech"),
    ("maktab uchun elektron platforma: dars jadvali uy vazifasi baho jurnali va ota ona kabineti", "edtech"),
    ("abituriyentlar uchun test tayyorgarlik: savol bazasi variant yechish va natija tahlili", "edtech"),
    ("образовательная платформа: преподаватель загружает уроки ученик проходит тесты и получает сертификат", "edtech"),
    ("система для учебного центра: группы посещаемость оценки оплата и отчет родителям", "edtech"),
    ("course platform where instructors upload lessons, students take quizzes, track progress and earn certificates", "edtech"),
    ("school management platform with timetable homework gradebook attendance and parent portal", "edtech"),

    # ---------------- fintech: pul, hisob, tranzaksiya, limit ---------------
    ("elektron hamyon: foydalanuvchi kartadan hisobini toldiradi boshqa foydalanuvchiga pul otkazadi tarix koradi", "fintech"),
    ("mikroqarz xizmati: ariza skoring tasdiqlash va oylik tolov jadvali bilan qarz berish", "fintech"),
    ("dokonlar uchun nasiya daftari: kim qancha qarzdor eslatma va tolov tarixi", "fintech"),
    ("shaxsiy moliya ilovasi: kartadan xarajatlarni yigib kategoriyalarga ajratadi va byudjet chegarasini eslatadi", "fintech"),
    ("valyuta ayirboshlash va otkazmalar xizmati kurs va komissiya hisobi bilan", "fintech"),
    ("jamgarma va investitsiya ilovasi: foydalanuvchi pul qoyadi foizini va portfelini kuzatadi", "fintech"),
    ("tolov shlyuzi: merchantlar bizning api orqali karta qabul qiladi biz tranzaksiyani protsessing qilamiz", "fintech"),
    ("электронный кошелек: пополнение с карты переводы между пользователями история транзакций и лимиты", "fintech"),
    ("сервис микрозаймов со скорингом заявкой одобрением и графиком погашения", "fintech"),
    ("digital wallet with card top up peer to peer transfers transaction history and spending limits", "fintech"),
    ("lending service with application scoring approval flow and monthly repayment schedule", "fintech"),

    # ---------------- delivery_logistics: kuryer, marshrut, yetkazish -------
    ("kuryer ilovasi: buyurtma keladi kuryer qabul qiladi xaritada manzilga boradi va yetkazganini belgilaydi", "delivery_logistics"),
    ("ovqat yetkazish: restoran buyurtmani qabul qiladi kuryer olib ketadi mijoz xaritada kuzatadi", "delivery_logistics"),
    ("yuk tashish platformasi: yuk egasi elon beradi haydovchi oladi marshrut va yuklash hujjatlari", "delivery_logistics"),
    ("dokon uchun yetkazish moduli: kuryerlarga marshrut taqsimlash va yetkazish vaqtini bashorat qilish", "delivery_logistics"),
    ("taksi buyurtma: yolovchi manzil qoyadi eng yaqin haydovchi topiladi narx oldindan hisoblanadi", "delivery_logistics"),
    ("pochta va jonatma kuzatuv tizimi: shtrix kod skanerlash va harakat tarixi", "delivery_logistics"),
    ("служба доставки: курьер принимает заказ едет по карте отмечает доставку клиент видит трекинг", "delivery_logistics"),
    ("логистическая платформа: грузовладелец публикует груз водитель берет маршрут и документы", "delivery_logistics"),
    ("courier delivery app where drivers accept orders navigate on a map and mark deliveries complete", "delivery_logistics"),
    ("last mile logistics with route assignment eta prediction and live package tracking", "delivery_logistics"),

    # ---------------- healthtech: bemor, shifokor, tashxis ------------------
    ("klinika uchun tizim: bemor kartasi tashxis tarixi retsept va shifokor jadvali", "healthtech"),
    ("telemeditsina xizmati: bemor shifokor bilan videoqongiroqda maslahatlashadi va retsept oladi", "healthtech"),
    ("laboratoriya uchun platforma: tahlil natijalari bemorga sms va shaxsiy kabinetga tushadi", "healthtech"),
    ("surunkali kasallikni kuzatish ilovasi: qon bosimi qand darajasi va dori qabulini eslatish", "healthtech"),
    ("stomatologiya klinikasi uchun: bemor tarixi davolash rejasi tish kartasi va tolov", "healthtech"),
    ("shifokorlar uchun mobil ilova: bemorlar royxati tashrif yozuvlari va tibbiy hujjat", "healthtech"),
    ("система для клиники: карта пациента история диагнозов рецепты и расписание врачей", "healthtech"),
    ("телемедицина: пациент консультируется с врачом по видео и получает рецепт", "healthtech"),
    ("clinic system with patient records diagnosis history prescriptions and doctor schedules", "healthtech"),
    ("chronic condition monitoring app tracking blood pressure glucose and medication adherence", "healthtech"),

    # ---------------- social_community: lenta, doʻstlar, muhokama ----------
    ("foydalanuvchilar post yozadi bir birini kuzatadi layk va izoh qoldiradi lenta algoritm bilan", "social_community"),
    ("qiziqish boyicha hamjamiyat: mavzular ochiladi odamlar muhokama qiladi moderatorlar nazorat qiladi", "social_community"),
    ("talabalar uchun ijtimoiy tarmoq: guruhlar elonlar va shaxsiy xabar almashish", "social_community"),
    ("mahalla platformasi: qoshnilar elon qoyadi savol beradi va bir biriga yordam soraydi", "social_community"),
    ("qisqa video ulashish ilovasi: cheksiz lenta layk izoh va ulashish", "social_community"),
    ("messenjer: shaxsiy va guruh chatlari fayl yuborish va oqilgan belgisi", "social_community"),
    ("социальная сеть: пользователи пишут посты подписываются друг на друга ставят лайки и комментируют", "social_community"),
    ("сообщество по интересам с темами обсуждениями и модерацией", "social_community"),
    ("social feed where users post follow each other like and comment with an algorithmic timeline", "social_community"),
    ("interest based community with threads discussions upvotes and moderator tools", "social_community"),

    # ---------------- saas_dashboard: b2b, tarif, hisobot -------------------
    ("kompaniyalar obuna boladi har biri oz kabinetida xodimlarini va hisobotlarini koradi tarif limitlari bor", "saas_dashboard"),
    ("b2b mahsulot: har bir firma alohida akkaunt rollar huquqlar va oylik tolov", "saas_dashboard"),
    ("savdo boyicha crm: lidlar voronka bosqichlari menejerlar va konversiya hisoboti", "saas_dashboard"),
    ("ombor va buxgalteriya tizimi: kirim chiqim qoldiq va oylik moliyaviy hisobot", "saas_dashboard"),
    ("hr platformasi: xodimlar davomat tatil arizasi va maosh hisobi", "saas_dashboard"),
    ("restoran tarmogi uchun boshqaruv paneli: filiallar menyu sotuvlar va xodimlar", "saas_dashboard"),
    ("b2b saas: компании подписываются у каждой свой кабинет роли лимиты тарифа и отчеты", "saas_dashboard"),
    ("crm для отдела продаж: лиды этапы воронки менеджеры и отчет по конверсии", "saas_dashboard"),
    ("multi tenant b2b saas where each company gets its own workspace roles plan limits and reports", "saas_dashboard"),
    ("internal business dashboard with kpis charts exports and role based access", "saas_dashboard"),

    # ---------------- booking_service: vaqt, slot, tasdiq -------------------
    ("mijoz bosh vaqtni tanlaydi usta tasdiqlaydi ikkalasiga eslatma boradi kelmay qolsa joy bosh qoladi", "booking_service"),
    ("sartaroshxona uchun onlayn navbat: usta jadvali xizmat turi va oldindan tolov", "booking_service"),
    ("shifokorga yozilish: bosh vaqtlar royxati tanlash tasdiqlash va sms eslatma", "booking_service"),
    ("sport zal uchun mashgulotga yozilish: guruh sigimi va bekor qilish qoidasi", "booking_service"),
    ("mehmonxona bron tizimi: xona turlari kalendar bandlik va oldindan tolov", "booking_service"),
    ("avtoservis uchun yozilish: usta vaqti xizmat davomiyligi va navbat boshqaruvi", "booking_service"),
    ("онлайн запись: клиент выбирает свободный слот мастер подтверждает обоим приходит напоминание", "booking_service"),
    ("бронирование номеров в отеле: календарь занятости типы номеров и предоплата", "booking_service"),
    ("appointment scheduling where clients pick an open slot, the provider confirms, and both get reminders", "booking_service"),
    ("resource reservation system with availability calendar cancellation policy and deposits", "booking_service"),

    # ---------------- ai_tool: model, prompt, generatsiya -------------------
    ("foydalanuvchi hujjat yuklaydi ai uni oqib savolga javob beradi", "ai_tool"),
    ("ai yordamchi: matn kiritiladi model tahlil qilib tayyor natija qaytaradi", "ai_tool"),
    ("sunniy intellekt bilan rasm generatsiya qilish uslub tanlash va yuklab olish", "ai_tool"),
    ("mijoz savollariga javob beradigan ai chatbot bilim bazasi asosida ishlaydi", "ai_tool"),
    ("ai kontent generator: mavzu beriladi maqola va post yozib beradi", "ai_tool"),
    ("nutqni matnga ogiruvchi va uni xulosalab beruvchi ai xizmat", "ai_tool"),
    ("сервис на базе ии: пользователь загружает документ модель отвечает на вопросы по нему", "ai_tool"),
    ("ии генератор контента: задаешь тему получаешь готовую статью", "ai_tool"),
    ("ai assistant that ingests user documents and answers questions grounded in them using rag", "ai_tool"),
    ("llm powered content generator producing articles and social posts from a topic", "ai_tool"),

    # ---------------- devtool_api: dasturchi, api kalit, sdk ---------------
    ("dasturchilar bizning api ga kalit olib murojaat qiladi sorovlar soni boyicha tolaydi", "devtool_api"),
    ("api xizmat: hujjat sdk kalit boshqaruvi va sorov limiti", "devtool_api"),
    ("dasturchilar uchun monitoring: log yigish xato kuzatuvi va ogohlantirish", "devtool_api"),
    ("sms va email yuborish uchun api dasturchilar oz ilovasiga ulaydi", "devtool_api"),
    ("integratsiya platformasi: turli servislarni webhook orqali bir biriga ulash", "devtool_api"),
    ("ci cd xizmati: git ga push qilinganda avtomatik test va deploy", "devtool_api"),
    ("api сервис: разработчики получают ключ обращаются к нам и платят за количество запросов", "devtool_api"),
    ("инструмент для разработчиков: сбор логов отслеживание ошибок и алерты", "devtool_api"),
    ("developer facing api where teams get keys, read docs, use our sdk and pay per request", "devtool_api"),
    ("observability platform collecting logs errors and alerts for engineering teams", "devtool_api"),

    # ---------------- mobile_app: store, push, offline ---------------------
    ("app store va play market ga chiqadigan ilova push bildirishnoma va offline rejim bilan", "mobile_app"),
    ("faqat mobil ilova qilmoqchiman web versiya kerak emas kamera va gps ishlatiladi", "mobile_app"),
    ("ios va android uchun bitta koddan ilova telefon kontaktlari va bildirishnoma bilan", "mobile_app"),
    ("mobil ilova: qurilma xotirasida malumot saqlansin internet bolmasa ham ishlasin", "mobile_app"),
    ("telefon ilovasi barmoq izi bilan kirish va biometrik himoya bilan", "mobile_app"),
    ("мобильное приложение для ios и android с пуш уведомлениями офлайн режимом и камерой", "mobile_app"),
    ("нативное приложение которое выходит в app store и play market с биометрией", "mobile_app"),
    ("mobile app shipped to app store and play market with push notifications and offline storage", "mobile_app"),
    ("cross platform mobile app using device camera gps contacts and biometric login", "mobile_app"),

    # ---------------- game: daraja, ochko, raqib ---------------------------
    ("oyinchi darajalarni ochadi ochko toplaydi va reyting jadvalida boshqalar bilan raqobatlashadi", "game"),
    ("real vaqtda ikki oyinchi bir biriga qarshi oynaydi natija reytingga yoziladi", "game"),
    ("viktorina oyini: savollar taymer ochko va haftalik reyting", "game"),
    ("kartochka oyini: kolodalar toplash jang va oyin ichida xarid", "game"),
    ("bolalar uchun tarbiyaviy oyin: darajalar mukofot va ota ona nazorati", "game"),
    ("игрок проходит уровни набирает очки и соревнуется в таблице лидеров", "game"),
    ("многопользовательская игра в реальном времени с матчмейкингом и рейтингом", "game"),
    ("game where players clear levels earn points and compete on a leaderboard", "game"),
    ("realtime multiplayer match with matchmaking ranked ladder and in game purchases", "game"),
]
