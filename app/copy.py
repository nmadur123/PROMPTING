"""Uch tildagi matn katalogi.

Nega bitta joyda: ilgari matnlar uchta servisga sochilgan va faqat o'zbekcha
edi, shuning uchun `lang=en` so'ralganda javob aralash chiqardi. Endi servis
kalit qaytaradi, matn shu yerdan tanlanadi.

Uslub qoidasi: har bir jumla bitta narsani aytadi va nima uchun shundayligini
tushuntiradi. "Yaxshi tanlov" degan gap emas, "nega aynan shu" degan javob.
Raqamlar `{}` orqali qo'yiladi — tarjimada joyi tilga qarab siljiydi.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

LANGS = ("uz", "ru", "en")
DEFAULT_LANG = "uz"

# Kalit -> {til: matn}. Matn ichida {name} ko'rinishidagi joy egallovchilar
# bo'lishi mumkin; ular `tr(...)` ga nomlangan argument sifatida beriladi.
COPY: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ #
    # Stack — qatlam nomlari
    # ------------------------------------------------------------------ #
    "cat.frontend":   {"uz": "Frontend", "ru": "Фронтенд", "en": "Frontend"},
    "cat.backend":    {"uz": "Backend", "ru": "Бэкенд", "en": "Backend"},
    "cat.database":   {"uz": "Ma'lumotlar bazasi", "ru": "База данных", "en": "Database"},
    "cat.auth":       {"uz": "Autentifikatsiya", "ru": "Аутентификация", "en": "Authentication"},
    "cat.realtime":   {"uz": "Realtime", "ru": "Realtime", "en": "Realtime"},
    "cat.cache":      {"uz": "Kesh", "ru": "Кеш", "en": "Cache"},
    "cat.search":     {"uz": "Qidiruv", "ru": "Поиск", "en": "Search"},
    "cat.ai":         {"uz": "AI qatlami", "ru": "AI-слой", "en": "AI layer"},
    "cat.queue":      {"uz": "Navbat", "ru": "Очередь", "en": "Queue"},
    "cat.storage":    {"uz": "Fayl saqlash", "ru": "Хранилище файлов", "en": "File storage"},
    "cat.payment":    {"uz": "To'lov", "ru": "Платежи", "en": "Payments"},
    "cat.monitoring": {"uz": "Monitoring", "ru": "Мониторинг", "en": "Monitoring"},
    "cat.cicd":       {"uz": "CI/CD", "ru": "CI/CD", "en": "CI/CD"},

    # ------------------------------------------------------------------ #
    # Stack — nega aynan shu
    # ------------------------------------------------------------------ #
    "why.fe.ecommerce": {
        "uz": "Mahsulot sahifalari Google'dan trafik keltiradi, shuning uchun ular oldindan "
              "yasalib qo'yiladi (ISR). Xaridor sahifani ochganda server hech narsa hisoblamaydi.",
        "ru": "Страницы товаров приносят трафик из поиска, поэтому они генерируются заранее "
              "(ISR). Когда покупатель открывает страницу, сервер ничего не считает.",
        "en": "Product pages earn their traffic from search, so they are generated ahead of time "
              "(ISR). When a buyer opens one, the server does no work at all.",
    },
    "why.fe.content": {
        "uz": "Maqola yozilgandan keyin o'zgarmaydi. Astro brauzerga deyarli JavaScript "
              "yubormaydi — sahifa telefonda ham darhol ochiladi.",
        "ru": "Статья после публикации не меняется. Astro почти не отправляет JavaScript в "
              "браузер — страница открывается мгновенно даже на телефоне.",
        "en": "An article does not change after it is written. Astro ships almost no JavaScript, "
              "so the page opens instantly even on a phone.",
    },
    "why.fe.dashboard": {
        "uz": "Panelni Google indekslamaydi, demak server render qilishning ma'nosi yo'q. "
              "shadcn/ui esa klaviatura va skrinriderga tayyor bloklar beradi.",
        "ru": "Панель не индексируется поиском, значит серверный рендер здесь бессмыслен. "
              "shadcn/ui даёт блоки, уже готовые для клавиатуры и скринридера.",
        "en": "A dashboard is not indexed, so server rendering buys nothing here. shadcn/ui "
              "gives you blocks that already work with a keyboard and a screen reader.",
    },
    "why.fe.devtool": {
        "uz": "Hujjat eng ko'p o'qiladigan sahifa va u statik bo'lishi kerak; panel esa "
              "interaktiv. Ikkisini ajratsangiz, hujjat serverga umuman tegmaydi.",
        "ru": "Документация — самая читаемая часть и должна быть статикой; панель интерактивна. "
              "Разделив их, вы убираете документацию с сервера полностью.",
        "en": "Docs are the most-read surface and should be static; the dashboard is interactive. "
              "Split them and the docs never touch your server.",
    },
    "why.fe.mobile": {
        "uz": "Bitta koddan ikkala do'kon uchun ilova chiqadi — startup uchun ikki jamoa "
              "saqlashdan arzon. Expo yangilanishni store navbatisiz yetkazadi.",
        "ru": "Один код — приложение для обоих сторов; для стартапа это дешевле двух команд. "
              "Expo доставляет обновление, не дожидаясь ревью стора.",
        "en": "One codebase ships to both stores, which is cheaper than two teams at your size. "
              "Expo pushes fixes without waiting in a store review queue.",
    },
    "why.fe.game": {
        "uz": "O'yin sikli React'dan tashqarida ishlaydi. Aks holda har kadrda qayta render "
              "bo'lib, FPS tushadi.",
        "ru": "Игровой цикл живёт вне React. Иначе каждый кадр вызывает перерисовку и FPS падает.",
        "en": "The game loop lives outside React. Otherwise every frame triggers a re-render and "
              "the frame rate drops.",
    },
    "why.fe.default": {
        "uz": "Ommaga ochiq sahifalarda SEO kerak, panelda kerak emas. Next.js ikkalasini bitta "
              "loyihada beradi, ya'ni ikkita frontend saqlamaysiz.",
        "ru": "Публичным страницам нужен SEO, панели — нет. Next.js закрывает оба случая в одном "
              "проекте, так что второй фронтенд не нужен.",
        "en": "Public pages need SEO; the dashboard does not. Next.js covers both in one project, "
              "so you are not maintaining two frontends.",
    },
    "why.be.ai": {
        "uz": "AI chaqiruvi 10-60 soniya kutadi. Async I/O shu kutishlarni bitta process'da "
              "ushlaydi — aks holda har so'rov bitta ishchini band qiladi.",
        "ru": "Вызов модели ждёт 10–60 секунд. Async I/O держит эти ожидания в одном процессе — "
              "иначе каждый запрос занимает отдельного воркера.",
        "en": "A model call waits 10–60 seconds. Async I/O holds those waits in one process; "
              "otherwise each request ties up a worker doing nothing.",
    },
    "why.be.go": {
        "uz": "Bu yukda Go bitta yadroda Python'dan bir necha barobar ko'p so'rov ko'taradi, "
              "ya'ni server hisobingiz shuncha kichrayadi.",
        "ru": "При такой нагрузке Go выдерживает на ядро в несколько раз больше запросов, чем "
              "Python, — счёт за серверы уменьшается во столько же раз.",
        "en": "At this load Go handles several times more requests per core than Python, and your "
              "server bill shrinks by the same factor.",
    },
    "why.be.node": {
        "uz": "Asosiy yuk — uzoq turadigan ulanishlar. Node event loop ularni arzon ushlaydi va "
              "frontend bilan til bir xil bo'ladi.",
        "ru": "Основная нагрузка — долгоживущие соединения. Event loop в Node держит их дёшево, "
              "и язык совпадает с фронтендом.",
        "en": "The load here is long-lived connections. Node's event loop holds them cheaply, and "
              "the language matches your frontend.",
    },
    "why.be.default": {
        "uz": "Tez yoziladi, OpenAPI hujjatini o'zi chiqaradi, Pydantic esa noto'g'ri ma'lumotni "
              "endpoint'ga kirgunicha to'xtatadi.",
        "ru": "Пишется быстро, сам отдаёт OpenAPI-документацию, а Pydantic отсекает некорректные "
              "данные ещё до входа в эндпоинт.",
        "en": "Fast to write, generates its own OpenAPI docs, and Pydantic stops bad data before "
              "it reaches your endpoint.",
    },
    "why.db.fintech": {
        "uz": "Pulda yarim bajarilgan amal bo'lmasligi kerak. PITR esa xato bo'lsa bazani "
              "istalgan daqiqaga qaytaradi.",
        "ru": "В деньгах не бывает наполовину выполненной операции. PITR возвращает базу на любую "
              "минуту, если что-то пошло не так.",
        "en": "Money has no half-finished operations. PITR rewinds the database to any minute if "
              "something goes wrong.",
    },
    "why.db.geo": {
        "uz": "Masofa bo'yicha qidiruv indeks bilan ishlashi kerak. PostGIS buni bazaning o'zida "
              "qiladi, alohida geo-servis saqlamaysiz.",
        "ru": "Поиск по расстоянию должен идти по индексу. PostGIS делает это внутри базы — "
              "отдельный гео-сервис не нужен.",
        "en": "Distance queries need an index behind them. PostGIS does that inside the database, "
              "so there is no separate geo service to run.",
    },
    "why.db.default": {
        "uz": "Startup ma'lumotining deyarli hammasi jadvalga tushadi, tushmagani esa JSONB "
              "ustuniga. Alohida NoSQL qo'shish keyin migratsiya bo'lib qaytadi.",
        "ru": "Почти все данные стартапа ложатся в таблицы, а остальное — в колонку JSONB. "
              "Отдельная NoSQL позже возвращается миграцией.",
        "en": "Nearly all startup data fits tables, and what does not fits a JSONB column. Adding "
              "a separate NoSQL store comes back later as a migration.",
    },
    "why.auth.tenant": {
        "uz": "Har bir so'rovda foydalanuvchi o'z firmasidan tashqariga chiqmasligini tekshirish "
              "kerak. Tashqi provayder buni sizning modelingiz bo'yicha qilib bermaydi.",
        "ru": "На каждом запросе нужно проверять, что пользователь не выходит за пределы своей "
              "компании. Внешний провайдер не знает вашу модель данных.",
        "en": "Every request has to check the user stays inside their own company. An external "
              "provider does not know your data model well enough to do that.",
    },
    "why.auth.phone": {
        "uz": "Bu auditoriya parol o'ylab topish bosqichida chiqib ketadi. SMS kod bilan "
              "ro'yxatdan o'tish oxirigacha yetadiganlar sezilarli ko'p.",
        "ru": "Эта аудитория отваливается на этапе придумывания пароля. С SMS-кодом до конца "
              "регистрации доходит заметно больше людей.",
        "en": "This audience drops off at the invent-a-password step. With an SMS code, noticeably "
              "more of them finish signing up.",
    },
    "why.auth.default": {
        "uz": "Bepul, hech kimga bog'lanmaydi va foydalanuvchi soniga qarab qimmatlashmaydi.",
        "ru": "Бесплатно, ни от кого не зависит и не дорожает с ростом числа пользователей.",
        "en": "Free, ties you to nobody, and does not get more expensive as users grow.",
    },
    "why.realtime": {
        "uz": "Server bir nechta nusxada ishlaganda xabar hammasiga yetib borishi kerak. "
              "Redis Pub/Sub shuni ta'minlaydi.",
        "ru": "Когда сервер работает в нескольких копиях, сообщение должно дойти до всех. "
              "Redis Pub/Sub это обеспечивает.",
        "en": "When the server runs as several copies, a message has to reach all of them. Redis "
              "Pub/Sub is what makes that happen.",
    },
    "why.cache": {
        "uz": "Bir xil javob qayta-qayta hisoblanmaydi. Katalog va sessiya keshlansa, bazaga "
              "tushadigan yuk sezilarli kamayadi.",
        "ru": "Один и тот же ответ перестаёт считаться заново. Кеш каталога и сессий заметно "
              "снижает нагрузку на базу.",
        "en": "The same answer stops being computed twice. Caching the catalogue and sessions "
              "takes a large share of load off the database.",
    },
    "why.search": {
        "uz": "Boshida alohida qidiruv serveri ortiqcha. Katalog o'sgach, Meilisearch xatoli "
              "yozuvni ham topadi va tez javob beradi.",
        "ru": "На старте отдельный поисковый сервер избыточен. Когда каталог вырастет, "
              "Meilisearch найдёт и с опечаткой, и ответит быстро.",
        "en": "A separate search server is overkill at the start. Once the catalogue grows, "
              "Meilisearch handles typos and stays fast.",
    },
    "why.ai": {
        "uz": "Bitta provayderga bog'lanib qolmaslik uchun OpenRouter. Hujjatlaringiz ustidagi "
              "qidiruvni esa o'zingizda saqlang — bu arzon va tez.",
        "ru": "OpenRouter — чтобы не зависеть от одного провайдера. Поиск по вашим документам "
              "держите у себя: это дёшево и быстро.",
        "en": "OpenRouter so you are not locked to one provider. Keep the search over your own "
              "documents in-house — it is cheap and fast.",
    },
    "why.queue": {
        "uz": "Uzoq davom etadigan ish HTTP so'rovni ushlab turmasligi kerak. Worker uni olib "
              "ketadi, foydalanuvchi natijani keyin oladi.",
        "ru": "Долгая задача не должна держать HTTP-запрос. Воркер забирает её, пользователь "
              "получает результат позже.",
        "en": "A long job should not hold an HTTP request open. A worker takes it, and the user "
              "collects the result afterwards.",
    },
    "why.storage": {
        "uz": "R2'da chiquvchi trafik bepul. Video va rasm ko'p loyihada bu oyiga yuzlab dollar "
              "farq qiladi. Fayllar hech qachon app serverida turmasin.",
        "ru": "У R2 исходящий трафик бесплатный. На проекте с видео и картинками это сотни "
              "долларов в месяц. Файлы не должны лежать на сервере приложения.",
        "en": "R2 charges nothing for egress. On a media-heavy project that is hundreds of dollars "
              "a month. Files should never sit on your app server.",
    },
    "why.monitoring": {
        "uz": "Xato foydalanuvchidan oldin sizga ko'rinishi kerak. Ikkalasining bepul rejasi "
              "shu bosqich uchun yetarli.",
        "ru": "Об ошибке вы должны узнать раньше пользователя. Бесплатных тарифов обоих сервисов "
              "на этом этапе хватает.",
        "en": "You should hear about a failure before your users do. The free tier of both covers "
              "this stage.",
    },
    "why.cicd": {
        "uz": "Kichik jamoada Kubernetes alohida odamning vaqtini yeydi. Docker Compose va bitta "
              "deploy skripti tushunarli va yetarli.",
        "ru": "В маленькой команде Kubernetes съедает время отдельного человека. Docker Compose и "
              "один скрипт деплоя понятны и достаточны.",
        "en": "In a small team Kubernetes costs you a person's time. Docker Compose and one deploy "
              "script are understandable and enough.",
    },
    "why.pay.uz": {
        "uz": "Mahalliy xaridor UzCard yoki Humo bilan to'laydi, Stripe esa bu kartalarni qabul "
              "qilmaydi. Ikkalasini qo'ysangiz qamrov deyarli to'liq bo'ladi.",
        "ru": "Местный покупатель платит UzCard или Humo, а Stripe эти карты не принимает. С "
              "обоими провайдерами покрытие получается почти полным.",
        "en": "Local buyers pay with UzCard or Humo, which Stripe does not accept. With both "
              "providers in place your coverage is close to complete.",
    },
    "why.pay.intl": {
        "uz": "Xalqaro kartalar uchun Stripe, mahalliy uchun Payme/Click. Ikkalasi bitta "
              "abstraksiya ostida bo'lsin, aks holda har biri kodga tarqalib ketadi.",
        "ru": "Stripe для международных карт, Payme/Click для местных. Держите обоих за одной "
              "абстракцией, иначе они расползутся по коду.",
        "en": "Stripe for international cards, Payme/Click for local ones. Keep both behind one "
              "abstraction or each will spread through your code.",
    },

    # ------------------------------------------------------------------ #
    # Anti-patterns — nima qilmaslik
    # ------------------------------------------------------------------ #
    "anti.k8s": {
        "uz": "Kubernetes'ni birinchi kundan olmang. Bitta server va Docker Compose bu hajmda "
              "bemalol yetadi, K8s esa alohida odamning ishiga aylanadi.",
        "ru": "Не берите Kubernetes с первого дня. Одного сервера и Docker Compose на этом объёме "
              "хватает, а K8s превращается в работу отдельного человека.",
        "en": "Do not reach for Kubernetes on day one. One server and Docker Compose carry this "
              "load, while K8s becomes somebody's full-time job.",
    },
    "anti.micro": {
        "uz": "Mikroservislarga bo'lmang. Monolit bilan boshlang — chegaralar qayerdaligi "
              "ma'lum bo'lgach ajratasiz, oldin emas.",
        "ru": "Не дробите на микросервисы. Начните с монолита: разделите тогда, когда границы "
              "станут понятны, а не заранее.",
        "en": "Do not split into microservices. Start with a monolith and separate once you know "
              "where the boundaries actually are.",
    },
    "anti.serverless": {
        "uz": "Serverless tanlamang — bu hajmda oddiy server arzonroq va xatoni topish osonroq.",
        "ru": "Не выбирайте serverless: на этом объёме обычный сервер дешевле, а отладка проще.",
        "en": "Skip serverless. At this size a plain server is cheaper and far easier to debug.",
    },
    "anti.elastic": {
        "uz": "Elasticsearch qo'shmang. PostgreSQL full-text sizning katalogingiz uchun hozircha "
              "yetarli va hech narsa talab qilmaydi.",
        "ru": "Не добавляйте Elasticsearch. PostgreSQL full-text пока покрывает ваш каталог и не "
              "требует обслуживания.",
        "en": "Do not add Elasticsearch. PostgreSQL full-text covers your catalogue for now and "
              "needs no upkeep.",
    },
    "anti.ws": {
        "uz": "WebSocket qo'shmang. Oddiy so'rov-javob yetarli bo'lsa, u faqat deploy va "
              "debug'ni murakkablashtiradi.",
        "ru": "Не добавляйте WebSocket. Если хватает обычного запроса-ответа, он лишь усложняет "
              "деплой и отладку.",
        "en": "Do not add WebSocket. If request-response is enough, it only complicates your "
              "deploy and your debugging.",
    },
    "anti.ai": {
        "uz": "Mahsulotga \"AI bor\" deb qo'shmang. Asosiy muammoni yechmaguningizcha u sizni "
              "ham, foydalanuvchini ham chalg'itadi.",
        "ru": "Не добавляйте AI ради галочки. Пока основная задача не решена, он отвлекает и вас, "
              "и пользователя.",
        "en": "Do not bolt on AI for its own sake. Until the core problem is solved it distracts "
              "you and your users alike.",
    },
    "anti.mongo": {
        "uz": "MongoDB'ni \"tezroq\" deb tanlamang. Bog'langan ma'lumotni hujjatda saqlash "
              "hisobotlar yozila boshlaganda qimmatga tushadi.",
        "ru": "Не выбирайте MongoDB «потому что быстрее». Связанные данные в документах дорого "
              "обходятся, как только дойдёт до отчётов.",
        "en": "Do not pick MongoDB because it sounds faster. Relational data in documents gets "
              "expensive the moment you start writing reports.",
    },

    # ------------------------------------------------------------------ #
    # Server — hisob izohlari
    # ------------------------------------------------------------------ #
    "srv.assume": {
        "uz": "Faraz: bitta yadro taxminan {rps} so'rov/sekund ko'taradi — bu bazaga murojaat "
              "qiladigan real endpoint uchun. Endpointlaringiz yengilroq bo'lsa, kamroq resurs "
              "ham yetadi; yuklama testi bilan aniqlang.",
        "ru": "Допущение: одно ядро выдерживает около {rps} запросов в секунду — для реального "
              "эндпоинта с обращением к базе. Если ваши эндпоинты легче, хватит и меньшего; "
              "проверьте нагрузочным тестом.",
        "en": "Assumption: one core handles about {rps} requests per second for a real endpoint "
              "that touches the database. If yours are lighter you need less — confirm with a "
              "load test.",
    },
    "srv.managed_db": {
        "uz": "Bazani alohida managed servisga chiqaring: backup va nosozlikda almashish o'zi "
              "ishlaydi, siz kechasi turmaysiz.",
        "ru": "Вынесите базу в managed-сервис: бэкапы и переключение при сбое работают сами, и вам "
              "не придётся вставать ночью.",
        "en": "Move the database to a managed service. Backups and failover run themselves, and "
              "you do not get woken up.",
    },
    "srv.cdn": {
        "uz": "Statik fayl va rasmlarni CDN orqali bering. Origin serverga tushadigan yuk ham, "
              "trafik hisobi ham keskin kamayadi.",
        "ru": "Отдавайте статику и картинки через CDN. Падает и нагрузка на origin, и счёт за "
              "трафик.",
        "en": "Serve static files and images from a CDN. It cuts both the load on your origin and "
              "the bandwidth bill.",
    },
    "srv.cache": {
        "uz": "Redis kesh qo'shing — takroriy o'qishlar bazaga bormaydi.",
        "ru": "Добавьте кеш Redis — повторные чтения перестанут доходить до базы.",
        "en": "Add a Redis cache so repeated reads stop reaching the database.",
    },
    "srv.heavy": {
        "uz": "Og'ir hisoblash aniqlandi (video, render yoki ML). Uni alohida worker serverga "
              "chiqaring, aks holda u foydalanuvchi so'rovlarini kutishga majbur qiladi.",
        "ru": "Обнаружены тяжёлые вычисления (видео, рендер или ML). Вынесите их на отдельный "
              "воркер, иначе они заставят пользовательские запросы ждать.",
        "en": "Heavy compute detected (video, rendering or ML). Put it on a separate worker or it "
              "will make user requests queue behind it.",
    },
    "srv.ai": {
        "uz": "AI chaqiruvlari uzoq davom etadi — ularni navbat va worker orqali bajaring.",
        "ru": "Вызовы модели длятся долго — выполняйте их через очередь и воркер.",
        "en": "Model calls run long — put them behind a queue and a worker.",
    },
    "srv.tier.budget": {
        "uz": "Hisoblangan {vcpu} vCPU / {ram} GB talabni aynan qoplaydi. Cho'qqi paytida zaxira "
              "kam qoladi, shuning uchun bu MVP va birinchi oylar uchun.",
        "ru": "Точно покрывает расчётные {vcpu} vCPU / {ram} ГБ. На пике запаса почти нет, так что "
              "это вариант для MVP и первых месяцев.",
        "en": "Covers the computed {vcpu} vCPU / {ram} GB exactly. There is little headroom at "
              "peak, so this is an MVP-and-first-months choice.",
    },
    "srv.tier.recommended": {
        "uz": "Taxminan 60% zaxira bilan: kutilmagan cho'qqini ham, 3-6 oylik o'sishni ham "
              "ko'taradi. Ko'pchilik loyiha shu bosqichda uzoq turadi.",
        "ru": "Примерно 60% запаса: выдержит и внезапный пик, и рост за 3–6 месяцев. Большинство "
              "проектов задерживаются здесь надолго.",
        "en": "About 60% headroom: it absorbs an unexpected peak and three to six months of "
              "growth. Most projects sit here for a long time.",
    },
    "srv.tier.same": {
        "uz": "Hisoblangan talab ({vcpu} vCPU / {ram} GB) kichik, shuning uchun eng arzon tarif "
              "ham yetarli zaxira bilan keladi. Qimmatrog'ini olishning hozircha ma'nosi yo'q.",
        "ru": "Расчётная потребность ({vcpu} vCPU / {ram} ГБ) невелика, поэтому даже самый дешёвый "
              "тариф идёт с запасом. Брать дороже пока незачем.",
        "en": "The computed requirement ({vcpu} vCPU / {ram} GB) is small, so even the cheapest "
              "plan comes with headroom. There is no reason to pay more yet.",
    },
    "srv.tier.scale": {
        "uz": "Uch barobar zaxira: trafik kutilganidan keskin oshsa ham serverni almashtirmaysiz.",
        "ru": "Тройной запас: даже если трафик резко превысит ожидания, менять сервер не придётся.",
        "en": "Three times the headroom: even if traffic overshoots badly you will not be swapping "
              "servers.",
    },
    "srv.tier.horizontal": {
        "uz": "Talab katalogdagi eng katta tarifdan ham yuqori. Bu yerda bitta server yetmaydi — "
              "load balancer va 2-3 ta app node kerak bo'ladi.",
        "ru": "Потребность выше самого крупного тарифа в каталоге. Одним сервером тут не обойтись: "
              "нужен балансировщик и 2–3 узла приложения.",
        "en": "The requirement exceeds the largest plan in the catalogue. One server will not do "
              "it — you need a load balancer and two or three app nodes.",
    },

    # ------------------------------------------------------------------ #
    # Masshtablash bosqichlari
    # ------------------------------------------------------------------ #
    "scale.1": {
        "uz": "1-bosqich: hammasi bitta serverda (app, baza, Redis). Docker Compose bilan "
              "boshqaring.",
        "ru": "Шаг 1: всё на одном сервере (приложение, база, Redis). Управляйте через Docker "
              "Compose.",
        "en": "Step 1: everything on one server (app, database, Redis), managed with Docker "
              "Compose.",
    },
    "scale.2": {
        "uz": "2-bosqich: bazani alohida chiqaring. Bu birinchi bo'g'iladigan joy.",
        "ru": "Шаг 2: вынесите базу отдельно. Это первое узкое место.",
        "en": "Step 2: move the database off the box. It is the first thing to bottleneck.",
    },
    "scale.3": {
        "uz": "3-bosqich: app'ni ikki va undan ko'p nusxaga ko'paytiring, oldiga load balancer "
              "qo'ying. App holatsiz bo'lishi shart.",
        "ru": "Шаг 3: поднимите два и более экземпляра приложения за балансировщиком. Приложение "
              "должно быть stateless.",
        "en": "Step 3: run two or more app instances behind a load balancer. The app has to be "
              "stateless for this to work.",
    },
    "scale.4": {
        "uz": "4-bosqich: o'qishni read-replica'ga, og'ir ishlarni navbat worker'iga ajrating.",
        "ru": "Шаг 4: чтение — на read-replica, тяжёлые задачи — в воркер очереди.",
        "en": "Step 4: send reads to a read replica and heavy jobs to a queue worker.",
    },
    "scale.cdn": {
        "uz": "Rasm va videoni origin'dan bermang: object storage va CDN trafik xarajatini bir "
              "necha barobar tushiradi.",
        "ru": "Не отдавайте картинки и видео с origin: object storage и CDN снижают расходы на "
              "трафик в разы.",
        "en": "Do not serve images and video from the origin: object storage plus a CDN cuts the "
              "bandwidth cost several times over.",
    },
    "scale.ws": {
        "uz": "Taxminan {n} ta bir vaqtdagi ulanish kutilyapti. WebSocket'ni alohida servisga "
              "ajrating, u asosiy app'ni bloklamasin.",
        "ru": "Ожидается около {n} одновременных соединений. Вынесите WebSocket в отдельный "
              "сервис, чтобы он не блокировал основное приложение.",
        "en": "Around {n} concurrent connections are expected. Put WebSocket in its own service so "
              "it cannot block the main app.",
    },
    "scale.spiky": {
        "uz": "Bu turdagi mahsulotda cho'qqi kunlik o'rtachadan bir necha barobar yuqori bo'ladi. "
              "Avtomatik masshtablash qo'ying yoki zaxira qoldiring.",
        "ru": "У продуктов этого типа пик в разы выше среднесуточного. Настройте автоскейлинг или "
              "оставьте запас.",
        "en": "Products of this shape peak several times above their daily average. Add autoscaling "
              "or leave headroom.",
    },
    "scale.db": {
        "uz": "Baza taxminan {gb} GB gacha o'sadi. Indekslar va partitioning strategiyasini "
              "oldindan rejalang — keyin qilish qimmatroq.",
        "ru": "База вырастет примерно до {gb} ГБ. Продумайте индексы и партиционирование заранее — "
              "потом это дороже.",
        "en": "The database will grow to roughly {gb} GB. Plan indexes and partitioning now; doing "
              "it later costs more.",
    },

    # ------------------------------------------------------------------ #
    # UI/UX — render strategiyasi
    # ------------------------------------------------------------------ #
    "ui.ecommerce.rule": {
        "uz": "Mahsulot va katalog sahifalari — ISR (60-300s yangilanish). Savat va to'lov — client.",
        "ru": "Страницы товаров и каталога — ISR (обновление 60–300 с). Корзина и оплата — клиент.",
        "en": "Product and catalogue pages on ISR (revalidate 60–300s). Cart and checkout on the client.",
    },
    "ui.ecommerce.why": {
        "uz": "Katalog har so'rovda bazaga bormaydi: minglab tashrif bitta so'rovga aylanadi.",
        "ru": "Каталог перестаёт ходить в базу на каждый запрос: тысячи визитов сводятся к одному.",
        "en": "The catalogue stops hitting the database per request: thousands of visits collapse into one.",
    },
    "ui.content.rule": {
        "uz": "Barcha kontent sahifalari — SSG yoki ISR. Faqat izoh va reaksiya qismi client.",
        "ru": "Все контентные страницы — SSG или ISR. Клиентскими остаются только комментарии и реакции.",
        "en": "All content pages on SSG or ISR. Only comments and reactions stay client-side.",
    },
    "ui.content.why": {
        "uz": "Maqola o'zgarmaydi, demak uni har safar qayta yasash sof isrof. CDN'dan beriladi.",
        "ru": "Статья не меняется, поэтому пересобирать её каждый раз — чистая трата. Отдаётся с CDN.",
        "en": "An article does not change, so rebuilding it every time is pure waste. It comes off the CDN.",
    },
    "ui.marketplace.rule": {
        "uz": "E'lon sahifasi — ISR; ro'yxat va filtr — server tomonda sahifalash bilan.",
        "ru": "Страница объявления — ISR; список и фильтры — с серверной пагинацией.",
        "en": "Listing pages on ISR; the index and filters with server-side pagination.",
    },
    "ui.marketplace.why": {
        "uz": "E'lonlar qidiruvdan trafik keltiradi. Filtrni client'da qilsangiz, butun bazani "
              "brauzerga yuborishga majbur bo'lasiz.",
        "ru": "Объявления приносят трафик из поиска. Фильтрация на клиенте заставит отправить в "
              "браузер всю базу.",
        "en": "Listings earn search traffic. Filtering on the client forces you to ship the whole "
              "table to the browser.",
    },
    "ui.dashboard.rule": {
        "uz": "SSR shart emas: client render va so'rov keshi. Faqat kirish sahifasi server render.",
        "ru": "SSR не нужен: клиентский рендер и кеш запросов. Серверным остаётся только вход.",
        "en": "No SSR needed: client rendering plus a query cache. Only the login page renders on the server.",
    },
    "ui.dashboard.why": {
        "uz": "Panel qidiruv tizimiga kerak emas, SSR bu yerda faqat server yukini oshiradi.",
        "ru": "Панель не нужна поисковику, и SSR здесь только добавляет нагрузку на сервер.",
        "en": "A dashboard is not for search engines, and SSR here only adds server load.",
    },
    "ui.booking.rule": {
        "uz": "Landing va xizmat sahifalari — statik. Bo'sh vaqt jadvali — client, qisqa keshli API.",
        "ru": "Лендинг и страницы услуг — статика. Расписание свободных слотов — клиент, API с "
              "коротким кешем.",
        "en": "Landing and service pages static. The availability grid on the client, behind a "
              "short-cached API.",
    },
    "ui.booking.why": {
        "uz": "Bo'sh vaqtlar tez o'zgaradi, qolgan hammasi o'zgarmaydi. Ikkisini aralashtirmang.",
        "ru": "Слоты меняются часто, всё остальное — нет. Не смешивайте эти два режима.",
        "en": "Slots change constantly; everything else does not. Do not mix the two.",
    },
    "ui.delivery.rule": {
        "uz": "Mijoz ilovasi — client render; xarita va joylashuv WebSocket orqali.",
        "ru": "Клиентское приложение — клиентский рендер; карта и геопозиция — через WebSocket.",
        "en": "The customer app renders on the client; map and location come over WebSocket.",
    },
    "ui.delivery.why": {
        "uz": "Joylashuv har soniyada yangilanadi. HTTP polling bu yerda serverni behuda yeydi.",
        "ru": "Позиция обновляется каждую секунду. HTTP-поллинг здесь впустую съедает сервер.",
        "en": "Position updates every second. HTTP polling here burns the server for nothing.",
    },
    "ui.social.rule": {
        "uz": "Lenta — client render va kursorli sahifalash. Profil sahifasi — ISR.",
        "ru": "Лента — клиентский рендер и курсорная пагинация. Страница профиля — ISR.",
        "en": "The feed on the client with cursor pagination. Profile pages on ISR.",
    },
    "ui.social.why": {
        "uz": "Offset sahifalash katta jadvalda asta-sekin sekinlashadi, kursor esa doim bir xil tez.",
        "ru": "Offset-пагинация на большой таблице постепенно замедляется, курсорная — всегда "
              "одинаково быстрая.",
        "en": "Offset pagination degrades as the table grows; cursor pagination stays constant.",
    },
    "ui.edtech.rule": {
        "uz": "Kurs va dars sahifalari — ISR. Video — HLS orqali CDN'dan.",
        "ru": "Страницы курсов и уроков — ISR. Видео — через HLS с CDN.",
        "en": "Course and lesson pages on ISR. Video over HLS from a CDN.",
    },
    "ui.edtech.why": {
        "uz": "Videoni o'z serveringizdan bersangiz, trafik hisobi qolgan hamma narsadan oshib ketadi.",
        "ru": "Если раздавать видео со своего сервера, счёт за трафик превысит все остальные статьи.",
        "en": "Serve video from your own box and the bandwidth line will dwarf everything else.",
    },
    "ui.ai.rule": {
        "uz": "Client render, natijani oqim bilan bo'lak-bo'lak ko'rsating.",
        "ru": "Клиентский рендер, результат показывайте потоком, по частям.",
        "en": "Client rendering, with the result streamed in as it arrives.",
    },
    "ui.ai.why": {
        "uz": "Foydalanuvchi 30 soniya bo'sh ekranga qaramaydi. Birinchi belgi bir soniyada chiqsin.",
        "ru": "Пользователь не будет 30 секунд смотреть в пустой экран. Первый символ должен "
              "появиться за секунду.",
        "en": "Nobody watches a blank screen for 30 seconds. The first character should land within a second.",
    },
    "ui.devtool.rule": {
        "uz": "Hujjatlar — to'liq statik. Panel — client render.",
        "ru": "Документация — полностью статика. Панель — клиентский рендер.",
        "en": "Docs fully static. The dashboard on the client.",
    },
    "ui.devtool.why": {
        "uz": "Hujjat eng ko'p o'qiladigan sahifa. Statik bo'lsa, server unda umuman ishtirok etmaydi.",
        "ru": "Документация читается чаще всего. Если она статична, сервер в ней не участвует вовсе.",
        "en": "Docs are the most-read pages. Static means the server never takes part.",
    },
    "ui.fintech.rule": {
        "uz": "Pul bilan bog'liq har bir ekran server render; client'da hech qanday hisob-kitob yo'q.",
        "ru": "Каждый экран, связанный с деньгами, — серверный рендер; на клиенте никаких расчётов.",
        "en": "Every money-related screen renders on the server; no arithmetic on the client.",
    },
    "ui.fintech.why": {
        "uz": "Client hisoblagan summani o'zgartirish mumkin. Yagona haqiqat manbai server bo'lishi shart.",
        "ru": "Сумму, посчитанную на клиенте, можно подменить. Единственным источником истины должен "
              "быть сервер.",
        "en": "A total computed on the client can be tampered with. The server has to be the only "
              "source of truth.",
    },
    "ui.health.rule": {
        "uz": "Bemor ma'lumoti faqat autentifikatsiyadan keyin, keshsiz.",
        "ru": "Данные пациента — только после аутентификации и без кеширования.",
        "en": "Patient data only after authentication, and never cached.",
    },
    "ui.health.why": {
        "uz": "Tibbiy ma'lumot CDN yoki brauzer keshida qolib ketmasligi kerak.",
        "ru": "Медицинские данные не должны оседать в кеше CDN или браузера.",
        "en": "Medical data must not linger in a CDN or browser cache.",
    },
    "ui.mobile.rule": {
        "uz": "Ro'yxatlarda virtualizatsiya, rasmlar uchun kesh.",
        "ru": "В списках — виртуализация, для изображений — кеш.",
        "en": "Virtualise long lists and cache images.",
    },
    "ui.mobile.why": {
        "uz": "Mobil qurilmada xotira chegarasi qattiq: virtualizatsiyasiz ilova yopilib ketadi.",
        "ru": "На мобильном жёсткий лимит памяти: без виртуализации приложение просто закроется.",
        "en": "Mobile memory limits are hard: without virtualisation the OS kills your app.",
    },
    "ui.game.rule": {
        "uz": "O'yin sikli canvas'da, React'dan tashqarida. UI qatlami alohida.",
        "ru": "Игровой цикл — на canvas, вне React. Слой UI — отдельно.",
        "en": "The game loop on canvas, outside React. The UI layer stays separate.",
    },
    "ui.game.why": {
        "uz": "React har kadrda qayta render qilsa, kadrlar soni tushadi.",
        "ru": "Если React перерисовывается каждый кадр, частота кадров падает.",
        "en": "If React re-renders every frame, the frame rate drops.",
    },
    "ui.default.rule": {
        "uz": "Ommaga ochiq sahifalar — ISR, foydalanuvchi paneli — client render.",
        "ru": "Публичные страницы — ISR, пользовательская панель — клиентский рендер.",
        "en": "Public pages on ISR, the user dashboard on the client.",
    },
    "ui.default.why": {
        "uz": "SEO kerak joyda statik, kerak bo'lmagan joyda server yuki nolga tushadi.",
        "ru": "Там, где нужен SEO, — статика; там, где не нужен, нагрузка на сервер падает до нуля.",
        "en": "Static where SEO matters, and zero server load where it does not.",
    },

    # ------------------------------------------------------------------ #
    # To'lov tavsiyalari
    # ------------------------------------------------------------------ #
    "pay.marketplace.provider": {
        "uz": "Payme + Click, bo'lib to'lash (split) rejimi bilan",
        "ru": "Payme + Click с режимом сплит-платежей",
        "en": "Payme + Click with split payouts",
    },
    "pay.marketplace.why": {
        "uz": "Marketplace'da pul sotuvchiga bo'lib o'tkaziladi. Buni merchant shartnomasida "
              "alohida so'rash kerak — keyin qo'shib bo'lmaydi.",
        "ru": "В маркетплейсе деньги расходятся продавцам долями. Это нужно отдельно запросить в "
              "договоре с провайдером — потом не добавить.",
        "en": "A marketplace splits money out to sellers. Ask for it in the merchant contract up "
              "front; it cannot be bolted on later.",
    },
    "pay.marketplace.integration": {
        "uz": "Bitta to'lov abstraksiyasi, ikkita adapter. Har buyurtma uchun holat mashinasi: "
              "yaratildi → ushlab turildi → chiqarildi yoki qaytarildi.",
        "ru": "Одна абстракция платежей, два адаптера. На каждый заказ — конечный автомат: создан "
              "→ удержан → выплачен или возвращён.",
        "en": "One payment abstraction, two adapters. A state machine per order: created → held → "
              "released or refunded.",
    },
    "pay.marketplace.caveat": {
        "uz": "Komissiya odatda 1-3%. Split uchun sotuvchilar ham identifikatsiyadan o'tishi shart.",
        "ru": "Комиссия обычно 1–3%. Для сплита продавцы тоже проходят идентификацию.",
        "en": "Fees usually run 1–3%. For splits your sellers must be identity-verified too.",
    },
    "pay.fintech.provider": {
        "uz": "Bank yoki protsessing bilan to'g'ridan-to'g'ri shartnoma + Payme/Click",
        "ru": "Прямой договор с банком или процессингом + Payme/Click",
        "en": "A direct bank or processor contract, plus Payme/Click",
    },
    "pay.fintech.why": {
        "uz": "Pul saqlash va o'tkazish litsenziya talab qiladi. Boshlash uchun mavjud "
              "protsessing ustida ishlang.",
        "ru": "Хранение и перевод денег требуют лицензии. Для старта работайте поверх "
              "существующего процессинга.",
        "en": "Holding and moving money needs a licence. To start, build on top of an existing "
              "processor.",
    },
    "pay.fintech.integration": {
        "uz": "Karta ma'lumoti sizning serveringizga umuman tushmasin — u faqat provayder "
              "sahifasida kiritilsin. Bu PCI DSS hajmini keskin kamaytiradi.",
        "ru": "Данные карты не должны попадать на ваш сервер — только на страницу провайдера. Это "
              "резко сокращает объём требований PCI DSS.",
        "en": "Card details must never reach your server — only the provider's page. That shrinks "
              "your PCI DSS scope dramatically.",
    },
    "pay.fintech.caveat": {
        "uz": "Markaziy bank talablarini oldindan tekshiring. Bu texnik emas, huquqiy to'siq.",
        "ru": "Заранее проверьте требования центрального банка. Это не техническое, а юридическое "
              "препятствие.",
        "en": "Check the central bank's requirements first. This is a legal blocker, not a "
              "technical one.",
    },
    "pay.local.provider": {"uz": "Payme + Click", "ru": "Payme + Click", "en": "Payme + Click"},
    "pay.local.integration": {
        "uz": "Merchant kabinetida ro'yxatdan o'ting, webhook URL bering va har to'lovni "
              "`order_id` bo'yicha idempotent qiling.",
        "ru": "Зарегистрируйтесь в кабинете мерчанта, укажите webhook URL и сделайте каждый платёж "
              "идемпотентным по `order_id`.",
        "en": "Register in the merchant console, give them a webhook URL, and make every payment "
              "idempotent on `order_id`.",
    },
    "pay.local.caveat": {
        "uz": "Webhook takroriy kelishi mumkin. Idempotentlik kaliti bo'lmasa, bitta to'lovni "
              "ikki marta hisoblab yuborasiz.",
        "ru": "Webhook может прийти повторно. Без ключа идемпотентности вы засчитаете один платёж "
              "дважды.",
        "en": "Webhooks can arrive twice. Without an idempotency key you will credit the same "
              "payment twice.",
    },
    "pay.intl.provider": {
        "uz": "Stripe (xalqaro) + Payme/Click (mahalliy)",
        "ru": "Stripe (международные) + Payme/Click (локальные)",
        "en": "Stripe (international) + Payme/Click (local)",
    },
    "pay.intl.integration": {
        "uz": "Stripe Checkout va `checkout.session.completed` webhook'i; mahalliy provayderlar "
              "uchun alohida adapter.",
        "ru": "Stripe Checkout и webhook `checkout.session.completed`; для локальных провайдеров — "
              "отдельный адаптер.",
        "en": "Stripe Checkout with the `checkout.session.completed` webhook; a separate adapter "
              "for the local providers.",
    },
    "pay.intl.caveat": {
        "uz": "Stripe O'zbekistonda to'g'ridan-to'g'ri hisob ochmaydi — chet el yuridik shaxsi "
              "kerak bo'lishi mumkin.",
        "ru": "Stripe не открывает счёт напрямую в Узбекистане — может потребоваться иностранное "
              "юрлицо.",
        "en": "Stripe does not open accounts directly in Uzbekistan — you may need a foreign "
              "entity.",
    },

    # ------------------------------------------------------------------ #
    # Aniqlashtirish so'rovi
    # ------------------------------------------------------------------ #
    "clarify": {
        "uz": "G'oyani biroz aniqroq yozing: kim foydalanadi, qanday muammoni yechadi va asosiy "
              "amal nima? Masalan: \"Toshkentdagi sartaroshxonalar uchun onlayn navbat — mijoz "
              "vaqt tanlaydi, usta tasdiqlaydi, SMS eslatma keladi.\"",
        "ru": "Опишите идею чуть конкретнее: кто пользуется, какую проблему решает и какое "
              "основное действие? Например: «Онлайн-запись для барбершопов — клиент выбирает "
              "время, мастер подтверждает, приходит SMS-напоминание».",
        "en": "Describe the idea a little more concretely: who uses it, what problem it solves, "
              "and what the main action is. For example: \"Online booking for barbershops — the "
              "client picks a slot, the barber confirms, an SMS reminder is sent.\"",
    },
}


def tr(key: str, lang: str = DEFAULT_LANG, **params: object) -> str:
    """Kalitni tanlangan tilga o'giradi.

    Kalit topilmasa, matn o'rniga kalitning o'zi qaytadi va log'ga yoziladi —
    jimgina bo'sh satr qaytarish xatoni yashiradi. Tarjima yo'q bo'lsa
    o'zbekchaga tushadi, chunki katalog o'zbekchadan boshlab to'ldirilgan.
    """
    entry = COPY.get(key)
    if entry is None:
        logger.warning("Tarjima kaliti topilmadi: %s", key)
        return key
    text = entry.get(lang) or entry.get(DEFAULT_LANG) or key
    if params:
        try:
            return text.format(**params)
        except (KeyError, IndexError):
            logger.warning("Tarjimada joy egallovchi mos kelmadi: %s (%s)", key, lang)
    return text


def missing_translations() -> dict[str, list[str]]:
    """Qaysi kalitda qaysi til yo'qligini qaytaradi — test uchun."""
    gaps: dict[str, list[str]] = {}
    for key, entry in COPY.items():
        absent = [lang for lang in LANGS if not entry.get(lang)]
        if absent:
            gaps[key] = absent
    return gaps
