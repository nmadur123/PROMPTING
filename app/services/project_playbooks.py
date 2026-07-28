"""Loyiha turiga xos domen bilimlari — promptni "bir xil shablon" bo'lishdan qutqaradi.

Ilgari barcha turlar uchun bitta skelet ishlatilardi: marketplace ham, o'yin ham
bir xil "papka tuzilishi, DB sxemasi, endpoint'lar" ro'yxatini olardi. Natijada
prompt umumiy va yuzaki chiqardi — model loyihaning aynan qaysi qismlari muhimligini
bilmasdi.

Bu yerda har bir tur uchun: qaysi obyektlar (entity) kerak, qaysi stsenariylar
kritik, qaysi funksiyalar shart, qanday xatolarga yo'l qo'yiladi va yuk ostida
nima birinchi bo'lib sinadi.

TIL HAQIDA: ro'yxatlar ataylab inglizcha texnik atamalarda yozilgan (`Listing`,
`escrow`, `idempotency key`). Bular kod identifikatorlari va soha standartlari —
ular tarjima qilinmaydi, chunki modeldan aynan shu nomlar bilan kod yozish
kutiladi. Atrofdagi izohlar esa foydalanuvchi tilida (prompt_builder skeletidan).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Playbook:
    """Bitta loyiha turi uchun texnik yo'riqnoma."""

    # Ma'lumotlar modelining o'zagi — modeldan shu obyektlarni yaratish kutiladi
    entities: list[str] = field(default_factory=list)
    # Kritik foydalanuvchi stsenariylari (uchidan uchigacha)
    flows: list[str] = field(default_factory=list)
    # Shu tur uchun majburiy funksiyalar — bularsiz mahsulot to'liq emas
    must_have: list[str] = field(default_factory=list)
    # Aynan shu turda tez-tez uchraydigan texnik xatolar
    pitfalls: list[str] = field(default_factory=list)
    # Yuk ortganda birinchi bo'lib sinadigan joylar
    hot_paths: list[str] = field(default_factory=list)


PLAYBOOKS: dict[str, Playbook] = {

    # ------------------------------------------------------------------ #
    "marketplace": Playbook(
        entities=[
            "User (role: buyer | seller | admin), SellerProfile (verification status, payout details)",
            "Listing (title, price, currency, stock, status: draft|pending|active|rejected|archived)",
            "Category (nested tree), ListingAttribute (per-category dynamic fields)",
            "Order, OrderItem (price snapshot at purchase time — never join to live Listing price)",
            "Payment, Payout, CommissionRule, LedgerEntry (double-entry: every money movement is two rows)",
            "Review (per completed order only), Dispute, Message/Thread (buyer ↔ seller)",
            "ModerationQueue, AuditLog",
        ],
        flows=[
            "Seller onboarding → identity/bank verification → first listing → moderation → published",
            "Buyer search → filter/sort → listing page → cart → checkout → payment → order tracking → delivery confirmation → review",
            "Money: buyer pays → funds held (escrow) → delivery confirmed → commission deducted → seller payout",
            "Dispute: buyer opens → evidence from both sides → admin decision → refund or release",
        ],
        must_have=[
            "Listing moderation queue — the platform is liable for what sellers publish",
            "Commission calculation as a separate, auditable rule set (percentage per category, minimum fee, seller tier)",
            "Financial ledger: no balance is ever computed by summing orders on the fly; every movement is an immutable ledger row",
            "Search with filters + sorting, backed by a real index (not LIKE '%…%')",
            "Rating aggregation that cannot be inflated (one review per completed order, no self-review)",
        ],
        pitfalls=[
            "Storing money as float — use integer minor units (tiyin/cent) or DECIMAL, never float",
            "Reading the current Listing price when displaying an old order — snapshot price into OrderItem",
            "Computing seller balance with SUM() over orders on every request — that query is the first thing to die",
            "No idempotency on payment webhooks — the provider WILL retry and you WILL double-credit",
            "Treating buyer and seller as separate tables — one user is frequently both",
        ],
        hot_paths=[
            "Search / listing feed — highest read volume, must be indexed and cached",
            "Listing detail page — cache aggressively, invalidate on price/stock change",
            "Checkout + payment webhook — must stay correct under concurrency (row-level locking on stock)",
        ],
    ),

    # ------------------------------------------------------------------ #
    "ecommerce": Playbook(
        entities=[
            "Product, ProductVariant (size/colour → own SKU, own stock, own price)",
            "Inventory (per warehouse/location), StockReservation (held during checkout, expires)",
            "Cart (guest cart by cookie + user cart, merged on login), CartItem",
            "Order, OrderItem (price + tax snapshot), Shipment, ShippingMethod, Address",
            "Payment, Refund, Discount/Coupon (usage limits, per-user limits), Category, Collection",
        ],
        flows=[
            "Browse → product page (variant selection) → add to cart → checkout (address, shipping, payment) → confirmation email",
            "Stock: reserve on checkout start → decrement on payment success → release on abandonment/timeout",
            "Return/refund: request → approve → restock → refund through the original payment method",
            "Abandoned cart recovery (scheduled job, not on request path)",
        ],
        must_have=[
            "Variant-level stock and pricing — product-level stock is wrong the moment you have sizes",
            "Stock reservation with expiry, so two buyers cannot buy the last item simultaneously",
            "Guest checkout — forcing registration kills conversion",
            "Order status state machine with allowed transitions, not a free-text status field",
            "Tax and shipping calculated server-side only; never trust a client-supplied total",
        ],
        pitfalls=[
            "Oversell: decrementing stock only after payment confirmation without reservation",
            "Recalculating the cart total on the client and trusting it at checkout",
            "Deleting products instead of soft-archiving — old orders lose their product reference",
            "Coupon logic without atomic usage counting — a coupon limited to 100 uses gets used 300 times under load",
        ],
        hot_paths=[
            "Catalogue/category listing and product detail — heaviest reads, cache + CDN",
            "Cart operations — frequent writes per session, keep them cheap",
            "Checkout stock reservation — the correctness-critical concurrency point",
        ],
    ),

    # ------------------------------------------------------------------ #
    "saas_dashboard": Playbook(
        entities=[
            "Organization/Tenant, Membership (user ↔ org with role), Invitation",
            "User, Session, ApiKey (scoped to org, hashed at rest)",
            "Role & Permission (RBAC: owner | admin | member | viewer), per-resource checks",
            "Subscription, Plan, UsageRecord (metered features), Invoice",
            "AuditLog (who changed what, when, from where), Webhook, Notification",
        ],
        flows=[
            "Sign up → create organization → invite team → set roles → first meaningful action",
            "Every request: authenticate user → resolve active organization → check membership → check permission → filter data by tenant",
            "Billing: plan change → proration → invoice → payment → feature gate update",
            "Data export (CSV/PDF) for any table the user can see — run as a background job, not inline",
        ],
        must_have=[
            "Tenant isolation enforced at the data-access layer (or Postgres Row Level Security), not only in UI conditionals",
            "Every list endpoint filtered by organization_id — a missing filter is a data breach, not a bug",
            "Audit log for every mutating action",
            "Feature gating driven by the subscription plan, checked server-side",
            "Table UX: server-side pagination, sorting and filtering — never ship all rows to the browser",
        ],
        pitfalls=[
            "Separate database per tenant — migrations become unmanageable past a few dozen tenants",
            "Checking permissions only in the frontend — the API must re-check every time",
            "Loading the full dataset into the browser and filtering client-side",
            "Forgetting organization_id on one endpoint — that is the classic multi-tenant leak",
            "N+1 queries on dashboard widgets — each widget quietly issues its own per-row query",
        ],
        hot_paths=[
            "Dashboard aggregate widgets — pre-compute or cache; do not aggregate raw tables per page load",
            "Large tables with filters — needs composite indexes matching the actual filter combinations",
            "Export jobs — must be queued, otherwise one export blocks a worker for minutes",
        ],
    ),

    # ------------------------------------------------------------------ #
    "booking_service": Playbook(
        entities=[
            "Resource (room/table/specialist/vehicle), Availability rule (recurring schedule)",
            "TimeSlot (generated from rules), Booking (status: pending|confirmed|cancelled|no_show|completed)",
            "BlackoutPeriod (holidays, maintenance), Customer, Payment/Deposit",
            "Reminder/Notification, CancellationPolicy",
        ],
        flows=[
            "Pick service → pick date → see free slots → choose slot → enter details → pay deposit (optional) → confirmation",
            "Reminder before the appointment (email/SMS/Telegram), scheduled job",
            "Cancel/reschedule within policy window → slot released back to availability",
            "Staff view: today's schedule, mark completed / no-show",
        ],
        must_have=[
            "Double-booking prevention with a database-level unique constraint or exclusion constraint on (resource, time range) — application-level checks alone lose the race",
            "Timezone handling: store UTC, render in the resource's local timezone; DST transitions must not shift bookings",
            "Availability generated from rules, not hand-created rows for every day forever",
            "Cancellation policy enforced automatically (free window, partial refund, no refund)",
        ],
        pitfalls=[
            "Checking availability then inserting in two separate statements — under concurrency both requests pass the check",
            "Storing local time without a timezone — bookings break twice a year on DST",
            "Generating an infinite number of TimeSlot rows in advance instead of computing them on demand from rules",
        ],
        hot_paths=[
            "Availability calendar query — heavy read, cache per (resource, date range) and invalidate on booking",
            "Slot booking transaction — the correctness-critical concurrency point",
        ],
    ),

    # ------------------------------------------------------------------ #
    "delivery_logistics": Playbook(
        entities=[
            "Order, Delivery, Courier (status, current location, active shift), Vehicle",
            "Route, RouteStop (sequence, ETA, actual arrival), Zone (delivery area polygon + pricing)",
            "LocationPing (courier GPS, high write volume — separate table, time-partitioned)",
            "DeliveryStatusEvent (append-only status history), ProofOfDelivery (photo/signature)",
        ],
        flows=[
            "Order placed → zone + price resolved → assigned to courier → picked up → in transit → delivered → proof captured",
            "Courier app: go online → receive assignment → accept/reject → navigate → update status at each stop",
            "Customer live tracking: courier position streamed/polled while delivery is active",
            "Reassignment when a courier rejects, goes offline, or misses the SLA",
        ],
        must_have=[
            "Status history as append-only events — never overwrite a single status column and lose the timeline",
            "Geospatial queries with a real index (PostGIS / earthdistance), not distance computed in application code over all couriers",
            "Assignment algorithm as an isolated, testable service (distance + load + rating), not inline in a controller",
            "Offline tolerance in the courier client: queue status updates locally and sync when the connection returns",
        ],
        pitfalls=[
            "Writing every GPS ping into the main orders table — it will dominate write volume and bloat the hot table",
            "Computing 'nearest courier' by loading all couriers and sorting in Python",
            "Sending each tracking update over a fresh HTTP request per second per customer — use WebSocket or sane polling intervals",
            "Trusting client-supplied delivery timestamps for SLA calculation",
        ],
        hot_paths=[
            "Location ping ingestion — highest write rate in the system, keep it off the transactional path",
            "Live tracking fan-out — one courier position may be watched by many clients",
            "Courier assignment — runs on every new order, must be fast and indexed",
        ],
    ),

    # ------------------------------------------------------------------ #
    "fintech": Playbook(
        entities=[
            "Account, LedgerEntry (double-entry: debit and credit rows, immutable), Transaction",
            "Card/PaymentMethod (tokenised — never store PAN), Beneficiary, TransferRequest",
            "KycProfile (document status, verification level), AmlFlag, RiskScore",
            "Statement, FeeRule, ExchangeRate (with validity period), IdempotencyKey",
        ],
        flows=[
            "Registration → KYC document upload → verification → account activation → limits assigned by verification level",
            "Transfer: validate limits → risk check → reserve funds → execute → ledger entries → notify both sides",
            "Every money movement is a transaction with an idempotency key and a full audit trail",
            "Reconciliation job: compare internal ledger against the provider's statement, flag mismatches",
        ],
        must_have=[
            "Double-entry ledger — balances are derived from immutable entries, never stored as a mutable column that gets UPDATE-ed",
            "Idempotency keys on every money-moving endpoint; a retried request must not move money twice",
            "Amounts as integer minor units or DECIMAL — floating point is disqualifying in financial code",
            "Transaction limits per verification level, enforced server-side",
            "Full audit trail: who, what, when, from which IP/device, with the before/after state",
            "PII and documents encrypted at rest; card data tokenised through the provider, never persisted locally",
        ],
        pitfalls=[
            "A mutable `balance` column updated with `balance = balance - amount` — under concurrency this silently loses money",
            "Using float/double anywhere near an amount",
            "No idempotency — the payment provider retries webhooks by design",
            "Deleting or editing a transaction record instead of writing a compensating entry",
            "Logging full card numbers, tokens, or documents into application logs",
        ],
        hot_paths=[
            "Balance lookup — must be fast despite being derived; use a maintained materialised balance updated inside the same transaction as the ledger entries",
            "Transaction history pagination — index on (account_id, created_at DESC)",
            "Webhook ingestion — must be idempotent, fast, and never block on downstream work",
        ],
    ),

    # ------------------------------------------------------------------ #
    "edtech": Playbook(
        entities=[
            "Course, Module, Lesson (video/text/quiz), Enrollment, Progress (per user per lesson)",
            "Quiz, Question, AnswerOption, Attempt (with score and timestamps), Certificate",
            "Instructor, Assignment, Submission, Grade, Cohort/Group",
            "VideoAsset (transcoded renditions, subtitles), Discussion/Comment",
        ],
        flows=[
            "Browse catalogue → enrol (free or paid) → consume lessons in order → quiz → progress updates → certificate on completion",
            "Video: upload → background transcode into multiple qualities → HLS/DASH playback with adaptive bitrate",
            "Instructor: create course → add modules/lessons → publish → track cohort progress → grade submissions",
            "Progress is saved continuously (resume playback at the exact second the learner stopped)",
        ],
        must_have=[
            "Video served through a CDN with adaptive bitrate — never stream original files from the application server",
            "Progress tracking granular enough to resume mid-lesson, throttled so it does not write on every second of playback",
            "Quiz answers validated server-side; correct answers never sent to the client before submission",
            "Certificate with a verifiable unique code",
        ],
        pitfalls=[
            "Serving video files directly from the app server — bandwidth alone will take the server down",
            "Sending the quiz answer key to the browser as part of the question payload",
            "Writing a progress row on every playback tick — that is thousands of writes per learner per lesson",
            "No transcoding: a 4K upload played on mobile burns the learner's data and your bandwidth",
        ],
        hot_paths=[
            "Video delivery — dominant bandwidth cost, must be CDN-offloaded",
            "Progress writes — highest write frequency, batch/throttle them",
            "Course catalogue and lesson listing — heavy reads, cache",
        ],
    ),

    # ------------------------------------------------------------------ #
    "healthtech": Playbook(
        entities=[
            "Patient, Practitioner, Appointment, Encounter/Visit",
            "MedicalRecord, Observation (vitals/labs), Prescription, Diagnosis (ICD code)",
            "Consent (explicit, per data category, revocable), AccessLog (every read of patient data)",
            "Document/Attachment (encrypted), Referral",
        ],
        flows=[
            "Patient registration → consent capture → appointment → consultation → record entry → prescription → follow-up",
            "Practitioner accesses a patient record: access is checked, logged, and limited to the treatment relationship",
            "Patient views and exports their own data; patient revokes consent",
        ],
        must_have=[
            "Access logging on READ as well as write — in medical data, who looked is as important as who changed",
            "Encryption at rest for records and attachments; TLS everywhere in transit",
            "Explicit, granular, revocable consent — stored, versioned, and enforced",
            "Role-based access limited to an active treatment relationship, not 'any doctor sees any patient'",
            "Data retention and deletion policy implemented, not just documented",
        ],
        pitfalls=[
            "Treating medical records like ordinary rows — no read audit, no encryption, no consent check",
            "Putting patient identifiers in URLs, logs, or analytics events",
            "Hard-deleting records that are legally required to be retained",
            "Sending diagnoses or results through unencrypted email/SMS",
        ],
        hot_paths=[
            "Patient record retrieval — must stay fast while writing an access-log row for every read",
            "Appointment calendar — same concurrency constraints as a booking system",
        ],
    ),

    # ------------------------------------------------------------------ #
    "social_community": Playbook(
        entities=[
            "User, Profile, Follow (directed edge), Block, Mute",
            "Post, Comment (nested), Reaction, Media, Hashtag, Mention",
            "Feed entry (materialised timeline), Notification, Report (moderation), ModerationAction",
            "DirectMessage/Conversation",
        ],
        flows=[
            "Sign up → build profile → follow accounts → personalised feed → post → engagement (reaction/comment/share) → notifications",
            "Feed generation: fan-out on write for normal accounts, fan-out on read for high-follower accounts (hybrid)",
            "Moderation: user reports → queue → reviewer action → notify → appeal",
            "Real-time: notifications and DMs over WebSocket",
        ],
        must_have=[
            "A concrete feed strategy chosen and justified (push/pull/hybrid) — 'ORDER BY created_at over all posts' is not a feed",
            "Block/mute honoured everywhere: feed, search, comments, notifications, DMs",
            "Moderation queue and reporting from day one — user-generated content without moderation becomes a legal problem",
            "Notification batching (5 people liked your post → one notification, not five)",
            "Rate limiting on posting, commenting and following to stop spam and follow-bots",
        ],
        pitfalls=[
            "Building the feed by querying all posts from all followed users on every page load",
            "Counting likes with COUNT(*) on every render instead of a maintained counter",
            "Unbounded nested comments loaded recursively — cap the depth and paginate replies",
            "Fan-out on write for an account with a million followers — that write becomes a million rows",
        ],
        hot_paths=[
            "Feed read — the single highest-volume query in the product",
            "Reaction/like writes — very high frequency, keep them append-only and aggregate asynchronously",
            "Notification fan-out — batch and queue it",
            "WebSocket connections — the RAM constraint, not the CPU one",
        ],
    ),

    # ------------------------------------------------------------------ #
    "content_media": Playbook(
        entities=[
            "Article/Post, Author, Category, Tag, MediaAsset (responsive renditions)",
            "Revision (edit history), PublicationSchedule, Comment",
            "Subscription/Paywall tier, ViewEvent (analytics, high volume, separate store)",
        ],
        flows=[
            "Author drafts → editor reviews → schedules → publishes → distributed (RSS, sitemap, social cards)",
            "Reader lands from search/social → reads → hits paywall (if metered) → subscribes",
            "Media upload → automatic responsive renditions (WebP/AVIF) → served via CDN",
        ],
        must_have=[
            "Server-side rendering or static generation for public pages — this is the one product type where SEO decides survival",
            "Full metadata: canonical URL, OpenGraph, structured data, sitemap, RSS",
            "Responsive images in modern formats via CDN; never serve the original upload",
            "Editorial workflow with revisions and scheduled publishing",
        ],
        pitfalls=[
            "Client-side rendering for public content — search engines and social previews suffer, traffic never arrives",
            "Serving original-resolution photos to mobile readers",
            "Writing a page-view row into the main database on every request",
            "No cache invalidation strategy — either stale content forever or cache that never helps",
        ],
        hot_paths=[
            "Article page reads — overwhelmingly the dominant traffic; full-page cache + CDN",
            "Homepage/section listing — cache with short TTL",
            "Image bandwidth — the biggest infrastructure cost line",
        ],
    ),

    # ------------------------------------------------------------------ #
    "ai_tool": Playbook(
        entities=[
            "User, Workspace, Conversation/Session, Message (role, content, token counts)",
            "Job (async generation: queued|running|succeeded|failed, with retry count)",
            "UsageRecord (tokens/credits consumed per user per model), CreditBalance, RateLimitBucket",
            "Prompt template, ModelConfig, Document/Embedding (if RAG), Feedback",
        ],
        flows=[
            "User submits input → validate + check credits → enqueue or stream → provider call → stream tokens to client → persist result → deduct usage",
            "Long generation runs as a background job; the client polls or subscribes rather than holding an HTTP request open",
            "Failure: provider error/timeout → retry with backoff → fall back to another model → refund credits if it ultimately fails",
        ],
        must_have=[
            "Streaming responses (SSE) — a user staring at a spinner for 30 seconds assumes it is broken",
            "Per-user rate limiting and credit accounting BEFORE the provider call, not after",
            "Provider timeout, retry with exponential backoff, and a fallback model",
            "Token/cost accounting per request, stored — otherwise the provider bill is unexplainable",
            "The provider API key lives only on the server; the browser never sees it",
        ],
        pitfalls=[
            "Calling the AI provider directly from the browser and exposing the key",
            "Holding a synchronous HTTP request open for the whole generation — it dies at the proxy timeout",
            "No spend cap per user — one script can burn the entire budget overnight",
            "Retrying a non-idempotent generation and charging the user twice",
            "Storing embeddings in a plain table and scanning them linearly instead of using a vector index",
        ],
        hot_paths=[
            "Provider calls — the dominant cost and latency; cache identical requests where semantics allow",
            "Streaming connections — long-lived, RAM- and file-descriptor-bound",
            "Usage/credit accounting — written on every request, must be cheap and correct under concurrency",
        ],
    ),

    # ------------------------------------------------------------------ #
    "devtool_api": Playbook(
        entities=[
            "ApiKey (hashed, scoped, revocable, with last-used timestamp), Project, Environment",
            "RequestLog (very high volume — time-partitioned, retention policy)",
            "RateLimitBucket, Quota, Webhook + WebhookDelivery (with retry state)",
            "Plan, UsageAggregate (per hour/day, pre-computed for billing)",
        ],
        flows=[
            "Developer signs up → creates project → gets API key → first successful call within minutes",
            "Every API request: authenticate key → check rate limit → check quota → execute → log → aggregate usage",
            "Webhook delivery: attempt → on failure retry with exponential backoff → dead-letter after N attempts → let the developer replay",
        ],
        must_have=[
            "API keys stored hashed, shown once at creation, revocable, and scoped",
            "Rate limiting with the standard response headers (limit, remaining, reset) and 429 with Retry-After",
            "Versioned API with a documented deprecation policy — breaking a public API breaks your customers' production",
            "OpenAPI specification generated from the code, not maintained by hand",
            "Idempotency keys on all mutating endpoints",
            "Consistent, machine-readable error format with stable error codes",
        ],
        pitfalls=[
            "Storing API keys in plain text",
            "Writing every request log row synchronously into the primary database — logs will outgrow business data within weeks",
            "Rate limiting in application memory when running multiple instances — the limit multiplies by instance count; use Redis",
            "Breaking changes without a version bump",
            "Retrying webhooks forever with no dead-letter queue",
        ],
        hot_paths=[
            "Auth + rate-limit check — runs before every single request, must be O(1) and cached",
            "Request logging — the highest write volume in the system, keep it async and off the primary store",
            "Usage aggregation — pre-compute; never aggregate raw logs at billing time",
        ],
    ),

    # ------------------------------------------------------------------ #
    "mobile_app": Playbook(
        entities=[
            "User, Device (push token, platform, app version), Session",
            "SyncState (per device: last sync cursor), OfflineQueue (pending local mutations)",
            "PushNotification, RemoteConfig/FeatureFlag, AppVersion (minimum supported)",
        ],
        flows=[
            "Install → onboarding → sign up/in → main action → push permission requested at the moment it makes sense (not on first launch)",
            "Sync: local write → optimistic UI → queued → sent when online → server resolves conflicts → local state reconciled",
            "Force-update flow when the installed version is below the minimum supported version",
        ],
        must_have=[
            "Offline-first: the app must be usable without a connection and sync afterwards",
            "Cursor-based pagination — offset pagination breaks as data shifts between requests",
            "A defined conflict resolution rule (last-write-wins with server timestamps, or per-field merge) — decided, not accidental",
            "Payload minimisation: mobile networks make every extra kilobyte a latency cost",
            "API versioning — old app versions stay installed on users' phones for months",
        ],
        pitfalls=[
            "Designing the API as if the client were always online on fast Wi-Fi",
            "Breaking the API and stranding every user who has not updated",
            "Sending huge JSON payloads with fields the app never renders",
            "Requesting push/location permission on first launch — users decline and it is hard to recover",
            "No minimum-version check, so a three-year-old build keeps hitting removed endpoints",
        ],
        hot_paths=[
            "Sync endpoint — called by every device on every foreground, must be incremental (delta, not full state)",
            "Push fan-out — batch through the platform provider, never one HTTP call per device inline",
        ],
    ),

    # ------------------------------------------------------------------ #
    "game": Playbook(
        entities=[
            "Player, PlayerProfile, Progress/SaveState, Inventory, Item, Currency (soft/hard)",
            "Match/Session, MatchResult, Leaderboard entry, Season",
            "Purchase/Transaction (store receipt validation), Achievement, DailyReward",
        ],
        flows=[
            "Launch → authenticate (guest allowed, upgradeable to an account) → load save state → play → submit result → progress and leaderboard update",
            "Matchmaking: queue → group by rating → create match → play → validate result server-side → rank update",
            "In-app purchase: store receipt → server-side validation with Apple/Google → grant items → record transaction",
        ],
        must_have=[
            "Server-authoritative game state for anything that matters (currency, progress, leaderboard) — the client is hostile input",
            "Receipt validation server-side against the store; never grant items on the client's word",
            "Leaderboard with anti-cheat plausibility checks (impossible scores, impossible timing)",
            "Guest play with later account linking — forcing registration before the first play loses most players",
        ],
        pitfalls=[
            "Trusting the client's reported score, currency, or inventory",
            "Recomputing the full leaderboard ranking on every submission",
            "Saving state on every frame or every action instead of at checkpoints",
            "Storing the hard currency balance as a mutable field with no transaction history",
        ],
        hot_paths=[
            "Save-state writes — very frequent, batch at checkpoints",
            "Leaderboard reads and rank computation — use a sorted structure (Redis sorted set), not ORDER BY over the whole table",
            "Match result submission — validation-critical and bursty",
        ],
    ),
}


# Turi aniqlanmagan yoki ro'yxatda yo'q holat uchun — umumiy, lekin bo'sh emas.
DEFAULT_PLAYBOOK = Playbook(
    entities=[
        "User, Session, Role/Permission",
        "The core domain entity of this product and its main child entity",
        "AuditLog, Notification",
    ],
    flows=[
        "Sign up → onboarding → the single main action of the product → see the result",
        "Return visit: sign in → find previous work → continue",
    ],
    must_have=[
        "Authentication and authorisation checked on the server for every request",
        "Server-side pagination on every list endpoint",
        "Input validation at the API boundary with a typed schema",
    ],
    pitfalls=[
        "Checking permissions only in the UI",
        "Returning unbounded lists without pagination",
        "N+1 queries on list endpoints",
    ],
    hot_paths=[
        "The main list/feed endpoint — highest read volume",
        "The main write action — must stay correct under concurrency",
    ],
)


def get(project_type: str) -> Playbook:
    """Turga mos yo'riqnoma; noma'lum tur uchun umumiy variant."""
    return PLAYBOOKS.get(project_type, DEFAULT_PLAYBOOK)
