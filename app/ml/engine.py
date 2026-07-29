"""ML yadro: loyiha turi klassifikatori + hujjatlar bo'yicha RAG qidiruv.

Ikkalasi ham scikit-learn ustida — tashqi API ham, embedding servisi ham
kerak emas. Model artifaktlari `data/artifacts/` ga joylanadi; artifakt
topilmasa, yadro ishga tushganda o'zi o'qitadi (birinchi start ~1-2 soniya).
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from app.config import ARTIFACTS_DIR
from app.ml.corpus import Chunk, load_chunks
from app.ml.training_data import PROJECT_LABELS, PROJECT_TRAINING

logger = logging.getLogger(__name__)

_CLASSIFIER_PATH = ARTIFACTS_DIR / "project_classifier.joblib"
_RETRIEVER_PATH = ARTIFACTS_DIR / "doc_retriever.joblib"

# Ishonch shu chegaradan past bo'lsa, tur "aniqlanmadi" deb qaraladi va
# foydalanuvchidan aniqlashtirish so'raladi.
CONFIDENCE_FLOOR = 0.20

_APOSTROPHES = "'‘’ʻʼ`´"


def normalize(text: str) -> str:
    """Solishtirish uchun matnni bir shaklga keltiradi.

    Tutuq belgilari olib tashlanadi: dataset tutuqsiz yozilgan, foydalanuvchi
    esa "o'zbek", "qo'shish" deb yozadi — aks holda mos kelmay qoladi.
    """
    t = (text or "").lower()
    for ch in _APOSTROPHES:
        t = t.replace(ch, "")
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------- #
# 1) Loyiha turi klassifikatori
# --------------------------------------------------------------------------- #


@dataclass
class ProjectPrediction:
    label: str
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    def title(self, lang: str = "uz") -> str:
        return PROJECT_LABELS.get(self.label, {}).get(lang, self.label)


class ProjectClassifier:
    """TF-IDF (char + word n-gram) + LogisticRegression, 14 ta loyiha turi."""

    def __init__(self, vectorizer: FeatureUnion, model: LogisticRegression) -> None:
        self.vectorizer = vectorizer
        self.model = model

    @staticmethod
    def build_pipeline() -> tuple[FeatureUnion, LogisticRegression]:
        """O'qitilmagan vektorizator va model — YAGONA manba.

        Baholash skripti ham shu yerdan oladi. Ilgari konfiguratsiya har bir
        skriptda qo'lda takrorlanardi: birini tuzatib ikkinchisini unutish
        kifoya edi va o'lchov production'dagi modeldan boshqa narsani
        o'lchay boshlardi — buni sezish deyarli imkonsiz.
        """
        # char_wb + word birlashmasi: char n-gram til aralashganda (uz+ru+en) va
        # o'zbek morfologiyasida ishonchli, word n-gram esa "komissiya",
        # "sertifikat" kabi sinfni aniq ajratuvchi atamalarni ushlaydi.
        # Konfiguratsiya scripts/tune_classifier.py bilan tanlangan.
        vectorizer = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ])
        # LinearSVC bir oz aniqroq, lekin ehtimollik bermaydi — ishonch darajasi
        # va muqobil variantlar UI uchun kerak, shuning uchun LogisticRegression.
        model = LogisticRegression(max_iter=4000, C=30.0)
        return vectorizer, model

    @classmethod
    def train(cls) -> "ProjectClassifier":
        texts = [normalize(t) for t, _ in PROJECT_TRAINING]
        labels = [l for _, l in PROJECT_TRAINING]
        vectorizer, model = cls.build_pipeline()
        X = vectorizer.fit_transform(texts)
        model.fit(X, labels)
        return cls(vectorizer, model)

    def predict(self, text: str, top_k: int = 3) -> ProjectPrediction:
        vec = self.vectorizer.transform([normalize(text)])
        proba = self.model.predict_proba(vec)[0]
        order = np.argsort(proba)[::-1]
        classes = self.model.classes_
        ranked = [(str(classes[i]), float(proba[i])) for i in order[:top_k]]
        best_label, best_conf = ranked[0]
        return ProjectPrediction(
            label=best_label,
            confidence=best_conf,
            alternatives=ranked[1:],
        )


# --------------------------------------------------------------------------- #
# 2) Hujjatlar bo'yicha RAG qidiruv
# --------------------------------------------------------------------------- #


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class DocRetriever:
    """Prompt-engineering korpusi ustida TF-IDF + cosine similarity qidiruv."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix, chunks: list[Chunk]) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: Optional[list[Chunk]] = None) -> "DocRetriever":
        chunks = chunks if chunks is not None else load_chunks()
        if not chunks:
            raise RuntimeError(
                "Korpus bo'sh. Avval `scripts/fetch_corpus.ps1` ni ishga tushiring."
            )
        corpus = [normalize(c.searchable) for c in chunks]
        vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"
        )
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer, matrix, chunks)

    def search(
        self,
        query: str,
        top_k: int = 6,
        family: Optional[str] = None,
        model: Optional[str] = None,
        min_score: float = 0.02,
        family_only: bool = False,
    ) -> list[RetrievedChunk]:
        """So'rovga eng mos bo'laklarni qaytaradi.

        `family` / `model` berilsa, o'sha oilaga tegishli bo'laklar oldinga
        suriladi. `family_only=True` bo'lsa esa faqat o'sha oila qoladi —
        bu kvota uchun kerak: Claude korpusi kattaroq bo'lgani uchun oddiy
        bonus bilan OpenAI hujjatlari umuman chiqmay qolishi mumkin.
        """
        q = self.vectorizer.transform([normalize(query)])
        scores = cosine_similarity(q, self.matrix).flatten()

        if family_only and family:
            # Boshqa oilani butunlay chetlatamiz (skorni nolga tushirmaymiz —
            # min_score filtri ishlashi uchun manfiy qiymat beramiz).
            for i, chunk in enumerate(self.chunks):
                if chunk.family != family:
                    scores[i] = -1.0
        elif family or model:
            # Model-maxsus hujjatga bonus: aynan tanlangan model uchun yozilgan
            # ko'rsatma umumiy qo'llanmadan muhimroq.
            boosted = scores.copy()
            for i, chunk in enumerate(self.chunks):
                if model and chunk.model == model:
                    boosted[i] *= 1.9
                elif family and chunk.family == family:
                    boosted[i] *= 1.25
            scores = boosted

        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]))
            for i in order
            if scores[i] > min_score
        ]


# --------------------------------------------------------------------------- #
# 3) Qoidaviy signal ajratish — klassifikator ushlamaydigan texnik talablar
# --------------------------------------------------------------------------- #

# (signal nomi, kalit so'zlar). Uch tilda — foydalanuvchi aralash yozadi.
_SIGNAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "realtime": (
        "real vaqt", "realtime", "real time", "jonli", "live", "chat", "xabar almashish",
        "websocket", "socket", "onlayn kuzatish", "в реальном времени", "чат", "трекинг",
        "notification", "bildirishnoma", "push",
    ),
    "media_heavy": (
        "video", "striming", "streaming", "audio", "podkast", "rasm", "surat", "foto",
        "photo", "image", "galereya", "видео", "фото", "аудио", "stream",
    ),
    "geo": (
        "xarita", "map", "gps", "lokatsiya", "geolokatsiya", "manzil", "marshrut",
        "карта", "геолокация", "маршрут", "location", "tracking",
    ),
    "payments": (
        "tolov", "tolov tizimi", "payme", "click", "uzum", "stripe", "karta", "pul",
        "toladi", "obuna", "subscription", "payment", "checkout", "оплата", "платеж",
        "подписка", "billing", "komissiya", "commission",
    ),
    "ai": (
        "ai", "suniy intellekt", "sun'iy intellekt", "yordamchi", "chatbot", "bot",
        "llm", "gpt", "claude", "gemini", "neyron", "ии", "искусственный интеллект",
        "нейросеть", "generatsiya", "generate", "rag",
    ),
    "multi_tenant": (
        "kompaniyalar uchun", "firmalar uchun", "b2b", "saas", "tashkilot", "korxona",
        "har bir firma", "multi tenant", "для компаний", "организаций", "tenant",
    ),
    "offline": ("offline", "internetsiz", "oflayn", "оффлайн", "без интернета"),
    "search_heavy": (
        "qidiruv", "filtr", "search", "filter", "katalog", "поиск", "фильтр", "каталог",
    ),
    "seo": (
        "seo", "google", "qidiruv tizimi", "landing", "blog", "maqola", "sayt",
        "продвижение", "трафик", "organic",
    ),
    "mobile": (
        "mobil", "ilova", "android", "ios", "flutter", "react native", "telefon",
        "мобильное", "приложение", "app store", "play market",
    ),
    "heavy_compute": (
        "video montaj", "render", "konvertatsiya", "convert", "transcode", "ml model",
        "training", "obrabotka", "обработка видео", "рендер",
    ),
    "compliance": (
        "shaxsiy malumot", "persondata", "gdpr", "tibbiy", "bemor", "medical", "hipaa",
        "bank", "litsenziya", "персональные данные", "медицинск",
    ),
}

_SCALE_PATTERNS = (
    # "10 000 foydalanuvchi", "1000 users", "5к пользователей"
    re.compile(r"(\d[\d\s.,]*)\s*(?:mln|million|млн)\s*(?:ta\s*)?(?:foydalanuvchi|user|users|пользоват)", re.I),
    re.compile(r"(\d[\d\s.,]*)\s*(?:ming|k|тыс)\s*(?:ta\s*)?(?:foydalanuvchi|user|users|пользоват)", re.I),
    re.compile(r"(\d[\d\s.,]*)\s*(?:ta\s*)?(?:foydalanuvchi|user|users|пользоват)", re.I),
)


def extract_signals(text: str) -> dict[str, bool]:
    """Matndan texnik talab signallarini qoidalar bilan ajratadi."""
    t = normalize(text)
    return {name: any(kw in t for kw in kws) for name, kws in _SIGNAL_KEYWORDS.items()}


def extract_expected_users(text: str) -> Optional[int]:
    """Matnda aytilgan kutilayotgan foydalanuvchi sonini topadi (oylik).

    Topilmasa None — bunda kalkulyator so'rovnomadagi qiymatga tayanadi.
    """
    t = normalize(text)
    for i, pattern in enumerate(_SCALE_PATTERNS):
        m = pattern.search(t)
        if not m:
            continue
        raw = m.group(1).replace(" ", "").replace(",", "").replace(".", "")
        if not raw.isdigit():
            continue
        n = int(raw)
        if i == 0:
            n *= 1_000_000
        elif i == 1:
            n *= 1_000
        return n
    return None


# --------------------------------------------------------------------------- #
# 4) Yagona yadro (lazy singleton)
# --------------------------------------------------------------------------- #


class MLEngine:
    """Klassifikator va retriever'ni bir joyda ushlaydi, thread-safe yuklaydi."""

    def __init__(self) -> None:
        self._classifier: Optional[ProjectClassifier] = None
        self._retriever: Optional[DocRetriever] = None
        self._lock = threading.Lock()

    # -- yuklash / o'qitish --

    def _load_or_train_classifier(self) -> ProjectClassifier:
        if _CLASSIFIER_PATH.exists():
            try:
                data = joblib.load(_CLASSIFIER_PATH)
                return ProjectClassifier(data["vectorizer"], data["model"])
            except Exception:  # noqa: BLE001 — buzilgan artifakt qayta o'qitiladi
                logger.warning("Klassifikator artifakti o'qilmadi, qayta o'qitilyapti", exc_info=True)
        clf = ProjectClassifier.train()
        self.save_classifier(clf)
        return clf

    def _load_or_build_retriever(self) -> DocRetriever:
        if _RETRIEVER_PATH.exists():
            try:
                data = joblib.load(_RETRIEVER_PATH)
                return DocRetriever(data["vectorizer"], data["matrix"], data["chunks"])
            except Exception:  # noqa: BLE001
                logger.warning("Retriever artifakti o'qilmadi, qayta qurilyapti", exc_info=True)
        retriever = DocRetriever.build()
        self.save_retriever(retriever)
        return retriever

    @staticmethod
    def save_classifier(clf: ProjectClassifier, path: Path = _CLASSIFIER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": clf.vectorizer, "model": clf.model}, path)

    @staticmethod
    def save_retriever(retriever: DocRetriever, path: Path = _RETRIEVER_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": retriever.vectorizer, "matrix": retriever.matrix, "chunks": retriever.chunks},
            path,
        )

    # -- foydalanish --

    @property
    def classifier(self) -> ProjectClassifier:
        if self._classifier is None:
            with self._lock:
                if self._classifier is None:
                    self._classifier = self._load_or_train_classifier()
        return self._classifier

    @property
    def retriever(self) -> DocRetriever:
        if self._retriever is None:
            with self._lock:
                if self._retriever is None:
                    self._retriever = self._load_or_build_retriever()
        return self._retriever

    def warmup(self) -> None:
        """Startda chaqiriladi — birinchi so'rov sekin bo'lmasin."""
        _ = self.classifier
        _ = self.retriever

    def stats(self) -> dict:
        return {
            "project_labels": len(PROJECT_LABELS),
            "training_examples": len(PROJECT_TRAINING),
            "doc_chunks": len(self.retriever.chunks),
            "doc_sources": len({c.doc_id for c in self.retriever.chunks}),
            "vocabulary_size": len(self.retriever.vectorizer.vocabulary_),
        }


engine = MLEngine()
