"""Klassifikator sifati pasayib ketmasligini qulflaydi.

Nega kerak: datasetga misol qo'shish aniqlikni oshirishi ham, TUSHIRISHI ham
mumkin. Noto'g'ri yorliqlangan yoki ikki sinfga ham to'g'ri keladigan bitta
misol chegarani surib yuboradi, va buni hech kim sezmaydi — mahsulot ishlayapti,
faqat tavsiyalar sekin-asta yomonlashadi.

Chegara ataylab joriy natijadan (91.1%) pastroq qo'yilgan: kross-validatsiya
tasodifiy bo'linishga bog'liq, shuning uchun bir-ikki foizlik tebranish normal.
Test "yaxshilanishini" emas, "yomonlashmasligini" tekshiradi.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from app.ml.engine import ProjectClassifier, engine, normalize
from app.ml.training_data import PROJECT_LABELS, PROJECT_TRAINING

# Joriy: accuracy 91.1%, macro F1 0.911. Zaxira ~3 foiz.
_MIN_ACCURACY = 0.88
_MIN_MACRO_F1 = 0.87
# Eng zaif sinf ham foydali bo'lsin — bittasi yiqilsa o'rtacha buni yashiradi.
_MIN_CLASS_F1 = 0.75


@pytest.fixture(scope="module")
def cv_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Kross-validatsiya bashoratlari. Modul bo'yicha bir marta hisoblanadi."""
    texts = np.array([normalize(t) for t, _ in PROJECT_TRAINING])
    labels = np.array([l for _, l in PROJECT_TRAINING])
    vectorizer, model = ProjectClassifier.build_pipeline()
    predicted = cross_val_predict(
        Pipeline([("features", vectorizer), ("clf", model)]),
        texts, labels,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
    )
    return labels, predicted


def test_accuracy_does_not_regress(cv_predictions):
    labels, predicted = cv_predictions
    accuracy = float((predicted == labels).mean())
    assert accuracy >= _MIN_ACCURACY, (
        f"aniqlik {accuracy:.1%} < {_MIN_ACCURACY:.0%}. Yangi misollar sifatini "
        "tekshiring: `python -m scripts.eval_classifier --errors`"
    )


def test_macro_f1_does_not_regress(cv_predictions):
    """Macro F1 — har bir sinf teng vaznda. Katta sinf kichigini yashirmasin."""
    labels, predicted = cv_predictions
    score = f1_score(labels, predicted, average="macro", zero_division=0)
    assert score >= _MIN_MACRO_F1, f"macro F1 {score:.3f} < {_MIN_MACRO_F1}"


def test_no_class_is_useless(cv_predictions):
    labels, predicted = cv_predictions
    classes = sorted(set(labels.tolist()))
    scores = f1_score(labels, predicted, average=None, labels=classes, zero_division=0)
    weak = {c: round(float(s), 3) for c, s in zip(classes, scores) if s < _MIN_CLASS_F1}
    assert not weak, f"F1 juda past sinflar: {weak}"


def test_every_label_has_a_human_name():
    """Yorliq qo'shilib, tarjimasi unutilsa UI'da ichki nom ko'rinib qoladi."""
    labels = {l for _, l in PROJECT_TRAINING}
    missing = sorted(labels - set(PROJECT_LABELS))
    assert not missing, f"PROJECT_LABELS da yo'q: {missing}"
    for label in labels:
        for lang in ("uz", "ru", "en"):
            assert PROJECT_LABELS[label].get(lang), f"{label}/{lang} tarjimasi yo'q"


def test_no_duplicate_training_texts():
    """Takroriy matn kross-validatsiyani aldaydi — o'qitish va testda bir xil satr."""
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for text, label in PROJECT_TRAINING:
        key = normalize(text)
        if key in seen:
            duplicates.append(f"{key[:60]!r} ({seen[key]} / {label})")
        seen[key] = label
    assert not duplicates, "takrorlangan misollar: " + "; ".join(duplicates[:5])


@pytest.mark.parametrize(
    "idea,expected",
    [
        # Bular baholashda aynan chalkashgan holatlar edi — hammasi
        # `marketplace` ga ketardi, chunki matnda "platforma" so'zi bor.
        ("platforma orqali mijoz bo'sh vaqt oralig'ini tanlaydi, usta tasdiqlaydi "
         "va bir kun oldin eslatma boradi", "booking_service"),
        ("mikrokredit platformasi: ariza, avtomatik skoring va to'lov jadvali "
         "bo'yicha qaytarish", "fintech"),
        ("o'quv platformasi: o'qituvchi dars yuklaydi, o'quvchi test topshiradi "
         "va progress ko'rinadi", "edtech"),
        ("platforma ikki tomonni bog'laydi: usta xizmatini e'lon qiladi, mijoz "
         "tanlaydi, biz komissiya olamiz", "marketplace"),
    ],
)
def test_platform_word_does_not_force_marketplace(idea, expected):
    engine.warmup()
    assert engine.classifier.predict(idea).label == expected
