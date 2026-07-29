"""Loyiha turi klassifikatorining aniqligini o'lchaydi.

Nega alohida skript: "ML kuchaytirildi" degan gapni faqat raqam tasdiqlaydi.
Datasetga misol qo'shish aniqlikni oshirishi ham, tushirishi ham mumkin —
o'lchovsiz qaysi biri bo'lganini bilib bo'lmaydi.

O'lchash usuli — stratifikatsiyalangan 5 qatlamli kross-validatsiya. Oddiy
train/test bo'linishi bu yerda ishonchsiz: sinfga ~20 tadan misol bor, ya'ni
bitta tasodifiy bo'linish natijani ±10% ga sakratib yuboradi.

Ishga tushirish:
    python -m scripts.eval_classifier
    python -m scripts.eval_classifier --errors     # xato tasniflarni ko'rsatadi
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.ml.engine import ProjectClassifier, normalize
from app.ml.training_data import PROJECT_TRAINING

_FOLDS = 5


def _build_estimator():
    """`ProjectClassifier.train()` bilan bir xil quvurni yig'adi.

    Ataylab qayta yig'iladi, o'qitilgan modeldan foydalanilmaydi: kross-
    validatsiya har qatlamda toza modeldan boshlashi kerak, aks holda test
    qismi o'qitishda ko'rilgan bo'lib chiqadi va aniqlik soxta oshadi.
    """
    from sklearn.pipeline import Pipeline

    vectorizer, model = ProjectClassifier.build_pipeline()
    return Pipeline([("features", vectorizer), ("clf", model)])


def main() -> int:
    # Dataset uch tilda va tutuq belgilari (ʻ) bor. Windows konsolining
    # standart kodlashi ularni chiqara olmaydi va chiqish o'rtada uzilib
    # qoladi — xatolar ro'yxati ayni shu sababdan yarim ko'rinardi.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--errors", action="store_true", help="xato tasniflarni chiqarish")
    args = parser.parse_args()

    texts = np.array([normalize(t) for t, _ in PROJECT_TRAINING])
    labels = np.array([l for _, l in PROJECT_TRAINING])

    counts = Counter(labels.tolist())
    print(f"Dataset: {len(texts)} misol, {len(counts)} sinf")
    smallest = min(counts.values())
    print(f"Sinf hajmi: {smallest}..{max(counts.values())} (eng kichigi: "
          f"{min(counts, key=counts.get)})")
    print()

    if smallest < _FOLDS:
        print(f"DIQQAT: eng kichik sinfda {smallest} misol bor, "
              f"{_FOLDS} qatlam uchun kam — qatlam soni kamaytirildi.")
    folds = min(_FOLDS, smallest)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    predicted = cross_val_predict(_build_estimator(), texts, labels, cv=cv)

    accuracy = float((predicted == labels).mean())
    print(f"=== {folds} qatlamli kross-validatsiya ===")
    print(f"Aniqlik (accuracy): {accuracy:.1%}")
    print()
    print(classification_report(labels, predicted, digits=3, zero_division=0))

    # Eng ko'p chalkashadigan juftliklar — dataset qayerda kuchsizligini aytadi.
    order = sorted(counts)
    matrix = confusion_matrix(labels, predicted, labels=order)
    pairs: list[tuple[int, str, str]] = []
    for i, actual in enumerate(order):
        for j, guessed in enumerate(order):
            if i != j and matrix[i][j]:
                pairs.append((int(matrix[i][j]), actual, guessed))
    pairs.sort(reverse=True)

    if pairs:
        print("Eng ko'p chalkashgan juftliklar (haqiqiy -> taxmin):")
        for n, actual, guessed in pairs[:10]:
            print(f"  {n:2}x  {actual:20} -> {guessed}")

    if args.errors:
        print("\nXato tasniflangan misollar:")
        for text, actual, guessed in zip(texts, labels, predicted):
            if actual != guessed:
                print(f"  [{actual} -> {guessed}] {text[:95]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
