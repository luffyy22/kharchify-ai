"""
Part A - Dish Category Classifier
=================================

Suggests a menu category (Starter, Main Course, Bread, ...) for a new dish name.

Pipeline
--------
1. ``clean_dishes``  - trim/collapse spaces, normalise case, fix category spelling
                       variants, drop missing values and duplicates, flag label conflicts.
2. ``split_data``    - stratified 80/20 train/test split with a fixed random seed.
3. ``make_model``    - two scikit-learn pipelines (both TF-IDF + Logistic Regression):
                         * "word"       : word uni/bi-grams            (baseline)
                         * "word+char"  : word n-grams + char n-grams  (handles unseen words)
4. ``evaluate``      - accuracy, macro-F1, per-category report, confusion matrix on the TEST set.
5. ``predict_category(dish_name)`` - the public helper requested in the task.

Run ``python part_a_classifier.py`` to reproduce every number in the write-up.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline, make_pipeline, make_union

SEED = 42
TEST_SIZE = 0.2
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

CATEGORIES = ["Starter", "Main Course", "Bread", "Rice", "Dessert", "Beverage"]

# Defensive: spelling variants a restaurant owner might type for the category column.
CATEGORY_ALIASES = {
    "starters": "Starter", "appetizer": "Starter", "appetizers": "Starter",
    "main": "Main Course", "mains": "Main Course", "main courses": "Main Course",
    "main-course": "Main Course", "maincourse": "Main Course",
    "breads": "Bread", "rice": "Rice", "rice & biryani": "Rice",
    "desserts": "Dessert", "sweet": "Dessert", "sweets": "Dessert",
    "beverages": "Beverage", "drinks": "Beverage", "drink": "Beverage",
}

# Below this top-class probability the UI should ask the owner to confirm the category.
# (Chosen from out-of-fold predictions in ``confidence_analysis``; see write-up.)
LOW_CONFIDENCE = 0.50


# --------------------------------------------------------------------------- #
# 1. Cleaning
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    """Trim and collapse any run of whitespace into a single space."""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_text(name: str) -> str:
    """Model-side normalisation: lower-case, punctuation -> space, single spaces.

    Letters and digits are kept ("Chicken 65" stays informative); "Hot & Sour" and
    "Hot and Sour" end up close to each other.
    """
    name = str(name).lower().replace("&", " and ")
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    return _squash(name)


def clean_dishes(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return ``(clean_df, report)``. ``report`` counts what was fixed or dropped."""
    report = {"rows_in": len(raw)}
    df = raw[["dish_name", "category"]].copy()

    # Blank strings are "missing" too, not just NaN.
    df = df.replace(r"^\s*$", np.nan, regex=True)
    before = len(df)
    df = df.dropna(subset=["dish_name", "category"])
    report["missing_dropped"] = before - len(df)

    df["dish_name"] = df["dish_name"].map(_squash)
    cat = df["category"].map(_squash)
    cat = cat.map(lambda c: CATEGORY_ALIASES.get(c.lower(), c.title()))
    df["category"] = cat
    report["categories_fixed"] = int((cat != raw.loc[df.index, "category"]).sum())

    df["key"] = df["dish_name"].map(normalize_text)

    # Same dish typed twice with *different* categories: label is untrustworthy -> drop all copies.
    n_cats = df.groupby("key")["category"].transform("nunique")
    conflicts = df[n_cats > 1]
    report["conflicting_labels_dropped"] = len(conflicts)
    report["conflicting_examples"] = sorted(conflicts["dish_name"].unique().tolist())
    df = df[n_cats == 1]

    # Same dish typed twice with the same category -> keep one.
    before = len(df)
    df = df.drop_duplicates(subset="key", keep="first")
    report["duplicates_dropped"] = before - len(df)

    unknown = sorted(set(df["category"]) - set(CATEGORIES))
    report["unexpected_categories"] = unknown  # not dropped, just surfaced
    report["rows_out"] = len(df)
    return df.drop(columns="key").reset_index(drop=True), report


def load_dishes(path: Path | str | None = None) -> tuple[pd.DataFrame, dict]:
    path = Path(path) if path else DATA_DIR / "dishes.csv"
    return clean_dishes(pd.read_csv(path))


# --------------------------------------------------------------------------- #
# 2. Split
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Stratified split so every category appears in both train and test."""
    return train_test_split(
        df["dish_name"], df["category"],
        test_size=test_size, stratify=df["category"], random_state=seed,
    )


# --------------------------------------------------------------------------- #
# 3. Models
# --------------------------------------------------------------------------- #
def make_model(kind: str = "word+char", C: float = 10.0) -> Pipeline:
    """Build an (unfitted) text-classification pipeline.

    kind="word"       TF-IDF over word 1-2-grams                       -> Logistic Regression
    kind="word+char"  the above + TF-IDF over character 2-4-grams      -> Logistic Regression
    """
    word = TfidfVectorizer(preprocessor=normalize_text, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb",
                           ngram_range=(2, 4), sublinear_tf=True)
    features = word if kind == "word" else make_union(word, char) if kind == "word+char" else None
    if features is None:
        raise ValueError(f"unknown model kind: {kind!r}")
    return make_pipeline(features, LogisticRegression(C=C, max_iter=5000))


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
def cv_scores(kind: str, X, y, n_splits: int = 5, n_repeats: int = 5, **kw) -> np.ndarray:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    return cross_val_score(make_model(kind, **kw), X, y, cv=cv, scoring="accuracy")


def repeated_split_scores(kind: str, df: pd.DataFrame, n_splits: int = 200, **kw) -> np.ndarray:
    """Test accuracy over many different random stratified 80/20 splits.

    With only ~26 test rows, one split is a noisy estimate; this shows how noisy.
    """
    scores = []
    for seed in range(n_splits):
        Xa, Xb, ya, yb = split_data(df, seed=seed)
        scores.append(accuracy_score(yb, make_model(kind, **kw).fit(Xa, ya).predict(Xb)))
    return np.array(scores)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion k/n (better than +/- for small n)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def evaluate(model: Pipeline, X_test, y_test) -> dict:
    """Score a *fitted* model on held-out data."""
    pred = model.predict(X_test)
    labels = [c for c in CATEGORIES if c in set(y_test) | set(pred)]
    return {
        "pred": pred,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", labels=labels, zero_division=0),
        "report": classification_report(y_test, pred, labels=labels, zero_division=0),
        "labels": labels,
        "confusion": confusion_matrix(y_test, pred, labels=labels),
    }


def out_of_fold_predictions(kind: str, X, y, C: float = 10.0) -> pd.DataFrame:
    """Predict every dish with a model that never saw it (5-fold). For error analysis only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(make_model(kind, C=C), X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    out = pd.DataFrame({
        "dish_name": list(X), "true": list(y),
        "pred": classes[proba.argmax(1)], "confidence": proba.max(1),
    })
    out["correct"] = out["true"] == out["pred"]
    return out


def confidence_analysis(oof: pd.DataFrame, thresholds=(0.4, 0.5, 0.6)) -> pd.DataFrame:
    """How accurate are predictions above vs below a confidence cut-off?"""
    rows = []
    for t in thresholds:
        hi, lo = oof[oof.confidence >= t], oof[oof.confidence < t]
        rows.append({
            "threshold": t,
            "auto-accepted %": round(100 * len(hi) / len(oof), 1),
            "accuracy when accepted": round(100 * hi.correct.mean(), 1) if len(hi) else np.nan,
            "accuracy when flagged": round(100 * lo.correct.mean(), 1) if len(lo) else np.nan,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. Public helper: predict_category(dish_name)
# --------------------------------------------------------------------------- #
_PRODUCTION_MODEL: Pipeline | None = None


def get_production_model() -> Pipeline:
    """Model used by ``predict_category``: best config, refit on ALL cleaned dishes.

    (Accuracy is measured on a held-out split *first*; refitting on everything afterwards is the
    usual practice - more data helps, and the reported number comes from the held-out model.)
    """
    global _PRODUCTION_MODEL
    if _PRODUCTION_MODEL is None:
        df, _ = load_dishes()
        _PRODUCTION_MODEL = make_model("word+char").fit(df["dish_name"], df["category"])
    return _PRODUCTION_MODEL


def predict_with_confidence(dish_name: str, top_k: int = 3, model: Pipeline | None = None) -> dict:
    """Category plus probabilities, and a flag telling the UI to ask for confirmation."""
    if not isinstance(dish_name, str) or not normalize_text(dish_name):
        raise ValueError("dish_name must be a non-empty string")
    model = model or get_production_model()
    proba = model.predict_proba([dish_name])[0]
    order = np.argsort(proba)[::-1][:top_k]
    classes = model.classes_
    return {
        "dish_name": _squash(dish_name),
        "category": str(classes[order[0]]),
        "confidence": float(proba[order[0]]),
        "top_k": [(str(classes[i]), float(proba[i])) for i in order],
        "needs_confirmation": bool(proba[order[0]] < LOW_CONFIDENCE),
    }


def predict_category(dish_name: str) -> str:
    """Suggest a category for a (possibly new) dish name, e.g. 'Butter Garlic Naan' -> 'Bread'."""
    return predict_with_confidence(dish_name)["category"]


# --------------------------------------------------------------------------- #
# Dishes that are NOT in dishes.csv. Labels were written down by hand, following the
# menu conventions visible in the data (e.g. Biryani -> Rice, Tandoori Chicken -> Starter),
# BEFORE running the model on them. They are never used for training or model selection.
# --------------------------------------------------------------------------- #
UNSEEN_DISHES = {
    "Butter Garlic Naan": "Bread", "Butter Kulcha": "Bread", "Paneer Paratha": "Bread",
    "Mango Milkshake": "Beverage", "Lemon Iced Tea": "Beverage", "Hot Chocolate": "Beverage",
    "Cappuccino": "Beverage", "Rose Lassi": "Beverage", "Lemonade": "Beverage",
    "Chicken Lollipop": "Starter", "Gobi Manchurian": "Starter", "Tandoori Paneer": "Starter",
    "Fish Tikka": "Starter", "Chicken Soup": "Starter",
    "Mutton Pulao": "Rice", "Veg Dum Biryani": "Rice", "Jeera Pulao": "Rice", "Egg Pulao": "Rice",
    "Rabri": "Dessert", "Tiramisu": "Dessert", "Chocolate Lava Cake": "Dessert",
    "Basundi": "Dessert", "Kala Jamun": "Dessert",
    "Dal Fry": "Main Course", "Mushroom Masala": "Main Course", "Chicken Lababdar": "Main Course",
    "Kadhai Mushroom": "Main Course", "Paneer Tikka Masala": "Main Course",
}


def unseen_check(model: Pipeline, known_names) -> pd.DataFrame:
    known = {normalize_text(n) for n in known_names}
    overlap = [d for d in UNSEEN_DISHES if normalize_text(d) in known]
    assert not overlap, f"these 'unseen' dishes are in the dataset: {overlap}"
    names = list(UNSEEN_DISHES)
    proba = model.predict_proba(names)
    out = pd.DataFrame({
        "dish_name": names, "expected": [UNSEEN_DISHES[n] for n in names],
        "predicted": model.classes_[proba.argmax(1)], "confidence": proba.max(1).round(2),
    })
    out["correct"] = out["expected"] == out["predicted"]
    return out


# --------------------------------------------------------------------------- #
# Plot helper
# --------------------------------------------------------------------------- #
def plot_confusion(conf: np.ndarray, labels: list[str], title: str, path: Path | None = None):
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, conf[i, j], ha="center", va="center",
                    color="white" if conf[i, j] > conf.max() / 2 else "black")
    fig.tight_layout()
    if path is not None:
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Command-line report
# --------------------------------------------------------------------------- #
def main() -> None:
    warnings.filterwarnings("ignore")
    df, rep = load_dishes()
    print("== Cleaning ==")
    for k, v in rep.items():
        print(f"  {k}: {v}")
    print("\nClass balance:\n" + df["category"].value_counts().to_string())

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows (stratified, seed={SEED})")

    print("\n== Model selection: 5x5 repeated CV on the TRAINING set only ==")
    for kind in ("word", "word+char"):
        s = cv_scores(kind, X_train, y_train)
        print(f"  {kind:10s} CV accuracy = {s.mean():.3f} +/- {s.std():.3f}")

    print("\n== Held-out TEST set (each model fitted on train only, scored once) ==")
    results = {}
    for kind in ("word", "word+char"):
        model = make_model(kind).fit(X_train, y_train)
        results[kind] = evaluate(model, X_test, y_test)
        r = results[kind]
        print(f"  {kind:10s} accuracy = {r['accuracy']:.3f} | macro-F1 = {r['macro_f1']:.3f}")
    best = "word+char"
    print(f"\nPer-category report for '{best}':\n{results[best]['report']}")
    plot_confusion(results[best]["confusion"], results[best]["labels"],
                   f"Confusion matrix ({best}, test set, n={len(y_test)})",
                   OUT_DIR / "part_a_confusion_matrix.png")

    print("== Stability check: 5x5 CV on ALL cleaned data ==")
    for kind in ("word", "word+char"):
        s = cv_scores(kind, df["dish_name"], df["category"])
        print(f"  {kind:10s} accuracy = {s.mean():.3f} +/- {s.std():.3f}")

    k = int(round(results[best]["accuracy"] * len(y_test)))
    lo, hi = wilson_interval(k, len(y_test))
    print(f"\n== How trustworthy is the single {len(y_test)}-row test number? ==")
    print(f"  {best}: {k}/{len(y_test)} correct -> 95% CI for true accuracy ~ {lo:.2f} to {hi:.2f}")
    for kind in ("word", "word+char"):
        sc = repeated_split_scores(kind, df)
        print(f"  {kind:10s} over {len(sc)} random splits: mean {sc.mean():.3f}, sd {sc.std():.3f}, "
              f"5th-95th pct {np.percentile(sc, 5):.2f}-{np.percentile(sc, 95):.2f}")
    sc = repeated_split_scores(best, df)
    print(f"  seed-{SEED} split ({results[best]['accuracy']:.3f}) is worse than "
          f"{100 * (sc > results[best]['accuracy'] + 1e-9).mean():.1f}% of random splits.")
    test_err = pd.DataFrame({"dish_name": X_test.values, "true": y_test.values,
                             "pred": results[best]["pred"]})
    print("\nTest-set mistakes:\n" + test_err[test_err.true != test_err.pred].to_string(index=False))

    oof = out_of_fold_predictions(best, df["dish_name"], df["category"])
    print("\n== Where it goes wrong (out-of-fold, all dishes) ==")
    print(oof[~oof.correct].sort_values("confidence", ascending=False).to_string(index=False))
    print("\nConfidence cut-off analysis:\n" + confidence_analysis(oof).to_string(index=False))

    print("\n== Dishes NOT in dishes.csv ==")
    chk = unseen_check(get_production_model(), df["dish_name"])
    print(chk.to_string(index=False))
    print(f"\nAccuracy on these {len(chk)} unseen dishes: {chk.correct.mean():.3f}")

    print("\n== predict_category() ==")
    for name in ["Butter Garlic Naan", "Mango Milkshake", "Chicken Lollipop", "Tiramisu", "Mutton Pulao"]:
        print(f"  {name:22s} -> {predict_category(name)}")


if __name__ == "__main__":
    sys.exit(main())
