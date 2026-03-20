"""
This experiment compares a blank spaCy pipeline against our trained legislative spaCy model.

Metrics:
- Accuracy: exact-set match accuracy per row
- Precision / Recall / F1: macro-averaged over extracted PERSON names
"""

from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path
import spacy


CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CODE_ROOT / "data"
MODELS_ROOT = CODE_ROOT / "models"

CSV_PATH = DATA_ROOT / "NER Legislative Labeled Dataset - labeling_batch_v1.csv"
TRAINED_MODEL_PATH = MODELS_ROOT / "spacy_legislative_ner"
PERSON_LABEL = "PERSON"

TITLES = re.compile(
    r"\b(Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Senator|Sen|"
    r"Assemblymember|Assemblywoman|Assemblyman|"
    r"Chairman|Chairwoman|Chair|Vice Chair|"
    r"Representative|Rep|Councilmember|Judge|Officer|"
    r"Director|Secretary|Commissioner|Madam|Sir)\.?\s*",
    re.IGNORECASE,
)


def normalize_name(name: str) -> str:
    name = TITLES.sub("", name).strip()
    name = name.strip(".,;:() ")
    return re.sub(r"\s+", " ", name).lower()


def parse_gold_names(raw_value: str) -> set[str]:
    names = set()
    for part in (raw_value or "").split("|"):
        cleaned = normalize_name(part)
        if cleaned:
            names.add(cleaned)
    return names


def load_dataset(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "sid": row.get("sid", ""),
                    "text": row.get("text", "") or "",
                    "gold_names": parse_gold_names(row.get("all_names", "")),
                }
            )
    return rows


def extract_names(nlp, text: str) -> set[str]:
    if "ner" not in nlp.pipe_names:
        return set()

    doc = nlp(text)
    names = set()
    for ent in doc.ents:
        if ent.label_ == PERSON_LABEL:
            normalized = normalize_name(ent.text)
            if normalized and len(normalized) > 1:
                names.add(normalized)
    return names


def score_predictions(
    rows: list[dict[str, object]], predictions: list[set[str]]
) -> dict[str, float]:
    exact_matches = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for row, pred in zip(rows, predictions):
        gold = row["gold_names"]
        if pred == gold:
            exact_matches += 1

        tp_total += len(pred & gold)
        fp_total += len(pred - gold)
        fn_total += len(gold - pred)

    accuracy = exact_matches / len(rows) if rows else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(label)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(
        f"  TP={int(metrics['tp'])}  FP={int(metrics['fp'])}  FN={int(metrics['fn'])}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--trained-model", type=Path, default=TRAINED_MODEL_PATH)
    args = parser.parse_args()

    rows = load_dataset(args.csv)
    print(f"Loaded {len(rows)} rows from {args.csv}")

    blank_nlp = spacy.blank("en")
    trained_nlp = spacy.load(args.trained_model)

    blank_predictions = [extract_names(blank_nlp, row["text"]) for row in rows]
    trained_predictions = [extract_names(trained_nlp, row["text"]) for row in rows]

    blank_metrics = score_predictions(rows, blank_predictions)
    trained_metrics = score_predictions(rows, trained_predictions)

    print()
    print("spaCy NER comparison")
    print("=" * 40)
    print_metrics("Blank spaCy pipeline (untrained)", blank_metrics)
    print_metrics("Trained legislative spaCy model", trained_metrics)


if __name__ == "__main__":
    main()
