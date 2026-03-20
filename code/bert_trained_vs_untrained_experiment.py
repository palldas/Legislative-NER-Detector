"""
This experiment runs a random-initialized BERT classifier against the saved fine-tuned
self-introduction classifier on the legislative labeled dataset.

Metrics:
- Accuracy
- Precision
- Recall
- F1
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import torch

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CODE_ROOT / "data"
MODELS_ROOT = CODE_ROOT / "models"

CSV_PATH = DATA_ROOT / "NER Legislative Labeled Dataset - labeling_batch_v1.csv"
TRAINED_MODEL_DIR = MODELS_ROOT / "bert_self_intro_classifier"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "sid": row.get("sid", ""),
                    "text": (row.get("text") or "").strip(),
                    "label": int((row.get("is_self_intro") or 0) or 0),
                }
            )
    return rows


def encode_texts(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )


def predict_labels(model, tokenizer, texts: list[str], batch_size: int = 16) -> list[int]:
    model.to(DEVICE)
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = encode_texts(tokenizer, batch_texts)
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            logits = model(**encoded).logits
            batch_preds = torch.argmax(logits, dim=1).tolist()
            predictions.extend(batch_preds)

    return predictions


def compute_binary_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)

    total = len(gold)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(label)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(
        f"TN={int(metrics['tn'])}  FP={int(metrics['fp'])}  "
        f"FN={int(metrics['fn'])}  TP={int(metrics['tp'])}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--trained-model", type=Path, default=TRAINED_MODEL_DIR)
    args = parser.parse_args()

    rows = load_dataset(args.csv)
    texts = [row["text"] for row in rows]
    gold = [row["label"] for row in rows]

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(args.trained_model)

    untrained_config = AutoConfig.from_pretrained(args.trained_model)
    untrained_model = AutoModelForSequenceClassification.from_config(untrained_config)

    trained_model = AutoModelForSequenceClassification.from_pretrained(args.trained_model)

    untrained_pred = predict_labels(untrained_model, tokenizer, texts)
    trained_pred = predict_labels(trained_model, tokenizer, texts)

    untrained_metrics = compute_binary_metrics(gold, untrained_pred)
    trained_metrics = compute_binary_metrics(gold, trained_pred)

    print()
    print("BERT self-intro comparison")
    print_metrics("Random-initialized BERT classifier (untrained)", untrained_metrics)
    print_metrics("Fine-tuned self-intro BERT classifier", trained_metrics)


if __name__ == "__main__":
    main()
