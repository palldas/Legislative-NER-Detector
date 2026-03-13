"""
Train a BERT classifier for self-introduction detection.

Running this script fine-tunes `bert-base-uncased` on the milestone 3 CSV
and saves the model/tokenizer bundle to `milestone3_bert_self_intro`.
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    raise SystemExit(
        "Missing dependency: torch\n"
        "Install with: pip install torch"
    )

try:
    from sklearn.model_selection import train_test_split
except ModuleNotFoundError:
    raise SystemExit(
        "Missing dependency: scikit-learn\n"
        "Install with: pip install scikit-learn"
    )

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError:
    raise SystemExit(
        "Missing dependency: transformers\n"
        "Install with: pip install transformers"
    )


CSV_PATH = Path("NER Legislative Labeled Dataset - labeling_batch_v1.csv")
MODEL_NAME = "bert-base-uncased"
MODEL_SAVE_DIR = Path("bert_self_intro_classifier")
RANDOM_SEED = 42
TEST_SIZE = 0.20
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class IntroDataset(Dataset):
    def __init__(self, texts, labels, tokenizer_obj, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer_obj
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise SystemExit(
            "Dataset not found: 'NER Legislative Labeled Dataset - labeling_batch_v1.csv'\n"
            "Place the CSV in the project root and rerun."
        )

    df = pd.read_csv(CSV_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["text"] = df["text"].fillna("")
    df["is_self_intro"] = df["is_self_intro"].fillna(0).astype(int)
    return df


def load_model():
    print(f"Loading base model: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to load Hugging Face model/tokenizer.\n"
            "If this is your first run, ensure internet access to download model files, "
            "or use a cached local model.\n"
            f"Details: {exc}"
        )

    return tokenizer, model


def main() -> None:
    set_seed(RANDOM_SEED)
    df = load_dataset()

    print(
        f"Loaded {len(df)} rows | {df['is_self_intro'].sum()} self-intros"
    )

    texts = df["text"].tolist()
    labels = df["is_self_intro"].tolist()

    train_texts, _, train_labels, _ = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    print(
        f"Training rows: {len(train_texts)} | "
        f"Self-intros in training split: {sum(train_labels)}"
    )

    tokenizer, model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    train_dataset = IntroDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    print("Training BERT classifier...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"  Epoch {epoch + 1}/{EPOCHS} - loss={avg_loss:.4f}")

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)
    print(f"Saved fine-tuned model to: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    main()
