"""
Train a legislative-domain spaCy NER model on the labeled CSV dataset.

The script learns a single entity label, PERSON, using the names annotated in:
`NER Legislative Labeled Dataset - labeling_batch_v1.csv`
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import spacy
from spacy.training import Example
from spacy.util import compounding, fix_random_seed, minibatch


CSV_PATH = Path("NER Legislative Labeled Dataset - labeling_batch_v1.csv")
MODEL_DIR = Path("spacy_legislative_ner")
LABEL = "PERSON"
RANDOM_SEED = 42

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
    name = name.replace("`", "'")
    name = re.sub(r"\s+", " ", name)
    return name.strip(".,;:() ")


def split_annotated_names(raw_value: str) -> list[str]:
    names = []
    for part in (raw_value or "").split("|"):
        cleaned = normalize_name(part)
        if cleaned:
            names.append(cleaned)
    return names


def candidate_variants(name: str) -> list[str]:
    variants = {normalize_name(name)}

    # Some rows include hints like "(nickname)"; train on the actual text form only.
    stripped = re.sub(r"\([^)]*\)", "", name).strip()
    if stripped:
        variants.add(normalize_name(stripped))

    words = normalize_name(name).split()
    if len(words) >= 2:
        variants.add(" ".join(words[:2]))
        variants.add(words[0])
        variants.add(words[-1])

    return [v for v in sorted(variants, key=len, reverse=True) if v]


def find_name_spans(text: str, names: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    taken = []

    for name in sorted(set(names), key=len, reverse=True):
        matched = False
        for variant in candidate_variants(name):
            pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(not (end <= s or start >= e) for s, e in taken):
                    continue
                spans.append((start, end, LABEL))
                taken.append((start, end))
                matched = True
                break
            if matched:
                break

    return sorted(spans, key=lambda item: item[0])


def load_training_examples(csv_path: Path) -> list[tuple[str, dict]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            "Dataset not found: 'NER Legislative Labeled Dataset - labeling_batch_v1.csv'"
        )

    examples = []
    skipped = 0

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            names = split_annotated_names(row.get("all_names", ""))
            self_intro_name = normalize_name(row.get("self_intro_name", ""))
            if self_intro_name:
                names.append(self_intro_name)

            spans = find_name_spans(text, names)
            if names and not spans:
                skipped += 1
                continue

            examples.append((text, {"entities": spans}))

    if not examples:
        raise ValueError("No trainable examples were created from the dataset.")

    print(
        f"Prepared {len(examples)} examples from {csv_path.name}. "
        f"Skipped {skipped} rows where names were annotated but not found in text."
    )
    return examples


def train_model(
    train_examples: list[tuple[str, dict]],
    dev_examples: list[tuple[str, dict]],
    output_dir: Path,
    iterations: int,
) -> Path:
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label(LABEL)

    optimizer = nlp.begin_training()

    for epoch in range(iterations):
        random.shuffle(train_examples)
        losses = {}

        batches = minibatch(train_examples, size=compounding(4.0, 16.0, 1.5))
        for batch in batches:
            batch_examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                batch_examples.append(Example.from_dict(doc, annotations))
            nlp.update(batch_examples, sgd=optimizer, drop=0.2, losses=losses)

        print(f"Epoch {epoch + 1}/{iterations} - losses: {losses}")

    output_dir.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_dir)
    print(f"Saved trained spaCy model to {output_dir}")

    if dev_examples:
        scores = evaluate_model(nlp, dev_examples)
        print(
            "Dev set scores - "
            f"precision: {scores['ents_p']:.3f}, "
            f"recall: {scores['ents_r']:.3f}, "
            f"f1: {scores['ents_f']:.3f}"
        )

    return output_dir


def evaluate_model(nlp, examples: list[tuple[str, dict]]) -> dict:
    scored = []
    for text, annotations in examples:
        doc = nlp.make_doc(text)
        scored.append(Example.from_dict(doc, annotations))
    return nlp.evaluate(scored)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output", type=Path, default=MODEL_DIR)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    args = parser.parse_args()

    fix_random_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    examples = load_training_examples(args.csv)
    random.shuffle(examples)

    split_at = int(len(examples) * (1 - args.dev_ratio))
    train_examples = examples[:split_at]
    dev_examples = examples[split_at:]

    print(
        f"Training on {len(train_examples)} examples, "
        f"evaluating on {len(dev_examples)} examples."
    )
    train_model(train_examples, dev_examples, args.output, args.iterations)


if __name__ == "__main__":
    main()
