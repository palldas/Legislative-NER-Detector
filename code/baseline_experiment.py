"""
Milestone 2 - Baseline NER & Self-Introduction Detection Experiment

We tried 4 baselines in this milestone on our labeled dataset:
Baseline 1: spaCy NER (en_core_web_sm -> PERSON entities)
Baseline 2: names-dataset (lexicon lookup of first/last names)
Baseline 3: NLTK NER (ne_chunk -> PERSON entities)
Baseline 4: Regex self-introduction detector
"""

import re
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from names_dataset import NameDataset
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

CODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = CODE_ROOT / "data"

CSV_PATH = DATA_ROOT / "NER Legislative Labeled Dataset - labeling_batch_v1.csv"
df = pd.read_csv(CSV_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df["all_names"] = df["all_names"].fillna("")
df["is_self_intro"] = df["is_self_intro"].fillna(0).astype(int)
df["self_intro_name"] = df["self_intro_name"].fillna("")
df["text"] = df["text"].fillna("")

print(f"Loaded {len(df)} rows  |  {df['is_self_intro'].sum()} self-intros  |  "
      f"{(df['all_names'] != '').sum()} rows with names\n")

TITLES = re.compile(
    r"\b(Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Senator|Sen|"
    r"Assemblymember|Assemblywoman|Assemblyman|"
    r"Chairman|Chairwoman|Chair|Vice Chair|"
    r"Representative|Rep|Councilmember|Judge|Officer|"
    r"Director|Secretary|Commissioner|Madam|Sir)\.?\s*",
    re.IGNORECASE
)

def parse_gold_names(raw: str) -> set:
    if not raw or pd.isna(raw):
        return set()
    names = set()
    for n in raw.split("|"):
        n = n.strip()
        if n:
            names.add(n.lower())
    return names

def normalize_name(name: str) -> str:
    name = TITLES.sub("", name).strip()
    name = name.strip(".,;:")
    return name.lower()

# spaCy NER
print("BASELINE 1: spaCy NER (en_core_web_sm -> PERSON)")

nlp = spacy.load("en_core_web_sm")

def spacy_extract_names(text: str) -> set:
    doc = nlp(text)
    names = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            n = normalize_name(ent.text)
            if n and len(n) > 1:
                names.add(n)
    return names

# names-dataset (lexicon lookup)
print("\nLoading names-dataset... ", end="", flush=True)
nd = NameDataset()
print("done.")

def names_dataset_extract(text: str) -> set:
    tokens = re.findall(r"[A-Z][a-z]+(?:-[A-Z][a-z]+)*", text)
    names = set()

    for tok in tokens:
        first_result = nd.search(tok)
        is_first = first_result and first_result.get("first_name") and \
                   first_result["first_name"].get("country")
        is_last  = first_result and first_result.get("last_name") and \
                   first_result["last_name"].get("country")
        if is_first or is_last:
            names.add(tok.lower())

    return names

# NLTK NER (ne_chunk)
print("\nBaseline 3: NLTK NER")

def nltk_extract_names(text: str) -> set:
    names = set()
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunks = ne_chunk(tagged)
        for subtree in chunks:
            if isinstance(subtree, Tree) and subtree.label() == "PERSON":
                name = " ".join(word for word, tag in subtree.leaves())
                n = normalize_name(name)
                if n and len(n) > 1:
                    names.add(n)
    except Exception:
        pass
    return names

# Regex Self-Introduction Detector
INTRO_PATTERNS = [
    r"my name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\-]+)*)",
    r"I[''']?m\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\-]+)*)",
    r"I am\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\-]+)*)",
    r"this is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\-]+)*)",
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z\-]+)+)\s+(?:on behalf of|with the|representing|from|here)",
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z\-]+)+),?\s+(?:the\s+)?(?:chair|director|executive)",
]

def regex_detect_intro(text: str) -> int:
    """Return 1 if any self-intro pattern matches, else 0."""
    for pat in INTRO_PATTERNS:
        if re.search(pat, text, re.IGNORECASE if pat.startswith("my name") or pat.startswith("I ") or pat.startswith("I[") or pat.startswith("this is") else 0):
            return 1
    return 0

print("\nRunning all baselines on dataset...")

spacy_preds = []
nd_preds = []
nltk_preds = []
regex_intro_preds = []

for i, row in df.iterrows():
    text = str(row["text"])
    spacy_preds.append(spacy_extract_names(text))
    nd_preds.append(names_dataset_extract(text))
    nltk_preds.append(nltk_extract_names(text))
    regex_intro_preds.append(regex_detect_intro(text))

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(df)} rows...")

print(f"  Processed {len(df)}/{len(df)} rows. Done.\n")

# evaluation - name extraction
def evaluate_name_extraction(pred_list, gold_list, label):
    """Compute per-row precision/recall/F1 and macro average."""
    precisions = []
    recalls = []
    f1s = []

    tp_total, fp_total, fn_total = 0, 0, 0

    for pred, gold in zip(pred_list, gold_list):
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)

        tp_total += tp
        fp_total += fp
        fn_total += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(gold) == 0 else 0.0)
        r = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if len(pred) == 0 else 0.0)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    # Micro-averaged
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    print(f"  {label}")
    print(f"Micro-Precision: {micro_p:.4f}")
    print(f"Micro-Recall: {micro_r:.4f}")
    print(f"Micro-F1: {micro_f1:.4f}")
    print(f"Macro-Precision: {np.mean(precisions):.4f}")
    print(f"Macro-Recall: {np.mean(recalls):.4f}")
    print(f"Macro-F1: {np.mean(f1s):.4f}")
    print(f"Total TP={tp_total}  FP={fp_total}  FN={fn_total}")
    print()

    return {
        "label": label,
        "micro_p": micro_p, "micro_r": micro_r, "micro_f1": micro_f1,
        "macro_p": np.mean(precisions), "macro_r": np.mean(recalls),
        "macro_f1": np.mean(f1s),
        "tp": tp_total, "fp": fp_total, "fn": fn_total
    }


gold_names = [parse_gold_names(row["all_names"]) for _, row in df.iterrows()]

print("NAME EXTRACTION RESULTS")

results_spacy = evaluate_name_extraction(spacy_preds, gold_names, "spaCy NER")
results_nd = evaluate_name_extraction(nd_preds, gold_names, "names-dataset Lookup")
results_nltk = evaluate_name_extraction(nltk_preds, gold_names, "NLTK NER")

# evaluation - self-introduction detection
print("SELF-INTRODUCTION DETECTION RESULTS")

gold_intro = df["is_self_intro"].tolist()

print("\n  Regex Self-Intro Detector:")
print(f"Accuracy: {accuracy_score(gold_intro, regex_intro_preds):.4f}")
print(f"Precision: {precision_score(gold_intro, regex_intro_preds, zero_division=0):.4f}")
print(f"Recall: {recall_score(gold_intro, regex_intro_preds, zero_division=0):.4f}")
print(f"F1: {f1_score(gold_intro, regex_intro_preds, zero_division=0):.4f}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(gold_intro, regex_intro_preds)
print(f"TN={cm[0][0]}  FP={cm[0][1]}")
print(f"FN={cm[1][0]}  TP={cm[1][1]}")
print()

# error analysis - sample errors
print("ERROR ANALYSIS — spaCy NER (sample)")

error_count = 0
for i, (pred, gold) in enumerate(zip(spacy_preds, gold_names)):
    fp = pred - gold
    fn = gold - pred
    if fp or fn:
        if error_count < 10:
            text_snippet = str(df.iloc[i]["text"])[:120]
            print(f"\n  Row {i} (sid={df.iloc[i]['sid']}):")
            print(f"Text: {text_snippet}...")
            print(f"Gold: {gold if gold else '{}'}")
            print(f"Predicted: {pred if pred else '{}'}")
            if fp: print(f"FALSE POS: {fp}")
            if fn: print(f"FALSE NEG: {fn}")
            error_count += 1

print(f"\n  (Showing {min(error_count, 10)} of total error rows)")


output_lines = []
output_lines.append("MILESTONE 2 — BASELINE EXPERIMENT RESULTS")
output_lines.append(f"\nDataset: {len(df)} rows, {df['is_self_intro'].sum()} self-intros, "
                     f"{(df['all_names'] != '').sum()} rows with names\n")

output_lines.append("\n--- NAME EXTRACTION ---\n")
summary_table = pd.DataFrame([results_spacy, results_nd, results_nltk])
summary_table = summary_table[["label", "micro_p", "micro_r", "micro_f1", "macro_p", "macro_r", "macro_f1", "tp", "fp", "fn"]]
output_lines.append(summary_table.to_string(index=False))

output_lines.append("\n\n--- SELF-INTRODUCTION DETECTION ---\n")
output_lines.append(f"Regex Detector:")
output_lines.append(f"Accuracy: {accuracy_score(gold_intro, regex_intro_preds):.4f}")
output_lines.append(f"Precision: {precision_score(gold_intro, regex_intro_preds, zero_division=0):.4f}")
output_lines.append(f"Recall: {recall_score(gold_intro, regex_intro_preds, zero_division=0):.4f}")
output_lines.append(f"F1: {f1_score(gold_intro, regex_intro_preds, zero_division=0):.4f}")
output_lines.append(f"\nConfusion Matrix:")
output_lines.append(f"TN={cm[0][0]}  FP={cm[0][1]}")
output_lines.append(f"FN={cm[1][0]}  TP={cm[1][1]}")

results_file = "baseline_results.txt"
with open(results_file, "w") as f:
    f.write("\n".join(output_lines))

print(f"\n\nResults saved to {results_file}")
