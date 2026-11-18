"""
run_eval_with_gold.py
---------------------
Populates `gold_PER` in Data/to_annotate.csv with verified annotations for the
100 sampled paragraphs and compares Raw spaCy NER vs. the full resolution pipeline
(Data/mentions_characters.csv).
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path
import pandas as pd

GOLD_ANNOTATIONS = {
    0: "",
    1: "Vilaça | Carlos | Cortês | Palma | Carlos | Palma Cavalão | Sr. Maia",
    2: "Carlos | Lola",
    3: "",
    4: "Carlos",
    5: "Pedro | Afonso | Maria | Carlos Eduardo | Carlos Eduardo",
    6: "Carlos",
    7: "Gouvarinho",
    8: "Tomás | Ega",
    9: "",
    10: "",
    11: "papá Monforte | Silva | Monforte | Monforte",
    12: "",
    13: "",
    14: "Carlos | Ega | Carlos",
    15: "Maria Eduarda",
    16: "Carlos",
    17: "Carlos",
    18: "Guimarães | Ega",
    19: "Sr. Vicente",
    20: "Azevedo | Sá Nunes | Cruges | Sá Nunes | Azevedo | Vilaça | Azevedo",
    21: "Cruges",
    22: "",
    23: "Carlos | Castro Gomes | Mac Gren",
    24: "Vilaça | Vilaça | Guimarães",
    25: "Gouvarinho | Joaninha Vilar",
    26: "Cruges | Silveirinha",
    27: "Carlos",
    28: "Craft | Alencar | Alencar | Carlos | Cohen | Ega",
    29: "Carlos | Maria Eduarda | Ega | Vilaça",
    30: "Carlos | Maria | Ega",
    31: "",
    32: "Carlos",
    33: "Baptista | Maria",
    34: "Vasques | Afonso | Fanny",
    35: "",
    36: "",
    37: "Teixeira | Pedro",
    38: "Dâmaso | Josefina do Salazar",
    39: "",
    40: "Carlos",
    41: "Castro Gomes",
    42: "João | João",
    43: "Carlos | Maria",
    44: "Ega | Matos",
    45: "Carlos",
    46: "",
    47: "",
    48: "Ega | Sr. Guimarães",
    49: "",
    50: "Alencar | Carlos",
    51: "Carlos",
    52: "Carlos | Dâmaso | Castro Gomes",
    53: "Dâmaso | Carlos",
    54: "D. Maria",
    55: "Ega | Vilaça",
    56: "Carlos | Susana | Susana",
    57: "Sousa Neto | Ega",
    58: "",
    59: "",
    60: "Maria Eduarda | Dâmaso | Carlos",
    61: "Carlos | Carlos",
    62: "",
    63: "Ega | Carlos | Vilaça | Ega",
    64: "",
    65: "",
    66: "Dâmaso",
    67: "Afonso | Pedro",
    68: "",
    69: "Carlos | Craft | Craft | Eusebiozinho",
    70: "Afonso | Dâmaso | Carlos | Dâmaso | Carlos",
    71: "Carlos",
    72: "",
    73: "Carlos",
    74: "",
    75: "",
    76: "Ega | Gouvarinho | Neves | Ega",
    77: "",
    78: "Carlos",
    79: "",
    80: "Maria Eduarda",
    81: "Carlos | D. Maria",
    82: "",
    83: "",
    84: "Carlos",
    85: "Charlie",
    86: "Raquel | Carlos | Ega",
    87: "Craft | Carlos | Craft",
    88: "Maria",
    89: "Ega | Sr. Guimarães | Dâmaso",
    90: "",
    91: "Sra. Josefa | Ega | Sra. Josefa",
    92: "Dâmaso | Carlos | Sr. Afonso da Maia | Ega | Ega",
    93: "Carlos | Craft",
    94: "",
    95: "",
    96: "",
    97: "Carlos",
    98: "",
    99: "Gouvarinho",
}

ARTICLES = {"o", "a", "os", "as", "um", "uma"}
TITLES = {
    "sr", "sra", "d", "dr", "dra", "dom", "dona", "senhor", "senhora", "mr", "mrs", "miss",
    "padre", "abade", "reverendo", "coronel", "general", "tia", "tio", "velho", "velha",
    "conde", "condessa", "marques", "marquesa"
}


def norm_mention(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    tokens = re.sub(r"[^a-z0-9]+", " ", s).split()
    while tokens and (tokens[0] in ARTICLES or tokens[0] in TITLES):
        tokens = tokens[1:]
    return " ".join(tokens)


def parse_pipe_list(cell):
    if pd.isna(cell) or not str(cell).strip():
        return Counter()
    items = [norm_mention(x) for x in str(cell).split("|")]
    return Counter(x for x in items if x)


def compute_metrics(gold_series, pred_series):
    tp = fp = fn = 0
    fp_ex, fn_ex = Counter(), Counter()
    for gold, pred in zip(gold_series, pred_series):
        matched = gold & pred
        tp += sum(matched.values())
        fp += sum((pred - gold).values())
        fn += sum((gold - pred).values())
        fp_ex.update(pred - gold)
        fn_ex.update(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Precision": p, "Recall": r, "F1": f1,
            "top_FP": fp_ex.most_common(10), "top_FN": fn_ex.most_common(10)}


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "Data"
    ann_path = data_dir / "to_annotate.csv"

    ann = pd.read_csv(ann_path)
    ann["gold_PER"] = ann.index.map(GOLD_ANNOTATIONS)
    ann.to_csv(ann_path, index=False, encoding="utf-8")

    mentions = pd.read_csv(data_dir / "mentions_characters.csv")
    mentions["paragraph_id"] = pd.to_numeric(mentions["paragraph_id"], errors="coerce")
    ann["paragraph_id"] = pd.to_numeric(ann["paragraph_id"], errors="coerce")

    target_pids = ann["paragraph_id"].tolist()
    sub = mentions[mentions["paragraph_id"].isin(set(target_pids))]

    # Dict comprehension evita o MultiIndex do pandas quando se devolve um Counter
    pipe_by_pid = {
        pid: Counter(norm_mention(m) for m in grp["mention"] if norm_mention(m))
        for pid, grp in sub.groupby("paragraph_id")
    }

    gold = [parse_pipe_list(x) for x in ann["gold_PER"]]
    spacy_pred = [parse_pipe_list(x) for x in ann["predicted_PER"]]
    pipe_pred = [pipe_by_pid.get(pid, Counter()) for pid in target_pids]

    r_spacy = compute_metrics(gold, spacy_pred)
    r_pipe = compute_metrics(gold, pipe_pred)

    summary = pd.DataFrame([
        {"Model": "spaCy Raw (PER)", **{k: r_spacy[k] for k in ["TP", "FP", "FN", "Precision", "Recall", "F1"]}},
        {"Model": "Pipeline (NER + Dict)", **{k: r_pipe[k] for k in ["TP", "FP", "FN", "Precision", "Recall", "F1"]}},
    ])
    print("\n=== Avaliação NER (100 parágrafos) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nTop FP (Pipeline):", r_pipe["top_FP"])
    print("Top FN (Pipeline):", r_pipe["top_FN"])


if __name__ == "__main__":
    main()