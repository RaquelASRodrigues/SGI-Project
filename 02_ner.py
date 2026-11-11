"""
02_ner.py
---------
Lê  Data/paragraphs.csv  e produz:
  Data/sentences.csv              frases, com posição dentro do parágrafo
  Data/entities_raw.csv           todas as entidades que o spaCy encontrou
  Data/candidates_capitalised.csv nomes em maiúscula (independente do spaCy), para apanhar o que falta
  Data/to_annotate.csv            amostra de parágrafos para validar à mão

Uso:
  pip install spacy pandas tqdm
  python -m spacy download pt_core_news_lg
  python 02_ner.py
  python 02_ner.py --model pt_core_news_md --n-process 1
O texto NÃO é posto em minúsculas: a maiúscula é o sinal principal dos nomes próprios.
"""
import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

MODELS = ["pt_core_news_lg", "pt_core_news_md", "pt_core_news_sm"]

# Nome próprio = palavra(s) com maiúscula, com partículas minúsculas pelo meio ("João da Ega")
NAME_RE = re.compile(
    r"(?<![\wÀ-ÿ])[A-ZÀ-Ý][\wÀ-ÿ'’\-]*"
    r"(?:\s+(?:(?:da|de|do|das|dos)\s+)?[A-ZÀ-Ý][\wÀ-ÿ'’\-]*)*"
)
SENTENCE_START_CHARS = (".", "!", "?", "…", "—", "–", ":", "«", "“", '"')


def load_model(name):
    import spacy
    names = [name] if name else MODELS
    for n in names:
        try:
            nlp = spacy.load(n, disable=["lemmatizer"])
            print(f"Modelo carregado: {n}")
            return nlp
        except OSError:
            continue
    raise SystemExit(
        "Nenhum modelo português encontrado. Instala com:\n"
        "  python -m spacy download pt_core_news_lg"
    )


def run_ner(nlp, df, n_process=1, batch_size=64):
    sentence_rows, entity_rows = [], []
    texts = df["text"].fillna("").tolist()
    docs = nlp.pipe(texts, batch_size=batch_size, n_process=n_process)

    for (_, par), doc in tqdm(zip(df.iterrows(), docs), total=len(df), desc="NER"):
        sent_index = {}
        for sid, sent in enumerate(doc.sents):
            sent_index[sent.start_char] = sid
            sentence_rows.append({
                "paragraph_id": par["paragraph_id"],
                "sent_id": sid,
                "chapter": par["chapter"],
                "start_char": sent.start_char,
                "end_char": sent.end_char,
                "text": sent.text,
            })
        for ent in doc.ents:
            entity_rows.append({
                "paragraph_id": par["paragraph_id"],
                "sent_id": sent_index.get(ent.sent.start_char, 0),
                "chapter": par["chapter"],
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "mention": ent.text,
                "label": ent.label_,
            })
    return pd.DataFrame(sentence_rows), pd.DataFrame(entity_rows)


def norm_key(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s).split()


def capitalised_candidates(df, min_mid=2):
    """Conta sequências em maiúscula. 'mid' = não está no início de frase,
    o que é um bom indício de nome próprio (evita 'Mas', 'Então', ...)."""
    total, mid = Counter(), Counter()
    for text in df["text"].fillna(""):
        for m in NAME_RE.finditer(text):
            cand = m.group(0)
            total[cand] += 1
            prefix = text[:m.start()].rstrip()
            at_start = prefix == "" or prefix.endswith(SENTENCE_START_CHARS)
            if not at_start:
                mid[cand] += 1
    rows = [{"candidate": c, "total": total[c], "mid_sentence": mid[c]}
            for c in total if mid[c] >= min_mid]
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["mid_sentence", "total"], ascending=False).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--model", default=None, help="ex.: pt_core_news_lg")
    ap.add_argument("--n-process", type=int, default=1)
    ap.add_argument("--n-annotate", type=int, default=100)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = pd.read_csv(data_dir / "paragraphs.csv")
    df["text"] = df["text"].fillna("")
    print(len(df), "parágrafos")

    nlp = load_model(args.model)
    sentences, ents = run_ner(nlp, df, n_process=args.n_process)

    sentences.to_csv(data_dir / "sentences.csv", index=False, encoding="utf-8")
    ents.to_csv(data_dir / "entities_raw.csv", index=False, encoding="utf-8")
    print(len(sentences), "frases |", len(ents), "entidades")

    print("\nEntidades por etiqueta:")
    print(ents["label"].value_counts().to_string())
    per = ents[ents["label"] == "PER"]["mention"].value_counts()
    print("\n60 menções PER mais frequentes:")
    print(per.head(60).to_string())

    # Candidatos por maiúsculas, com indicação do que o spaCy também apanhou como PER
    cands = capitalised_candidates(df)
    spacy_per = {" ".join(norm_key(m)) for m in per.index}
    cands["also_spacy_per"] = cands["candidate"].map(lambda c: " ".join(norm_key(c)) in spacy_per)
    cands.to_csv(data_dir / "candidates_capitalised.csv", index=False, encoding="utf-8")
    missed = cands[~cands["also_spacy_per"]].head(40)
    print("\nNomes em maiúscula frequentes que o spaCy NÃO marcou como PER (rever):")
    print(missed.to_string(index=False))

    # Amostra para anotação manual (precisão/recall do NER)
    per_by_par = (ents[ents["label"] == "PER"].groupby("paragraph_id")["mention"]
                  .apply(lambda s: " | ".join(s)))
    pool = df[df["n_words"] >= 10]
    sample = pool.sample(min(args.n_annotate, len(pool)), random_state=0)
    sample = sample[["paragraph_id", "chapter", "text"]].copy()
    sample["predicted_PER"] = sample["paragraph_id"].map(per_by_par).fillna("")
    sample["gold_PER"] = ""          # a preencher à mão: nomes de pessoas no parágrafo, separados por " | "
    sample.to_csv(data_dir / "to_annotate.csv", index=False, encoding="utf-8")
    print(f"\nGuardada amostra de {len(sample)} parágrafos em to_annotate.csv")


if __name__ == "__main__":
    main()
