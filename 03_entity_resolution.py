"""
03_entity_resolution.py
-----------------------
Lê  Data/paragraphs.csv, sentences.csv, entities_raw.csv  (do 02_ner.py)
e a configuração em characters_config.py.

Produz:
  Data/entities_resolved.csv     todas as menções (resolvidas ou não)
  Data/mentions_characters.csv   só as menções resolvidas -> entrada da rede de co-ocorrência
  Data/unresolved_mentions.csv   menções por resolver, com exemplo e sugestão -> o teu TRABALHO
  Data/characters_summary.csv    nº de menções, parágrafos e capítulos por personagem

O que faz, por ordem:
  1. Junta as entidades do spaCy com uma procura por dicionário (apanha nomes que o spaCy falhou).
  2. Normaliza cada menção (minúsculas, sem acentos, sem artigo nem título).
  3. Procura no dicionário de aliases; formas ambíguas usam as regras de CONTEXT_RULES.
  4. Nunca adivinha: o que não se resolve fica em unresolved_mentions.csv com uma sugestão.

Uso:
  python 03_entity_resolution.py
  python 03_entity_resolution.py --show "Maria Eduarda"   # onde aparece, por capítulo (para validar regras)
  python 03_entity_resolution.py --min-count 3 --labels PER,MISC,ORG
"""
import argparse
import bisect
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

import characters_config as cfg


# ------------------------- normalização -------------------------

def norm_text(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_mention(s):
    """Minúsculas, sem acentos/pontuação, sem artigos nem títulos no início."""
    tokens = norm_text(s).split()
    while tokens and (tokens[0] in cfg.ARTICLES or tokens[0] in cfg.TITLE_TOKENS):
        tokens = tokens[1:]
    return " ".join(tokens)


# ------------------------- dicionário -------------------------

def load_character_aliases(nodes_path):
    """Load canonical names and semicolon-separated aliases from ground truth."""
    nodes = pd.read_csv(nodes_path, dtype=str).fillna("")
    required = {"id", "name", "aliases"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing columns in {nodes_path}: {', '.join(sorted(missing))}")
    if nodes["id"].duplicated().any() or nodes["name"].duplicated().any():
        raise ValueError("Ground-truth IDs and canonical names must be unique.")

    characters = {}
    for row in nodes.itertuples(index=False):
        aliases = [alias.strip() for alias in row.aliases.split(";") if alias.strip()]
        characters[row.name.strip()] = aliases
    return characters


def build_lookup(characters):
    """lookup: forma normalizada -> nome canónico.
    raw_forms: forma tal como escrita -> nome canónico (para a procura por dicionário)."""
    lookup, raw_forms = {}, {}
    for canon, aliases in characters.items():
        for form in [canon] + list(aliases):
            key = norm_mention(form)
            if not key or key in cfg.CONTEXT_RULES:
                continue                     # formas ambíguas só se resolvem por regra
            if key in lookup and lookup[key] != canon:
                print(f"AVISO: '{form}' está em '{lookup[key]}' e em '{canon}'. Mantive o primeiro.")
                continue
            lookup[key] = canon
            if "(" not in form:
                raw_forms[form] = canon
    return lookup, raw_forms


def build_dict_regex(raw_forms):
    forms = sorted(raw_forms, key=len, reverse=True)       # formas mais longas primeiro
    if not forms:
        return None
    body = "|".join(re.escape(f) for f in forms)
    return re.compile(r"(?<![\wÀ-ÿ])(?:" + body + r")(?![\wÀ-ÿ])")


def prev_word(text, start):
    words = re.findall(r"[\wÀ-ÿ]+", text[max(0, start - 40):start])
    return norm_text(words[-1]) if words else ""


def apply_rule(rule, chapter, text, start):
    if rule["type"] == "chapter":
        return rule["before"] if chapter <= rule["upto"] else rule["after"]
    if rule["type"] == "prev_word":
        return rule["map"].get(prev_word(text, start), rule.get("default"))
    raise ValueError(f"Tipo de regra desconhecido: {rule['type']}")


def resolve(mention, chapter, text, start, lookup):
    key = norm_mention(mention)
    if not key:
        return None, "empty"
    if key in cfg.CONTEXT_RULES:
        name = apply_rule(cfg.CONTEXT_RULES[key], chapter, text, start)
        return (name, "context_rule") if name else (None, "ambiguous")
    if key in lookup:
        return lookup[key], "alias"
    return None, "unresolved"


def suggest(key, lookup):
    """Sugere personagens parecidas (nunca aplica automaticamente)."""
    toks = set(key.split())
    if not toks:
        return ""
    out = []
    for k, canon in lookup.items():
        kt = set(k.split())
        if toks < kt or (kt < toks and len(kt) >= 2):
            if canon not in out:
                out.append(canon)
    return " | ".join(out[:3])


# ------------------------- fusão de candidatos -------------------------

def select_non_overlapping(cands):
    """Entre menções sobrepostas fica a resolvida; depois a mais longa; depois a do spaCy."""
    cands = sorted(cands, key=lambda c: (c["character"] is None,
                                         -(c["end"] - c["start"]),
                                         c["source"] != "spacy",
                                         c["start"]))
    chosen = []
    for c in cands:
        if all(c["end"] <= o["start"] or c["start"] >= o["end"] for o in chosen):
            chosen.append(c)
    return sorted(chosen, key=lambda c: c["start"])


# ------------------------- ferramenta de validação -------------------------

def show(name, ents, text_by_pid):
    key = norm_mention(name)
    sub = ents[ents["mention"].map(norm_mention) == key]
    if sub.empty:
        print(f"Nenhuma menção '{name}' em entities_raw.csv")
        return
    print(f"Menções de '{name}' por capítulo:")
    print(sub.groupby("chapter").size().to_string())
    for chapter, grp in sub.groupby("chapter"):
        for r in grp.head(2).itertuples():
            t = text_by_pid[r.paragraph_id]
            print(f"\n[cap. {chapter}] ...{t[max(0, r.start_char - 70):r.end_char + 70]}...")


# ------------------------- principal -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="Data")
    ap.add_argument("--min-count", type=int, default=2, help="mínimo de menções no relatório de não resolvidas")
    ap.add_argument("--labels", default="PER,MISC,ORG", help="etiquetas do spaCy a listar como não resolvidas")
    ap.add_argument("--show", default=None, help="mostra onde aparece uma menção, por capítulo")
    args = ap.parse_args()

    d = Path(args.data_dir)
    paras = pd.read_csv(d / "paragraphs.csv")
    paras["text"] = paras["text"].fillna("")
    text_by_pid = dict(zip(paras["paragraph_id"], paras["text"]))
    chapter_by_pid = dict(zip(paras["paragraph_id"], paras["chapter"]))
    sentences = pd.read_csv(d / "sentences.csv")
    ents = pd.read_csv(d / "entities_raw.csv")
    ents["mention"] = ents["mention"].astype(str)

    if args.show:
        show(args.show, ents, text_by_pid)
        return

    starts_by_pid = {pid: sorted(g["start_char"]) for pid, g in sentences.groupby("paragraph_id")}

    def sent_for(pid, pos):
        starts = starts_by_pid.get(pid, [0])
        return max(0, bisect.bisect_right(starts, pos) - 1)

    characters = load_character_aliases(d / "os_maias_ground_truth_nodes.csv")
    lookup, raw_forms = build_lookup(characters)
    rx = build_dict_regex(raw_forms)
    print(f"{len(lookup)} formas normalizadas | {len(characters)} personagens do ground truth | "
          f"{len(cfg.CONTEXT_RULES)} regras de contexto")

    # 1. candidatos: spaCy + dicionário
    cands = defaultdict(list)
    for r in ents.itertuples():
        text = text_by_pid[r.paragraph_id]
        chapter = chapter_by_pid[r.paragraph_id]
        char, method = resolve(r.mention, chapter, text, r.start_char, lookup)
        cands[r.paragraph_id].append({
            "start": r.start_char, "end": r.end_char, "mention": r.mention, "label": r.label,
            "source": "spacy", "character": char, "method": method,
        })
    if rx:
        for pid, text in text_by_pid.items():
            for m in rx.finditer(text):
                cands[pid].append({
                    "start": m.start(), "end": m.end(), "mention": m.group(0), "label": "DICT",
                    "source": "dict", "character": raw_forms[m.group(0)], "method": "dict_alias",
                })

    # 2. escolher sem sobreposições e escrever
    rows = []
    for pid, items in cands.items():
        for c in select_non_overlapping(items):
            rows.append({
                "paragraph_id": pid,
                "sent_id": sent_for(pid, c["start"]),
                "chapter": chapter_by_pid[pid],
                "start_char": c["start"], "end_char": c["end"],
                "mention": c["mention"], "label": c["label"], "source": c["source"],
                "character": c["character"], "method": c["method"],
            })
    out = pd.DataFrame(rows).sort_values(["paragraph_id", "start_char"]).reset_index(drop=True)
    out.to_csv(d / "entities_resolved.csv", index=False, encoding="utf-8")

    resolved = out[out["character"].notna()].copy()
    resolved.to_csv(d / "mentions_characters.csv", index=False, encoding="utf-8")

    # 3. cobertura
    spacy_per = out[(out["source"] == "spacy") & (out["label"] == "PER")]
    if len(spacy_per):
        print(f"\nPER do spaCy resolvidas: {spacy_per['character'].notna().mean():.1%} "
              f"({spacy_per['character'].notna().sum()}/{len(spacy_per)})")
    print(f"Menções de personagens: {len(resolved)} "
          f"({(resolved['source'] == 'dict').sum()} só pelo dicionário)")
    n_amb = (out["method"] == "ambiguous").sum()
    if n_amb:
        print(f"Ambíguas sem regra de capítulo/contexto que resolva: {n_amb} (ver unresolved_mentions.csv)")

    # 4. relatório de não resolvidas
    labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    un = out[out["character"].isna() & out["label"].isin(labels)].copy()
    un["key"] = un["mention"].map(norm_mention)
    un = un[un["key"] != ""]
    rep = []
    for key, g in un.groupby("key"):
        if len(g) < args.min_count:
            continue
        first = g.iloc[0]
        t = text_by_pid[first["paragraph_id"]]
        rep.append({
            "normalised": key,
            "most_common_form": g["mention"].mode().iloc[0],
            "label": g["label"].mode().iloc[0],
            "count": len(g),
            "n_chapters": g["chapter"].nunique(),
            "method": g["method"].mode().iloc[0],
            "suggestion": suggest(key, lookup),
            "example": t[max(0, first["start_char"] - 50):first["end_char"] + 50].replace("\n", " "),
        })
    rep = pd.DataFrame(rep)
    if len(rep):
        rep = rep.sort_values("count", ascending=False).reset_index(drop=True)
    rep.to_csv(d / "unresolved_mentions.csv", index=False, encoding="utf-8")
    print(f"\nPor resolver (>= {args.min_count} menções): {len(rep)} formas. As 25 mais frequentes:")
    if len(rep):
        print(rep.head(25)[["most_common_form", "label", "count", "suggestion"]].to_string(index=False))

    # 5. resumo por personagem (inclui as que não aparecem nenhuma vez)
    if len(resolved):
        summ = resolved.groupby("character").agg(
            n_mentions=("mention", "size"),
            n_paragraphs=("paragraph_id", "nunique"),
            first_chapter=("chapter", "min"),
            last_chapter=("chapter", "max"),
            n_chapters=("chapter", "nunique"),
        )
    else:
        summ = pd.DataFrame(columns=["n_mentions", "n_paragraphs", "first_chapter",
                                     "last_chapter", "n_chapters"])
    summ = summ.reindex(sorted(characters)).fillna(0).astype(int)
    summ = summ.sort_values("n_mentions", ascending=False)
    summ.to_csv(d / "characters_summary.csv", encoding="utf-8")
    print("\nPersonagens por nº de menções (as 25 primeiras):")
    print(summ.head(25).to_string())
    zero = summ[summ["n_mentions"] == 0].index.tolist()
    if zero:
        print(f"\nSem nenhuma menção ({len(zero)}): {zero}")
        print("Ou não estão no texto, ou falta uma forma em ALIASES. Verifica com --show.")


if __name__ == "__main__":
    main()
