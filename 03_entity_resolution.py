"""
03_entity_resolution.py
-----------------------
Combines spaCy entities with character names and aliases found directly in
the source text. Character IDs are loaded from the ground-truth nodes CSV.

Outputs:
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


# ------------------------- normalisation -------------------------

def norm_text(s):
    """Lowercase text, remove accents, and replace punctuation with spaces."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def norm_mention(s):
    """Normalize a mention and remove leading articles and titles."""
    tokens = norm_text(s).split()
    while tokens and (tokens[0] in cfg.ARTICLES or tokens[0] in cfg.TITLE_TOKENS):
        tokens = tokens[1:]
    return " ".join(tokens)


# ------------------------- lookup -------------------------

def load_characters(nodes_path):
    """Load character IDs, canonical names, and semicolon-separated aliases."""
    nodes = pd.read_csv(nodes_path, dtype=str).fillna("")
    required = {"id", "name", "aliases"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"Missing required columns in {nodes_path}: {', '.join(sorted(missing))}")
    
    nodes["id"] = nodes["id"].str.strip()
    nodes["name"] = nodes["name"].str.strip()
    if (nodes["id"] == "").any() or (nodes["name"] == "").any():
        raise ValueError("Every character must have a non-empty ID and name.")
    if nodes["id"].duplicated().any() or nodes["name"].duplicated().any():
        raise ValueError("Character IDs and canonical names must be unique.")

    characters = {}
    for row in nodes.itertuples(index=False):
        characters[row.name] = {
            "id": row.id,
            "name": row.name,
            "aliases": [x.strip() for x in row.aliases.split(";") if x.strip()],
        }
    return characters


def build_lookup(characters):
    """Build normalized and exact-form lookups, excluding context-dependent forms."""
    lookup, raw_forms = {}, {}
    for canonical, character in characters.items():
        for form in [canonical] + character["aliases"]:
            key = norm_mention(form)
            if not key or key in cfg.CONTEXT_RULES:
                continue
            if key in lookup and lookup[key] != canonical:
                print(
                    f"WARNING: '{form}' maps to both '{lookup[key]}' and "
                    f"'{canonical}'. Keeping the first mapping."
                )
                continue
            lookup[key] = canonical
            if "(" not in form:
                raw_forms[form] = canonical
    return lookup, raw_forms


def build_dict_regex(raw_forms):
    """Build a case-sensitive regex, preferring longer forms."""
    forms = sorted(raw_forms, key=len, reverse=True)       # formas mais longas primeiro
    if not forms:
        return None
    alternatives = "|".join(re.escape(x) for x in forms)
    return re.compile(r"(?<![\wÀ-ÿ])(?:" + alternatives + r")(?![\wÀ-ÿ])")


def prev_word(text, start):
    """Return the normalized word immediately before a mention."""
    words = re.findall(r"[\wÀ-ÿ]+", text[max(0, start - 40):start])
    return norm_text(words[-1]) if words else ""


def apply_context_rule(rule, chapter, text, start):
    """Apply a chapter-based or previous-word disambiguation rule."""
    if rule["type"] == "chapter":
        return rule["before"] if chapter <= rule["upto"] else rule["after"]
    if rule["type"] == "prev_word":
        return rule["map"].get(prev_word(text, start), rule.get("default"))
    raise ValueError(f"Unknown context rule type: {rule['type']}")


def resolve_mention(mention, chapter, text, start, lookup):
    """Resolve a mention to a canonical character name, if possible."""
    key = norm_mention(mention)
    if not key:
        return None, "empty"
    if key in cfg.CONTEXT_RULES:
        name = apply_context_rule(cfg.CONTEXT_RULES[key], chapter, text, start)
        return (name, "context_rule") if name else (None, "ambiguous")
    if key in lookup:
        return lookup[key], "alias"
    return None, "unresolved"


def suggest_characters(key, lookup):
    """Suggest similar character forms without applying a match automatically."""
    tokens = set(key.split())
    suggestions = []
    for candidate_key, canonical in lookup.items():
        candidate_tokens = set(candidate_key.split())
        if tokens < candidate_tokens or (
            candidate_tokens < tokens and len(candidate_tokens) >= 2
        ):
            if canonical not in suggestions:
                suggestions.append(canonical)
    return " | ".join(suggestions[:3])


# ------------------------- fusion of candidates -------------------------

def select_non_overlapping(candidates):
    """Prefer resolved, longer, then spaCy candidates when spans overlap."""
    candidates = sorted(
        candidates,
        key=lambda c: (
            c["character"] is None,
            -(c["end"] - c["start"]),
            c["source"] != "spacy",
            c["start"],
        ),
    )
    selected = []
    for candidate in candidates:
        if all(
            candidate["end"] <= old["start"] or candidate["start"] >= old["end"]
            for old in selected
        ):
            selected.append(candidate)
    return sorted(selected, key=lambda c: c["start"])


# ------------------------- validation -------------------------

def show_mentions(name, entities, text_by_pid):
    """Display spaCy occurrences of a mention, grouped by chapter."""
    key = norm_mention(name)
    subset = entities[entities["mention"].map(norm_mention) == key]
    if subset.empty:
        print(f"No occurrence of '{name}' was found in entities_raw.csv.")
        return
    print(f"Occurrences of '{name}' by chapter:")
    print(subset.groupby("chapter").size().to_string())
    for chapter, group in subset.groupby("chapter"):
        for row in group.head(2).itertuples():
            text = text_by_pid[row.paragraph_id]
            print(f"\n[Chapter {chapter}] ...{text[max(0, row.start_char-70):row.end_char+70]}...")


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser(description="Resolve character mentions in Os Maias.")
    parser.add_argument("--data-dir", default="Data")
    parser.add_argument("--min-count", type=int, default=2) #TODO: MÍNIMO 2 NAO, queremos personagens q aparecem só 1 vez
    parser.add_argument("--labels", default="PER,MISC,ORG")
    parser.add_argument("--show", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    paragraphs = pd.read_csv(data_dir / "paragraphs.csv")
    paragraphs["text"] = paragraphs["text"].fillna("")
    text_by_pid = dict(zip(paragraphs["paragraph_id"], paragraphs["text"]))
    chapter_by_pid = dict(zip(paragraphs["paragraph_id"], paragraphs["chapter"]))
    sentences = pd.read_csv(data_dir / "sentences.csv")
    entities = pd.read_csv(data_dir / "entities_raw.csv")
    entities["mention"] = entities["mention"].astype(str)

    if args.show:
        show_mentions(args.show, entities, text_by_pid)
        return

    sentence_starts = {
        pid: sorted(group["start_char"])
        for pid, group in sentences.groupby("paragraph_id")
    }

    def sentence_for(pid, position):
        starts = sentence_starts.get(pid, [0])
        return max(0, bisect.bisect_right(starts, position) - 1)

    characters = load_characters(data_dir / "os_maias_ground_truth_nodes.csv")
    lookup, raw_forms = build_lookup(characters)
    dictionary_regex = build_dict_regex(raw_forms)
    print(
        f"Loaded {len(characters)} characters, {len(lookup)} normalized forms, "
        f"and {len(cfg.CONTEXT_RULES)} context rules."
    )

    # Collect candidates from spaCy and from direct dictionary matching.
    candidates_by_pid = defaultdict(list)
    for row in entities.itertuples():
        pid = row.paragraph_id
        text = text_by_pid[pid]
        canonical, method = resolve_mention(
            row.mention, chapter_by_pid[pid], text, row.start_char, lookup
        )
        candidates_by_pid[pid].append({
            "start": row.start_char, "end": row.end_char,
            "mention": row.mention, "label": row.label,
            "source": "spacy", "character": canonical, "method": method,
        })

    if dictionary_regex:
        for pid, text in text_by_pid.items():
            for match in dictionary_regex.finditer(text):
                candidates_by_pid[pid].append({
                    "start": match.start(), "end": match.end(),
                    "mention": match.group(0), "label": "DICT",
                    "source": "dict", "character": raw_forms[match.group(0)],
                    "method": "dict_alias",
                })

    # Merge overlapping spans and attach the stable character ID.
    columns = [
        "paragraph_id", "sent_id", "chapter", "start_char", "end_char",
        "mention", "label", "source", "character_id", "character", "method",
    ]
    rows = []
    for pid, candidates in candidates_by_pid.items():
        for candidate in select_non_overlapping(candidates):
            name = candidate["character"]
            character_id = characters[name]["id"] if name is not None else None
            rows.append({
                "paragraph_id": pid,
                "sent_id": sentence_for(pid, candidate["start"]),
                "chapter": chapter_by_pid[pid],
                "start_char": candidate["start"],
                "end_char": candidate["end"],
                "mention": candidate["mention"],
                "label": candidate["label"],
                "source": candidate["source"],
                "character_id": character_id,
                "character": name,
                "method": candidate["method"],
            })

    output = pd.DataFrame(rows, columns=columns)
    if not output.empty:
        output = output.sort_values(["paragraph_id", "start_char"]).reset_index(drop=True)
    output.to_csv(data_dir / "entities_resolved.csv", index=False, encoding="utf-8")

    resolved = output[output["character_id"].notna()].copy()
    resolved.to_csv(data_dir / "mentions_characters.csv", index=False, encoding="utf-8")

    # Report resolution coverage.
    spacy_per = output[(output["source"] == "spacy") & (output["label"] == "PER")]
    if not spacy_per.empty:
        n_resolved = spacy_per["character_id"].notna().sum()
        print(f"\nResolved spaCy PER mentions: {n_resolved/len(spacy_per):.1%} ({n_resolved}/{len(spacy_per)}).")
    n_dictionary = (resolved["source"] == "dict").sum()
    print(f"Resolved character mentions: {len(resolved)} ({n_dictionary} detected only by the dictionary).")
    n_ambiguous = (output["method"] == "ambiguous").sum()
    if n_ambiguous:
        print(f"Unresolved ambiguous mentions: {n_ambiguous}. See unresolved_mentions.csv.")

    # Create a frequency-based report of unresolved spaCy entities.
    labels = {x.strip() for x in args.labels.split(",") if x.strip()}
    unresolved = output[
        output["character_id"].isna() & output["label"].isin(labels)
    ].copy()
    unresolved["key"] = unresolved["mention"].map(norm_mention)
    unresolved = unresolved[unresolved["key"] != ""]
    report_rows = []
    for key, group in unresolved.groupby("key"):
        if len(group) < args.min_count:
            continue
        first = group.iloc[0]
        text = text_by_pid[first["paragraph_id"]]
        start = max(0, first["start_char"] - 50)
        end = first["end_char"] + 50
        report_rows.append({
            "normalised": key,
            "most_common_form": group["mention"].mode().iloc[0],
            "label": group["label"].mode().iloc[0],
            "count": len(group),
            "n_chapters": group["chapter"].nunique(),
            "method": group["method"].mode().iloc[0],
            "suggestion": suggest_characters(key, lookup),
            "example": text[start:end].replace("\n", " "),
        })

    report_columns = [
        "normalised", "most_common_form", "label", "count",
        "n_chapters", "method", "suggestion", "example",
    ]
    report = pd.DataFrame(report_rows, columns=report_columns)
    if not report.empty:
        report = report.sort_values("count", ascending=False).reset_index(drop=True)
    report.to_csv(data_dir / "unresolved_mentions.csv", index=False, encoding="utf-8")
    print(f"\nUnresolved forms with at least {args.min_count} mentions: {len(report)}. Top 25:")
    if not report.empty:
        print(report.head(25)[["most_common_form", "label", "count", "suggestion"]].to_string(index=False))

    # Summarize every ground-truth character, including those with no mentions.
    if not resolved.empty:
        summary = resolved.groupby(["character_id", "character"]).agg(
            n_mentions=("mention", "size"),
            n_paragraphs=("paragraph_id", "nunique"),
            first_chapter=("chapter", "min"),
            last_chapter=("chapter", "max"),
            n_chapters=("chapter", "nunique"),
        )
    else:
        summary = pd.DataFrame(columns=[
            "n_mentions", "n_paragraphs", "first_chapter", "last_chapter", "n_chapters"
        ])
        summary.index = pd.MultiIndex.from_arrays([[], []], names=["character_id", "character"])

    all_characters = pd.DataFrame([
        {"character_id": item["id"], "character": item["name"]}
        for item in characters.values()
    ]).set_index(["character_id", "character"])
    summary = all_characters.join(summary, how="left").fillna(0).reset_index()
    summary = summary.astype({
        "n_mentions": int, "n_paragraphs": int, "first_chapter": int,
        "last_chapter": int, "n_chapters": int,
    })
    summary = summary.sort_values("n_mentions", ascending=False)
    summary.to_csv(data_dir / "characters_summary.csv", index=False, encoding="utf-8")

    print("\nCharacters ranked by number of mentions (top 25):")
    print(summary.head(25).to_string(index=False))
    no_mentions = summary.loc[summary["n_mentions"] == 0, "character"].tolist()
    if no_mentions:
        print(f"\nCharacters with no detected mentions ({len(no_mentions)}): {no_mentions}")
        print("They may not occur in the text, or their names/aliases may be missing from the nodes CSV.")


if __name__ == "__main__":
    main()
