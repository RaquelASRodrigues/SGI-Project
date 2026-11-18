"""Prepare and apply manual review decisions for unresolved spaCy PER mentions.

Run after ``03_entity_resolution.py``:

    python review_per_mentions.py --prepare

By default, only ambiguous/unresolved PER mention forms occurring at least
twice are included. Use ``--min-mentions 3`` to require three occurrences.

Edit ``Data/per_review.csv`` in a spreadsheet:
    include = yes / no / blank
    character = optional canonical character name; blank uses the NER mention

Then apply the decisions:

    python review_per_mentions.py --apply

Included records are added to ``Data/mentions_characters.csv``. Excluded and
pending records are not added. The review file is refreshed without discarding
existing decisions when ``--prepare`` is run again.
"""

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["paragraph_id", "start_char", "end_char", "mention"]
REVIEW_COLUMNS = [
    "paragraph_id",
    "chapter",
    "start_char",
    "end_char",
    "mention",
    "method",
    "suggestion",
    "context",
    "include",
    "character",
    "review_status",
]
REVIEW_METHODS = {"ambiguous", "unresolved"}
ARTICLES = {"o", "a", "os", "as", "um", "uma"}
TITLE_TOKENS = {
    "sr", "sra", "d", "dr", "dra", "dom", "dona",
    "senhor", "senhora", "mr", "mrs",
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read project CSVs written as UTF-8, including optional Excel BOMs."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def occurrence_key(frame: pd.DataFrame) -> pd.Series:
    """Build a stable key that survives CSV type inference."""
    return frame[KEY_COLUMNS].astype(str).agg("\x1f".join, axis=1)


def prepare_review(data_dir: Path, min_mentions: int = 2) -> None:
    entities = read_csv(data_dir / "entities_resolved.csv")
    paragraphs = read_csv(data_dir / "paragraphs.csv")
    unresolved = read_csv(data_dir / "unresolved_mentions.csv")

    required_entities = set(KEY_COLUMNS + ["chapter", "source", "label", "method"])
    missing = required_entities - set(entities.columns)
    if missing:
        raise ValueError(f"entities_resolved.csv is missing columns: {sorted(missing)}")
    if not {"paragraph_id", "text"} <= set(paragraphs.columns):
        raise ValueError("paragraphs.csv must contain paragraph_id and text columns.")
    if not {"normalised", "suggestion"} <= set(unresolved.columns):
        raise ValueError("unresolved_mentions.csv must contain normalised and suggestion columns.")

    candidates = entities[
        entities["source"].eq("spacy")
        & entities["label"].eq("PER")
        & entities["method"].isin(REVIEW_METHODS)
    ].copy()
    if min_mentions < 2:
        raise ValueError("min_mentions must be at least 2.")

    mention_counts = candidates["mention"].map(normalize_name).value_counts()
    repeated_forms = set(mention_counts[mention_counts >= min_mentions].index)
    candidates = candidates[candidates["mention"].map(normalize_name).isin(repeated_forms)].copy()
    if candidates.duplicated(KEY_COLUMNS).any():
        raise ValueError("Ambiguous/unresolved PER occurrences have duplicate occurrence keys.")

    paragraph_text = dict(zip(paragraphs["paragraph_id"].astype(str), paragraphs["text"]))
    suggestion_by_form = (
        unresolved.drop_duplicates("normalised")
        .set_index("normalised")["suggestion"]
        .to_dict()
    )

    contexts = []
    suggestions = []
    for row in candidates.itertuples(index=False):
        text = paragraph_text.get(str(row.paragraph_id), "")
        try:
            start = int(row.start_char)
            end = int(row.end_char)
        except ValueError as exc:
            raise ValueError(
                f"Invalid character offsets for paragraph {row.paragraph_id}: "
                f"{row.start_char}, {row.end_char}"
            ) from exc
        contexts.append(text[max(0, start - 120):min(len(text), end + 120)])
        suggestions.append(
            suggestion_by_form.get(normalize_name(row.mention), "")
        )

    review = pd.DataFrame({
        "paragraph_id": candidates["paragraph_id"].astype(str),
        "chapter": candidates["chapter"],
        "start_char": candidates["start_char"].astype(str),
        "end_char": candidates["end_char"].astype(str),
        "mention": candidates["mention"],
        "method": candidates["method"],
        "suggestion": suggestions,
        "context": contexts,
    })
    review["include"] = ""
    review["character"] = ""
    review["review_status"] = "active"
    review = review[REVIEW_COLUMNS]

    review_path = data_dir / "per_review.csv"
    if review_path.exists():
        old = read_csv(review_path)
        missing_old = set(KEY_COLUMNS + ["include", "character"]) - set(old.columns)
        if missing_old:
            raise ValueError(f"Existing per_review.csv is missing columns: {sorted(missing_old)}")
        if old.duplicated(KEY_COLUMNS).any():
            raise ValueError("Existing per_review.csv has duplicate occurrence keys.")

        old["__key"] = occurrence_key(old)
        review["__key"] = occurrence_key(review)
        old_decisions = old.set_index("__key")[["include", "character"]]
        review["include"] = review["__key"].map(old_decisions["include"]).fillna("")
        review["character"] = review["__key"].map(old_decisions["character"]).fillna("")

        current_keys = set(review["__key"])
        stale = old[~old["__key"].isin(current_keys)].copy()
        if not stale.empty:
            stale = stale[stale["mention"].map(normalize_name).isin(repeated_forms)].copy()
            for column in REVIEW_COLUMNS:
                if column not in stale.columns:
                    stale[column] = ""
            stale["review_status"] = "stale"
            review = pd.concat(
                [review.drop(columns="__key"), stale[REVIEW_COLUMNS]],
                ignore_index=True,
            )
        else:
            review = review.drop(columns="__key")

    review.to_csv(review_path, index=False, encoding="utf-8-sig")
    active_count = int(review["review_status"].eq("active").sum())
    stale_count = int(review["review_status"].eq("stale").sum())
    print(f"Review spreadsheet ready: {review_path}")
    print(f"Active occurrences: {active_count}; preserved stale decisions: {stale_count}")
    print("Fill include with yes or no; leave blank to keep that occurrence out of the network.")


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    tokens = re.sub(r"[^a-z0-9]+", " ", value.casefold()).split()
    while tokens and tokens[0] in ARTICLES | TITLE_TOKENS:
        tokens.pop(0)
    return " ".join(tokens)


def build_character_lookup(nodes: pd.DataFrame) -> dict[str, tuple[str, str]]:
    required = {"id", "name", "aliases"}
    missing = required - set(nodes.columns)
    if missing:
        raise ValueError(f"os_maias_ground_truth_nodes.csv is missing columns: {sorted(missing)}")

    forms: dict[str, set[tuple[str, str]]] = {}
    for row in nodes.itertuples(index=False):
        names = [row.name] + [alias.strip() for alias in row.aliases.split(";") if alias.strip()]
        for name in names:
            key = normalize_name(name)
            forms.setdefault(key, set()).add((row.id, row.name))

    return {
        key: next(iter(matches))
        for key, matches in forms.items()
        if len(matches) == 1
    }


def apply_review(data_dir: Path) -> None:
    entities = read_csv(data_dir / "entities_resolved.csv")
    review = read_csv(data_dir / "per_review.csv")
    nodes = read_csv(data_dir / "os_maias_ground_truth_nodes.csv")

    required_entities = set(KEY_COLUMNS + ["chapter", "sent_id", "label", "source", "character_id", "character", "method"])
    missing = required_entities - set(entities.columns)
    if missing:
        raise ValueError(f"entities_resolved.csv is missing columns: {sorted(missing)}")
    if set(KEY_COLUMNS + ["include", "character"]) - set(review.columns):
        raise ValueError("per_review.csv must contain occurrence key, include, and character columns.")

    reviewable = entities[
        entities["source"].eq("spacy")
        & entities["label"].eq("PER")
        & entities["method"].isin(REVIEW_METHODS)
    ].copy()
    if review.duplicated(KEY_COLUMNS).any():
        raise ValueError("per_review.csv has duplicate occurrence keys.")

    review_by_key = review.copy()
    review_by_key["__key"] = occurrence_key(review_by_key)
    decisions = dict(zip(review_by_key["__key"], zip(review_by_key["include"], review_by_key["character"])))
    reviewable["__key"] = occurrence_key(reviewable)

    missing_reviews = set(reviewable["__key"]) - set(decisions)
    if missing_reviews:
        raise ValueError(
            f"{len(missing_reviews)} current unresolved PER occurrences are missing from per_review.csv. "
            "Run with --prepare first."
        )

    decision_values = reviewable["__key"].map(lambda key: decisions[key][0].strip().casefold())
    invalid = sorted(set(decision_values) - {"", "yes", "no"})
    if invalid:
        raise ValueError(f"Invalid include values: {invalid}. Use yes, no, or blank.")

    selected = reviewable[decision_values.eq("yes")].copy()
    overrides = selected["__key"].map(lambda key: decisions[key][1].strip())
    selected["character"] = [
        override if override else mention
        for override, mention in zip(overrides, selected["mention"])
    ]

    character_lookup = build_character_lookup(nodes)
    selected_ids = []
    canonical_names = []
    unmatched_names = set()
    for name in selected["character"]:
        match = character_lookup.get(normalize_name(name))
        if match is None:
            selected_ids.append("")
            canonical_names.append(name)
            unmatched_names.add(name)
        else:
            character_id, canonical_name = match
            selected_ids.append(character_id)
            canonical_names.append(canonical_name)
    selected["character_id"] = selected_ids
    selected["character"] = canonical_names
    selected["method"] = "manual_review"

    output_columns = list(entities.columns)
    resolved_entities = entities[entities["character_id"].astype(str).str.strip().ne("")].copy()
    combined = pd.concat(
        [resolved_entities, selected[output_columns]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(KEY_COLUMNS + ["character_id", "character"], keep="last")
    combined = combined.sort_values(["paragraph_id", "start_char"], kind="stable").reset_index(drop=True)
    output_path = data_dir / "mentions_characters.csv"
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")

    if "review_status" in review.columns:
        active_review = review[review["review_status"].eq("active")]
    else:
        active_review = review
    pending_count = int(active_review["include"].astype(str).str.strip().eq("").sum())
    excluded_count = int(active_review["include"].astype(str).str.strip().str.casefold().eq("no").sum())
    print(f"Updated mentions file: {output_path}")
    print(f"Added reviewed occurrences: {len(selected)}; excluded: {excluded_count}; pending: {pending_count}")
    if unmatched_names:
        print(
            "Included names without a ground-truth character ID: "
            + ", ".join(sorted(unmatched_names))
            + ". Add them as nodes or aliases if ID-based exports are required."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true", help="create/refresh Data/per_review.csv")
    action.add_argument("--apply", action="store_true", help="apply decisions to Data/mentions_characters.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=2,
        help="for --prepare, keep only PER mentions normalized with at least this many occurrences (default: 2)",
    )
    args = parser.parse_args()

    if args.prepare:
        prepare_review(args.data_dir, args.min_mentions)
    else:
        apply_review(args.data_dir)


if __name__ == "__main__":
    main()
