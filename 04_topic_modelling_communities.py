"""Experiência 1: perfis temáticos das comunidades de personagens.

O modelo é treinado globalmente sobre parágrafos associados a pelo menos uma
comunidade. Um parágrafo pode pertencer a várias comunidades. A associação é
feita pelo número de personagens *distintas* da comunidade mencionadas no
parágrafo; não pelo número de menções.

Exemplo:
    python 04_topic_modelling_communities.py --threshold 1 --n-topics 8

Os resultados são escritos em ``outputs/topic_modelling_experiment1`` por
defeito. Para comparar os critérios de selecção, corra o script com
``--threshold 1``, ``--threshold 2`` e ``--threshold 3``, cada um com um
directório de saída diferente.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Lista pequena e explícita para português; os termos específicos da obra não
# são removidos, excepto nomes de personagens e aliases (ver anonymise_text).
# PORTUGUESE_STOPWORDS = {
#     "a", "à", "ao", "aos", "as", "às", "com", "como", "da", "das", "de",
#     "dela", "dele", "deles", "depois", "do", "dos", "duma", "dum", "dumas", "duns",
#     "e", "ela", "ele", "em", "num", "numa", "nuns", "numas", 
#     "entre", "era", "essa", "esse", "esta", "este", "eu", "foi", "foram", "há",
#     "lhe", "lhes", "mais", "mas", "me", "mesmo", "mim", "na", "não", "nas",
#     "nem", "no", "nos", "o", "os", "ou", "para", "pela", "pelo", "por", "que",
#     "se", "sem", "ser", "seu", "seus", "sua", "suas", "também", "te", "tem",
#     "tinha", "um", "uma", "você", "vos", "já", "é", "são", "àquela", "aquele",
#     "aquela", "aqueles", "aquelas", "isto", "isso", "ainda", "muito", "muito",
#     "sobre", "onde", "quando", "tão", "tanto", "assim", "então", "porque", "pois", "dessa", "desse",
#     "tudo", "todo","toda", "todos", "nada", "cada", "outro", "outra", "outros", "outras", "além", "através",
#     "bem", "cima", "baixo", "longe", "perto", "aí", "aqui", "ali", "lá", "logo",
#     "enfim", "agora", "antes", "sempre", "nunca", "talvez", "quase",
# }

# Additional stopwords specific to the literary corpus.
CUSTOM_STOPWORDS = {
    # Generic narrative vocabulary
    "coisa", "coisas", "homem", "mulher", "pessoa", "pessoas",
    "senhor", "senhora", "amigo", "amiga",

    # Frequent physical descriptions
    "olho", "olhos", "mão", "mãos", "braço", "braços",
    "rosto", "face", "ombro", "ombros",

    # Generic spatial and temporal vocabulary
    "lado", "porta", "mesa", "sala", "quarto",
    "momento", "instante", "fundo", "vez", "dia", "noite", "ano", "tempo",
    "voz", "palavra", "modo", "maneira", "ar", "cousa",

    # Generic narrative adjectives
    "grande", "pequeno", "pequena", "novo", "nova", "bom", "boa",

    # Generic verbs that may survive as nominal forms
    "fazer", "dizer", "ter", "estar", "ser",
}

ALLOWED_POS = {"NOUN", "ADJ"}


def preprocess_texts(texts: pd.Series, spacy_model: str) -> list[str]:
    """Lematiza texto e mantém apenas substantivos/adjetivos informativos."""
    try:
        nlp = spacy.load(spacy_model)
    except OSError as exc:
        raise RuntimeError(
            f"O modelo spaCy '{spacy_model}' não está instalado. Instale-o com:\n"
            f"  python -m spacy download {spacy_model}"
        ) from exc

    # pt_core_news_lg usa actualmente ``morphologizer`` (e não ``tagger``).
    if not (nlp.has_pipe("tagger") or nlp.has_pipe("morphologizer")) or not nlp.has_pipe("lemmatizer"):
        raise RuntimeError(
            f"O modelo spaCy '{spacy_model}' não inclui POS tagging e lematização. "
            "Use, por exemplo, pt_core_news_lg."
        )
    processed_texts = []
    for doc in nlp.pipe(
        texts.fillna("").astype(str),
        batch_size=64
    ):
        tokens = []
        for token in doc:
            if not token.is_alpha:
                continue
            if token.pos_ not in ALLOWED_POS:
                continue
            lemma = token.lemma_.lower().strip()
            if len(lemma) < 3:
                continue
            if token.is_stop or lemma in CUSTOM_STOPWORDS:
                continue
            tokens.append(lemma)
        processed_texts.append(" ".join(tokens))
    return processed_texts


def normalise(value: object) -> str:
    """Normaliza texto para correspondência de nomes, preservando só letras/números."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def split_values(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip()]


def read_csv(path: Path) -> pd.DataFrame:
    """Lê os CSV do projecto (UTF-8; aceita BOM)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def build_character_lookup(nodes: pd.DataFrame) -> dict[str, str]:
    """Cria índice de nome canónico/alias -> character_id."""
    lookup: dict[str, str] = {}
    for row in nodes.itertuples(index=False):
        character_id = str(row.id)
        for name in [row.name, character_id, *split_values(row.aliases)]:
            key = normalise(name)
            if not key:
                continue
            # Não adivinhamos em aliases ambíguos: o ficheiro de comunidades
            # deve usar o nome canónico nesses casos.
            if key in lookup and lookup[key] != character_id:
                lookup[key] = "__AMBIGUOUS__"
            else:
                lookup[key] = character_id
    return lookup


def load_communities(path: Path, lookup: dict[str, str]) -> dict[str, set[str]]:
    communities = read_csv(path)
    required = {"community_id", "characters"}
    if not required.issubset(communities.columns):
        raise ValueError(f"{path} tem de conter as colunas {sorted(required)}.")

    resolved: dict[str, set[str]] = {}
    errors: list[str] = []
    for row in communities.itertuples(index=False):
        ids: set[str] = set()
        for name in split_values(row.characters):
            character_id = lookup.get(normalise(name))
            if character_id is None:
                errors.append(f"{row.community_id}: personagem desconhecida: {name}")
            elif character_id == "__AMBIGUOUS__":
                errors.append(f"{row.community_id}: alias ambíguo: {name}; use o nome canónico")
            else:
                ids.add(character_id)
        if ids:
            resolved[str(row.community_id)] = ids

    if errors:
        raise ValueError("Não foi possível resolver comunidades:\n- " + "\n- ".join(errors))
    return resolved


def make_name_pattern(nodes: pd.DataFrame, mentions: pd.DataFrame) -> re.Pattern[str] | None:
    """Constrói um padrão para eliminar nomes, aliases e formas observadas pela NER."""
    forms: set[str] = set()
    for row in nodes.itertuples(index=False):
        forms.add(str(row.name))
        forms.update(split_values(row.aliases))
    forms.update(mentions["mention"].dropna().astype(str))
    forms = {form.strip() for form in forms if len(form.strip()) >= 2}
    if not forms:
        return None
    alternatives = "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", flags=re.IGNORECASE)


def anonymise_text(text: str, name_pattern: re.Pattern[str] | None) -> str:
    text = name_pattern.sub(" ", text) if name_pattern else text
    return re.sub(r"\s+", " ", text).strip()


def topic_terms(model: LatentDirichletAllocation, vectorizer: CountVectorizer, n_terms: int) -> pd.DataFrame:
    vocabulary = vectorizer.get_feature_names_out()
    rows = []
    for topic_id, weights in enumerate(model.components_):
        indices = weights.argsort()[-n_terms:][::-1]
        for rank, index in enumerate(indices, start=1):
            rows.append({
                "topic_id": topic_id,
                "rank": rank,
                "term": vocabulary[index],
                "weight": float(weights[index]),
            })
    return pd.DataFrame(rows)

def membership_mask(documents: pd.DataFrame, community_id: str) -> pd.Series:
    return documents["community_id_list"].map(lambda ids: community_id in ids)


def make_community_profiles(
    documents: pd.DataFrame,
    communities: dict[str, set[str]],
    topic_column: str,
    probability_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Agrega atribuições de tópicos por comunidade, preservando sobreposição."""
    rows = []
    topic_ids = sorted(documents[topic_column].unique())
    for community_id in communities:
        community_docs = documents.loc[membership_mask(documents, community_id)]
        for topic_id in topic_ids:
            row = {
                "community_id": community_id,
                "topic_id": int(topic_id),
                "n_paragraphs": len(community_docs),
                "n_paragraphs_with_topic": int((community_docs[topic_column] == topic_id).sum()),
                "dominant_topic_share": float((community_docs[topic_column] == topic_id).mean()),
            }
            if probability_columns is not None and topic_id >= 0:
                row["mean_topic_probability"] = float(community_docs[probability_columns[topic_id]].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def run_bertopic(documents: pd.DataFrame, communities: dict[str, set[str]], args) -> None:
    """Executa BERTopic nos mesmos parágrafos usados por LDA."""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "BERTopic não está instalado. Instale as dependências com:\n"
            "  python -m pip install bertopic sentence-transformers hdbscan umap-learn"
        ) from exc

    # Use anonymised original paragraphs for semantic embeddings.
    semantic_docs = documents["clean_text"].tolist()

    # Use preprocessed text for topic representation.
    representation_docs = documents["processed_text"].tolist()

    embedding_model = SentenceTransformer(args.embedding_model)

    embeddings = embedding_model.encode(
        semantic_docs,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )

    vectorizer = CountVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]{3,}\b",
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_df=args.max_df,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        language="portuguese",
        min_topic_size=args.min_topic_size,
        nr_topics=args.n_topics + 1,  # +1 to allow for outliers
        top_n_words=args.top_terms,
        calculate_probabilities=False,
        verbose=True,
    )

    topics, probabilities = topic_model.fit_transform(
        representation_docs,
        embeddings,
    )

    documents["bertopic_topic"] = topics
    documents["bertopic_probability"] = probabilities

    profiles = make_community_profiles(documents, communities, "bertopic_topic")
    profiles.to_csv(
        args.output_dir / "bertopic_community_topic_profiles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save document assignments.
    documents.drop(columns="community_id_list").to_csv(
        args.output_dir / "bertopic_documents.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save topic information.
    topic_info = topic_model.get_topic_info()

    topic_info.to_csv(
        args.output_dir / "bertopic_topics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save individual topic terms.
    topic_rows = []

    for topic_id in topic_model.get_topic_info()["Topic"]:
        if topic_id == -1:
            continue

        for rank, (term, weight) in enumerate(
            topic_model.get_topic(topic_id),
            start=1,
        ):
            topic_rows.append({
                "topic_id": topic_id,
                "rank": rank,
                "term": term,
                "weight": weight,
            })

    pd.DataFrame(topic_rows).to_csv(
        args.output_dir / "bertopic_topic_terms.csv",
        index=False,
        encoding="utf-8-sig",
    )

    topic_model.save(str(args.output_dir / "bertopic_model"), serialization="safetensors", save_ctfidf=True)

    print(topic_info.to_string(index=False))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paragraphs", type=Path, default=Path("Data/paragraphs.csv"))
    parser.add_argument("--mentions", type=Path, default=Path("Data/mentions_characters.csv"))
    parser.add_argument("--nodes", type=Path, default=Path("Data/os_maias_ground_truth_nodes.csv"))
    parser.add_argument("--communities", type=Path, default=Path("Data/louvain_communities_by_community.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/topic_modelling_experiment1"))
    parser.add_argument("--threshold", type=int, default=1, help="Mínimo de personagens distintas da comunidade por parágrafo.")
    parser.add_argument("--n-topics", type=int, default=8)
    parser.add_argument("--min-df", type=int, default=3, help="Número mínimo de documentos que contêm um termo.")
    parser.add_argument("--max-df", type=float, default=0.50)
    parser.add_argument("--top-terms", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model", choices=("lda", "bertopic", "both"), default="lda",
                        help="Modelo a executar. 'both' corre LDA e BERTopic nos mesmos documentos.")
    parser.add_argument("--spacy-model", default="pt_core_news_lg")
    parser.add_argument("--embedding-model", default="iara-project/BERTimbau-large-matryoshka-sts-pt",
                        help="Modelo SentenceTransformers usado apenas por BERTopic.")
    parser.add_argument("--min-topic-size", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold < 1 or args.n_topics < 2:
        raise ValueError("--threshold deve ser >= 1 e --n-topics deve ser >= 2.")

    paragraphs = read_csv(args.paragraphs)
    mentions = read_csv(args.mentions)
    nodes = read_csv(args.nodes)
    for frame, columns, name in [
        (paragraphs, {"paragraph_id", "chapter", "text"}, "paragraphs"),
        (mentions, {"paragraph_id", "character_id", "mention"}, "mentions"),
        (nodes, {"id", "name", "aliases"}, "nodes"),
    ]:
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{name} não contém as colunas obrigatórias: {sorted(missing)}")

    communities = load_communities(args.communities, build_character_lookup(nodes))
    paragraph_characters = (
        mentions.dropna(subset=["character_id"])
        .groupby("paragraph_id")["character_id"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )

    memberships: dict[int, list[str]] = defaultdict(list)
    counts_by_membership: dict[tuple[int, str], int] = {}
    for paragraph_id, characters in paragraph_characters.items():
        for community_id, community_characters in communities.items():
            count = len(characters & community_characters)
            if count >= args.threshold:
                paragraph_id = int(paragraph_id)
                memberships[paragraph_id].append(community_id)
                counts_by_membership[(paragraph_id, community_id)] = count

    documents = paragraphs[paragraphs["paragraph_id"].isin(memberships)].copy()
    if len(documents) < args.n_topics:
        raise ValueError(
            f"Só foram seleccionados {len(documents)} parágrafos; são necessários pelo menos "
            f"{args.n_topics} para {args.n_topics} tópicos. Reduza --n-topics ou --threshold."
        )
    name_pattern = make_name_pattern(nodes, mentions)
    documents["clean_text"] = documents["text"].fillna("").map(lambda text: anonymise_text(text, name_pattern))
    documents["processed_text"] = preprocess_texts(documents["clean_text"], args.spacy_model)
    documents = documents[documents["processed_text"].str.strip().ne("")].copy()
    if len(documents) < args.n_topics:
        raise ValueError(
            f"Só restaram {len(documents)} parágrafos após o pré-processamento; "
            "reduza --n-topics ou reveja os filtros."
        )

    documents["community_id_list"] = documents["paragraph_id"].map(lambda identifier: memberships[int(identifier)])
    documents["community_ids"] = documents["community_id_list"].map("; ".join)
    documents["community_character_counts"] = documents.apply(
        lambda row: json.dumps({community: counts_by_membership[(int(row.paragraph_id), community)]
                                for community in memberships[int(row.paragraph_id)]}, ensure_ascii=False),
        axis=1,
    )

    vectorizer = CountVectorizer(
        # lowercase=True,
        # stop_words=list(PORTUGUESE_STOPWORDS),
        # token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]{3,}\b",
        # min_df=args.min_df,
        # max_df=args.max_df,
        lowercase=False,
        token_pattern=r"(?u)\b[a-záàâãéêíóôõúüç]{3,}\b",
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_range=(1, 2),
    )
    matrix = None
    if args.model in {"lda", "both"}:
        matrix = vectorizer.fit_transform(documents["processed_text"])
        if matrix.shape[1] == 0:
            raise ValueError("O vocabulário ficou vazio; reduza --min-df ou reveja a limpeza do texto.")

        model = LatentDirichletAllocation(
            n_components=args.n_topics,
            learning_method="batch",
            max_iter=50,
            random_state=args.random_state,
            n_jobs=-1,
        )
        topic_distribution = model.fit_transform(matrix)
        documents["dominant_topic"] = topic_distribution.argmax(axis=1)
        probability_columns = []
        for topic_id in range(args.n_topics):
            column = f"topic_{topic_id}_probability"
            documents[column] = topic_distribution[:, topic_id]
            probability_columns.append(column)
        profiles = make_community_profiles(documents, communities, "dominant_topic", probability_columns)
    selection = pd.DataFrame([
        {
            "community_id": community_id,
            "community_size": len(character_ids),
            "candidate_paragraphs": sum(community_id in ids for ids in memberships.values()),
            "selected_paragraphs": int(membership_mask(documents, community_id).sum()),
        }
        for community_id, character_ids in communities.items()
    ])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    documents.drop(columns="community_id_list").to_csv(
        args.output_dir / "selected_paragraphs.csv", index=False, encoding="utf-8-sig"
    )
    selection.to_csv(args.output_dir / "community_selection_summary.csv", index=False, encoding="utf-8-sig")
    if args.model in {"lda", "both"}:
        profiles.to_csv(args.output_dir / "community_topic_profiles.csv", index=False, encoding="utf-8-sig")
        topic_terms(model, vectorizer, args.top_terms).to_csv(args.output_dir / "topics.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump({
            "threshold": args.threshold,
            "n_topics": args.n_topics,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "random_state": args.random_state,
            "model": args.model,
            "spacy_model": args.spacy_model,
            "embedding_model": args.embedding_model if args.model in {"bertopic", "both"} else None,
            "n_selected_documents": len(documents),
            "n_vocabulary_terms": matrix.shape[1] if matrix is not None else None,
            "communities_file": str(args.communities),
        }, file, indent=2, ensure_ascii=False)

    if args.model in {"lda", "both"}:
        print(f"LDA treinado com {len(documents)} parágrafos e {matrix.shape[1]} termos.")
    if args.model in {"bertopic", "both"}:
        run_bertopic(documents, communities, args)
    print(f"Resultados: {args.output_dir}")
    print(selection.to_string(index=False))


if __name__ == "__main__":
    main()
