"""
Character co-occurrence network functions for Os Maias.

The module uses the canonical `character` column in the mentions CSV.
It builds:
  1. Raw weighted network: paragraph_bonus * shared_paragraphs
     + sentence_bonus * shared_sentences.
  2. Jaccard network: shared_paragraphs / union of paragraph appearances.
  3. Cosine network: shared_paragraphs / sqrt(paragraph_frequency_i *
     paragraph_frequency_j).

The Jaccard and cosine measures are paragraph-presence similarities. They
intentionally do not include the sentence bonus, so they are alternative
frequency-adjusted definitions of edge strength.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_cooccurrences(mentions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate pairwise shared paragraph and sentence counts.

    A character is counted once per paragraph/sentence, regardless of how
    many times its name occurs within that unit.
    """
    required = {"paragraph_id", "sent_id", "character"}
    missing = required - set(mentions.columns)
    if missing:
        raise ValueError(f"Mentions DataFrame is missing: {sorted(missing)}")

    paragraph_characters = (
        mentions.groupby("paragraph_id")["character"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )

    sentence_characters = (
        mentions.groupby(["paragraph_id", "sent_id"])["character"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )

    shared_paragraphs = defaultdict(int)
    for characters in paragraph_characters.values():
        for a, b in combinations(characters, 2):
            shared_paragraphs[(a, b)] += 1

    shared_sentences = defaultdict(int)
    for characters in sentence_characters.values():
        for a, b in combinations(characters, 2):
            shared_sentences[(a, b)] += 1

    pairs = sorted(set(shared_paragraphs) | set(shared_sentences))
    rows = [
        {
            "character_1": a,
            "character_2": b,
            "shared_paragraphs": int(shared_paragraphs[(a, b)]),
            "shared_sentences": int(shared_sentences[(a, b)]),
        }
        for a, b in pairs
    ]
    edges = pd.DataFrame(
        rows,
        columns=[
            "character_1", "character_2",
            "shared_paragraphs", "shared_sentences"
        ],
    )

    # Character-level frequency table, based on distinct textual units.
    stats = []
    for character, group in mentions.groupby("character"):
        stats.append({
            "character": character,
            "total_mentions": int(len(group)),
            "paragraph_frequency": int(group["paragraph_id"].nunique()),
            "sentence_frequency": int(
                group[["paragraph_id", "sent_id"]].drop_duplicates().shape[0]
            ),
            "chapter_count": int(group["chapter"].nunique()),
        })
    nodes = pd.DataFrame(stats)

    if not edges.empty:
        frequency_map = nodes.set_index("character")["paragraph_frequency"]
        edges["paragraph_frequency_1"] = edges["character_1"].map(frequency_map)
        edges["paragraph_frequency_2"] = edges["character_2"].map(frequency_map)

    return nodes, edges


def build_networks(
    mentions: pd.DataFrame,
    paragraph_bonus: float = 1.0,
    sentence_bonus: float = 2.0,
) -> tuple[nx.Graph, nx.Graph, nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    Build raw, Jaccard and cosine character co-occurrence networks.

    Returns:
        G_raw, G_jaccard, G_cosine, nodes_df, edges_df
    """
    if paragraph_bonus < 0 or sentence_bonus < 0:
        raise ValueError("Weight bonuses must be non-negative.")

    nodes_df, edges_df = calculate_cooccurrences(mentions)

    graphs = {
        "raw": nx.Graph(name="Raw paragraph + sentence co-occurrence"),
        "jaccard": nx.Graph(name="Jaccard paragraph overlap"),
        "cosine": nx.Graph(name="Cosine paragraph overlap"),
    }

    # All networks have the same nodes and node attributes.
    for _, row in nodes_df.iterrows():
        attrs = {
            "total_mentions": int(row["total_mentions"]),
            "paragraph_frequency": int(row["paragraph_frequency"]),
            "sentence_frequency": int(row["sentence_frequency"]),
            "chapter_count": int(row["chapter_count"]),
        }
        for graph in graphs.values():
            graph.add_node(str(row["character"]), **attrs)

    if not edges_df.empty:
        edges_df = edges_df.copy()
        edges_df["raw_weight"] = (
            paragraph_bonus * edges_df["shared_paragraphs"]
            + sentence_bonus * edges_df["shared_sentences"]
        )

        union = (
            edges_df["paragraph_frequency_1"]
            + edges_df["paragraph_frequency_2"]
            - edges_df["shared_paragraphs"]
        )
        edges_df["jaccard_weight"] = (
            edges_df["shared_paragraphs"] / union
        )

        denominator = np.sqrt(
            edges_df["paragraph_frequency_1"]
            * edges_df["paragraph_frequency_2"]
        )
        edges_df["cosine_weight"] = (
            edges_df["shared_paragraphs"] / denominator
        )

        for _, row in edges_df.iterrows():
            a = str(row["character_1"])
            b = str(row["character_2"])
            common_attrs = {
                "shared_paragraphs": int(row["shared_paragraphs"]),
                "shared_sentences": int(row["shared_sentences"]),
            }
            graphs["raw"].add_edge(
                a, b, weight=float(row["raw_weight"]), **common_attrs
            )
            graphs["jaccard"].add_edge(
                a, b, weight=float(row["jaccard_weight"]), **common_attrs
            )
            graphs["cosine"].add_edge(
                a, b, weight=float(row["cosine_weight"]), **common_attrs
            )

    return (
        graphs["raw"],
        graphs["jaccard"],
        graphs["cosine"],
        nodes_df,
        edges_df,
    )


def centrality_comparison(
    G_raw: nx.Graph,
    G_jaccard: nx.Graph,
    G_cosine: nx.Graph,
) -> pd.DataFrame:
    """Compare weighted degree (strength) and ranks across the three graphs."""
    graph_map = {
        "raw_strength": G_raw,
        "jaccard_strength": G_jaccard,
        "cosine_strength": G_cosine,
    }

    result = pd.DataFrame({"character": list(G_raw.nodes())})

    for column, graph in graph_map.items():
        strengths = dict(graph.degree(weight="weight"))
        result[column] = result["character"].map(strengths).fillna(0.0)
        result[column.replace("_strength", "_rank")] = (
            result[column].rank(ascending=False, method="min")
        )

    return result.sort_values("raw_rank").reset_index(drop=True)


def network_summary(graph: nx.Graph, name: str) -> dict:
    """Return basic descriptive statistics for a network."""
    n = graph.number_of_nodes()
    strengths = dict(graph.degree(weight="weight"))
    return {
        "network": name,
        "nodes": n,
        "edges": graph.number_of_edges(),
        "density": nx.density(graph) if n > 1 else 0.0,
        "connected_components": (
            nx.number_connected_components(graph) if n else 0
        ),
        "average_degree": (
            sum(dict(graph.degree()).values()) / n if n else 0.0
        ),
        "average_strength": (
            sum(strengths.values()) / n if n else 0.0
        ),
    }


def plot_network(
    graph: nx.Graph,
    title: str,
    pos: Optional[dict] = None,
    top_n_edges: Optional[int] = None,
    figsize: tuple[int, int] = (18, 14),
    label_font_size: int = 8,
):
    """
    Plot a network. If top_n_edges is given, display only that many strongest
    edges; nodes with no retained edges are omitted from the drawing.
    """
    display_graph = graph.copy()

    if top_n_edges is not None:
        strongest = sorted(
            graph.edges(data=True),
            key=lambda edge: edge[2].get("weight", 0),
            reverse=True,
        )[:top_n_edges]
        display_graph = nx.Graph()
        display_graph.add_nodes_from(graph.nodes(data=True))
        display_graph.add_edges_from(strongest)
        display_graph.remove_nodes_from(list(nx.isolates(display_graph)))

    if pos is None:
        pos = nx.spring_layout(
            graph, weight="weight", seed=42, k=0.8, iterations=100
        )

    # Keep only positions for nodes actually displayed.
    draw_pos = {node: pos[node] for node in display_graph.nodes() if node in pos}

    strengths = dict(display_graph.degree(weight="weight"))
    node_sizes = [
        max(80, strengths[node] * 2)
        for node in display_graph.nodes()
    ]

    edge_weights = [
        data.get("weight", 1.0)
        for _, _, data in display_graph.edges(data=True)
    ]
    max_weight = max(edge_weights) if edge_weights else 1.0
    edge_widths = [
        0.5 + 4.0 * (data.get("weight", 1.0) / max_weight)
        for _, _, data in display_graph.edges(data=True)
    ]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(
        display_graph, draw_pos, ax=ax,
        width=edge_widths, alpha=0.3, edge_color="gray"
    )
    nx.draw_networkx_nodes(
        display_graph, draw_pos, ax=ax,
        node_size=node_sizes, alpha=0.85
    )
    nx.draw_networkx_labels(
        display_graph, draw_pos, ax=ax,
        font_size=label_font_size, font_weight="bold"
    )
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax


def export_networks(
    output_dir: str | Path,
    G_raw: nx.Graph,
    G_jaccard: nx.Graph,
    G_cosine: nx.Graph,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    """Export three GraphML files and the node/edge/comparison CSV tables."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    nx.write_graphml(G_raw, output / "network_raw.graphml")
    nx.write_graphml(G_jaccard, output / "network_jaccard.graphml")
    nx.write_graphml(G_cosine, output / "network_cosine.graphml")

    nodes_out = nodes_df.copy()
    nodes_out["degree_raw"] = nodes_out["character"].map(dict(G_raw.degree()))
    nodes_out["strength_raw"] = nodes_out["character"].map(
        dict(G_raw.degree(weight="weight"))
    )
    nodes_out.to_csv(output / "nodes.csv", index=False)

    edges_df.to_csv(output / "edge_comparison.csv", index=False)
    comparison_df.to_csv(output / "centrality_comparison.csv", index=False)

    summaries = pd.DataFrame([
        network_summary(G_raw, "raw"),
        network_summary(G_jaccard, "jaccard"),
        network_summary(G_cosine, "cosine"),
    ])
    summaries.to_csv(output / "network_summaries.csv", index=False)
