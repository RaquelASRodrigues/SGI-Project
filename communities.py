"""Analysis helpers for k-clique communities in the Os Maias character graph."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


METRICS = (
    "num_communities",
    "largest_community_size",
    "giant_fraction",
    "covered_nodes",
    "coverage_pct",
    "overlap_nodes",
    "mean_community_size",
)


def load_weighted_graph(
    edges_path: str | Path, nodes_path: str | Path
) -> tuple[nx.Graph, dict[str, str]]:
    """Load the undirected weighted edge list and return it with an ID/name map."""
    edges = pd.read_csv(edges_path, dtype={"source": str, "target": str})
    nodes = pd.read_csv(nodes_path, dtype={"id": str})
    if not {"source", "target", "weight"}.issubset(edges.columns):
        raise ValueError("Edge CSV must contain source, target, and weight columns.")
    if not {"id", "name"}.issubset(nodes.columns):
        raise ValueError("Node CSV must contain id and name columns.")

    name_map = dict(zip(nodes["id"], nodes["name"].fillna(nodes["id"])))
    graph = nx.Graph(name="Os Maias weighted character co-occurrence")
    graph.add_nodes_from(name_map)
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), weight=float(row.weight))
        name_map.setdefault(str(row.source), str(row.source))
        name_map.setdefault(str(row.target), str(row.target))
    return graph, name_map


def threshold_graph(graph: nx.Graph, w_min: float) -> nx.Graph:
    """Return a copy containing all nodes and only edges with weight >= w_min."""
    filtered = nx.Graph(**graph.graph)
    filtered.add_nodes_from(graph.nodes(data=True))
    filtered.add_edges_from(
        (u, v, data.copy())
        for u, v, data in graph.edges(data=True)
        if data.get("weight", 1) >= w_min
    )
    return filtered


def max_clique_size(graph: nx.Graph) -> int:
    """Return the size of the largest clique, or zero for an empty graph."""
    return max((len(clique) for clique in nx.find_cliques(graph)), default=0)


def get_k_clique_communities(graph: nx.Graph, k: int) -> list[set[str]]:
    """Compute deterministically ordered k-clique percolation communities."""
    communities = [
        set(community)
        for community in nx.community.k_clique_communities(graph, k)
    ]
    return sorted(communities, key=lambda group: (-len(group), tuple(sorted(group))))


def community_metrics(
    communities: Iterable[set[str]], total_nodes: int
) -> dict[str, float | int]:
    """Calculate coverage, overlap, size, and giant-community metrics."""
    groups = list(communities)
    memberships: dict[str, int] = {}
    for group in groups:
        for node in group:
            memberships[node] = memberships.get(node, 0) + 1
    covered = len(memberships)
    largest = max((len(group) for group in groups), default=0)
    return {
        "num_communities": len(groups),
        "largest_community_size": largest,
        "giant_fraction": largest / covered if covered else 0.0,
        "covered_nodes": covered,
        "coverage_pct": 100.0 * covered / total_nodes if total_nodes else 0.0,
        "overlap_nodes": sum(count >= 2 for count in memberships.values()),
        "mean_community_size": (
            float(np.mean([len(group) for group in groups]) if groups else 0.0)
        ),
    }


def degree_preserving_shuffles(
    graph: nx.Graph,
    n_shuffles: int = 100,
    swaps_per_edge: int = 10,
    seed: int = 4200,
    max_tries_multiplier: int = 100,
) -> list[nx.Graph]:
    """Generate seeded double-edge-swap graphs, preserving each node degree."""
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("Degree-preserving shuffles require a simple undirected graph.")
    nswap = swaps_per_edge * graph.number_of_edges()
    if nswap and graph.number_of_edges() < 2:
        raise ValueError("At least two edges are needed for double-edge swaps.")
    shuffles = []
    for index in range(n_shuffles):
        shuffled = graph.copy()
        if nswap:
            nx.double_edge_swap(
                shuffled,
                nswap=nswap,
                max_tries=max(1000, nswap * max_tries_multiplier),
                seed=seed + index,
            )
        shuffles.append(shuffled)
    return shuffles


def _null_summary(
    real_value: float | int, null_values: list[float | int]
) -> dict[str, float]:
    """Summarize a metric's null distribution, guarding a zero standard deviation."""
    values = np.asarray(null_values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    z_score = (float(real_value) - mean) / std if std > 0 else np.nan
    return {
        "null_mean": mean,
        "null_std": std,
        "z_score": z_score,
        "p_ge": float(np.mean(values >= real_value)),
        "p_le": float(np.mean(values <= real_value)),
    }


def run_k_clique_sweep(
    graph: nx.Graph,
    thresholds: Iterable[int] = (1, 2, 3, 5),
    n_shuffles: int = 100,
    swaps_per_edge: int = 10,
    seed: int = 4200,
) -> pd.DataFrame:
    """Run real and degree-preserving null k-clique metrics over k and thresholds."""
    largest_k = max_clique_size(graph)
    rows = []
    for threshold_index, w_min in enumerate(thresholds):
        filtered = threshold_graph(graph, w_min)
        print(
            f"w_min={w_min}: generating {n_shuffles} null graphs "
            f"({swaps_per_edge} swaps per edge)...",
            flush=True,
        )
        null_graphs = degree_preserving_shuffles(
            filtered,
            n_shuffles=n_shuffles,
            swaps_per_edge=swaps_per_edge,
            seed=seed + threshold_index * 100_000,
        )
        null_metrics_by_k: dict[int, list[dict[str, float | int]]] = {
            k: [] for k in range(3, largest_k + 1)
        }
        for replicate, null_graph in enumerate(null_graphs, start=1):
            for k in null_metrics_by_k:
                null_groups = get_k_clique_communities(null_graph, k)
                null_metrics_by_k[k].append(
                    community_metrics(null_groups, graph.number_of_nodes())
                )
            if replicate % 10 == 0 or replicate == n_shuffles:
                print(
                    f"  w_min={w_min}: analyzed null graph "
                    f"{replicate}/{n_shuffles}",
                    flush=True,
                )

        for k in null_metrics_by_k:
            real_groups = get_k_clique_communities(filtered, k)
            real_metrics = community_metrics(real_groups, graph.number_of_nodes())
            row: dict[str, float | int | str] = {
                "k": k,
                "w_min": w_min,
                "status": "ok" if real_groups else "none",
            }
            row.update(real_metrics)
            null_rows = null_metrics_by_k[k]
            for metric in METRICS:
                summary = _null_summary(
                    real_metrics[metric],
                    [null_row[metric] for null_row in null_rows],
                )
                for statistic, value in summary.items():
                    row[f"{metric}_{statistic}"] = value
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["w_min", "k"]).reset_index(drop=True)


def candidate_table(
    results: pd.DataFrame,
    min_communities: int = 3,
    max_giant_fraction: float = 0.9,
    min_coverage_pct: float = 50.0,
    min_overlap_nodes: int = 3,
    min_abs_z: float = 2.0,
    max_empirical_p: float = 0.05,
) -> pd.DataFrame:
    """Mark and rank candidates by criteria passed, coverage, then overlap."""
    out = results.copy()
    significant_z = (
        out["covered_nodes_z_score"].abs().ge(min_abs_z)
        | out["num_communities_z_score"].abs().ge(min_abs_z)
    )
    empirical_p = pd.concat(
        [
            out["covered_nodes_p_ge"],
            out["covered_nodes_p_le"],
            out["num_communities_p_ge"],
            out["num_communities_p_le"],
        ],
        axis=1,
    ).min(axis=1)
    significant_p = empirical_p.le(max_empirical_p)
    checks = pd.DataFrame(
        {
            "enough_communities": out["num_communities"].ge(min_communities),
            "giant_fraction_ok": out["giant_fraction"].le(max_giant_fraction),
            "coverage_ok": out["coverage_pct"].ge(min_coverage_pct),
            "overlap_ok": out["overlap_nodes"].ge(min_overlap_nodes),
            "different_from_null": significant_z | significant_p,
        },
        index=out.index,
    )
    out["criteria_passed"] = checks.sum(axis=1)
    out["candidate"] = (
        checks.all(axis=1) & out["status"].eq("ok")
    )
    return out.sort_values(
        ["criteria_passed", "coverage_pct", "overlap_nodes"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def find_group_column(nodes: pd.DataFrame) -> str | None:
    """Find a non-empty family, group, or social-circle annotation column."""
    group_names = {"family", "group", "social_circle", "socialcircle"}
    for column in nodes.columns:
        normalized = str(column).strip().lower().replace("-", "_").replace(" ", "_")
        if not any(token in normalized for token in group_names):
            continue
        values = nodes[column].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        if values.nunique() >= 2:
            return str(column)
    return None


def _nmi_and_shuffle_baseline(
    nodes: list[str],
    clique_labels: dict[str, int],
    louvain_labels: list[dict[str, int]],
    rng: np.random.Generator,
    n_label_shuffles: int,
) -> tuple[float, float, float, float]:
    """Return real mean/std and permuted-label baseline mean/std."""
    labels = np.asarray([clique_labels[node] for node in nodes])
    real_scores = np.asarray(
        [
            normalized_mutual_info_score(
                labels, [partition[node] for node in nodes]
            )
            for partition in louvain_labels
        ]
    )
    shuffled_scores = np.empty(n_label_shuffles, dtype=float)
    for index in range(n_label_shuffles):
        permuted = rng.permutation(labels)
        shuffled_scores[index] = np.mean(
            [
                normalized_mutual_info_score(
                    permuted, [partition[node] for node in nodes]
                )
                for partition in louvain_labels
            ]
        )
    return (
        float(real_scores.mean()),
        float(real_scores.std(ddof=0)),
        float(shuffled_scores.mean()),
        float(shuffled_scores.std(ddof=0)),
    )


def run_nmi_sweep(
    graph: nx.Graph,
    results: pd.DataFrame,
    n_label_shuffles: int = 200,
    seeds: Iterable[int] = (41, 42, 43, 44, 45),
    seed: int = 5200,
    node_data: pd.DataFrame | None = None,
    group_column: str | None = None,
    group_settings: Iterable[tuple[int, int]] = ((5, 2), (8, 3)),
) -> pd.DataFrame:
    """Compare k-clique/Louvain NMI, shuffled-label baselines, and overlap sensitivity."""
    seeds = tuple(seeds)
    group_settings = set(group_settings)
    rows = []
    for row_number, result in enumerate(results.itertuples(index=False)):
        k, w_min = int(result.k), int(result.w_min)
        row: dict[str, object] = {
            "k": k,
            "w_min": w_min,
            "num_communities": int(result.num_communities),
            "coverage_pct": float(result.coverage_pct),
            "nmi_nodes": np.nan,
            "non_overlap_nodes": np.nan,
            "nmi_status": (
                "masked_single_community"
                if int(result.num_communities) == 1
                else "masked_no_communities"
            ),
        }
        for variant in ("covered", "no_overlap"):
            for statistic in ("real_mean", "real_std", "shuffle_mean", "shuffle_std"):
                row[f"nmi_{variant}_{statistic}"] = np.nan
            row[f"louvain_pairwise_nmi_{variant}"] = np.nan
        row["group_nmi"] = np.nan
        if int(result.num_communities) < 2:
            rows.append(row)
            continue

        filtered = threshold_graph(graph, w_min)
        groups = get_k_clique_communities(filtered, k)
        memberships: dict[str, list[int]] = {}
        clique_labels: dict[str, int] = {}
        for community_index, community in enumerate(groups):
            for node in community:
                memberships.setdefault(node, []).append(community_index)
                clique_labels.setdefault(node, community_index)
        covered = sorted(clique_labels)
        exclusive = [node for node in covered if len(memberships[node]) == 1]
        louvain_runs = [
            list(
                nx.community.louvain_communities(
                    filtered, seed=run_seed, weight="weight"
                )
            )
            for run_seed in seeds
        ]
        louvain_labels = [
            {
                node: community_index
                for community_index, community in enumerate(partition)
                for node in community
            }
            for partition in louvain_runs
        ]
        pairwise = [
            normalized_mutual_info_score(
                [left[node] for node in covered],
                [right[node] for node in covered],
            )
            for left, right in combinations(louvain_labels, 2)
        ]
        rng = np.random.default_rng(seed + row_number)
        covered_stats = _nmi_and_shuffle_baseline(
            covered, clique_labels, louvain_labels, rng, n_label_shuffles
        )
        no_overlap_stats = _nmi_and_shuffle_baseline(
            exclusive, clique_labels, louvain_labels, rng, n_label_shuffles
        ) if exclusive else (np.nan, np.nan, np.nan, np.nan)
        row.update(
            {
                "nmi_nodes": len(covered),
                "non_overlap_nodes": len(exclusive),
                "nmi_status": "ok",
                "nmi_covered_real_mean": covered_stats[0],
                "nmi_covered_real_std": covered_stats[1],
                "nmi_covered_shuffle_mean": covered_stats[2],
                "nmi_covered_shuffle_std": covered_stats[3],
                "nmi_no_overlap_real_mean": no_overlap_stats[0],
                "nmi_no_overlap_real_std": no_overlap_stats[1],
                "nmi_no_overlap_shuffle_mean": no_overlap_stats[2],
                "nmi_no_overlap_shuffle_std": no_overlap_stats[3],
                "louvain_pairwise_nmi_covered": float(np.mean(pairwise)),
                "louvain_pairwise_nmi_no_overlap": (
                    float(
                        np.mean(
                            [
                                normalized_mutual_info_score(
                                    [left[node] for node in exclusive],
                                    [right[node] for node in exclusive],
                                )
                                for left, right in combinations(louvain_labels, 2)
                            ]
                        )
                    )
                    if exclusive
                    else np.nan
                ),
            }
        )
        if (
            node_data is not None
            and group_column is not None
            and (k, w_min) in group_settings
        ):
            group_labels = node_data.set_index("id")[group_column].to_dict()
            valid = [
                node
                for node in covered
                if pd.notna(group_labels.get(node))
                and str(group_labels[node]).strip()
            ]
            if len(valid) >= 2:
                row["group_nmi"] = normalized_mutual_info_score(
                    [clique_labels[node] for node in valid],
                    [group_labels[node] for node in valid],
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["w_min", "k"]).reset_index(drop=True)


def plot_nmi_sweep(nmi_results: pd.DataFrame):
    """Plot real and overlap-excluded NMI with permuted-label bands by threshold."""
    thresholds = sorted(nmi_results["w_min"].dropna().unique())
    columns = min(2, max(1, len(thresholds)))
    rows = int(np.ceil(len(thresholds) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(7 * columns, 4.5 * rows), squeeze=False
    )
    for ax, w_min in zip(axes.flat, thresholds):
        group = nmi_results[
            (nmi_results["w_min"] == w_min)
            & (nmi_results["nmi_status"] == "ok")
        ].sort_values("k")
        if group.empty:
            ax.set_title(f"w_min={w_min} (no comparable settings)")
            ax.axis("off")
            continue
        k_values = group["k"].to_numpy()
        coverage = group["coverage_pct"].to_numpy()
        for variant, color, marker, label in (
            ("covered", "tab:blue", "o", "All covered nodes"),
            ("no_overlap", "tab:orange", "^", "Excluding overlap nodes"),
        ):
            real = group[f"nmi_{variant}_real_mean"].to_numpy(dtype=float)
            mean = group[f"nmi_{variant}_shuffle_mean"].to_numpy(dtype=float)
            std = group[f"nmi_{variant}_shuffle_std"].to_numpy(dtype=float)
            ax.fill_between(
                k_values,
                np.maximum(0, mean - 2 * std),
                np.minimum(1, mean + 2 * std),
                color=color,
                alpha=0.14 if variant == "covered" else 0.08,
            )
            ax.plot(
                k_values, real, color=color, marker=marker, label=label
            )
            for x, y, pct in zip(k_values, real, coverage):
                if np.isfinite(y):
                    offset = 7 if variant == "covered" else -13
                    ax.annotate(
                        f"{pct:.0f}%",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, offset),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )
        reference = group["louvain_pairwise_nmi_covered"].mean()
        ax.axhline(
            reference,
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"Louvain pairwise mean={reference:.2f}",
        )
        ax.set(title=f"w_min={w_min}", xlabel="k", ylabel="Normalized mutual information")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(thresholds):]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_metric_sweep(
    results: pd.DataFrame,
    metrics: Iterable[str] = (
        "num_communities",
        "giant_fraction",
        "coverage_pct",
        "overlap_nodes",
    ),
):
    """Plot each real metric against k with a null mean +/- 2 std band."""
    metrics = list(metrics)
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(11, 3.2 * len(metrics)), sharex=True
    )
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        for w_min, group in results.groupby("w_min", sort=True):
            group = group.sort_values("k")
            mean_col, std_col = f"{metric}_null_mean", f"{metric}_null_std"
            lower = group[mean_col] - 2 * group[std_col]
            upper = group[mean_col] + 2 * group[std_col]
            line, = ax.plot(
                group["k"], group[metric], marker="o", label=f"w_min={w_min}"
            )
            ax.fill_between(
                group["k"], lower, upper, color=line.get_color(), alpha=0.16
            )
            ax.plot(group["k"], group[mean_col], color=line.get_color(), ls="--")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("k")
    fig.tight_layout()
    return fig, axes


def plot_metric_heatmap(
    results: pd.DataFrame,
    candidates: pd.DataFrame,
    metric: str,
    ax=None,
):
    """Draw a k-by-threshold heatmap and mark settings meeting the criteria."""
    pivot = results.pivot(index="w_min", columns="k", values=metric).sort_index()
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.5))
    image = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xlabel("k")
    ax.set_ylabel("w_min")
    ax.set_title(metric.replace("_", " "))
    fig = ax.figure
    fig.colorbar(image, ax=ax, label=metric)
    chosen = candidates[candidates["candidate"]]
    for row in chosen.itertuples():
        ax.scatter(
            list(pivot.columns).index(row.k),
            list(pivot.index).index(row.w_min),
            marker="*",
            s=110,
            facecolors="none",
            edgecolors="white",
            linewidths=1.5,
        )
    return ax


def validate_k_clique_communities(
    graph: nx.Graph, k: int, communities: Iterable[set[str]] | None = None
) -> bool:
    """Assert each returned community contains a complete k-clique."""
    groups = (
        list(communities)
        if communities is not None
        else get_k_clique_communities(graph, k)
    )
    for group in groups:
        subgraph = graph.subgraph(group)
        if not any(len(clique) >= k for clique in nx.find_cliques(subgraph)):
            raise AssertionError(f"Community has no {k}-clique: {sorted(group)}")
    return True


def describe_communities(
    graph: nx.Graph, k: int, name_map: dict[str, str]
) -> dict[str, object]:
    """Return communities, overlaps, and uncovered nodes, using display names."""
    groups = get_k_clique_communities(graph, k)
    memberships: dict[str, list[int]] = {}
    for index, group in enumerate(groups, start=1):
        for node in group:
            memberships.setdefault(node, []).append(index)
    return {
        "communities": [
            [name_map.get(node, node) for node in sorted(group, key=lambda n: name_map.get(n, n))]
            for group in groups
        ],
        "overlap_nodes": {
            name_map.get(node, node): memberships[node]
            for node in sorted(memberships)
            if len(memberships[node]) >= 2
        },
        "uncovered_nodes": sorted(
            (name_map.get(node, node) for node in graph.nodes if node not in memberships)
        ),
    }


def compare_with_louvain(
    graph: nx.Graph,
    k: int,
    seeds: Iterable[int] = (41, 42, 43, 44, 45),
) -> dict[str, float]:
    """Compare k-clique with five seeded Louvain runs on covered nodes."""
    clique_groups = get_k_clique_communities(graph, k)
    covered = sorted(set().union(*clique_groups)) if clique_groups else []
    if not covered:
        return {
            "louvain_num_communities_mean": np.nan,
            "louvain_modularity_mean": np.nan,
            "kclique_louvain_nmi_mean": np.nan,
            "louvain_pairwise_nmi_mean": np.nan,
        }
    # NMI needs one label per node; overlapping nodes use their first (largest)
    # deterministically ordered k-clique community.
    clique_labels = {}
    for index, group in enumerate(clique_groups):
        for node in group:
            clique_labels.setdefault(node, index)
    runs = [
        list(nx.community.louvain_communities(graph, seed=run_seed, weight="weight"))
        for run_seed in seeds
    ]
    lou_labels = [
        {node: index for index, group in enumerate(groups) for node in group}
        for groups in runs
    ]
    nmis_to_clique = [
        normalized_mutual_info_score(
            [clique_labels[node] for node in covered],
            [labels[node] for node in covered],
        )
        for labels in lou_labels
    ]
    pairwise_nmi = [
        normalized_mutual_info_score(
            [left[node] for node in covered], [right[node] for node in covered]
        )
        for left, right in combinations(lou_labels, 2)
    ]
    modularities = [
        nx.community.modularity(graph, groups, weight="weight") for groups in runs
    ]
    return {
        "louvain_num_communities_mean": float(np.mean([len(run) for run in runs])),
        "louvain_modularity_mean": float(np.mean(modularities)),
        "kclique_louvain_nmi_mean": float(np.mean(nmis_to_clique)),
        "louvain_pairwise_nmi_mean": float(np.mean(pairwise_nmi)),
    }


def plot_community_network(
    graph: nx.Graph,
    k: int,
    name_map: dict[str, str],
    pos: dict[str, np.ndarray],
    w_min: int,
    top_n_labels: int = 15,
):
    """Draw a thresholded network, coloring communities and outlining overlaps."""
    groups = get_k_clique_communities(graph, k)
    memberships: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        for node in group:
            memberships.setdefault(node, []).append(index)
    colors = plt.get_cmap("tab20")
    node_colors = [
        colors(memberships[node][0] % 20) if node in memberships else "#c9c9c9"
        for node in graph
    ]
    strength = dict(graph.degree(weight="weight"))
    sizes = [80 + 14 * np.sqrt(strength[node]) for node in graph]
    fig, ax = plt.subplots(figsize=(13, 10))
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.22, width=0.7)
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=node_colors, node_size=sizes, alpha=0.9
    )
    overlap = [node for node, groups_for_node in memberships.items() if len(groups_for_node) > 1]
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=overlap,
        node_color=[node_colors[list(graph).index(node)] for node in overlap],
        node_size=[sizes[list(graph).index(node)] * 1.15 for node in overlap],
        node_shape="o",
        edgecolors="black",
        linewidths=2,
        ax=ax,
    )
    top_nodes = sorted(strength, key=strength.get, reverse=True)[:top_n_labels]
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: name_map.get(node, node) for node in top_nodes},
        font_size=8,
        ax=ax,
    )
    ax.set_title(
        f"k={k}, w_min={w_min}; gray nodes are left out, outlined nodes overlap"
    )
    ax.axis("off")
    fig.tight_layout()
    return fig, ax
