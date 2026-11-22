"""From-scratch Ahn et al. link clustering for the Os Maias character graph.

The main analysis is unweighted after edge-weight thresholding. Weights are used
only for threshold selection, character strength, and visualization.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import textwrap
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score

from communities import get_k_clique_communities, load_weighted_graph, threshold_graph


MIN_LINKS = 3
DEFAULT_THRESHOLDS = (1, 2, 3)
DEFAULT_NULL_REPLICATES = 100
DEFAULT_SWAPS_PER_EDGE = 10
DEFAULT_NULL_SEED = 4200
LOUVAIN_SEEDS = (41, 42, 43, 44, 45)
LABEL_SHUFFLE_SEED = 5200
LAYOUT_SEED = 42
ADAPTIVE_SEEDS = tuple(range(8))
ADAPTIVE_NULL_REPLICATES = 20
ADAPTIVE_NULL_SEED = 7200
ADAPTIVE_WALK_SEED = 6200
ADAPTIVE_STEPS = 5000
ADAPTIVE_SETTINGS = {
    "cold": {"temperature_scale": 0.1, "cool_start_fraction": 0.25},
    "auto": {"temperature_scale": 1.0, "cool_start_fraction": 2 / 3},
    "hot": {"temperature_scale": 100.0, "cool_start_fraction": 2 / 3},
}


def _canonical_link(u: str, v: str) -> tuple[str, str]:
    """Return an undirected link with its endpoint IDs in stable order."""
    return (u, v) if u <= v else (v, u)


def link_similarities(
    graph: nx.Graph, links: list[tuple[str, str]]
) -> list[tuple[float, int, int]]:
    """Compute Jaccard similarities only for links sharing an endpoint."""
    inclusive = {
        node: set(graph.neighbors(node)) | {node}
        for node in graph.nodes
    }
    incident: dict[str, list[int]] = defaultdict(list)
    for index, (u, v) in enumerate(links):
        incident[u].append(index)
        incident[v].append(index)

    similarities = []
    for shared, indices in incident.items():
        for left, right in combinations(indices, 2):
            left_link, right_link = links[left], links[right]
            i = left_link[1] if left_link[0] == shared else left_link[0]
            j = right_link[1] if right_link[0] == shared else right_link[0]
            left_neighborhood, right_neighborhood = inclusive[i], inclusive[j]
            similarity = len(left_neighborhood & right_neighborhood) / len(
                left_neighborhood | right_neighborhood
            )
            similarities.append((similarity, left, right))
    similarities.sort(key=lambda item: (-item[0], item[1], item[2]))
    return similarities


def partition_density(
    edge_groups: Iterable[Iterable[tuple[str, str]]], total_edges: int
) -> float:
    """Compute the Ahn et al. partition density; groups of <=2 nodes add zero."""
    if total_edges == 0:
        return 0.0
    weighted_density = 0.0
    for group in edge_groups:
        edges = list(group)
        m_c = len(edges)
        n_c = len({node for edge in edges for node in edge})
        if n_c > 2:
            weighted_density += (
                m_c * (m_c - (n_c - 1)) / ((n_c - 2) * (n_c - 1))
            )
    return float(2.0 * weighted_density / total_edges)


def _edge_components(root_by_edge: list[int]) -> list[list[int]]:
    """Group edge indices by union-find root in deterministic order."""
    groups: dict[int, list[int]] = defaultdict(list)
    for edge_index, root in enumerate(root_by_edge):
        groups[root].append(edge_index)
    return sorted(groups.values(), key=lambda group: group[0])


def _membership_map(
    node_communities: list[set[str]],
) -> dict[str, list[int]]:
    """Map each node to its 1-based community IDs."""
    membership: dict[str, list[int]] = defaultdict(list)
    for index, nodes in enumerate(node_communities, start=1):
        for node in nodes:
            membership[node].append(index)
    return dict(membership)


def build_link_dendrogram(
    links: list[tuple[str, str]], merges: pd.DataFrame
) -> dict[str, object]:
    """Reconstruct the merge forest, retaining each node's link set and height."""
    n_links = len(links)
    nodes: dict[int, dict[str, object]] = {}
    parent: dict[int, int] = {}
    for link_id, link in enumerate(links):
        nodes[link_id] = {
            "id": link_id,
            "children": None,
            "similarity": np.nan,
            "link_ids": frozenset({link_id}),
            "links": frozenset({link}),
            "node_set": frozenset(link),
            "n_nodes": 2,
            "partition_contribution": 0.0,
        }

    for row in merges.itertuples(index=False):
        left, right, merged = (
            int(row.left_cluster),
            int(row.right_cluster),
            int(row.merged_cluster),
        )
        link_ids = nodes[left]["link_ids"] | nodes[right]["link_ids"]
        link_set = nodes[left]["links"] | nodes[right]["links"]
        node_set = nodes[left]["node_set"] | nodes[right]["node_set"]
        n_nodes = len(node_set)
        n_edges = len(link_ids)
        contribution = (
            n_edges * (n_edges - (n_nodes - 1))
            / ((n_nodes - 2) * (n_nodes - 1))
            if n_nodes > 2
            else 0.0
        )
        nodes[merged] = {
            "id": merged,
            "children": (left, right),
            "similarity": float(row.similarity),
            "link_ids": frozenset(link_ids),
            "links": frozenset(link_set),
            "node_set": frozenset(node_set),
            "n_nodes": n_nodes,
            "partition_contribution": contribution,
        }
        parent[left] = merged
        parent[right] = merged
    roots = sorted(set(nodes) - set(parent))
    return {"nodes": nodes, "parent": parent, "roots": roots}


def cluster_links(graph: nx.Graph, min_links: int = MIN_LINKS) -> dict[str, object]:
    """Build the single-linkage dendrogram and select its maximum-density cut.

    Each similarity level is applied atomically before evaluating density, so
    ties cannot make the selected cut depend on an arbitrary pair ordering.
    All components are retained in ``communities``; only those meeting
    ``min_links`` are exposed as real communities and used for node metrics.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("Link clustering requires a simple undirected graph.")
    if nx.number_of_selfloops(graph):
        raise ValueError("Self-loops are not supported by link clustering.")
    if min_links < 1:
        raise ValueError("min_links must be positive.")

    links = sorted(_canonical_link(u, v) for u, v in graph.edges())
    n_links = len(links)
    similarities = link_similarities(graph, links)
    parent = list(range(n_links))
    rank = [0] * n_links
    cluster_id = list(range(n_links))
    merges: list[dict[str, float | int]] = []

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def merge(left: int, right: int, similarity: float) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        left_cluster, right_cluster = cluster_id[left_root], cluster_id[right_root]
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1
        merged_cluster = n_links + len(merges)
        cluster_id[left_root] = merged_cluster
        merges.append(
            {
                "similarity": similarity,
                "left_cluster": left_cluster,
                "right_cluster": right_cluster,
                "merged_cluster": merged_cluster,
            }
        )

    cut_rows = [
        {
            "similarity": np.nan,
            "partition_density": 0.0,
            "num_components": n_links,
        }
    ]
    best_density = 0.0
    best_similarity = np.nan
    best_groups = [[index] for index in range(n_links)]
    best_cut_nodes = list(range(n_links))

    start = 0
    while start < len(similarities):
        similarity = similarities[start][0]
        end = start + 1
        while end < len(similarities) and similarities[end][0] == similarity:
            end += 1
        for _, left, right in similarities[start:end]:
            merge(left, right, similarity)

        edge_groups = _edge_components([find(i) for i in range(n_links)])
        density = partition_density(
            ([links[index] for index in group] for group in edge_groups),
            n_links,
        )
        cut_rows.append(
            {
                "similarity": similarity,
                "partition_density": density,
                "num_components": len(edge_groups),
            }
        )
        if density > best_density:
            best_density = density
            best_similarity = similarity
            best_groups = edge_groups
            best_cut_nodes = sorted(
                {cluster_id[find(index)] for index in range(n_links)}
            )
        start = end

    groups = sorted(
        best_groups,
        key=lambda group: (-len(group), tuple(links[index] for index in group)),
    )
    communities = [[links[index] for index in group] for group in groups]
    node_communities = [
        {node for edge in community for node in edge}
        for community in communities
    ]
    real_indices = [
        index for index, edges in enumerate(communities) if len(edges) >= min_links
    ]
    real_communities = [communities[index] for index in real_indices]
    real_node_communities = [node_communities[index] for index in real_indices]
    small_communities = [
        community for community in communities if len(community) < min_links
    ]
    small_nodes = {node for group in small_communities for edge in group for node in edge}
    real_membership = _membership_map(real_node_communities)
    membership = _membership_map(node_communities)
    tree = build_link_dendrogram(links, pd.DataFrame(merges))

    return {
        "links": links,
        "similarity_pairs": len(similarities),
        "merges": pd.DataFrame(
            merges,
            columns=["similarity", "left_cluster", "right_cluster", "merged_cluster"],
        ),
        "density_curve": pd.DataFrame(cut_rows),
        "best_similarity": best_similarity,
        "best_partition_density": best_density,
        "best_cut_nodes": best_cut_nodes,
        "dendrogram": tree,
        "communities": communities,
        "node_communities": node_communities,
        "membership": membership,
        "real_communities": real_communities,
        "real_node_communities": real_node_communities,
        "real_membership": real_membership,
        "small_communities": small_communities,
        "small_nodes": small_nodes,
        "nodes_only_in_small_communities": small_nodes - set(real_membership),
        "min_links": min_links,
    }


def validate_result(result: dict[str, object]) -> bool:
    """Assert the cut is an exact edge partition and its density is reproducible."""
    links = result["links"]
    communities = result["communities"]
    available_clusters = set(range(len(links)))
    for merge_index, row in enumerate(result["merges"].itertuples(index=False)):
        assert row.left_cluster in available_clusters
        assert row.right_cluster in available_clusters
        assert row.left_cluster != row.right_cluster
        assert row.merged_cluster == len(links) + merge_index
        available_clusters.add(row.merged_cluster)
    counts = Counter(edge for group in communities for edge in group)
    assert set(counts) == set(links)
    assert all(count == 1 for count in counts.values())
    direct_density = partition_density(communities, len(links))
    assert np.isclose(direct_density, result["best_partition_density"])
    tree_nodes = result["dendrogram"]["nodes"]
    cut_link_ids = [
        link_id
        for node_id in result["best_cut_nodes"]
        for link_id in tree_nodes[node_id]["link_ids"]
    ]
    assert sorted(cut_link_ids) == list(range(len(links)))
    return True


def _communities_from_cut(
    result: dict[str, object], cut_nodes: Iterable[int]
) -> list[list[tuple[str, str]]]:
    """Translate a valid tree cut into sorted sets of graph links."""
    links = result["links"]
    tree_nodes = result["dendrogram"]["nodes"]
    return [
        sorted(links[link_id] for link_id in tree_nodes[node_id]["link_ids"])
        for node_id in sorted(cut_nodes)
    ]


def result_for_cut(
    result: dict[str, object], cut_nodes: Iterable[int]
) -> dict[str, object]:
    """Return community metrics and link groups for any valid dendrogram cut."""
    cut_nodes = tuple(sorted(cut_nodes))
    communities = _communities_from_cut(result, cut_nodes)
    node_communities = [
        {node for edge in community for node in edge}
        for community in communities
    ]
    real_indices = [
        index
        for index, edges in enumerate(communities)
        if len(edges) >= result["min_links"]
    ]
    real_communities = [communities[index] for index in real_indices]
    real_node_communities = [
        node_communities[index] for index in real_indices
    ]
    small_communities = [
        community for community in communities if len(community) < result["min_links"]
    ]
    small_nodes = {
        node for group in small_communities for edge in group for node in edge
    }
    return {
        **result,
        "communities": communities,
        "node_communities": node_communities,
        "membership": _membership_map(node_communities),
        "real_communities": real_communities,
        "real_node_communities": real_node_communities,
        "real_membership": _membership_map(real_node_communities),
        "small_communities": small_communities,
        "small_nodes": small_nodes,
        "nodes_only_in_small_communities": small_nodes
        - set(_membership_map(real_node_communities)),
        "best_cut_nodes": sorted(cut_nodes),
        "best_partition_density": partition_density(
            communities, len(result["links"])
        ),
        "best_similarity": np.nan,
    }


def _result_metrics(result: dict[str, object]) -> dict[str, float | int]:
    """Calculate observed partition, coverage, overlap, and small-cluster metrics."""
    real = result["real_communities"]
    membership = result["real_membership"]
    real_links = sum(map(len, real))
    total_links = len(result["links"])
    link_counts = [len(edges) for edges in real]
    overlap_counts = [len(ids) for ids in membership.values() if len(ids) >= 2]
    return {
        "D_star": float(result["best_partition_density"]),
        "cut_similarity": float(result["best_similarity"]),
        "num_communities": len(real),
        "links_in_real_pct": 100.0 * real_links / total_links if total_links else 0.0,
        "covered_nodes": len(membership),
        "overlap_nodes": len(overlap_counts),
        "communities_per_overlap_node": (
            float(np.mean(overlap_counts)) if overlap_counts else np.nan
        ),
        "mean_communities_per_covered_node": (
            float(np.mean(list(map(len, membership.values())))) if membership else 0.0
        ),
        "largest_community_link_fraction": (
            max(link_counts) / total_links if total_links else 0.0
        ),
        "small_community_count": len(result["small_communities"]),
        "small_community_links": sum(map(len, result["small_communities"])),
        "small_community_nodes": len(result["small_nodes"]),
        "nodes_only_in_small_communities": len(
            result["nodes_only_in_small_communities"]
        ),
    }


def automatic_adaptive_temperature(n_links: int, single_density: float) -> float:
    """Use the link-community auto-temperature rule documented by the repository."""
    if n_links < 1:
        raise ValueError("At least one link is needed to set the temperature.")
    factor = 0.3 if single_density > 0.7 else 0.01 if single_density < 0.01 else 1.0
    return float(np.clip(0.1 * factor / n_links, 1e-8, 1e-3))


def _temperature_at(
    temperature: float, step: int, cooling_start: int
) -> float:
    """Hold T through cooling_start, then follow the repository's log schedule."""
    if step <= cooling_start:
        return temperature
    return temperature / np.log(step - cooling_start + 2)


def run_adaptive_walk(
    result: dict[str, object],
    setting: str,
    seed: int,
    n_steps: int = ADAPTIVE_STEPS,
) -> dict[str, object]:
    """Run the specified single-community up/down Metropolis walk on the tree."""
    if setting not in ADAPTIVE_SETTINGS:
        raise ValueError(f"Unknown adaptive temperature setting: {setting}")
    if n_steps < 1:
        raise ValueError("n_steps must be positive.")
    n_links = len(result["links"])
    if n_links == 0:
        raise ValueError("Adaptive cuts require at least one link.")

    tree = result["dendrogram"]
    nodes = tree["nodes"]
    parents = tree["parent"]
    config = ADAPTIVE_SETTINGS[setting]
    temperature = automatic_adaptive_temperature(
        n_links, result["best_partition_density"]
    ) * config["temperature_scale"]
    cooling_start = int(n_steps * config["cool_start_fraction"])
    rng = np.random.default_rng(seed)
    current = set(result["best_cut_nodes"])
    current_list = sorted(current)
    current_contribution = sum(nodes[node]["partition_contribution"] for node in current)
    best = set(current)
    best_density = 2.0 * current_contribution / n_links
    accepted = 0
    proposed = 0

    for step in range(n_steps):
        selected = current_list[int(rng.integers(len(current_list)))]
        children = nodes[selected]["children"]
        actions = []
        if selected in parents:
            actions.append("up")
        if children is not None:
            actions.append("down")
        if not actions:
            continue
        proposed += 1
        action = actions[int(rng.integers(len(actions)))]

        if action == "down":
            left, right = children
            proposed_cut = (current - {selected}) | {left, right}
            delta_contribution = (
                nodes[left]["partition_contribution"]
                + nodes[right]["partition_contribution"]
                - nodes[selected]["partition_contribution"]
            )
        else:
            parent = parents[selected]
            parent_links = nodes[parent]["link_ids"]
            removed = [
                node
                for node in current
                if nodes[node]["link_ids"] <= parent_links
            ]
            proposed_cut = (current - set(removed)) | {parent}
            delta_contribution = (
                nodes[parent]["partition_contribution"]
                - sum(nodes[node]["partition_contribution"] for node in removed)
            )

        delta_density = 2.0 * delta_contribution / n_links
        temperature_at_step = _temperature_at(temperature, step, cooling_start)
        if delta_density >= 0 or rng.random() < np.exp(
            delta_density / temperature_at_step
        ):
            current = proposed_cut
            if action == "down":
                current_list.remove(selected)
                current_list.extend((left, right))
            else:
                removed_set = set(removed)
                current_list = [
                    node for node in current_list if node not in removed_set
                ]
                current_list.append(parent)
            current_contribution += delta_contribution
            accepted += 1
            current_density = 2.0 * current_contribution / n_links
            if current_density > best_density + 1e-14:
                best_density = current_density
                best = set(current)

    best_partition = result_for_cut(result, best)
    final_partition = result_for_cut(result, current)
    assert np.isclose(best_density, best_partition["best_partition_density"])
    assert validate_result(best_partition)
    assert validate_result(final_partition)
    return {
        "setting": setting,
        "seed": seed,
        "temperature": temperature,
        "cooling_start": cooling_start,
        "n_steps": n_steps,
        "best_cut_nodes": tuple(sorted(best)),
        "best_D": best_partition["best_partition_density"],
        "best_num_clusters": len(best),
        "best_partition": best_partition,
        "final_cut_nodes": tuple(sorted(current)),
        "final_D": final_partition["best_partition_density"],
        "final_num_clusters": len(current),
        "n_proposed": proposed,
        "n_accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "step_acceptance_rate": accepted / n_steps,
    }


def _adaptive_run_row(
    threshold: int, walk: dict[str, object], single_result: dict[str, object]
) -> dict[str, object]:
    metrics = _result_metrics(walk["best_partition"])
    return {
        "w_min": threshold,
        "setting": walk["setting"],
        "seed": walk["seed"],
        "temperature": walk["temperature"],
        "cooling_start": walk["cooling_start"],
        "D_adaptive": walk["best_D"],
        "D_single": single_result["best_partition_density"],
        "D_gain": walk["best_D"] - single_result["best_partition_density"],
        "all_cut_communities": walk["best_num_clusters"],
        "final_communities": walk["final_num_clusters"],
        "acceptance_rate": walk["acceptance_rate"],
        "step_acceptance_rate": walk["step_acceptance_rate"],
        "n_proposed": walk["n_proposed"],
        "n_accepted": walk["n_accepted"],
        **{
            f"adaptive_{key}": value
            for key, value in metrics.items()
            if key
            in {
                "num_communities",
                "links_in_real_pct",
                "covered_nodes",
                "overlap_nodes",
                "communities_per_overlap_node",
                "largest_community_link_fraction",
            }
        },
    }


def run_adaptive_sensitivity(
    single_results: dict[int, dict[str, object]],
    seeds: Iterable[int] = ADAPTIVE_SEEDS,
    n_steps: int = ADAPTIVE_STEPS,
    seed: int = ADAPTIVE_WALK_SEED,
) -> tuple[pd.DataFrame, dict[tuple[int, str, int], dict[str, object]]]:
    """Run paired seed/settings sensitivity walks for each threshold dendrogram."""
    seeds = tuple(seeds)
    rows = []
    walks = {}
    for threshold_index, (threshold, single_result) in enumerate(single_results.items()):
        for setting_index, setting in enumerate(ADAPTIVE_SETTINGS):
            for replicate_seed in seeds:
                walk_seed = seed + threshold_index * 100_000 + int(replicate_seed)
                walk = run_adaptive_walk(
                    single_result, setting, walk_seed, n_steps=n_steps
                )
                rows.append(_adaptive_run_row(threshold, walk, single_result))
                walks[(threshold, setting, walk_seed)] = walk
        print(
            f"w_min={threshold}: completed {len(ADAPTIVE_SETTINGS) * len(seeds)} "
            "adaptive sensitivity walks",
            flush=True,
        )
    return pd.DataFrame(rows), walks


def adaptive_setting_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize D ranges, movement, and coarse-collapse diagnostics by setting."""
    rows = []
    for (threshold, setting), group in runs.groupby(["w_min", "setting"], sort=True):
        collapse_rate = float(group["final_communities"].le(2).mean())
        median_acceptance = float(group["acceptance_rate"].median())
        rows.append(
            {
                "w_min": threshold,
                "setting": setting,
                "D_min": float(group["D_adaptive"].min()),
                "D_median": float(group["D_adaptive"].median()),
                "D_max": float(group["D_adaptive"].max()),
                "median_D_gain": float(group["D_gain"].median()),
                "median_acceptance_rate": median_acceptance,
                "final_collapse_rate": collapse_rate,
                "too_warm": collapse_rate >= 0.5,
                "too_cold": median_acceptance < 0.01,
            }
        )
    return pd.DataFrame(rows)


def choose_adaptive_setting(
    runs: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    """Choose the highest-median-D setting, avoiding warm/cold diagnostics."""
    summaries = adaptive_setting_summary(runs)
    eligible = summaries[~summaries["too_warm"] & ~summaries["too_cold"]]
    pool = eligible if not eligible.empty else summaries
    selected = pool.sort_values(
        ["D_median", "median_D_gain", "setting"],
        ascending=[False, False, True],
    ).iloc[0]["setting"]
    return str(selected), summaries


def summarize_adaptive_partition(
    single_result: dict[str, object],
    setting_runs: pd.DataFrame,
    walks: dict[tuple[int, str, int], dict[str, object]],
    threshold: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """Select the most representative seed (closest to median D) for a setting."""
    selected_setting, setting_summary = choose_adaptive_setting(setting_runs)
    chosen_runs = setting_runs[setting_runs["setting"] == selected_setting]
    median_density = float(chosen_runs["D_adaptive"].median())
    representative = chosen_runs.assign(
        median_distance=(chosen_runs["D_adaptive"] - median_density).abs()
    ).sort_values(["median_distance", "seed"])
    seed = int(representative.iloc[0]["seed"])
    walk = walks[(threshold, selected_setting, seed)]
    adaptive_result = walk["best_partition"]
    single_metrics = _result_metrics(single_result)
    adaptive_metrics = _result_metrics(adaptive_result)
    summary = {
        "w_min": threshold,
        "single_D": single_metrics["D_star"],
        "adaptive_D_median": median_density,
        "adaptive_D_min": float(chosen_runs["D_adaptive"].min()),
        "adaptive_D_max": float(chosen_runs["D_adaptive"].max()),
        "adaptive_representative_D": float(walk["best_D"]),
        "adaptive_setting": selected_setting,
        "representative_seed": seed,
        "single_communities": single_metrics["num_communities"],
        "adaptive_communities": adaptive_metrics["num_communities"],
        "single_links_in_real_pct": single_metrics["links_in_real_pct"],
        "adaptive_links_in_real_pct": adaptive_metrics["links_in_real_pct"],
        "single_covered_nodes": single_metrics["covered_nodes"],
        "adaptive_covered_nodes": adaptive_metrics["covered_nodes"],
        "single_overlap_nodes": single_metrics["overlap_nodes"],
        "adaptive_overlap_nodes": adaptive_metrics["overlap_nodes"],
        "single_small_community_count": single_metrics["small_community_count"],
        "adaptive_small_community_count": adaptive_metrics["small_community_count"],
        "single_small_community_links": single_metrics["small_community_links"],
        "adaptive_small_community_links": adaptive_metrics[
            "small_community_links"
        ],
        "single_small_community_nodes": single_metrics["small_community_nodes"],
        "adaptive_small_community_nodes": adaptive_metrics["small_community_nodes"],
        "single_nodes_only_in_small_communities": single_metrics[
            "nodes_only_in_small_communities"
        ],
        "adaptive_nodes_only_in_small_communities": adaptive_metrics[
            "nodes_only_in_small_communities"
        ],
        "adaptive_communities_per_overlap_node": adaptive_metrics[
            "communities_per_overlap_node"
        ],
        "setting_selection_rule": (
            "highest median D across 8 seeds; exclude settings with >=50% "
            "final cuts at <=2 clusters or median acceptance below 1%"
        ),
    }
    return summary, adaptive_result


def run_adaptive_null(
    graph: nx.Graph,
    threshold: int,
    n_shuffles: int = ADAPTIVE_NULL_REPLICATES,
    seeds: Iterable[int] = ADAPTIVE_SEEDS,
    n_steps: int = ADAPTIVE_STEPS,
    shuffle_seed: int = ADAPTIVE_NULL_SEED,
    walk_seed: int = ADAPTIVE_WALK_SEED + 1_000_000,
) -> pd.DataFrame:
    """Optimize each degree-preserving null over the same settings and seed count."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be at least one.")
    seeds = tuple(seeds)
    filtered = threshold_graph(graph, threshold)
    rows = []
    for replicate in range(n_shuffles):
        shuffled = _rewire_graph(
            filtered,
            DEFAULT_SWAPS_PER_EDGE,
            shuffle_seed + replicate,
        )
        single_result = cluster_links(shuffled)
        null_runs = []
        for setting_index, setting in enumerate(ADAPTIVE_SETTINGS):
            for replicate_seed in seeds:
                walk = run_adaptive_walk(
                    single_result,
                    setting,
                    walk_seed
                    + replicate * 100_000
                    + setting_index * 1_000
                    + int(replicate_seed),
                    n_steps=n_steps,
                )
                null_runs.append(_adaptive_run_row(threshold, walk, single_result))
        null_run_table = pd.DataFrame(null_runs)
        selected_setting, _ = choose_adaptive_setting(null_run_table)
        chosen = null_run_table[null_run_table["setting"] == selected_setting]
        rows.append(
            {
                "replicate": replicate + 1,
                "D_adaptive": float(chosen["D_adaptive"].median()),
                "selected_setting": selected_setting,
                "median_final_communities": float(chosen["final_communities"].median()),
            }
        )
        if (replicate + 1) % 5 == 0 or replicate + 1 == n_shuffles:
            print(
                f"  w_min={threshold}: optimized null graph "
                f"{replicate + 1}/{n_shuffles}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_adaptive_null(
    observed_density: float, null: pd.DataFrame
) -> dict[str, float | int]:
    """Report mean, percentile interval, and +1 corrected upper-tail p-value."""
    values = null["D_adaptive"].to_numpy(dtype=float)
    return {
        "adaptive_null_replicates": len(values),
        "adaptive_null_mean": float(values.mean()),
        "adaptive_null_q025": float(np.quantile(values, 0.025)),
        "adaptive_null_q975": float(np.quantile(values, 0.975)),
        "adaptive_null_p_upper": float(
            (1 + np.count_nonzero(values >= observed_density)) / (len(values) + 1)
        ),
    }


def compare_changed_communities(
    single_result: dict[str, object],
    adaptive_result: dict[str, object],
    name_map: dict[str, str],
    max_examples: int = 3,
) -> pd.DataFrame:
    """Find adaptive real communities merging or splitting single-cut branches."""
    single_by_link = {
        edge: index
        for index, group in enumerate(single_result["communities"], start=1)
        for edge in group
    }
    adaptive_by_link = {
        edge: index
        for index, group in enumerate(adaptive_result["communities"], start=1)
        for edge in group
    }
    rows = []
    for edges in adaptive_result["real_communities"]:
        sources = sorted({single_by_link[edge] for edge in edges})
        if len(sources) < 2:
            continue
        adaptive_id = adaptive_by_link[edges[0]]
        source_edges = [
            edge
            for source in sources
            for edge in single_result["communities"][source - 1]
        ]
        rows.append(
            {
                "change": "adaptive merge",
                "single communities": ", ".join(f"S{i}" for i in sources),
                "adaptive communities": f"A{adaptive_id}",
                "single link counts": ", ".join(
                    str(len(single_result["communities"][i - 1])) for i in sources
                ),
                "adaptive link counts": str(len(edges)),
                "single members": ", ".join(
                    name_map.get(node, node)
                    for node in sorted(
                        {x for edge in source_edges for x in edge},
                        key=lambda value: name_map.get(value, value).casefold(),
                    )
                ),
                "adaptive members": ", ".join(
                    name_map.get(node, node)
                    for node in sorted(
                        {x for edge in edges for x in edge},
                        key=lambda value: name_map.get(value, value).casefold(),
                    )
                ),
                "links affected": len(edges),
            }
        )
    for edges in single_result["real_communities"]:
        single_id = single_by_link[edges[0]]
        destinations = sorted({adaptive_by_link[edge] for edge in edges})
        destination_groups = [
            adaptive_result["communities"][index - 1] for index in destinations
        ]
        real_destinations = [
            index
            for index in destinations
            if len(adaptive_result["communities"][index - 1])
            >= adaptive_result["min_links"]
        ]
        if len(real_destinations) < 2:
            continue
        rows.append(
            {
                "change": "adaptive split",
                "single communities": f"S{single_id}",
                "adaptive communities": ", ".join(
                    f"A{i}" for i in real_destinations
                ),
                "single link counts": str(len(edges)),
                "adaptive link counts": ", ".join(
                    str(len(adaptive_result["communities"][i - 1]))
                    for i in real_destinations
                ),
                "single members": ", ".join(
                    name_map.get(node, node)
                    for node in sorted(
                        {x for edge in edges for x in edge},
                        key=lambda value: name_map.get(value, value).casefold(),
                    )
                ),
                "adaptive members": " | ".join(
                    ", ".join(
                        name_map.get(node, node)
                        for node in sorted(
                            {x for edge in group for x in edge},
                            key=lambda value: name_map.get(value, value).casefold(),
                        )
                    )
                    for group in destination_groups
                    if len(group) >= adaptive_result["min_links"]
                ),
                "links affected": len(edges),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "change", "single communities", "adaptive communities",
                "single link counts", "adaptive link counts", "single members",
                "adaptive members", "links affected",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("links affected", ascending=False)
        .groupby("change", group_keys=False)
        .head(max_examples)
        .reset_index(drop=True)
    )


def plot_changed_communities(changed: pd.DataFrame):
    """Show member-name labels side by side for communities changed by the cut."""
    if changed.empty:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(0.5, 0.5, "No real communities changed between cuts.", ha="center")
        ax.axis("off")
        fig.tight_layout()
        return fig, np.asarray([ax])

    panels = []
    for _, row in changed.iterrows():
        single_members = textwrap.fill(row["single members"], width=48)
        adaptive_members = textwrap.fill(
            row["adaptive members"].replace(" | ", "\n\n"), width=48
        )
        panels.append((single_members, adaptive_members))
    line_count = max(
        max(text.count("\n") + 1 for text in panel) for panel in panels
    )
    height = max(3.2, 1.15 + 0.19 * line_count) * len(changed)
    fig, axes = plt.subplots(
        len(changed), 2, figsize=(14, height), squeeze=False
    )
    for index, (_, row) in enumerate(changed.iterrows()):
        for column, (members, title) in enumerate(
            (
                (
                    panels[index][0],
                    f"Single-height {row['single communities']} "
                    f"({row['single link counts']} links)",
                ),
                (
                    panels[index][1],
                    f"Adaptive {row['adaptive communities']} "
                    f"({row['adaptive link counts']} links)",
                ),
            )
        ):
            ax = axes[index, column]
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.text(
                0.02, 0.96, members, ha="left", va="top",
                fontsize=8, transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
    fig.suptitle(
        "Changed link communities (character names label memberships)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig, axes


def _null_statistics(observed: float, values: np.ndarray) -> dict[str, float]:
    """Summarize a null metric with population z-score and corrected tail p-values."""
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    p_lower = (1 + np.count_nonzero(values <= observed)) / (len(values) + 1)
    p_upper = (1 + np.count_nonzero(values >= observed)) / (len(values) + 1)
    return {
        "null_mean": mean,
        "null_std": std,
        "z_score": (observed - mean) / std if std > 0 else np.nan,
        "p_value": float(min(1.0, 2 * min(p_lower, p_upper))),
        "p_upper": float(p_upper),
    }


def _rewire_graph(graph: nx.Graph, swaps_per_edge: int, seed: int) -> nx.Graph:
    """Return one simple graph after exactly swaps_per_edge * m double swaps."""
    if graph.is_directed() or graph.is_multigraph():
        raise TypeError("Degree-preserving shuffles require a simple undirected graph.")
    shuffled = graph.copy()
    nswap = swaps_per_edge * shuffled.number_of_edges()
    if nswap and shuffled.number_of_edges() < 2:
        raise ValueError("At least two edges are needed for double-edge swaps.")
    if nswap:
        nx.double_edge_swap(
            shuffled,
            nswap=nswap,
            max_tries=max(1000, 100 * nswap),
            seed=seed,
        )
    return shuffled


def test_partition_density(
    graph: nx.Graph,
    observed: dict[str, object] | None = None,
    n_shuffles: int = DEFAULT_NULL_REPLICATES,
    swaps_per_edge: int = DEFAULT_SWAPS_PER_EDGE,
    seed: int = DEFAULT_NULL_SEED,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Compare D*, real-community count and overlap with degree-preserving nulls."""
    if n_shuffles < 1:
        raise ValueError("n_shuffles must be at least one.")
    observed_result = observed or cluster_links(graph)
    observed_metrics = _result_metrics(observed_result)
    null_rows = []
    for replicate in range(n_shuffles):
        shuffled = _rewire_graph(graph, swaps_per_edge, seed + replicate)
        null_result = cluster_links(shuffled)
        null_rows.append(
            {
                "replicate": replicate + 1,
                **_result_metrics(null_result),
            }
        )
        if (replicate + 1) % 10 == 0 or replicate + 1 == n_shuffles:
            print(f"  analyzed null graph {replicate + 1}/{n_shuffles}", flush=True)

    null = pd.DataFrame(null_rows)
    metric_names = (
        "D_star",
        "num_communities",
        "overlap_nodes",
        "mean_communities_per_covered_node",
    )
    summary: dict[str, float | int] = {}
    for metric in metric_names:
        statistics = _null_statistics(
            float(observed_metrics[metric]),
            null[metric].to_numpy(dtype=float),
        )
        for statistic, value in statistics.items():
            summary[f"{metric}_{statistic}"] = value
    return summary, null


def analyze_thresholds(
    graph: nx.Graph,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
    n_shuffles: int = DEFAULT_NULL_REPLICATES,
    swaps_per_edge: int = DEFAULT_SWAPS_PER_EDGE,
    seed: int = DEFAULT_NULL_SEED,
) -> tuple[pd.DataFrame, dict[int, dict[str, object]], dict[int, pd.DataFrame]]:
    """Run the same unweighted link clustering and null test at each threshold."""
    rows = []
    results: dict[int, dict[str, object]] = {}
    null_distributions: dict[int, pd.DataFrame] = {}
    for threshold_index, threshold in enumerate(thresholds):
        filtered = threshold_graph(graph, threshold)
        print(
            f"w_min={threshold}: clustering {filtered.number_of_edges()} links; "
            f"generating {n_shuffles} null graphs",
            flush=True,
        )
        result = cluster_links(filtered)
        validate_result(result)
        null_summary, null = test_partition_density(
            filtered,
            observed=result,
            n_shuffles=n_shuffles,
            swaps_per_edge=swaps_per_edge,
            seed=seed + threshold_index * 100_000,
        )
        rows.append(
            {
                "w_min": threshold,
                "nodes": filtered.number_of_nodes(),
                "links": filtered.number_of_edges(),
                "similarity_pairs": result["similarity_pairs"],
                **_result_metrics(result),
                **null_summary,
            }
        )
        results[threshold] = result
        null_distributions[threshold] = null

    summary = pd.DataFrame(rows)
    n_tests = len(summary)
    if n_tests:
        summary["D_star_p_upper_adjusted"] = np.minimum(
            1.0, summary["D_star_p_upper"] * n_tests
        )
    return summary, results, null_distributions


def recommend_setting(summary: pd.DataFrame) -> tuple[int, str]:
    """Select a significant, covered, non-dominated setting when one exists."""
    if summary.empty:
        raise ValueError("Cannot recommend a threshold from an empty summary.")
    significant = summary["D_star_p_upper_adjusted"].le(0.05)
    well_covered = summary["links_in_real_pct"].ge(50.0)
    not_dominated = summary["largest_community_link_fraction"].le(0.50)
    eligible = summary[significant & well_covered & not_dominated]
    if not eligible.empty:
        row = eligible.sort_values(
            ["D_star", "links_in_real_pct"], ascending=False
        ).iloc[0]
        reason = (
            "selected among thresholds with Bonferroni-adjusted upper-tail p <= 0.05, "
            "at least 50% real-community link coverage, and no community holding "
            "more than half of all links; highest D* among those candidates"
        )
    else:
        row = summary.sort_values(
            ["D_star_p_upper_adjusted", "D_star", "links_in_real_pct"],
            ascending=[True, False, False],
        ).iloc[0]
        reason = (
            "no threshold met all significance, coverage, and largest-community "
            "criteria; selected the strongest adjusted upper-tail evidence, then "
            "D* and coverage. Treat the choice as exploratory"
        )
    return int(row["w_min"]), reason


def community_table(
    graph: nx.Graph,
    result: dict[str, object],
    name_map: dict[str, str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Summarize real link communities and their strongest characters."""
    memberships = result["real_membership"]
    rows = []
    for community_id, edges in enumerate(result["real_communities"], start=1):
        nodes = {node for edge in edges for node in edge}
        strength: Counter[str] = Counter()
        for u, v in edges:
            strength[u] += graph[u][v].get("weight", 1.0)
            strength[v] += graph[u][v].get("weight", 1.0)
        ordered = sorted(
            nodes,
            key=lambda node: (-strength[node], name_map.get(node, node).casefold()),
        )
        rows.append(
            {
                "community": community_id,
                "number of links": len(edges),
                "number of nodes": len(nodes),
                "top characters by weighted degree": ", ".join(
                    name_map.get(node, node) for node in ordered[:top_n]
                ),
                "overlap characters": ", ".join(
                    name_map.get(node, node)
                    for node in ordered
                    if len(memberships.get(node, [])) > 1
                ),
                "strongest character": name_map.get(ordered[0], ordered[0]),
            }
        )
    return pd.DataFrame(rows)


def community_link_table(
    graph: nx.Graph,
    result: dict[str, object],
    name_map: dict[str, str],
) -> pd.DataFrame:
    """Return a long table of links in reported (>=3-link) communities."""
    rows = []
    for community_id, edges in enumerate(result["real_communities"], start=1):
        for u, v in edges:
            rows.append(
                {
                    "community": community_id,
                    "link source": name_map.get(u, u),
                    "link target": name_map.get(v, v),
                    "weight": graph[u][v].get("weight", 1.0),
                }
            )
    return pd.DataFrame(rows)


def character_community_table(
    graph: nx.Graph,
    result: dict[str, object],
    name_map: dict[str, str],
) -> tuple[pd.DataFrame, float]:
    """Count real-community memberships and links per character; correlate with degree."""
    memberships = result["real_membership"]
    rows = []
    for node in graph.nodes:
        degree = graph.degree(node)
        count = len(memberships.get(node, []))
        rows.append(
            {
                "character": name_map.get(node, node),
                "character_id": node,
                "communities": count,
                "degree": degree,
                "communities per link": count / degree if degree else np.nan,
                "weighted degree": graph.degree(node, weight="weight"),
                "overlap": count >= 2,
            }
        )
    table = pd.DataFrame(rows)
    nonisolated = table[table["degree"] > 0]
    if (
        len(nonisolated) > 1
        and nonisolated["degree"].nunique() > 1
        and nonisolated["communities"].nunique() > 1
    ):
        rho = float(
            spearmanr(nonisolated["degree"], nonisolated["communities"]).statistic
        )
    else:
        rho = np.nan
    table = table.sort_values(
        ["communities", "weighted degree", "character"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return table, rho


def compare_kclique_overlaps(
    graph: nx.Graph,
    result: dict[str, object],
    name_map: dict[str, str],
    settings: Iterable[tuple[int, int]] = ((5, 2), (8, 3)),
) -> pd.DataFrame:
    """Compare link-community overlap characters with selected k-clique overlaps."""
    link_overlap = {
        node for node, groups in result["real_membership"].items() if len(groups) > 1
    }
    rows = []
    for k, threshold in settings:
        groups = get_k_clique_communities(threshold_graph(graph, threshold), k)
        membership: dict[str, int] = defaultdict(int)
        for group in groups:
            for node in group:
                membership[node] += 1
        clique_overlap = {node for node, count in membership.items() if count > 1}
        union = link_overlap | clique_overlap
        intersection = link_overlap & clique_overlap
        rows.append(
            {
                "comparison": f"k={k}, w_min={threshold}",
                "link overlap nodes": len(link_overlap),
                "k-clique overlap nodes": len(clique_overlap),
                "intersection": len(intersection),
                "union": len(union),
                "Jaccard": len(intersection) / len(union) if union else np.nan,
                "in both": ", ".join(
                    name_map.get(node, node)
                    for node in sorted(intersection, key=lambda x: name_map.get(x, x))
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_with_louvain(
    graph: nx.Graph,
    result: dict[str, object],
    seeds: Iterable[int] = LOUVAIN_SEEDS,
    n_label_shuffles: int = 1000,
    seed: int = LABEL_SHUFFLE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare dominant real link labels with Louvain and a shuffled-label baseline."""
    seeds = tuple(seeds)
    if n_label_shuffles < 1:
        raise ValueError("n_label_shuffles must be at least one.")
    covered = sorted(result["real_membership"])
    if len(covered) < 2:
        return pd.DataFrame(), pd.DataFrame()
    community_labels: dict[str, list[int]] = defaultdict(list)
    for community_id, edges in enumerate(result["real_communities"], start=1):
        for u, v in edges:
            community_labels[u].append(community_id)
            community_labels[v].append(community_id)
    dominant = {}
    for node in covered:
        incident_counts = Counter(community_labels[node])
        dominant[node] = min(
            incident_counts,
            key=lambda community_id: (-incident_counts[community_id], community_id),
        )
    link_labels = [dominant[node] for node in covered]

    louvain_runs = [
        nx.community.louvain_communities(graph, seed=run_seed, weight="weight")
        for run_seed in seeds
    ]
    louvain_labels = [
        {
            node: community_id
            for community_id, group in enumerate(groups)
            for node in group
        }
        for groups in louvain_runs
    ]
    link_louvain_scores = [
        normalized_mutual_info_score(
            link_labels, [labels[node] for node in covered]
        )
        for labels in louvain_labels
    ]
    pairwise_scores = [
        normalized_mutual_info_score(
            [left[node] for node in covered],
            [right[node] for node in covered],
        )
        for left, right in combinations(louvain_labels, 2)
    ]
    rng = np.random.default_rng(seed)
    shuffled_scores = np.empty(n_label_shuffles, dtype=float)
    labels_array = np.asarray(link_labels)
    for index in range(n_label_shuffles):
        permuted = rng.permutation(labels_array)
        shuffled_scores[index] = np.mean(
            [
                normalized_mutual_info_score(
                    permuted, [labels[node] for node in covered]
                )
                for labels in louvain_labels
            ]
        )
    observed_mean = float(np.mean(link_louvain_scores))
    null_mean = float(shuffled_scores.mean())
    summary = pd.DataFrame(
        [
            {
                "covered characters": len(covered),
                "link-vs-Louvain NMI mean": observed_mean,
                "link-vs-Louvain NMI std": float(np.std(link_louvain_scores, ddof=0)),
                "Louvain-vs-Louvain NMI ceiling": float(np.mean(pairwise_scores)),
                "shuffled-label NMI mean": null_mean,
                "shuffled-label NMI std": float(np.std(shuffled_scores, ddof=0)),
                "shuffled-label p_upper": float(
                    (1 + np.count_nonzero(shuffled_scores >= observed_mean))
                    / (n_label_shuffles + 1)
                ),
            }
        ]
    )
    per_seed = pd.DataFrame(
        {
            "seed": list(seeds),
            "link-vs-Louvain NMI": link_louvain_scores,
        }
    )
    return summary, per_seed


def plot_density_sweep(
    summary: pd.DataFrame,
    results: dict[int, dict[str, object]],
    null_distributions: dict[int, pd.DataFrame],
):
    """Plot each D curve with a horizontal null D* mean +/- two standard deviations."""
    thresholds = summary["w_min"].tolist()
    columns = min(3, max(1, len(thresholds)))
    rows = int(np.ceil(len(thresholds) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, threshold in zip(axes, thresholds):
        result = results[threshold]
        curve = result["density_curve"].dropna(subset=["similarity"])
        null_values = null_distributions[threshold]["D_star"].to_numpy(dtype=float)
        null_mean = float(null_values.mean())
        null_std = float(null_values.std(ddof=0))
        ax.plot(
            curve["similarity"],
            curve["partition_density"],
            color="tab:blue",
            marker=".",
            linewidth=1.1,
            label="Observed D(cut)",
        )
        ax.axhline(null_mean, color="tab:orange", linestyle="--", label="Null mean D*")
        ax.axhspan(
            null_mean - 2 * null_std,
            null_mean + 2 * null_std,
            color="tab:orange",
            alpha=0.16,
            label="Null D* mean +/- 2 std",
        )
        if np.isfinite(result["best_similarity"]):
            ax.axvline(
                result["best_similarity"],
                color="tab:red",
                linestyle=":",
                label=f"maximum at S={result['best_similarity']:.3f}",
            )
        ax.invert_xaxis()
        ax.set(
            title=f"w_min={threshold}; D*={result['best_partition_density']:.3f}",
            xlabel="Similarity cut (descending merge order)",
            ylabel="Partition density",
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    for ax in axes[len(thresholds):]:
        ax.axis("off")
    fig.tight_layout()
    return fig, axes


def plot_adaptive_comparison(summary: pd.DataFrame):
    """Compare single-height D with median adaptive D and its eight-seed range."""
    data = summary.sort_values("w_min")
    x = data["w_min"].to_numpy(dtype=float)
    adaptive = data["adaptive_D_median"].to_numpy(dtype=float)
    lower = adaptive - data["adaptive_D_min"].to_numpy(dtype=float)
    upper = data["adaptive_D_max"].to_numpy(dtype=float) - adaptive
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, data["single_D"], marker="o", label="Single-height cut")
    ax.errorbar(
        x,
        adaptive,
        yerr=np.vstack([lower, upper]),
        marker="s",
        capsize=4,
        label="Adaptive cut (median; min-max over 8 seeds)",
    )
    ax.set(
        xticks=x,
        xlabel="Minimum edge weight",
        ylabel="Partition density D",
        title="Single-height versus adaptive partition density",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def compute_link_layout(
    graph: nx.Graph,
    result: dict[str, object],
    seed: int = LAYOUT_SEED,
    intra_boost: float = 4.0,
    outward: float = 0.25,
    jitter: float = 0.12,
) -> dict[str, np.ndarray]:
    """Lay out covered nodes first and place other characters near positioned neighbors."""
    membership = result["real_membership"]
    covered = [node for node in graph if node in membership]
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(covered)
    for u, v, data in graph.subgraph(covered).edges(data=True):
        shared_communities = set(membership[u]) & set(membership[v])
        weight = np.log1p(data.get("weight", 1.0))
        layout_graph.add_edge(
            u,
            v,
            layout_weight=weight * (intra_boost if shared_communities else 1.0),
        )

    if layout_graph:
        pos = nx.spring_layout(
            layout_graph,
            weight="layout_weight",
            seed=seed,
            k=5.0 / np.sqrt(len(layout_graph)),
            iterations=800,
        )
        centre = np.mean(list(pos.values()), axis=0)
    else:
        pos = {}
        centre = np.zeros(2)
    rng = np.random.default_rng(seed)
    pending = [node for node in graph if node not in pos]
    for _ in range(max(5, len(pending))):
        for node in list(pending):
            neighbors = [
                (other, np.log1p(graph[node][other].get("weight", 1.0)))
                for other in graph[node]
                if other in pos
            ]
            if not neighbors:
                continue
            weights = np.asarray([weight for _, weight in neighbors])
            coordinates = np.asarray([pos[other] for other, _ in neighbors])
            point = (coordinates * weights[:, None]).sum(axis=0) / weights.sum()
            direction = point - centre
            direction /= np.linalg.norm(direction) + 1e-9
            pos[node] = point + outward * direction + rng.normal(0, jitter, 2)
            pending.remove(node)
        if not pending:
            break
    remaining_graph = graph.subgraph(pending)
    remaining_components = list(nx.connected_components(remaining_graph))
    for component_index, component in enumerate(remaining_components):
        angle = 2 * np.pi * component_index / max(len(remaining_components), 1)
        if len(component) > 1:
            local_graph = graph.subgraph(component).copy()
            local = nx.spring_layout(
                local_graph,
                seed=seed + component_index + 1,
                weight="weight",
            )
            local_centre = np.mean(list(local.values()), axis=0)
            offset = centre + (1.4 + 0.25 * component_index) * np.asarray(
                [np.cos(angle), np.sin(angle)]
            )
            for node, point in local.items():
                pos[node] = offset + 0.45 * (point - local_centre)
        else:
            node = next(iter(component))
            pos[node] = centre + 1.2 * np.asarray([np.cos(angle), np.sin(angle)])
    return pos


def plot_link_communities(
    graph: nx.Graph,
    result: dict[str, object],
    name_map: dict[str, str],
    w_min: int,
    pos: dict[str, np.ndarray] | None = None,
    seed: int = LAYOUT_SEED,
    figsize: tuple[float, float] = (14, 11),
):
    """Draw links by community, tiny clusters in gray and overlap nodes as pies."""
    if pos is None:
        pos = compute_link_layout(graph, result, seed=seed)
    communities = result["real_communities"]
    membership = result["real_membership"]
    palette = plt.get_cmap("tab10" if len(communities) <= 10 else "tab20")
    colors = [palette(index % palette.N) for index in range(len(communities))]
    leaders = []
    for edges in communities:
        strength: Counter[str] = Counter()
        for u, v in edges:
            weight = graph[u][v].get("weight", 1.0)
            strength[u] += weight
            strength[v] += weight
        leader = min(
            strength,
            key=lambda node: (-strength[node], name_map.get(node, node).casefold()),
        )
        leaders.append(leader)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    for color, edges in zip(colors, communities):
        nx.draw_networkx_edges(
            graph, pos, edgelist=edges, ax=ax, alpha=0.7, width=1.4, edge_color=[color]
        )
    small_edges = [
        edge for community in result["small_communities"] for edge in community
    ]
    if small_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=small_edges,
            ax=ax,
            alpha=0.38,
            width=0.8,
            edge_color="#c8c8c8",
        )

    weighted_degree = dict(graph.degree(weight="weight"))
    uncovered = [node for node in graph if node not in membership]
    single = [node for node in graph if len(membership.get(node, [])) == 1]
    overlap = [node for node in graph if len(membership.get(node, [])) > 1]
    size = lambda node: 40 + 10 * np.sqrt(weighted_degree.get(node, 0))
    nx.draw_networkx_nodes(
        graph, pos, nodelist=uncovered, node_color="#d9d9d9",
        node_size=[size(node) * 0.65 for node in uncovered],
        edgecolors="#999999", linewidths=0.4, ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=single,
        node_color=[colors[membership[node][0] - 1] for node in single],
        node_size=[size(node) for node in single],
        edgecolors="#222222",
        linewidths=0.6,
        ax=ax,
    )
    span = max(
        np.ptp([point[0] for point in pos.values()]),
        np.ptp([point[1] for point in pos.values()]),
        1e-6,
    )
    for node in overlap:
        radius = span * 0.0038 * np.sqrt(size(node) / 40) * 1.5
        ax.pie(
            [1] * len(membership[node]),
            colors=[colors[index - 1] for index in membership[node]],
            center=pos[node],
            radius=radius,
            wedgeprops={"edgecolor": "black", "linewidth": 1.2},
            frame=True,
        )
    for node in dict.fromkeys(leaders + overlap):
        ax.annotate(
            name_map.get(node, node),
            pos[node],
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
            zorder=10,
        )

    handles = [
        Patch(
            facecolor=color,
            edgecolor="#222222",
            label=(
                f"L{index}: {name_map.get(leader, leader)} "
                f"({len(edges)} links)"
            ),
        )
        for index, (color, leader, edges) in enumerate(
            zip(colors, leaders, communities), start=1
        )
    ]
    if small_edges:
        handles.append(
            Patch(
                facecolor="#c8c8c8",
                label=f"Small clusters (<{MIN_LINKS} links; {len(small_edges)} links)",
            )
        )
    handles.append(Patch(facecolor="#d9d9d9", label="No real-community membership"))
    ax.legend(
        handles=handles,
        title="Link communities",
        loc="lower left",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
    )
    ax.set_title(
        f"Link communities in 'Os Maias' (w_min={w_min} | "
        f"D*={result['best_partition_density']:.3f}, "
        f"{len(communities)} communities, {len(overlap)} overlap characters)",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        0.99,
        0.01,
        "pie node = overlap of 2+ real link communities\n"
        "node size = weighted degree; gray edges = clusters of 1-2 links",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, pos


def plot_top_character_communities(
    table: pd.DataFrame, top_n: int = 15, figsize: tuple[float, float] = (10, 6)
):
    """Plot the characters with the most real link-community memberships."""
    top = table[table["communities"] > 0].head(top_n).sort_values("communities")
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["character"], top["communities"], color="tab:blue")
    ax.set(xlabel="Number of real link communities", ylabel="", title=f"Top {top_n} characters by community count")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def create_output_dir(root: str | Path, stem: str) -> Path:
    """Create a fresh numbered output directory without overwriting prior runs."""
    root = Path(root)
    suffix = 1
    while True:
        name = stem if suffix == 1 else f"{stem}_{suffix}"
        output = root / name
        try:
            output.mkdir()
            return output
        except FileExistsError:
            suffix += 1


def main() -> None:
    """Run the default threshold sweep and save summary/community tables."""
    root = Path(__file__).resolve().parent
    graph, name_map = load_weighted_graph(
        root / "Data" / "os_maias_cooccurrence_edges.csv",
        root / "Data" / "os_maias_ground_truth_nodes.csv",
    )
    summary, results, _ = analyze_thresholds(graph)
    output_dir = create_output_dir(root, "link_communities_outputs")
    summary.to_csv(output_dir / "link_communities_results.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for threshold, result in results.items():
        community_table(graph if threshold == 1 else threshold_graph(graph, threshold),
                        result, name_map).to_csv(
            output_dir / f"link_communities_w{threshold}.csv", index=False
        )
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
