"""
================================================================================
COMBINED WEIGHTED NETWORK & NORMALIZED COMMUNITY ANALYSIS FOR "OS MAIAS"
================================================================================

WHAT THIS SCRIPT DOES:
    Combines the weighted network analysis (Section 7) and the normalized-weight
    community detection & diagnostics pipeline into a single script:

    PART I — WEIGHTED NETWORK ANALYSIS (SECTION 7):
1. Degree (k_i) vs. Strength (s_i = sum w_ij): Computes Spearman rank
         correlation and highlights rank-shift outliers.
2. Weight Distribution & Strongest Ties: Summarizes tie weights and lists
         the Top-15 strongest character co-occurrence pairs.
3. Unweighted (BFS) vs. Weighted Paths (Dijkstra, d_ij = 1 / w_ij):
         Compares minimum-hop paths against strong-tie shortest paths.
4. Exports log-log scatter plot: images/degree_vs_strength_os_maias.png

    PART II — NORMALIZED COMMUNITY DETECTION & DIAGNOSTICS:
5. Weight Normalization: Computes Cosine (Salton) and Generalized Jaccard
         edge weights to mitigate protagonist/hub dominance.
6. Multi-Seed Louvain (30 seeds): Compares Raw, Cosine, and Jaccard
         schemes via Modularity (Q) and pairwise NMI stability.
7. Resolution Sweep (gamma in [0.6, 1.4]) & Null Model Validation:
         Supports gamma = 1.0 and tests significance via 100 weight shuffles.
8. Exports diagnostics & network plots plus node community assignments.

INPUT FILES REQUIRED (located in Data/):
- os_maias_cooccurrence_edges.csv : Edge list ['source', 'target', 'weight'].
- mentions_characters.csv         : Entity mapping ['character_id', 'character'].
- os_maias_ground_truth_nodes.csv : Optional metadata ['id', 'name', ...].

OUTPUTS PRODUCED:
- images/degree_vs_strength_os_maias.png  : Degree vs. Strength log-log plot.
- images/louvain_diagnostics_os_maias.png : 4-panel Q & NMI decision plots.
- images/louvain_communities_normalized.png: Normalized community graph.
- Data/louvain_communities_normalized.csv  : Node-level community assignments.
- Data/louvain_communities_by_community.csv: Community-level character lists.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score


# ==============================================================================
# 1. CONFIGURATION, DATA LOADING & GRAPH CONSTRUCTION
# ==============================================================================
DATA_DIR = os.path.abspath("Data")
PROJECT_DIR = os.path.dirname(DATA_DIR)
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

edges_path = os.path.join(DATA_DIR, "os_maias_cooccurrence_edges.csv")
mentions_path = os.path.join(DATA_DIR, "mentions_characters.csv")
gt_path = os.path.join(DATA_DIR, "os_maias_ground_truth_nodes.csv")

edges_df = pd.read_csv(edges_path)
mentions_df = pd.read_csv(mentions_path)

# Map character_id -> canonical character name
id_to_name = (
    mentions_df.dropna(subset=["character_id", "character"])
    .assign(
        character_id=lambda d: d["character_id"].astype(str).str.strip(),
        character=lambda d: d["character"].astype(str).str.strip(),
    )
    .drop_duplicates("character_id")
    .set_index("character_id")["character"]
    .to_dict()
)

if os.path.exists(gt_path):
    gt_df = pd.read_csv(gt_path, dtype=str)
    if {"id", "name"}.issubset(gt_df.columns):
        gt_map = dict(zip(gt_df["id"].str.strip(), gt_df["name"].str.strip()))
        id_to_name.update(gt_map)
else:
    gt_df = None

# Build base undirected graph with raw weights
G = nx.Graph()
for _, row in edges_df.iterrows():
    u = id_to_name.get(str(row["source"]).strip(), str(row["source"]).strip())
    v = id_to_name.get(str(row["target"]).strip(), str(row["target"]).strip())
    w = float(row["weight"])
    if u != v:
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

for name in id_to_name.values():
    if name not in G:
        G.add_node(name)

# Compute inverse-weight distances (1/w) and node-normalized weights (Cosine & Jaccard)
raw_strength = dict(G.degree(weight="weight"))
for u, v, d in G.edges(data=True):
    w_raw = d["weight"]
    s_u = raw_strength[u]
    s_v = raw_strength[v]

    d["distance"] = 1.0 / w_raw
    d["weight_cosine"] = w_raw / np.sqrt(s_u * s_v) if (s_u > 0 and s_v > 0) else 0.0
    denom_jaccard = s_u + s_v - w_raw
    d["weight_jaccard"] = w_raw / denom_jaccard if denom_jaccard > 0 else 0.0

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ==============================================================================
# PART I: WEIGHTED NETWORK ANALYSIS (SECTION 7)
# ==============================================================================

# --- 2. Degree (k_i) vs. Strength (s_i) & Spearman Rank Correlation ---
node_stats = pd.DataFrame(
    {
        "character": list(G.nodes()),
        "degree": [G.degree(n) for n in G.nodes()],
        "strength": [raw_strength[n] for n in G.nodes()],
    }
)
node_stats["avg_weight_per_link"] = node_stats["strength"] / node_stats["degree"].replace(0, np.nan)
node_stats["degree_rank"] = node_stats["degree"].rank(ascending=False, method="min")
node_stats["strength_rank"] = node_stats["strength"].rank(ascending=False, method="min")
# Positive rank_diff means much higher Strength rank than Degree rank (intense focused ties)
node_stats["rank_diff"] = node_stats["degree_rank"] - node_stats["strength_rank"]

rho, p_val = spearmanr(node_stats["degree"], node_stats["strength"])

print("\n--- 1. DEGREE VS. STRENGTH ---")
print(f"Spearman rank correlation (degree vs. strength): rho = {rho:.4f} (p = {p_val:.2e})")

print("\nTop 10 Characters by Strength (s_i):")
print(
    node_stats.sort_values("strength", ascending=False)[
        ["character", "degree", "degree_rank", "strength", "strength_rank", "avg_weight_per_link"]
    ]
    .head(10)
    .to_string(index=False)
)

print("\nBiggest Positive Rank Shifts (Higher Strength Rank than Degree Rank — Intense Focused Ties):")
print(
    node_stats[node_stats["degree"] >= 3]
    .sort_values("rank_diff", ascending=False)[
        ["character", "degree", "degree_rank", "strength", "strength_rank", "rank_diff"]
    ]
    .head(8)
    .to_string(index=False)
)


# --- 3. Weight Distribution & Strongest Ties ---
edge_list = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
weights_arr = np.array([d["weight"] for _, _, d in edge_list])
n_weight_1 = int(np.sum(weights_arr == 1))

print("\n--- 2. STRONGEST TIES ---")
print(
    f"Total links: {len(weights_arr)} | Min weight: {weights_arr.min():.0f} | "
    f"Max weight: {weights_arr.max():.0f} | Median: {np.median(weights_arr):.1f} | "
    f"Links with weight == 1: {n_weight_1} ({100 * n_weight_1 / len(weights_arr):.1f}%)"
)
print("\nTop 15 Strongest Character Ties:")
for rank, (u, v, d) in enumerate(edge_list[:15], start=1):
    print(f"  {rank:2d}. {u} — {v}: weight = {d['weight']:.0f} (distance = {d['distance']:.4f})")


# --- 4. Unweighted (BFS) vs. Weighted Shortest Paths (Dijkstra, d = 1/w) ---
print("\n--- 3. SHORTEST PATHS: UNWEIGHTED (HOPS) VS. WEIGHTED (DIJKSTRA 1/w) ---")
candidate_pairs = [
    ("Maria Monforte", "Conde de Gouvarinho"),
    ("Pedro da Maia", "Raquel Cohen"),
    ("Maria Monforte", "Maria Eduarda"),
    ("Afonso da Maia", "Jacob Cohen"),
    ("Tancredo", "Dâmaso Salcede"),
]

for src, dst in candidate_pairs:
    if src in G and dst in G and nx.has_path(G, src, dst):
        bfs_path = nx.shortest_path(G, source=src, target=dst, weight=None)
        dijk_path = nx.dijkstra_path(G, source=src, target=dst, weight="distance")
        dijk_dist = nx.dijkstra_path_length(G, source=src, target=dst, weight="distance")

        step_weights = [int(G[dijk_path[i]][dijk_path[i + 1]]["weight"]) for i in range(len(dijk_path) - 1)]
        dijk_str = dijk_path[0]
        for node_next, w_step in zip(dijk_path[1:], step_weights):
            dijk_str += f" --(w={w_step})--> {node_next}"

        print(f"\nPair: {src}  -->  {dst}")
        print(f"  Unweighted BFS ({len(bfs_path)-1} hops): {' -> '.join(bfs_path)}")
        print(f"  Weighted Dijkstra ({len(dijk_path)-1} hops, dist={dijk_dist:.4f}): {dijk_str}")


# --- 5. Figure 1: Degree vs. Strength (Log-Log Scatter with Outliers) ---
fig_ds, ax_ds = plt.subplots(figsize=(10, 7))

active = node_stats[node_stats["degree"] > 0].copy()
ax_ds.scatter(
    active["degree"],
    active["strength"],
    s=45 + 0.08 * active["strength"],
    color="#2980b9",
    alpha=0.78,
    edgecolors="#2c3e50",
    linewidths=0.7,
)

x_vals = np.linspace(active["degree"].min(), active["degree"].max(), 100)
median_ratio = (active["strength"] / active["degree"]).median()
ax_ds.plot(
    x_vals,
    median_ratio * x_vals,
    linestyle="--",
    color="#7f8c8d",
    linewidth=1.3,
    label=f"Median Strength/Degree Ratio ({median_ratio:.1f}x)",
)

to_label = set(active.nlargest(8, "strength")["character"]) | set(
    active[active["degree"] >= 4].nlargest(4, "rank_diff")["character"]
)

for _, row in active[active["character"].isin(to_label)].iterrows():
    ax_ds.annotate(
        row["character"],
        (row["degree"], row["strength"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=8.5,
        fontweight="bold" if row["strength"] > 500 else "normal",
        color="#1a1a1a",
    )

ax_ds.set_xscale("log")
ax_ds.set_yscale("log")
ax_ds.set_xlabel("Degree k_i (Number of Unique Co-occurring Characters, log scale)", fontsize=10.5)
ax_ds.set_ylabel("Strength s_i (Total Weighted Co-occurrences, log scale)", fontsize=10.5)
ax_ds.set_title(
    f"Degree vs. Strength in 'Os Maias' (Spearman ρ = {rho:.3f})",
    fontsize=12.5,
    fontweight="bold",
    pad=10,
)
ax_ds.grid(True, which="both", linestyle=":", alpha=0.45)
ax_ds.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)

plt.tight_layout()
ds_out_path = os.path.join(IMAGES_DIR, "degree_vs_strength_os_maias.png")
plt.savefig(ds_out_path, dpi=300, bbox_inches="tight")


# ==============================================================================
# PART II: NORMALIZED WEIGHT COMMUNITY DETECTION & DIAGNOSTICS
# ==============================================================================

# --- 6. Multi-Seed Louvain Evaluation Across Weight Schemes ---
def partition_to_labels(partition, node_order):
    """Convert a list of node sets into an integer label vector."""
    node_to_comm = {}
    for comm_idx, comm in enumerate(partition):
        for node in comm:
            node_to_comm[node] = comm_idx
    return [node_to_comm[n] for n in node_order]


def evaluate_louvain_seeds(graph, weight_attr="weight_cosine", n_runs=30, resolution=1.0):
    """Run Louvain across multiple seeds and compute point-by-point Q and NMI metrics."""
    nodes_order = sorted(graph.nodes())
    partitions = []
    q_values = []
    k_values = []

    for seed in range(n_runs):
        comms = nx.community.louvain_communities(
            graph, weight=weight_attr, resolution=resolution, seed=seed
        )
        comms = sorted(comms, key=len, reverse=True)
        q = nx.community.modularity(graph, comms, weight=weight_attr)
        partitions.append(comms)
        q_values.append(q)
        k_values.append(len(comms))

    labels_list = [partition_to_labels(p, nodes_order) for p in partitions]

    nmi_matrix = np.ones((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            score = normalized_mutual_info_score(labels_list[i], labels_list[j])
            nmi_matrix[i, j] = score
            nmi_matrix[j, i] = score

    mean_nmi_per_seed = (nmi_matrix.sum(axis=1) - 1.0) / (n_runs - 1)
    upper_tri_nmi = nmi_matrix[np.triu_indices(n_runs, k=1)]

    best_idx = int(np.argmax(q_values))
    nmi_to_best = nmi_matrix[:, best_idx]

    return {
        "partitions": partitions,
        "labels_list": labels_list,
        "q_values": np.array(q_values),
        "k_values": np.array(k_values),
        "nmi_matrix": nmi_matrix,
        "mean_nmi_per_seed": mean_nmi_per_seed,
        "nmi_to_best": nmi_to_best,
        "overall_mean_nmi": float(np.mean(upper_tri_nmi)),
        "overall_min_nmi": float(np.min(upper_tri_nmi)),
        "overall_max_nmi": float(np.max(upper_tri_nmi)),
        "best_idx": best_idx,
        "best_partition": partitions[best_idx],
        "best_Q": float(q_values[best_idx]),
    }


N_RUNS = 30
schemes = {
    "Raw Weights": "weight",
    "Cosine Normalized": "weight_cosine",
    "Jaccard Normalized": "weight_jaccard",
}

results_by_scheme = {
    label: evaluate_louvain_seeds(G, weight_attr=attr, n_runs=N_RUNS, resolution=1.0)
    for label, attr in schemes.items()
}

print("\n--- COMPARISON ACROSS WEIGHT SCHEMES (30 SEEDS, RESOLUTION = 1.0) ---")
for label, res in results_by_scheme.items():
    print(
        f"{label:20s} | Q = {res['q_values'].mean():.4f} ± {res['q_values'].std():.4f} "
        f"(max={res['best_Q']:.4f}) | k = {int(np.median(res['k_values']))} "
        f"[{res['k_values'].min()}-{res['k_values'].max()}] | "
        f"Mean NMI = {res['overall_mean_nmi']:.4f}"
    )

nodes_sorted = sorted(G.nodes())
raw_best_labels = partition_to_labels(results_by_scheme["Raw Weights"]["best_partition"], nodes_sorted)
cos_best_labels = partition_to_labels(results_by_scheme["Cosine Normalized"]["best_partition"], nodes_sorted)
jac_best_labels = partition_to_labels(results_by_scheme["Jaccard Normalized"]["best_partition"], nodes_sorted)

print(
    f"\nAgreement between best partitions: "
    f"NMI(Raw vs Cosine) = {normalized_mutual_info_score(raw_best_labels, cos_best_labels):.4f} | "
    f"NMI(Cosine vs Jaccard) = {normalized_mutual_info_score(cos_best_labels, jac_best_labels):.4f}"
)

SELECTED_SCHEME = "Cosine Normalized"
SELECTED_ATTR = "weight_cosine"
sel_res = results_by_scheme[SELECTED_SCHEME]
best_partition = sel_res["best_partition"]
best_Q = sel_res["best_Q"]
best_idx = sel_res["best_idx"]

print(f"\n--- BEST PARTITION DETAILS ({SELECTED_SCHEME}, Seed {best_idx}, Q = {best_Q:.4f}) ---")
cos_strength = dict(G.degree(weight=SELECTED_ATTR))
comm_rows = []
comm_summary = {}

for idx, comm in enumerate(best_partition, start=1):
    members_sorted = sorted(comm, key=lambda n: raw_strength[n], reverse=True)
    leader = members_sorted[0]
    comm_summary[idx] = {"leader": leader, "size": len(comm), "top3": members_sorted[:3]}
    print(f"\nCommunity {idx} [Leader: {leader}] ({len(comm)} characters):")
    print(
        "  Top members:",
        ", ".join(f"{m} (raw={int(raw_strength[m])}, cos={cos_strength[m]:.2f})" for m in members_sorted[:6]),
    )
    if len(members_sorted) > 6:
        print("  Remaining  :", ", ".join(members_sorted[6:]))
    for m in members_sorted:
        comm_rows.append(
            {
                "character": m,
                "community_id": idx,
                "community_leader": leader,
                "raw_strength": raw_strength[m],
                "cosine_strength": round(cos_strength[m], 4),
                "degree": G.degree(m),
            }
        )

comm_df = pd.DataFrame(comm_rows)
comm_csv_path = os.path.join(DATA_DIR, "louvain_communities_normalized.csv")
comm_df.to_csv(comm_csv_path, index=False, encoding="utf-8-sig")

# Export characters grouped by community (community_id,characters)
comm_by_group_rows = [
    {
        "community_id": f"community_{idx - 1}",
        "characters": "; ".join(sorted(comm, key=lambda n: raw_strength[n], reverse=True)),
    }
    for idx, comm in enumerate(best_partition, start=1)
]
comm_by_group_df = pd.DataFrame(comm_by_group_rows)
comm_by_group_csv_path = os.path.join(DATA_DIR, "louvain_communities_by_community.csv")
comm_by_group_df.to_csv(comm_by_group_csv_path, index=False, encoding="utf-8-sig")


# --- 7. Resolution Parameter Sweep (Supporting gamma = 1.0) ---
gammas = np.round(np.arange(0.6, 1.45, 0.1), 2)
gamma_q_mean, gamma_q_std = [], []
gamma_nmi_mean, gamma_k_median = [], []

for g_val in gammas:
    res_g = evaluate_louvain_seeds(G, weight_attr=SELECTED_ATTR, n_runs=20, resolution=float(g_val))
    gamma_q_mean.append(res_g["q_values"].mean())
    gamma_q_std.append(res_g["q_values"].std())
    gamma_nmi_mean.append(res_g["overall_mean_nmi"])
    gamma_k_median.append(np.median(res_g["k_values"]))


# --- 8. Null Model Validation on Normalized Weights ---
def run_weight_shuffled_null(G_orig, weight_attr="weight_cosine", n_shuffles=100, random_seed=42):
    """Permute edge weights across fixed real network topology."""
    rng = np.random.default_rng(random_seed)
    orig_weights = [d[weight_attr] for _, _, d in G_orig.edges(data=True)]
    null_Qs = []

    for i in range(n_shuffles):
        G_rand = G_orig.copy()
        shuffled_w = rng.permutation(orig_weights)
        for (u, v), w in zip(G_rand.edges(), shuffled_w):
            G_rand[u][v]["w_tmp"] = float(w)
        c_null = nx.community.louvain_communities(G_rand, weight="w_tmp", seed=i)
        null_Qs.append(nx.community.modularity(G_rand, c_null, weight="w_tmp"))

    return np.array(null_Qs)


N_SHUFFLES = 100
null_Q_cos = run_weight_shuffled_null(G, weight_attr=SELECTED_ATTR, n_shuffles=N_SHUFFLES)
z_cos = (best_Q - null_Q_cos.mean()) / null_Q_cos.std()

print(f"\n--- NULL MODEL VALIDATION ({SELECTED_SCHEME}, {N_SHUFFLES} SHUFFLES) ---")
print(
    f"Q_real = {best_Q:.4f} vs. Q_null = {null_Q_cos.mean():.4f} ± {null_Q_cos.std():.4f} "
    f"(Z-score = {z_cos:+.2f})"
)


# --- 9. Figure 2: Point-by-Point Diagnostics (Q & NMI Decision Support) ---
fig_diag, axes_d = plt.subplots(2, 2, figsize=(16, 11))
seeds_x = np.arange(N_RUNS)

scheme_styles = {
    "Raw Weights": ("#7f8c8d", "o", "--"),
    "Cosine Normalized": ("#2980b9", "s", "-"),
    "Jaccard Normalized": ("#16a085", "^", "-."),
}
for label, res in results_by_scheme.items():
    color, marker, ls = scheme_styles[label]
    axes_d[0, 0].plot(
        seeds_x,
        res["q_values"],
        color=color,
        marker=marker,
        linestyle=ls,
        markersize=4.5,
        linewidth=1.4,
        alpha=0.88,
        label=f"{label} (mean={res['q_values'].mean():.3f})",
    )
axes_d[0, 0].scatter(
    [best_idx],
    [best_Q],
    color="#e74c3c",
    s=130,
    zorder=6,
    edgecolors="black",
    label=f"Selected Best (Seed {best_idx}, Q={best_Q:.3f})",
)
axes_d[0, 0].set_title("A. Modularity (Q) Across Random Seeds", fontsize=11.5, fontweight="bold")
axes_d[0, 0].set_xlabel("Random Seed Index (0–29)")
axes_d[0, 0].set_ylabel("Modularity (Q)")
axes_d[0, 0].grid(True, linestyle=":", alpha=0.5)
axes_d[0, 0].legend(fontsize=8.5, loc="best")

axes_d[0, 1].plot(
    seeds_x,
    sel_res["mean_nmi_per_seed"],
    color="#8e44ad",
    marker="o",
    markersize=5,
    linewidth=1.6,
    label="Mean Pairwise NMI vs. All Other Seeds",
)
axes_d[0, 1].plot(
    seeds_x,
    sel_res["nmi_to_best"],
    color="#e67e22",
    marker="d",
    linestyle="--",
    markersize=4.5,
    linewidth=1.3,
    label=f"NMI vs. Selected Best Partition (Seed {best_idx})",
)
axes_d[0, 1].axhline(
    sel_res["overall_mean_nmi"],
    color="#2c3e50",
    linestyle=":",
    linewidth=1.5,
    label=f"Overall Mean NMI = {sel_res['overall_mean_nmi']:.3f}",
)
axes_d[0, 1].scatter([best_idx], [sel_res["mean_nmi_per_seed"][best_idx]], color="#e74c3c", s=120, zorder=6)
axes_d[0, 1].set_ylim(0.5, 1.05)
axes_d[0, 1].set_title(f"B. Partition Stability (NMI) per Seed — {SELECTED_SCHEME}", fontsize=11.5, fontweight="bold")
axes_d[0, 1].set_xlabel("Random Seed Index (0–29)")
axes_d[0, 1].set_ylabel("Normalized Mutual Information (NMI)")
axes_d[0, 1].grid(True, linestyle=":", alpha=0.5)
axes_d[0, 1].legend(fontsize=8.5, loc="lower right")

ax_c = axes_d[1, 0]
ax_c_twin = ax_c.twinx()

l1 = ax_c.errorbar(
    gammas,
    gamma_q_mean,
    yerr=gamma_q_std,
    color="#2980b9",
    marker="s",
    capsize=3,
    linewidth=1.8,
    label="Modularity Q (left axis)",
)
l2 = ax_c.plot(
    gammas,
    gamma_nmi_mean,
    color="#8e44ad",
    marker="o",
    linestyle="--",
    linewidth=1.8,
    label="Mean NMI Stability (left axis)",
)
l3 = ax_c_twin.plot(
    gammas,
    gamma_k_median,
    color="#27ae60",
    marker="^",
    linestyle=":",
    linewidth=1.8,
    label="Median Communities k (right axis)",
)
ax_c.axvline(1.0, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5)
ax_c.set_title(f"C. Resolution Parameter Sweep (γ) — {SELECTED_SCHEME}", fontsize=11.5, fontweight="bold")
ax_c.set_xlabel("Resolution Parameter (γ)")
ax_c.set_ylabel("Modularity (Q) / Mean NMI")
ax_c_twin.set_ylabel("Median Number of Communities (k)", color="#27ae60")
ax_c.grid(True, linestyle=":", alpha=0.5)

lines_c = [l1.lines[0], l2[0], l3[0]]
labels_c = [l.get_label() for l in lines_c]
ax_c.legend(lines_c, labels_c, fontsize=8.5, loc="lower left")

axes_d[1, 1].hist(
    null_Q_cos,
    bins=14,
    color="#95a5a6",
    edgecolor="#2c3e50",
    alpha=0.85,
    label=f"Weight-Shuffled Null Model ({N_SHUFFLES} shuffles)",
)
axes_d[1, 1].axvline(
    best_Q,
    color="#e74c3c",
    linestyle="--",
    linewidth=2.5,
    label=f"Real Cosine Q = {best_Q:.3f} (Z = {z_cos:+.2f})",
)
axes_d[1, 1].set_title(f"D. Statistical Significance ({SELECTED_SCHEME})", fontsize=11.5, fontweight="bold")
axes_d[1, 1].set_xlabel("Cosine-Weighted Modularity (Q)")
axes_d[1, 1].set_ylabel("Frequency")
axes_d[1, 1].grid(axis="y", linestyle=":", alpha=0.5)
axes_d[1, 1].legend(fontsize=8.5, loc="best")

plt.tight_layout()
diag_img_path = os.path.join(IMAGES_DIR, "louvain_diagnostics_os_maias.png")
plt.savefig(diag_img_path, dpi=300, bbox_inches="tight")


# --- 10. Figure 3: Normalized Community Network Visualization ---
node_to_comm_id = dict(zip(comm_df["character"], comm_df["community_id"]))
n_comms = len(best_partition)

angles = np.linspace(0, 2 * np.pi, n_comms, endpoint=False)
radius = 2.7
comm_centers = {
    c_id: np.array([radius * np.cos(ang), radius * np.sin(ang)])
    for c_id, ang in zip(range(1, n_comms + 1), angles)
}

G_layout = nx.Graph()
G_layout.add_nodes_from(G.nodes())
for u, v in G.edges():
    same_comm = node_to_comm_id[u] == node_to_comm_id[v]
    G_layout.add_edge(u, v, layout_weight=2.3 if same_comm else 0.22)

rng_layout = np.random.default_rng(42)
init_pos = {
    node: comm_centers[node_to_comm_id[node]] + rng_layout.normal(0, 0.35, size=2)
    for node in G.nodes()
}

pos = nx.spring_layout(G_layout, pos=init_pos, weight="layout_weight", k=0.40, iterations=95, seed=42)

cmap = plt.cm.Set2 if n_comms <= 8 else plt.cm.tab10
comm_colors = {c_id: cmap((c_id - 1) / max(n_comms, 1)) for c_id in range(1, n_comms + 1)}
node_colors = [comm_colors[node_to_comm_id[n]] for n in G.nodes()]
node_sizes = [55 + 0.16 * raw_strength[n] for n in G.nodes()]

fig_net, ax_net = plt.subplots(figsize=(12, 8.5))

intra_edges = [(u, v) for u, v in G.edges() if node_to_comm_id[u] == node_to_comm_id[v]]
inter_edges = [(u, v) for u, v in G.edges() if node_to_comm_id[u] != node_to_comm_id[v]]

nx.draw_networkx_edges(G, pos, edgelist=inter_edges, ax=ax_net, alpha=0.08, edge_color="#7f8c8d", width=0.6)
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=ax_net, alpha=0.24, edge_color="#34495e", width=0.9)

nx.draw_networkx_nodes(
    G,
    pos,
    ax=ax_net,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.95,
    edgecolors="#2c3e50",
    linewidths=0.8,
)

# Radial outward label placement to prevent overlaps
labeled_nodes = set()
for c_id, info in comm_summary.items():
    for rank, char_name in enumerate(info["top3"]):
        if char_name in pos and char_name not in labeled_nodes:
            labeled_nodes.add(char_name)
            x, y = pos[char_name]
            norm = np.hypot(x, y) + 1e-6
            ux, uy = x / norm, y / norm
            offset = 0.14 + 0.08 * rank
            ax_net.text(
                x + ux * offset,
                y + uy * offset,
                char_name,
                fontsize=8.5 if rank == 0 else 7.5,
                fontweight="bold" if rank == 0 else "normal",
                ha="center",
                va="center",
                color="#1a1a1a",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85),
                zorder=5,
            )

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=f"C{c_id}: {info['leader']} (n={info['size']})",
        markerfacecolor=comm_colors[c_id],
        markeredgecolor="#2c3e50",
        markersize=9,
    )
    for c_id, info in comm_summary.items()
]
ax_net.legend(
    handles=legend_handles,
    loc="lower left",
    frameon=True,
    facecolor="white",
    framealpha=0.92,
    fontsize=8.5,
    title=f"Communities ({SELECTED_SCHEME})",
    title_fontsize=9,
)
ax_net.margins(0.12)
ax_net.set_title(
    f"Louvain Communities in 'Os Maias' — {SELECTED_SCHEME} (Q = {best_Q:.3f} | Mean NMI = {sel_res['overall_mean_nmi']:.2f})",
    fontsize=12.5,
    fontweight="bold",
    pad=12,
)
ax_net.axis("off")

plt.tight_layout()
net_img_path = os.path.join(IMAGES_DIR, "louvain_communities_normalized.png")
plt.savefig(net_img_path, dpi=300, bbox_inches="tight")

print(f"\nSaved Degree vs. Strength plot to: {ds_out_path}")
print(f"Saved diagnostics figure to: {diag_img_path}")
print(f"Saved normalized network figure to: {net_img_path}")
print(f"Saved normalized community table to: {comm_csv_path}")
print(f"Saved characters-by-community table to: {comm_by_group_csv_path}")
plt.show()