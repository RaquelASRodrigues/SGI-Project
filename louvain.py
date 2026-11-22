"""
================================================================================
COMMUNITY DETECTION & NULL MODEL VALIDATION FOR "OS MAIAS" CO-OCCURRENCE NETWORK
================================================================================

WHAT THIS SCRIPT DOES:
    Performs community detection on the character co-occurrence network of Eça de
    Queirós' novel "Os Maias" using the Louvain modularity optimization algorithm.
    It evaluates partition stability across multiple random seeds, tests statistical
    significance against degree- and topology-preserving null models, and generates
    a publication-ready visualization saved to disk.

HOW IT WORKS:
    1. Graph Construction: Loads weighted co-occurrence edges and maps raw entity
       IDs to canonical character names.
    2. Multi-Seed Louvain & Stability (NMI): Runs Louvain across 30 random seeds
       to account for stochastic tie-breaking, computing Modularity (Q) statistics
       and pairwise Normalized Mutual Information (NMI) across all runs.
    3. Statistical Validation (Null Models):
       - Unweighted Degree-Preserving Null Model: Uses Maslov-Sneppen double-edge
         swaps (10 * M swaps) to preserve exact node degrees while randomizing edges.
       - Topology-Preserving Weight-Shuffled Null Model: Preserves the real network
         topology while permuting edge weights across existing links, avoiding the
         artificial inflation of Q caused by assigning hub weights to leaf pairs.
    4. Visualization & Export: Computes a community-aware spring layout with
       non-overlapping leader labels and exports a high-resolution PNG figure.

INPUT FILES REQUIRED (located in DATA_DIR):
    - os_maias_cooccurrence_edges.csv : Edge list with columns ['source', 'target', 'weight'].
    - mentions_characters.csv         : Entity mapping with ['character_id', 'character'].
    - characters_summary.csv          : Summary statistics per character.
    - os_maias_ground_truth_nodes.csv : Optional metadata with ['id', 'name', ...].

OUTPUTS PRODUCED:
    - Console summary of modularity Q, NMI stability, community rosters, and Z-scores.
    - High-resolution figure saved to: ../images/louvain_communities_os_maias.png
    - Node-level community assignment table saved to: Data/louvain_communities.csv
================================================================================
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import normalized_mutual_info_score


# ==============================================================================
# 1. CONFIGURATION & DATA LOADING
# ==============================================================================
DATA_DIR = r"C:\SGI_Final\SGI-Project\Data"
PROJECT_DIR = os.path.dirname(DATA_DIR)
IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

edges_path = os.path.join(DATA_DIR, "os_maias_cooccurrence_edges.csv")
mentions_path = os.path.join(DATA_DIR, "mentions_characters.csv")
gt_path = os.path.join(DATA_DIR, "os_maias_ground_truth_nodes.csv")

edges_df = pd.read_csv(edges_path)
mentions_df = pd.read_csv(mentions_path)

# Build ID -> canonical character name dictionary
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

# Construct weighted undirected graph using readable character names
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

# Ensure all known characters exist in the graph even if isolated
for name in id_to_name.values():
    if name not in G:
        G.add_node(name)

print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# ==============================================================================
# 2. MULTI-SEED LOUVAIN & PARTITION STABILITY (NMI)
# ==============================================================================
def partition_to_labels(partition, node_order):
    """Convert a list of node sets into an integer cluster label array."""
    node_to_comm = {}
    for comm_idx, comm in enumerate(partition):
        for node in comm:
            node_to_comm[node] = comm_idx
    return [node_to_comm[n] for n in node_order]


nodes_list = sorted(G.nodes())
N_RUNS = 30
partitions = []
modularities = []
num_comms = []

for seed in range(N_RUNS):
    comms = nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    q = nx.community.modularity(G, comms, weight="weight")
    partitions.append(comms)
    modularities.append(q)
    num_comms.append(len(comms))

# Compute pairwise NMI across all 30 runs (435 unique pairs)
labels_list = [partition_to_labels(p, nodes_list) for p in partitions]
nmi_scores = [
    normalized_mutual_info_score(labels_list[i], labels_list[j])
    for i in range(N_RUNS)
    for j in range(i + 1, N_RUNS)
]

best_idx = int(np.argmax(modularities))
best_partition = partitions[best_idx]
best_Q = modularities[best_idx]

print("\n--- LOUVAIN RESULTS (30 SEEDS) ---")
print(
    f"Modularity Q: mean = {np.mean(modularities):.4f} ± {np.std(modularities):.4f} "
    f"(min = {np.min(modularities):.4f}, max = {best_Q:.4f})"
)
print(f"Communities detected: {min(num_comms)} to {max(num_comms)} (median = {int(np.median(num_comms))})")
print(
    f"Partition stability (NMI): mean = {np.mean(nmi_scores):.4f} "
    f"(range = [{np.min(nmi_scores):.4f}, {np.max(nmi_scores):.4f}])"
)

# Inspect best partition and label each community by its highest-strength member
print(f"\n--- BEST PARTITION DETAILS (Seed {best_idx}, Q = {best_Q:.4f}) ---")
node_strength = dict(G.degree(weight="weight"))
comm_rows = []
comm_summary = {}

for idx, comm in enumerate(best_partition, start=1):
    members_sorted = sorted(comm, key=lambda n: node_strength[n], reverse=True)
    leader = members_sorted[0]
    comm_summary[idx] = {"leader": leader, "size": len(comm), "top3": members_sorted[:3]}
    print(f"\nCommunity {idx} [Leader: {leader}] ({len(comm)} characters):")
    print("  Top members:", ", ".join(f"{m} ({int(node_strength[m])})" for m in members_sorted[:8]))
    if len(members_sorted) > 8:
        print("  Remaining  :", ", ".join(members_sorted[8:]))
    for m in members_sorted:
        comm_rows.append(
            {
                "character": m,
                "community_id": idx,
                "community_leader": leader,
                "weighted_degree": node_strength[m],
                "degree": G.degree(m),
            }
        )

comm_df = pd.DataFrame(comm_rows)
comm_csv_path = os.path.join(DATA_DIR, "louvain_communities.csv")
comm_df.to_csv(comm_csv_path, index=False, encoding="utf-8-sig")


# ==============================================================================
# 3. NULL MODEL STATISTICAL VALIDATION
# ==============================================================================
def run_unweighted_degree_null(G_orig, n_shuffles=30, random_seed=42):
    """
    Generate degree-preserving random graphs via Maslov-Sneppen double-edge swaps
    and compute unweighted Louvain modularity on each randomized graph.
    """
    rng = np.random.default_rng(random_seed)
    m = G_orig.number_of_edges()
    null_Qs = []

    for i in range(n_shuffles):
        G_rand = G_orig.copy()
        nx.double_edge_swap(
            G_rand,
            nswap=10 * m,
            max_tries=100 * m,
            seed=int(rng.integers(0, 1_000_000)),
        )
        c_null = nx.community.louvain_communities(G_rand, weight=None, seed=i)
        null_Qs.append(nx.community.modularity(G_rand, c_null, weight=None))

    return np.array(null_Qs)


def run_weight_shuffled_null(G_orig, n_shuffles=30, random_seed=42):
    """
    Keep real network edges fixed and randomly permute edge weights across them.
    This isolates the effect of weight distribution without assigning massive
    hub weights to isolated peripheral leaf pairs.
    """
    rng = np.random.default_rng(random_seed)
    orig_weights = [d["weight"] for _, _, d in G_orig.edges(data=True)]
    null_Qs = []

    for i in range(n_shuffles):
        G_rand = G_orig.copy()
        shuffled_w = rng.permutation(orig_weights)
        for (u, v), w in zip(G_rand.edges(), shuffled_w):
            G_rand[u][v]["weight"] = float(w)
        c_null = nx.community.louvain_communities(G_rand, weight="weight", seed=i)
        null_Qs.append(nx.community.modularity(G_rand, c_null, weight="weight"))

    return np.array(null_Qs)


Q_real_unweighted = nx.community.modularity(G, best_partition, weight=None)
null_Q_unw = run_unweighted_degree_null(G, n_shuffles=30)
z_unw = (Q_real_unweighted - null_Q_unw.mean()) / null_Q_unw.std()

null_Q_w_fixed_topo = run_weight_shuffled_null(G, n_shuffles=30)
z_w_fixed_topo = (best_Q - null_Q_w_fixed_topo.mean()) / null_Q_w_fixed_topo.std()

print("\n--- NULL MODEL COMPARISON (30 SHUFFLES) ---")
print(
    f"[Unweighted Degree-Preserving] Q_real = {Q_real_unweighted:.4f} vs. "
    f"Q_null = {null_Q_unw.mean():.4f} ± {null_Q_unw.std():.4f} (Z-score = {z_unw:+.2f})"
)
print(
    f"[Weighted Topology-Preserving] Q_real = {best_Q:.4f} vs. "
    f"Q_null = {null_Q_w_fixed_topo.mean():.4f} ± {null_Q_w_fixed_topo.std():.4f} (Z-score = {z_w_fixed_topo:+.2f})"
)

# Check NMI against true categorical metadata columns only (exclude IDs/aliases)
if gt_df is not None and "name" in gt_df.columns:
    gt_merged = comm_df.merge(gt_df, left_on="character", right_on="name", how="inner")
    excluded_cols = {"id", "name", "aliases", "alias"}
    candidate_cols = [c for c in gt_df.columns if c.lower() not in excluded_cols]
    for col in candidate_cols:
        valid = gt_merged.dropna(subset=[col])
        n_unique = valid[col].nunique()
        if 1 < n_unique < 0.5 * len(valid):
            nmi_gt = normalized_mutual_info_score(valid["community_id"], valid[col])
            print(f"NMI between Louvain and categorical attribute '{col}': {nmi_gt:.4f}")


# ==============================================================================
# 4. COMMUNITY-AWARE NETWORK VISUALIZATION & EXPORT
# ==============================================================================
node_to_comm_id = dict(zip(comm_df["character"], comm_df["community_id"]))
n_comms = len(best_partition)

# Place community anchors evenly on a circle to prevent central hairball collapse
angles = np.linspace(0, 2 * np.pi, n_comms, endpoint=False)
radius = 2.6
comm_centers = {
    c_id: np.array([radius * np.cos(ang), radius * np.sin(ang)])
    for c_id, ang in zip(range(1, n_comms + 1), angles)
}

# Build an auxiliary layout graph with balanced intra/inter-community attraction
G_layout = nx.Graph()
G_layout.add_nodes_from(G.nodes())
for u, v in G.edges():
    same_comm = node_to_comm_id[u] == node_to_comm_id[v]
    G_layout.add_edge(u, v, layout_weight=2.2 if same_comm else 0.25)

rng_layout = np.random.default_rng(42)
init_pos = {}
for node in G.nodes():
    c_id = node_to_comm_id[node]
    jitter = rng_layout.normal(0, 0.35, size=2)
    init_pos[node] = comm_centers[c_id] + jitter

pos = nx.spring_layout(
    G_layout,
    pos=init_pos,
    weight="layout_weight",
    k=0.38,
    iterations=90,
    seed=42,
)

# Configure color palette
cmap = plt.cm.Set2 if n_comms <= 8 else plt.cm.tab10
comm_colors = {c_id: cmap((c_id - 1) / max(n_comms, 1)) for c_id in range(1, n_comms + 1)}
node_colors = [comm_colors[node_to_comm_id[n]] for n in G.nodes()]
node_sizes = [55 + 0.16 * node_strength[n] for n in G.nodes()]

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.55, 1]})

# --- Panel A: Community Network Graph ---
intra_edges = [(u, v) for u, v in G.edges() if node_to_comm_id[u] == node_to_comm_id[v]]
inter_edges = [(u, v) for u, v in G.edges() if node_to_comm_id[u] != node_to_comm_id[v]]

nx.draw_networkx_edges(G, pos, edgelist=inter_edges, ax=axes[0], alpha=0.08, edge_color="#7f8c8d", width=0.6)
nx.draw_networkx_edges(G, pos, edgelist=intra_edges, ax=axes[0], alpha=0.22, edge_color="#34495e", width=0.9)

nx.draw_networkx_nodes(
    G,
    pos,
    ax=axes[0],
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.95,
    edgecolors="#2c3e50",
    linewidths=0.8,
)

# Label top 3 leaders per community, nudging labels radially outward from center
labeled_nodes = set()
for c_id, info in comm_summary.items():
    for rank, char_name in enumerate(info["top3"]):
        if char_name in pos and char_name not in labeled_nodes:
            labeled_nodes.add(char_name)
            x, y = pos[char_name]
            norm = np.hypot(x, y) + 1e-6
            ux, uy = x / norm, y / norm
            offset = 0.14 + 0.07 * rank
            axes[0].text(
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
axes[0].legend(
    handles=legend_handles,
    loc="lower left",
    frameon=True,
    facecolor="white",
    framealpha=0.9,
    fontsize=8.5,
    title="Detected Communities",
    title_fontsize=9,
)
axes[0].set_title(
    f"Louvain Communities in 'Os Maias' (Weighted Q = {best_Q:.3f} | NMI = {np.mean(nmi_scores):.2f})",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
axes[0].axis("off")

# --- Panel B: Unweighted Degree-Preserving Null Model Significance ---
axes[1].hist(
    null_Q_unw,
    bins=10,
    color="#95a5a6",
    edgecolor="#2c3e50",
    alpha=0.85,
    label="Degree-Preserving Null Model (30 swaps)",
)
axes[1].axvline(
    Q_real_unweighted,
    color="#e74c3c",
    linestyle="--",
    linewidth=2.5,
    label=f"Real Network Q = {Q_real_unweighted:.3f}",
)
axes[1].set_xlabel("Unweighted Modularity (Q)", fontsize=10.5)
axes[1].set_ylabel("Frequency", fontsize=10.5)
axes[1].set_title(
    f"Topological Significance Test (Z = {z_unw:+.2f})",
    fontsize=12,
    fontweight="bold",
    pad=10,
)
axes[1].grid(axis="y", linestyle=":", alpha=0.5)
axes[1].legend(frameon=True, facecolor="white", fontsize=9)

plt.tight_layout()

output_img_path = os.path.join(IMAGES_DIR, "louvain_communities_os_maias.png")
plt.savefig(output_img_path, dpi=300, bbox_inches="tight")
print(f"\nSaved visualization to: {output_img_path}")
print(f"Saved community assignments to: {comm_csv_path}")
plt.show()