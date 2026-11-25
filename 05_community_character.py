"""Experiência 2 (v2): perfis temáticos por personagem (BERTopic) vs. comunidades Louvain.

Alterações face à v1:
  1) Perfis "encolhidos" para a média global (prior de Dirichlet) -> --representation dirichlet
     (embedding Hellinger = raiz quadrada do perfil) ou log-odds com z-score informativo contra
     o fundo (Monroe et al.) -> --representation logodds.
  2) Clustering por Ward (sobre a raiz do perfil), spectral e k-medoids, só com personagens
     com pelo menos --min-paragraphs parágrafos (por defeito 30; aceita vários valores).
  3) Perfil de referência de cada comunidade = média simples dos perfis dos *outros* membros,
     calculados SEM os parágrafos em que a personagem avaliada aparece (remove a circularidade
     dos parágrafos partilhados).

Requer a saída do 04_topic_modelling_communities.py (--model bertopic, --threshold 1).

Exemplo:
    python 05_character_topic_profiles.py \
        --bertopic-documents outputs/topic_modelling_experiment1/bertopic_documents.csv \
        --min-paragraphs 20 30 --alpha 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Reutiliza as funções do script 04 (mesmo carregamento de dados/comunidades).
_spec = importlib.util.spec_from_file_location(
    "tm04", Path(__file__).parent / "04_topic_modelling_communities.py"
)
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)


# ----------------------------------------------------------------------------
# Representações
# ----------------------------------------------------------------------------
def shrunk_profile(counts: np.ndarray, G: np.ndarray, alpha: float) -> np.ndarray:
    """Perfil com prior de Dirichlet centrado na distribuição global: (y + alpha*g) / (n + alpha)."""
    counts = np.atleast_2d(counts).astype(float)
    g = G / G.sum()
    return (counts + alpha * g) / (counts.sum(axis=1, keepdims=True) + alpha)


def embed(counts: np.ndarray, G: np.ndarray, alpha: float, representation: str) -> np.ndarray:
    """Contagens (linhas = personagens) -> vetores onde a distância euclidiana é a usada em tudo."""
    counts = np.atleast_2d(counts).astype(float)
    if representation == "dirichlet":
        return np.sqrt(shrunk_profile(counts, G, alpha))  # Hellinger (norma 1)

    # log-odds com prior de Dirichlet informativo e z-score (Monroe et al., 2008):
    # personagem vs. todas as outras (fundo = G - contagens da própria personagem).
    g = G / G.sum()
    a = alpha * g
    a0 = alpha
    rest = np.clip(G[None, :] - counts, 0, None)
    n = counts.sum(axis=1, keepdims=True)
    n_r = rest.sum(axis=1, keepdims=True)
    delta = (np.log((counts + a) / (n + a0 - counts - a))
             - np.log((rest + a) / (n_r + a0 - rest - a)))
    var = 1.0 / (counts + a) + 1.0 / (rest + a)
    z = delta / np.sqrt(var)
    # Normaliza para norma 1: a magnitude do z cresce com a evidência (n) e voltaria a
    # misturar "quantidade de dados" com "tema".
    return z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------
def kmedoids(D: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 30, max_iter: int = 100) -> np.ndarray:
    n = len(D)
    best_cost, best_labels = np.inf, None
    for _ in range(n_init):
        medoids = rng.choice(n, size=k, replace=False)
        for _ in range(max_iter):
            labels = D[:, medoids].argmin(axis=1)
            new = medoids.copy()
            for c in range(k):
                members = np.where(labels == c)[0]
                if len(members):
                    new[c] = members[D[np.ix_(members, members)].sum(axis=1).argmin()]
            if np.array_equal(new, medoids):
                break
            medoids = new
        labels = D[:, medoids].argmin(axis=1)
        cost = D[np.arange(n), medoids[labels]].sum()
        if cost < best_cost:
            best_cost, best_labels = cost, labels
    return best_labels + 1


def run_clustering(method: str, X: np.ndarray, D: np.ndarray, Z: np.ndarray, k: int, seed: int) -> np.ndarray:
    if method == "ward":
        return fcluster(Z, t=k, criterion="maxclust")
    if method == "spectral":
        sigma = np.median(D[np.triu_indices_from(D, k=1)])
        affinity = np.exp(-(D ** 2) / (2 * sigma ** 2))
        return SpectralClustering(n_clusters=k, affinity="precomputed", random_state=seed).fit_predict(affinity) + 1
    if method == "kmedoids":
        return kmedoids(D, k, np.random.default_rng(seed))
    raise ValueError(method)


def perm_p(observed: float, null: list[float]) -> float:
    return (1 + sum(x >= observed for x in null)) / (1 + len(null))


def knn_purity(D: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    purity = []
    for i in range(len(D)):
        neighbours = np.argsort(D[i])[1:k + 1]
        purity.append(np.mean(labels[neighbours] == labels[i]))
    return float(np.mean(purity))


# ----------------------------------------------------------------------------
# Referência por comunidade sem parágrafos partilhados
# ----------------------------------------------------------------------------
def nearest_community(X, E, Wt, lab_idx, n_comm):
    """Distância de cada personagem i à média simples dos *outros* membros de cada comunidade.

    E[i, j] = embedding da personagem j calculado SEM os parágrafos em que i aparece.
    Wt[i, j] = 1 se j (j != i) tem parágrafos suficientes para servir de referência a i.
    """
    H = np.eye(n_comm)[lab_idx]                              # n x C
    cnt = np.einsum("ij,jc->ic", Wt, H)                      # nº de referências por (i, comunidade)
    sums = np.einsum("ij,jc,ijk->ick", Wt, H, E, optimize=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        R = sums / cnt[:, :, None]
    d = np.linalg.norm(X[:, None, :] - R, axis=2)
    d[cnt == 0] = np.inf
    return d


def hit_rates(d: np.ndarray, lab_idx: np.ndarray, n_comm: int) -> tuple[np.ndarray, float, float]:
    nearest = d.argmin(axis=1)
    nearest[np.isinf(d.min(axis=1))] = -1                    # sem referência utilizável = falha
    hits = nearest == lab_idx
    macro = float(np.mean([hits[lab_idx == c].mean() for c in range(n_comm)]))
    return nearest, float(hits.mean()), macro


# ----------------------------------------------------------------------------
# Auxiliares de saída
# ----------------------------------------------------------------------------
def topic_labels(path: Path, topic_ids: list[int], n_words: int = 3) -> dict[int, str]:
    labels = {t: f"T{t}" for t in topic_ids}
    if path.exists():
        terms = tm.read_csv(path).sort_values(["topic_id", "rank"])
        for topic_id, group in terms.groupby("topic_id"):
            if int(topic_id) in labels:
                labels[int(topic_id)] = f"T{int(topic_id)}: " + ", ".join(group["term"].head(n_words))
    return labels


def make_plots(Z, lift, leaf_labels, comm_labels, topic_names, out_dir):
    communities = sorted(set(comm_labels))
    cmap = plt.get_cmap("tab10")
    colour = {c: cmap(i % 10) for i, c in enumerate(communities)}
    leaf_to_comm = dict(zip(leaf_labels, comm_labels))

    fig, ax = plt.subplots(figsize=(max(9, len(leaf_labels) * 0.45), 6))
    dendrogram(Z, labels=leaf_labels, leaf_rotation=90, color_threshold=0, above_threshold_color="grey", ax=ax)
    for tick in ax.get_xmajorticklabels():
        tick.set_color(colour[leaf_to_comm[tick.get_text()]])
    ax.set_title("Ward sobre √perfil (cor = comunidade Louvain)")
    ax.legend(handles=[plt.Line2D([], [], color=colour[c], marker="o", ls="", label=c) for c in communities],
              loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dendrogram_characters.png", dpi=200)
    plt.close(fig)

    order = sorted(range(len(leaf_labels)), key=lambda i: (comm_labels[i], leaf_labels[i]))
    log_lift = np.clip(np.log2(lift[order]), -2, 2)
    fig, ax = plt.subplots(figsize=(max(8, len(topic_names) * 1.1), max(5, len(order) * 0.3)))
    im = ax.imshow(log_lift, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{leaf_labels[i]} [{comm_labels[i]}]" for i in order], fontsize=7)
    ax.set_xticks(range(len(topic_names)))
    ax.set_xticklabels(topic_names, rotation=60, ha="right", fontsize=7)
    fig.colorbar(im, ax=ax, label="log2(perfil encolhido / frequência global)")
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_character_topics_lift.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Uma execução para um dado --min-paragraphs
# ----------------------------------------------------------------------------
def run_one(min_par: int, ctx: dict, args: argparse.Namespace) -> dict | None:
    out_dir = args.output_dir / f"minpar_{min_par}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_state)

    ids_all, counts_all, G = ctx["ids"], ctx["counts"], ctx["G"]
    M_all, T, comm_all = ctx["M"], ctx["T"], ctx["comm"]
    n_valid = counts_all.sum(axis=1)

    sel = [i for i in range(len(ids_all)) if n_valid[i] >= min_par]
    sizes = pd.Series(comm_all[sel]).value_counts()
    eligible = set(sizes[sizes >= 2].index)
    dropped_single = sorted(set(sizes.index) - eligible)
    sel = [i for i in sel if comm_all[i] in eligible]
    comms = sorted(eligible)
    if len(comms) < 2 or len(sel) < 6:
        print(f"[min_paragraphs={min_par}] poucas personagens/comunidades ({len(sel)} / {len(comms)}); a saltar.")
        return None

    ids = [ids_all[i] for i in sel]
    names = [ctx["names"].get(i, i) for i in ids]
    labels = comm_all[sel]
    lab_idx = np.array([comms.index(c) for c in labels])
    n_comm = len(comms)
    counts = counts_all[sel]
    M = M_all[sel]
    n = len(sel)
    print(f"\n=== min_paragraphs={min_par}: {n} personagens, {n_comm} comunidades "
          f"(comunidades com <2 membros removidas: {dropped_single or 'nenhuma'}) ===")

    X = embed(counts, G, args.alpha, args.representation)
    D = squareform(pdist(X))

    # --- 3) referências sem parágrafos partilhados ---
    E = np.zeros((n, n, X.shape[1]))
    Wt = np.zeros((n, n))
    for a in range(n):
        shared = (M * M[a][None, :]) @ T                      # contagens por tópico dos parágrafos partilhados com a
        minus = np.clip(counts - shared, 0, None)
        E[a] = embed(minus, G, args.alpha, args.representation)
        Wt[a] = (minus.sum(axis=1) >= args.min_ref_paragraphs).astype(float)
        Wt[a, a] = 0.0

    d_comm = nearest_community(X, E, Wt, lab_idx, n_comm)
    nearest, hit_micro, hit_macro = hit_rates(d_comm, lab_idx, n_comm)
    null_micro, null_macro = [], []
    for _ in range(args.n_permutations):
        shuffled = rng.permutation(lab_idx)
        _, m1, m2 = hit_rates(nearest_community(X, E, Wt, shuffled, n_comm), shuffled, n_comm)
        null_micro.append(m1)
        null_macro.append(m2)

    # --- medidas sem clustering ---
    sil = float(silhouette_score(D, lab_idx, metric="precomputed"))
    knn = knn_purity(D, lab_idx, k=3)
    null_sil, null_knn = [], []
    for _ in range(args.n_permutations):
        shuffled = rng.permutation(lab_idx)
        null_sil.append(silhouette_score(D, shuffled, metric="precomputed"))
        null_knn.append(knn_purity(D, shuffled, k=3))

    # --- 2) clustering ---
    k = args.k or n_comm
    Z = linkage(X, method="ward")
    cluster_labels, clustering_rows = {}, {}
    for method in args.cluster_methods:
        cl = run_clustering(method, X, D, Z, k, args.random_state)
        cluster_labels[method] = cl
        ari = adjusted_rand_score(lab_idx, cl)
        nmi = normalized_mutual_info_score(lab_idx, cl)
        null_ari = [adjusted_rand_score(rng.permutation(lab_idx), cl) for _ in range(args.n_permutations)]
        null_nmi = [normalized_mutual_info_score(rng.permutation(lab_idx), cl) for _ in range(args.n_permutations)]
        clustering_rows[method] = {
            "ARI": float(ari), "ARI_perm_mean": float(np.mean(null_ari)), "ARI_p": perm_p(ari, null_ari),
            "NMI": float(nmi), "NMI_perm_mean": float(np.mean(null_nmi)), "NMI_p": perm_p(nmi, null_nmi),
            "cluster_sizes": sorted(pd.Series(cl).value_counts().tolist(), reverse=True),
        }
        pd.crosstab(pd.Series(labels, index=ids, name="louvain"), pd.Series(cl, index=ids, name=f"cluster_{method}")) \
            .to_csv(out_dir / f"contingency_louvain_vs_{method}.csv", encoding="utf-8-sig")

    # --- tabelas ---
    topic_ids, tnames = ctx["topic_ids"], ctx["topic_names"]
    g = G / G.sum()
    prof = shrunk_profile(counts, G, args.alpha)
    lift = prof / g

    table = pd.DataFrame(prof, index=ids, columns=[f"topic_{t}" for t in topic_ids])
    table.insert(0, "name", names)
    table.insert(1, "louvain_community", labels)
    table.insert(2, "n_paragraphs", ctx["total_par"].loc[ids].values)
    table.insert(3, "n_paragraphs_with_topic", counts.sum(axis=1).astype(int))
    table.insert(4, "outlier_share", 1 - counts.sum(axis=1) / ctx["total_par"].loc[ids].values)
    table.insert(5, "dominant_topic", [topic_ids[j] for j in prof.argmax(axis=1)])
    table.insert(6, "most_distinctive_topic", [topic_ids[j] for j in lift.argmax(axis=1)])
    table.insert(7, "nearest_community_by_topics", [comms[j] if j >= 0 else "" for j in nearest])
    table.insert(8, "matches_own_community", nearest == lab_idx)
    for method, cl in cluster_labels.items():
        table[f"cluster_{method}"] = cl
    table.index.name = "character_id"
    table.to_csv(out_dir / "character_topic_profiles.csv", encoding="utf-8-sig")

    rows = []
    for c, comm in enumerate(comms):
        m = lab_idx == c
        intra = D[np.ix_(m, m)][np.triu_indices(m.sum(), k=1)]
        inter = D[np.ix_(m, ~m)].ravel()
        mean_prof = prof[m].mean(axis=0)
        top_share = np.argsort(mean_prof)[::-1][:3]
        top_lift = np.argsort(mean_prof / g)[::-1][:3]
        rows.append({
            "community_id": comm, "n_characters": int(m.sum()),
            "mean_intra_distance": float(intra.mean()), "mean_inter_distance": float(inter.mean()),
            "share_closest_to_own_community": float(np.mean(nearest[m] == c)),
            "top_topics_by_share": " | ".join(tnames[topic_ids[j]] for j in top_share),
            "top_topics_by_lift": " | ".join(tnames[topic_ids[j]] for j in top_lift),
        })
    community_table = pd.DataFrame(rows)
    community_table.to_csv(out_dir / "community_coherence.csv", index=False, encoding="utf-8-sig")

    summary = {
        "min_paragraphs": min_par, "representation": args.representation, "alpha": args.alpha,
        "min_ref_paragraphs": args.min_ref_paragraphs,
        "n_characters": n, "n_communities": n_comm, "k_clusters": int(k),
        "communities_dropped_single_member": dropped_single,
        "nearest_community_hit_rate": hit_micro, "hit_rate_perm_mean": float(np.mean(null_micro)),
        "hit_rate_p": perm_p(hit_micro, null_micro),
        "nearest_community_hit_rate_macro": hit_macro, "hit_rate_macro_perm_mean": float(np.mean(null_macro)),
        "hit_rate_macro_p": perm_p(hit_macro, null_macro),
        "knn3_purity": knn, "knn3_perm_mean": float(np.mean(null_knn)), "knn3_p": perm_p(knn, null_knn),
        "silhouette_louvain": sil, "silhouette_perm_mean": float(np.mean(null_sil)),
        "silhouette_p": perm_p(sil, null_sil),
        "clustering": clustering_rows,
    }
    with (out_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    leaf_labels = [f"{nm} (n={int(c)})" for nm, c in zip(names, counts.sum(axis=1))]
    make_plots(Z, lift, leaf_labels, list(labels), [tnames[t] for t in topic_ids], out_dir)

    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "clustering"}, indent=2, ensure_ascii=False))
    for method, row in clustering_rows.items():
        print(f"  [{method}] ARI={row['ARI']:.3f} (nulo {row['ARI_perm_mean']:.3f}, p={row['ARI_p']:.3f}) | "
              f"NMI={row['NMI']:.3f} (nulo {row['NMI_perm_mean']:.3f}, p={row['NMI_p']:.3f}) | "
              f"tamanhos {row['cluster_sizes']}")
    print(community_table.to_string(index=False))
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bertopic-documents", type=Path,
                   default=Path("outputs/louvain_final_final/bertopic_documents.csv"))
    p.add_argument("--topic-terms", type=Path, default=None,
                   help="bertopic_topic_terms.csv (por defeito, na mesma pasta de --bertopic-documents).")
    p.add_argument("--mentions", type=Path, default=Path("Data/mentions_characters.csv"))
    p.add_argument("--nodes", type=Path, default=Path("Data/os_maias_ground_truth_nodes.csv"))
    p.add_argument("--communities", type=Path, default=Path("Data/louvain_communities_by_community.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/character_topic_profiles_v2_15par"))
    p.add_argument("--min-paragraphs", type=int, nargs="+", default=[15],
                   help="Mínimo de parágrafos com tópico (fora de outliers). Aceita vários valores (sensibilidade).")
    p.add_argument("--representation", choices=("dirichlet", "logodds"), default="dirichlet")
    p.add_argument("--alpha", type=float, default=20.0,
                   help="Força do prior (nº de pseudo-parágrafos com a distribuição global).")
    p.add_argument("--min-ref-paragraphs", type=int, default=5,
                   help="Mínimo de parágrafos (após remover os partilhados) para um membro servir de referência.")
    p.add_argument("--cluster-methods", nargs="+", choices=("ward", "spectral", "kmedoids"),
                   default=["ward", "spectral", "kmedoids"])
    p.add_argument("--k", type=int, default=None, help="Nº de clusters; por defeito, nº de comunidades analisadas.")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topic_terms_path = args.topic_terms or args.bertopic_documents.parent / "bertopic_topic_terms.csv"

    nodes = tm.read_csv(args.nodes)
    communities = tm.load_communities(args.communities, tm.build_character_lookup(nodes))
    char2comm = {cid: comm for comm, ids in communities.items() for cid in ids}
    names = dict(zip(nodes["id"].astype(str), nodes["name"]))

    docs = tm.read_csv(args.bertopic_documents)[["paragraph_id", "bertopic_topic"]].drop_duplicates("paragraph_id")
    mentions = tm.read_csv(args.mentions).dropna(subset=["character_id"]).copy()
    mentions["character_id"] = mentions["character_id"].astype(str)
    pairs = (mentions[["paragraph_id", "character_id"]].drop_duplicates()
             .merge(docs, on="paragraph_id", how="inner"))
    pairs = pairs[pairs["character_id"].isin(char2comm)]
    total_par = pairs.groupby("character_id").size()

    valid = pairs[pairs["bertopic_topic"] >= 0]
    topic_ids = sorted(int(t) for t in valid["bertopic_topic"].unique())
    counts_df = (pd.crosstab(valid["character_id"], valid["bertopic_topic"])
                 .reindex(columns=topic_ids, fill_value=0))
    ids = list(counts_df.index)
    counts = counts_df.values.astype(float)
    G = counts.sum(axis=0)                                   # fundo global (nível pares parágrafo-personagem)

    # Matriz personagem x parágrafo e parágrafo x tópico (para remover parágrafos partilhados).
    par_topic = valid.drop_duplicates("paragraph_id").set_index("paragraph_id")["bertopic_topic"]
    par_index = {pid: j for j, pid in enumerate(par_topic.index)}
    T = np.zeros((len(par_index), len(topic_ids)))
    t_index = {t: j for j, t in enumerate(topic_ids)}
    for pid, topic in par_topic.items():
        T[par_index[pid], t_index[int(topic)]] = 1.0
    char_index = {c: i for i, c in enumerate(ids)}
    M = np.zeros((len(ids), len(par_index)))
    for pid, cid in zip(valid["paragraph_id"], valid["character_id"]):
        M[char_index[cid], par_index[pid]] = 1.0

    ctx = {
        "ids": ids, "counts": counts, "G": G, "M": M, "T": T,
        "comm": np.array([char2comm[c] for c in ids]), "names": names, "total_par": total_par,
        "topic_ids": topic_ids, "topic_names": topic_labels(topic_terms_path, topic_ids),
    }
    print(f"{len(ids)} personagens com pelo menos um parágrafo com tópico; "
          f"representação={args.representation}, alpha={args.alpha}.")

    summaries = [s for s in (run_one(m, ctx, args) for m in args.min_paragraphs) if s]
    if summaries:
        flat = []
        for s in summaries:
            row = {k_: v for k_, v in s.items() if k_ not in ("clustering", "communities_dropped_single_member")}
            for method, r in s["clustering"].items():
                row.update({f"{method}_{k_}": (str(v) if isinstance(v, list) else v) for k_, v in r.items()})
            flat.append(row)
        pd.DataFrame(flat).to_csv(args.output_dir / "sensitivity_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\nResultados: {args.output_dir}")


if __name__ == "__main__":
    main()
