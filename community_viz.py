"""Plot k-clique communities (with overlaps as pie nodes) and tabulate their members."""
from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from communities import get_k_clique_communities, threshold_graph


def membership_map(groups):
    """node -> list of community numbers (1-based)."""
    out = {}
    for i, group in enumerate(groups, start=1):
        for node in group:
            out.setdefault(node, []).append(i)
    return out


def community_tables(H, k, name_map):
    """Return (summary_df, long_df).

    summary_df: one row per community (size, members sorted by weighted degree, overlap members).
    long_df: one row per (community, character) - handy for CSV/Excel.
    """
    groups = get_k_clique_communities(H, k)
    mem = membership_map(groups)
    strength = dict(H.degree(weight="weight"))
    nm = lambda n: name_map.get(n, n)

    long_rows, summary_rows = [], []
    for i, group in enumerate(groups, start=1):
        ordered = sorted(group, key=lambda n: -strength[n])
        for n in ordered:
            long_rows.append({
                "community": i, "character": nm(n),
                "weighted_degree": strength[n],
                "overlap": len(mem[n]) > 1,
                "all_communities": ", ".join(map(str, mem[n])),
            })
        summary_rows.append({
            "community": i, "size": len(group),
            "members (by weighted degree)": ", ".join(
                nm(n) + (" *" if len(mem[n]) > 1 else "") for n in ordered),
            "overlap members": ", ".join(nm(n) for n in ordered if len(mem[n]) > 1),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(long_rows)


def plot_communities(H, k, w_min, name_map, pos=None, n_labels=8, seed=42, figsize=(16, 12)):
    """Network plot: node colour = community, overlap nodes = pie chart of their communities."""
    groups = get_k_clique_communities(H, k)
    mem = membership_map(groups)
    palette = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
    col = lambda i: palette((i - 1) % palette.N)
    # Isolated characters (no edge left after thresholding) would be thrown to the corners
    # by the force layout, so they are left out of the drawing and only counted in the legend.
    isolated = [n for n in H if H.degree(n) == 0]
    H = H.subgraph([n for n in H if n not in isolated]).copy()
    if pos is None:
        L = H.copy()  # log-damp the weights so the few huge ties do not collapse the layout
        for u, v, d in L.edges(data=True):
            d["w"] = np.log1p(d["weight"])
        pos = nx.spring_layout(L, weight="w", seed=seed, k=4.5 / np.sqrt(len(L)), iterations=400)

    strength = dict(H.degree(weight="weight"))
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # edges: faint overall, stronger inside a community
    inside = {(u, v) for g in groups for u in g for v in g if H.has_edge(u, v)}
    other = [e for e in H.edges if e not in inside and (e[1], e[0]) not in inside]
    nx.draw_networkx_edges(H, pos, edgelist=other, ax=ax, alpha=0.08, width=0.5, edge_color="gray")
    nx.draw_networkx_edges(H, pos, edgelist=list(inside), ax=ax, alpha=0.35, width=0.9)

    # nodes
    left_out = [n for n in H if n not in mem]
    single = [n for n in H if n in mem and len(mem[n]) == 1]
    overlap = [n for n in H if n in mem and len(mem[n]) > 1]
    size = lambda n: 60 + 12 * np.sqrt(strength[n])
    nx.draw_networkx_nodes(H, pos, nodelist=left_out, node_color="#d0d0d0",
                           node_size=[size(n) * 0.5 for n in left_out], ax=ax)
    nx.draw_networkx_nodes(H, pos, nodelist=single,
                           node_color=[col(mem[n][0]) for n in single],
                           node_size=[size(n) for n in single],
                           edgecolors="white", linewidths=0.6, ax=ax)
    span = max(np.ptp([p[0] for p in pos.values()]), np.ptp([p[1] for p in pos.values()]))
    for n in overlap:  # pie chart node: one wedge per community
        r = span * 0.0045 * np.sqrt(size(n) / 60)
        ax.pie([1] * len(mem[n]), colors=[col(i) for i in mem[n]], center=pos[n], radius=r,
               wedgeprops=dict(edgecolor="black", linewidth=1.4), frame=True)
    ax.set_xlim(*ax.get_xlim()); ax.set_ylim(*ax.get_ylim())

    # labels: all overlap nodes + strongest covered nodes
    covered = [n for n in H if n in mem]
    top = set(sorted(covered, key=lambda n: -strength[n])[:n_labels]) | set(overlap)
    for n in top:
        ax.annotate(name_map.get(n, n), pos[n], xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8, fontweight="bold" if n in overlap else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    handles = [Patch(color=col(i), label=f"Community {i} ({len(g)} characters)")
               for i, g in enumerate(groups, start=1)]
    handles.append(Patch(color="#d0d0d0", label=f"Connected but in no community ({len(left_out)})"))
    if isolated:
        handles.append(Patch(color="white", label=f"Not drawn: {len(isolated)} characters with no edge at w_min={w_min}"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True)
    ax.set_title(f"k-clique communities: k={k}, w_min={w_min}\n"
                 "pie nodes = characters in 2+ communities; node size = weighted degree",
                 fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, pos


def plot_community_table(summary_df, k, w_min, wrap=95, figsize_w=14):
    """Render the summary table as a figure (wrapped text), colour-coded like the network plot."""
    n = len(summary_df)
    palette = plt.get_cmap("tab10" if n <= 10 else "tab20")
    rows = []
    for r in summary_df.itertuples(index=False):
        rows.append([str(r[0]), str(r[1]), textwrap.fill(r[2], wrap),
                     textwrap.fill(r[3], 30)])
    heights = [max(c.count("\n") + 1 for c in row) for row in rows]
    fig, ax = plt.subplots(figsize=(figsize_w, 1.0 + 0.24 * sum(heights) + 0.3 * n))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["#", "Size", "Members (by weighted degree; * = overlap)", "Overlap members"],
                   colWidths=[0.04, 0.05, 0.66, 0.25], loc="upper center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    total = sum(heights)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#333"); cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.05)
        else:
            cell.set_height(0.9 * heights[r - 1] / total)
            if c == 0:
                cell.set_facecolor(palette((r - 1) % palette.N)); cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#f7f7f7" if r % 2 else "white")
    ax.set_title(f"Characters per community (k={k}, w_min={w_min})", fontsize=12, pad=10)
    fig.tight_layout()
    return fig, ax


def plot_communities_clean(G, H, k, w_min, name_map, seed=42, n_extra_labels=2,
                           intra_boost=4.0, outward=0.25, jitter=0.12, community_names=None, figsize=(13, 10)):
    """Louvain-style figure for k-clique communities.

    G : full weighted graph (used only for the layout, so no character is thrown to the corners)
    H : thresholded graph the communities were computed on
    Layout trick: edges inside a community are pulled together (intra_boost) so groups form
    visible clusters; overlap characters end up between their communities.
    """
    groups = get_k_clique_communities(H, k)
    mem = membership_map(groups)
    strength = dict(H.degree(weight="weight"))
    nm = lambda n: name_map.get(n, n)
    palette = plt.get_cmap("tab10" if len(groups) <= 10 else "tab20")
    col = lambda i: palette((i - 1) % palette.N)

    # Layout in two steps so the communities are readable:
    # 1) force layout of the covered characters only, edges inside a community boosted;
    # 2) every other character is placed near its covered neighbours, pushed slightly outward.
    covered = [n for n in G if n in mem]
    L = nx.Graph()
    L.add_nodes_from(covered)
    for u, v, d in G.subgraph(covered).edges(data=True):
        shared = set(mem[u]) & set(mem[v])
        L.add_edge(u, v, w=np.log1p(d["weight"]) * (intra_boost if shared else 1.0))
    pos = nx.spring_layout(L, weight="w", seed=seed, k=5.0 / np.sqrt(len(L)), iterations=800)
    centre = np.mean([pos[n] for n in covered], axis=0)
    rng = np.random.default_rng(seed)
    pending = [n for n in G if n not in pos]
    for _ in range(5):  # a few passes so nodes attached only to other outsiders still get placed
        for n in list(pending):
            nb = [(m, np.log1p(G[n][m]["weight"])) for m in G[n] if m in pos]
            if not nb:
                continue
            w = np.array([x[1] for x in nb]); xy = np.array([pos[x[0]] for x in nb])
            p = (xy * w[:, None]).sum(0) / w.sum()
            direction = p - centre
            direction = direction / (np.linalg.norm(direction) + 1e-9)
            pos[n] = p + direction * outward + rng.normal(0, jitter, 2)
            pending.remove(n)
    for n in pending:  # characters with no edges at all
        angle = rng.uniform(0, 2 * np.pi)
        pos[n] = centre + 1.6 * np.array([np.cos(angle), np.sin(angle)])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.06, width=0.5, edge_color="#6b7b8c")
    inside = [(u, v) for g in groups for u in g for v in g if u < v and H.has_edge(u, v)]
    nx.draw_networkx_edges(H, pos, edgelist=inside, ax=ax, alpha=0.18, width=0.8, edge_color="#6b7b8c")

    size = lambda n: 40 + 11 * np.sqrt(strength.get(n, 0))
    left_out = [n for n in G if n not in mem]
    single = [n for n in G if n in mem and len(mem[n]) == 1]
    overlap = [n for n in G if n in mem and len(mem[n]) > 1]
    nx.draw_networkx_nodes(G, pos, nodelist=left_out, node_color="#d9d9d9",
                           node_size=[size(n) * 0.6 for n in left_out],
                           edgecolors="#9a9a9a", linewidths=0.4, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=single, node_color=[col(mem[n][0]) for n in single],
                           node_size=[size(n) for n in single],
                           edgecolors="#222", linewidths=0.6, ax=ax)
    span = max(np.ptp([p[0] for p in pos.values()]), np.ptp([p[1] for p in pos.values()]))
    for n in overlap:
        r = span * 0.0032 * np.sqrt(size(n) / 40) * 1.5
        ax.pie([1] * len(mem[n]), colors=[col(i) for i in mem[n]], center=pos[n], radius=r,
               wedgeprops=dict(edgecolor="black", linewidth=1.5), frame=True)
    ax.set_xlim(min(p[0] for p in pos.values()) - 0.1, max(p[0] for p in pos.values()) + 0.1)
    ax.set_ylim(min(p[1] for p in pos.values()) - 0.1, max(p[1] for p in pos.values()) + 0.1)

    # leaders (bold) + a few extra strong members per community + overlap characters
    leaders, used = {}, set()
    for i, g in enumerate(groups, start=1):
        pool = [n for n in g if len(mem[n]) == 1 and n not in used] or [n for n in g if n not in used] or list(g)
        leaders[i] = max(pool, key=lambda n: strength[n]); used.add(leaders[i])
    to_label = set(leaders.values()) | set(overlap)
    for i, g in enumerate(groups, start=1):
        extra = sorted((n for n in g if n not in to_label), key=lambda n: -strength[n])[:n_extra_labels]
        to_label |= set(extra)
    for n in to_label:
        bold = n in leaders.values() or n in overlap
        ax.annotate(nm(n), pos[n], xytext=(0, 9 + 0.35 * np.sqrt(size(n))), textcoords="offset points",
                    ha="center", fontsize=9 if bold else 8, fontweight="bold" if bold else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75), zorder=10)

    handles = [Patch(facecolor=col(i), edgecolor="#222", label=f"C{i}: {(community_names or {}).get(i, nm(leaders[i]))} (n={len(g)})")
               for i, g in enumerate(groups, start=1)]
    handles.append(Patch(facecolor="#d9d9d9", edgecolor="#9a9a9a", label=f"In no community (n={len(left_out)})"))
    leg = ax.legend(handles=handles, title="Detected communities", loc="lower left",
                    fontsize=9, title_fontsize=10, frameon=True)
    n_cov = len(mem)
    ax.set_title(f"k-clique communities in 'Os Maias' (k={k}, w_min={w_min} | "
                 f"{n_cov}/{G.number_of_nodes()} characters covered, {len(overlap)} overlap)",
                 fontsize=13, fontweight="bold")
    ax.text(0.99, 0.01, "pie node = character in 2+ communities\nnode size = weighted degree",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555")
    ax.axis("off")
    fig.tight_layout()
    return fig, ax, pos