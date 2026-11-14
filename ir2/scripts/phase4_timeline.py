"""Phase 4: embedding timeline of cs.IR (tests_backtest.md Part B).

A. SPECTER2: for each keyword-defined subtopic and year, the centroid of its cs.IR abstracts' cached SPECTER2
   vectors (only points with >= MIN_DOCS abstracts). One PCA fitted on all centroids together, so the axes are
   shared across years. Robustness: refit without 2026 (partial year) and with >= 50 abstracts per point, compared
   with the main layout by Procrustes disparity (0 = same shape).
B. word2vec (optional part of the plan): nearest neighbours of a few concepts per cutoff (seed 0), and how far each
   concept's vector moves between consecutive cutoffs after orthogonal Procrustes alignment on the shared vocabulary
   (Hamilton et al. 2016), compared with the median word.

Usage: ir2/.venv/bin/python ir2/scripts/phase4_timeline.py
Writes ir2/figures/timeline_specter2.png (overview), ir2/figures/timeline_specter2_panels.png (one panel per
       subtopic), ir2/results/timeline.md, ir2/results/timeline_centroids.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from scipy.spatial import procrustes  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

MIN_DOCS = 20                     # same threshold as the backtest's "concept exists" rule
YEARS = range(2010, L.LAST_YEAR + 1)
# Keyword sets, matched as whole tokens on the normalized unphrased text (like the backtest indicators).
# The first five extend the backtest targets; evaluation and LLM/RAG are added to cover the field.
SUBTOPICS = {
    "recommender systems": ["recommender", "recommenders", "recommendation", "recommendations",
                            "collaborative filtering"],
    "query expansion & reformulation": ["query expansion", "query reformulation", "query rewriting",
                                        "query suggestion", "relevance feedback"],
    "learning to rank": ["learning to rank", "ranknet", "lambdamart", "lambdarank", "listwise", "pairwise ranking"],
    "dense & neural retrieval": ["dense retrieval", "dense retriever", "dense retrievers", "dense passage retrieval",
                                 "dual encoder", "dual encoders", "bi encoder", "bi encoders", "neural ranking",
                                 "neural retrieval", "colbert"],
    "conversational search": ["conversational search", "conversational recommendation", "conversational recommender",
                              "clarifying question", "clarifying questions", "conversational information seeking"],
    "IR evaluation": ["test collection", "test collections", "relevance judgments", "relevance judgements",
                      "relevance assessments", "evaluation metrics", "evaluation measures", "offline evaluation",
                      "cranfield"],
    "LLMs & RAG": ["large language model", "large language models", "llm", "llms", "retrieval augmented", "rag"],
}
# categorical slots 1-7 of the dataviz skill's validated default palette, fixed order (light mode)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
INK, MUTED = "#2b2b2a", "#77766f"
W2V_CONCEPTS = ["query_expansion", "recommender", "learning_to_rank", "retrieval"]


def centroids(docs, emb_of_row, min_docs, years):
    rows = []
    ir = np.flatnonzero(docs["is_ir"].values)
    text = docs["text"].values
    year = docs["year"].values
    for name, kws in SUBTOPICS.items():
        m = L.indicator_regex(kws)
        hit = np.array([r for r in ir if m.search(text[r])])
        for y in years:
            sel = hit[year[hit] == y]
            if len(sel) >= min_docs:
                c = emb_of_row(sel).mean(axis=0)
                rows.append({"subtopic": name, "year": y, "n": len(sel), "vec": c / np.linalg.norm(c), "raw": c})
    return pd.DataFrame(rows)


def layout(c):
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(np.stack(c.vec.values))
    return xy, pca.explained_variance_ratio_


def disparity(c_main, xy_main, c_alt, xy_alt):
    key = lambda c: list(zip(c.subtopic, c.year))
    a = dict(zip(key(c_main), xy_main))
    b = dict(zip(key(c_alt), xy_alt))
    common = [k for k in a if k in b]
    return procrustes(np.array([a[k] for k in common]), np.array([b[k] for k in common]))[2], len(common)


def plot(c, xy, var, path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, name in enumerate(SUBTOPICS):
        m = (c.subtopic == name).values
        if not m.any():
            continue
        pts, ys = xy[m], c.year.values[m]
        col = COLORS[i]
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2, solid_capstyle="round", zorder=2,
                label=f"{name} ({ys[0]}–{ys[-1]})")
        ax.scatter(pts[:, 0], pts[:, 1], s=30, color=col, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter(*pts[0], s=70, facecolor="white", edgecolor=col, linewidth=2, zorder=4)   # start: hollow
        if len(pts) > 1:                                                                  # end: arrowhead
            ax.annotate("", pts[-1], xytext=pts[-2], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=16))
        for (x, y), yr, dy in [(pts[0], ys[0], -11), (pts[-1], ys[-1], 6)]:
            ax.annotate(str(yr), (x, y), xytext=(5, dy), textcoords="offset points", fontsize=7.5, color=MUTED)
    ax.set_xlabel(f"PC1 ({var[0]:.0%} of variance)", color=INK)
    ax.set_ylabel(f"PC2 ({var[1]:.0%} of variance)", color=INK)
    ax.set_title("cs.IR subtopics over time: yearly centroids of SPECTER2 abstract vectors\n"
                 f"one PCA fitted on all centroids · a point needs ≥ {MIN_DOCS} abstracts that year\n"
                 "hollow dot = first year, arrow = last year", fontsize=10, color=INK)
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#bdbcb5")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.margins(x=0.08, y=0.08)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1), labelcolor=INK,
              title="subtopic (years plotted)", title_fontsize=8.5, alignment="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_panels(c, xy, var, path):
    """Small multiples: one panel per subtopic, the focal trajectory in one colour over the others in grey.
    Identity comes from the panel title, not from colour (the 7-colour overview fails the all-pairs CVD check)."""
    names = [n for n in SUBTOPICS if (c.subtopic == n).any()]
    ncol = 4
    nrow = -(-len(names) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        for other in names:
            m = (c.subtopic == other).values
            ax.plot(xy[m, 0], xy[m, 1], color="#d6d5cf", lw=1.2, zorder=1)
        m = (c.subtopic == name).values
        pts, ys = xy[m], c.year.values[m]
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS[0], lw=2, zorder=2)
        ax.scatter(pts[:, 0], pts[:, 1], s=22, color=COLORS[0], edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter(*pts[0], s=55, facecolor="white", edgecolor=COLORS[0], linewidth=2, zorder=4)
        if len(pts) > 1:
            ax.annotate("", pts[-1], xytext=pts[-2], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[0], lw=2, mutation_scale=13))
        for (x, y), yr, dy in [(pts[0], ys[0], -11), (pts[-1], ys[-1], 6)]:
            ax.annotate(str(yr), (x, y), xytext=(5, dy), textcoords="offset points", fontsize=7.5, color=MUTED)
        ax.set_title(name, fontsize=10, color=INK, loc="left")
        ax.grid(color="#eeede8", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bdbcb5")
        ax.tick_params(colors=MUTED, labelsize=7)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.supxlabel(f"PC1 ({var[0]:.0%} of variance)", color=INK, fontsize=9)
    fig.supylabel(f"PC2 ({var[1]:.0%} of variance)", color=INK, fontsize=9)
    fig.suptitle("cs.IR subtopics over time (SPECTER2 yearly centroids, one shared PCA) · grey = the other "
                 "subtopics · hollow dot = first year, arrow = last year", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def w2v_part():
    from gensim.models import KeyedVectors
    kv = {T: KeyedVectors.load(str(L.cutoff_dir(T) / "w2v_s0.kv")) for T in L.CUTOFFS}
    out = ["## B. word2vec: neighbours and movement across cutoffs (seed 0)", "",
           "Spaces of different cutoffs are not aligned, so each pair of consecutive cutoffs is aligned by "
           "orthogonal Procrustes on the normalized input vectors of their shared vocabulary (HistWords). "
           "*Self-similarity* = cosine between a word's aligned vector at T and its vector at the next cutoff; "
           "lower = the word's usage moved more. It is compared with the median over all shared words.", "",
           "| concept | " + " | ".join(f"{a}→{b}" for a, b in zip(L.CUTOFFS, L.CUTOFFS[1:])) + " |",
           "|---|" + ":-:|" * (len(L.CUTOFFS) - 1)]
    sims = {w: [] for w in W2V_CONCEPTS}
    med = []
    for a, b in zip(L.CUTOFFS, L.CUTOFFS[1:]):
        shared = [w for w in kv[a].index_to_key if w in kv[b].key_to_index]
        A = np.stack([kv[a].get_vector(w, norm=True) for w in shared])
        B = np.stack([kv[b].get_vector(w, norm=True) for w in shared])
        R, _ = orthogonal_procrustes(A, B)
        s = np.einsum("ij,ij->i", A @ R, B) / np.linalg.norm(A @ R, axis=1)
        med.append(float(np.median(s)))
        idx = {w: i for i, w in enumerate(shared)}
        for w in W2V_CONCEPTS:
            sims[w].append(f"{s[idx[w]]:.2f}" if w in idx else "—")
    for w in W2V_CONCEPTS:
        out.append(f"| `{w}` | " + " | ".join(sims[w]) + " |")
    out.append("| *median word* | " + " | ".join(f"{m:.2f}" for m in med) + " |")
    out += ["", "Nearest neighbours (input vectors, top 8, terms in the candidate vocabulary):", ""]
    for w in W2V_CONCEPTS:
        out += [f"**`{w}`**", "", "| T | neighbours |", "|---|---|"]
        for T in L.CUTOFFS:
            if w not in kv[T].key_to_index:
                out.append(f"| {T} | not in vocabulary |")
                continue
            nb = [t for t, _ in kv[T].most_similar(w, topn=40) if L.is_candidate_term(t)][:8]
            out.append(f"| {T} | {', '.join(nb)} |")
        out.append("")
    return out


def main():
    docs = L.load_docs()
    emb = np.load(L.DATA / "specter2.npy")
    pos = np.full(len(docs), -1)
    pos[np.load(L.DATA / "specter2_rows.npy")] = np.arange(len(emb))
    emb_of_row = lambda rows: emb[pos[rows]].astype(np.float32)
    assert (pos[docs["is_ir"].values] >= 0).all(), "every cs.IR abstract must have a SPECTER2 vector"

    c = centroids(docs, emb_of_row, MIN_DOCS, YEARS)
    xy, var = layout(c)
    plot(c, xy, var, L.FIGURES / "timeline_specter2.png")
    plot_panels(c, xy, var, L.FIGURES / "timeline_specter2_panels.png")
    c.assign(pc1=xy[:, 0], pc2=xy[:, 1]).drop(columns=["vec", "raw"]).to_csv(L.RESULTS / "timeline_centroids.csv", index=False)

    # robustness
    c26 = c[c.year < L.LAST_YEAR].reset_index(drop=True)
    d26, n26 = disparity(c, xy, c26, layout(c26)[0])
    c50 = centroids(docs, emb_of_row, 50, YEARS)
    d50, n50 = disparity(c, xy, c50, layout(c50)[0])

    # similarities in the full 768-d space, measured relative to the average cs.IR abstract: SPECTER2 vectors are
    # anisotropic (raw centroid cosines are all 0.95-0.99), so the mean cs.IR vector is subtracted first
    mu = emb_of_row(np.flatnonzero(docs["is_ir"].values)).mean(axis=0)
    unit = lambda v: v / np.linalg.norm(v)
    V = {(s, y): unit(r - mu) for s, y, r in zip(c.subtopic, c.year, c.raw)}
    md = ["# Phase 4: embedding timeline of cs.IR", "",
          f"Subtopics are keyword sets matched on cs.IR abstracts (first-version year; an abstract can belong to "
          f"several). A point = the mean SPECTER2 vector of a subtopic's abstracts in one year, kept only with "
          f"≥ {MIN_DOCS} abstracts. Figures: `figures/timeline_specter2_panels.png` (one panel per subtopic) and `figures/timeline_specter2.png` (overview). Caveat: SPECTER2 was pretrained on "
          f"recent papers.", "",
          "Similarities below are cosines of centroids **minus the mean cs.IR abstract vector**: raw SPECTER2 "
          "centroids all have cosine 0.95–0.99 with each other, because all abstracts share a common direction. "
          "Centering does not change the PCA plot.", "",
          "## A. SPECTER2 centroids", "",
          f"One PCA on all {len(c)} centroids: PC1 explains {var[0]:.1%} and PC2 {var[1]:.1%} of their variance "
          f"({var.sum():.1%} together). The 2-D plot is a projection; distances below are in the full space.", "",
          "| subtopic | keywords | first year | last year | abstracts (first → last year) | "
          "centered cosine first↔last centroid |", "|---|---|:-:|:-:|:-:|:-:|"]
    for name, kws in SUBTOPICS.items():
        x = c[c.subtopic == name]
        if x.empty:
            md.append(f"| {name} | {', '.join(kws)} | — | — | never ≥ {MIN_DOCS} | — |")
            continue
        f, l = x.iloc[0], x.iloc[-1]
        md.append(f"| {name} | {', '.join(kws)} | {f.year} | {l.year} | {f.n} → {l.n} | "
                  f"{float(V[(name, f.year)] @ V[(name, l.year)]):.2f} |")
    pairs = [("dense & neural retrieval", "LLMs & RAG"), ("query expansion & reformulation", "LLMs & RAG"),
             ("query expansion & reformulation", "dense & neural retrieval"),
             ("recommender systems", "conversational search"), ("learning to rank", "dense & neural retrieval"),
             ("recommender systems", "LLMs & RAG")]
    yrs = sorted(c.year.unique())
    md += ["", "### Centered cosine similarity between subtopic centroids, by year", "",
           "| pair | " + " | ".join(str(y) for y in yrs) + " |", "|---|" + ":-:|" * len(yrs)]
    for a, b in pairs:
        md.append(f"| {a} ↔ {b} | " + " | ".join(
            f"{float(V[(a, y)] @ V[(b, y)]):.2f}" if (a, y) in V and (b, y) in V else "—" for y in yrs) + " |")
    md += ["", "### Points per subtopic and year (abstracts)", "",
           c.pivot(index="subtopic", columns="year", values="n").fillna(0).astype(int).to_markdown(), "",
           "### Robustness of the 2-D layout", "",
           "Procrustes disparity between the main layout and a refit on a variant, on their common points "
           "(0 = identical shape up to rotation, scaling and reflection; 1 = unrelated).", "",
           "| variant | common points | disparity |", "|---|:-:|:-:|",
           f"| without {L.LAST_YEAR} (partial year) | {n26} | {d26:.3f} |",
           f"| ≥ 50 abstracts per point | {n50} | {d50:.3f} |", ""]
    md += w2v_part()
    (L.RESULTS / "timeline.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
