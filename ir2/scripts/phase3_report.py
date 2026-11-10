"""Phase 3d: tables and figures from the backtest csv files.

Usage: ir2/.venv/bin/python ir2/scripts/phase3_report.py [--hero recommender_systems 2018] [--setting phrases]
Writes ir2/results/backtest_summary.md, ir2/results/hero_top20.md and ir2/figures/fig3a_*.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

METHODS = ["w2v_out", "w2v_in", "specter2", "cooc2", "frequency", "random"]
LABELS = {"w2v_out": "word2vec output·input (paper)", "w2v_in": "word2vec input·input",
          "specter2": "SPECTER2 abstract means", "cooc2": "2nd-order co-occurrence",
          "frequency": "frequency", "random": "random (1000 draws)"}
COLORS = {"w2v_out": "#1f4e9c", "w2v_in": "#6f9fd8", "specter2": "#2a9d8f", "cooc2": "#e76f51",
          "frequency": "#7a7a7a", "random": "#b5b5b5"}


def fmt(m, s):
    return f"{m:.3f}" if np.isnan(s) else f"{m:.3f} ± {s:.3f}"


def summary(r, setting):
    """Mean over scored (target, T) pairs; for word2vec, mean ± std across the 3 seeds of that mean."""
    r = r[(r.setting == setting) & r.w2v_target_in_vocab]
    pairs = r[["target", "T"]].drop_duplicates()
    out = [f"### Candidates: `{setting}` — {len(pairs)} scored (target, T) pairs", "",
           "| method | P@10 | P@20 | P@50 | P@50 frequency-matched | P@50 − freq-matched |", "|---|:-:|:-:|:-:|:-:|:-:|"]
    for m in METHODS:
        x = r[r.method == m]
        if x.empty:
            continue
        by_seed = x.groupby("seed")[["p10", "p20", "p50", "p50_freq_matched"]].mean()
        if "p50_freq_matched" in by_seed:
            by_seed["lift"] = by_seed["p50"] - by_seed["p50_freq_matched"]
        mu, sd = by_seed.mean(), by_seed.std() if len(by_seed) > 1 else by_seed.mean() * np.nan
        fm = fmt(mu["p50_freq_matched"], sd["p50_freq_matched"]) if not np.isnan(mu.get("p50_freq_matched", np.nan)) else "—"
        lift = fmt(mu["lift"], sd["lift"]) if not np.isnan(mu.get("lift", np.nan)) else "—"
        out.append(f"| {LABELS[m]} | {fmt(mu.p10, sd.p10)} | {fmt(mu.p20, sd.p20)} | {fmt(mu.p50, sd.p50)} | {fm} | {lift} |")
    base = r.drop_duplicates(["target", "T"]).base_rate.mean()
    out += ["", f"Mean base rate (share of all candidates found): {base:.3f}.", ""]
    return out


def per_pair(r, setting):
    r = r[(r.setting == setting)]
    piv = r.groupby(["target", "T", "method"]).p50.mean().unstack("method")
    piv["base_rate"] = r.groupby(["target", "T"]).base_rate.first()
    cols = [m for m in METHODS if m in piv] + ["base_rate"]
    return ["| target | T | " + " | ".join(cols) + " |", "|---|---|" + ":-:|" * len(cols)] + \
        [f"| {t} | {T} | " + " | ".join("—" if np.isnan(v) else f"{v:.2f}" for v in row[cols]) + " |"
         for (t, T), row in piv.iterrows()]


def fig3a(c, setting, path):
    c = c[(c.setting == setting) & c.w2v_target_in_vocab]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    w = c[c.method == "w2v_out"].groupby(["T", "n"]).frac.mean().reset_index()   # mean over targets and seeds
    for T, g in w.groupby("T"):
        ax.plot(g.n, 100 * g.frac, color=COLORS["w2v_out"], lw=0.8, alpha=0.45)
        ax.annotate(str(T), (g.n.iloc[-1], 100 * g.frac.iloc[-1]), fontsize=7, color=COLORS["w2v_out"], alpha=0.7,
                    xytext=(2, 0), textcoords="offset points", va="center")
    for m in ["w2v_out", "w2v_in", "specter2", "cooc2", "frequency", "random"]:
        g = c[c.method == m].groupby(["T", "n"]).frac.mean().groupby("n").agg(["mean", "count"])
        g = g[g["count"] >= 3]                          # mean only where >= 3 cutoffs have that horizon
        if g.empty:
            continue
        ax.plot(g.index, 100 * g["mean"], color=COLORS[m], lw=3 if m == "w2v_out" else 1.8,
                ls="-" if m.startswith("w2v") else "--", label=LABELS[m] + (" (mean)" if m == "w2v_out" else ""))
    ax.set_xlabel("years after cutoff T (n = 1 means found in year T)")
    ax.set_ylabel("% of top-50 candidates found")
    ax.set_title(f"Backtest on cs.IR, cutoffs {min(L.CUTOFFS)}–{max(L.CUTOFFS)}, candidates: {setting}\n"
                 "thin lines: word2vec output·input per cutoff (mean over targets and seeds)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def hero(t, target, T, setting, k=20):
    x = t[(t.setting == setting) & (t.target == target) & (t["T"] == T)]
    out = [f"# Hero list: `{target}`, cutoff T = {T}, candidates `{setting}`", "",
           f"Trained on cs.IR ∪ cs.CL abstracts before {T}. Candidates never appear in an abstract with a "
           f"`{target}` indicator before {T}. ✓ = appears with an indicator in ≥ 3 cs.IR abstracts from {T} on "
           f"(year the third one appeared).", "",
           "| rank | word2vec output·input (paper) | found | frequency baseline | found |", "|:-:|---|:-:|---|:-:|"]
    a = x[x.method == "w2v_out"].sort_values("rank").head(k)
    b = x[x.method == "frequency"].sort_values("rank").head(k)
    mark = lambda r: f"✓ {int(r.first_year)}" if r.found else "✗"
    for (_, ra), (_, rb) in zip(a.iterrows(), b.iterrows()):
        out.append(f"| {ra['rank']} | `{ra.term}` | {mark(ra)} | `{rb.term}` | {mark(rb)} |")
    out += ["", f"word2vec: {int(a.found.sum())}/{len(a)} found · frequency: {int(b.found.sum())}/{len(b)} found", ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", nargs=2, default=["learning_to_rank", "2020"], metavar=("TARGET", "T"))
    ap.add_argument("--hero-setting", default="phrases")
    args = ap.parse_args()
    r = pd.read_csv(L.RESULTS / "backtest.csv")
    c = pd.read_csv(L.RESULTS / "backtest_curves.csv")
    t = pd.read_csv(L.RESULTS / "backtest_top.csv")
    st = pd.read_csv(L.RESULTS / "backtest_status.csv")

    md = ["# Phase 3: backtest results", "",
          "Precision@k = share of the top-k candidates that co-occur with a target indicator in ≥ 3 cs.IR abstracts "
          f"from T to {L.LAST_YEAR}. word2vec rows: mean over scored pairs, ± std across 3 seeds. "
          "\"Frequency-matched\" = expected P@50 if the same top-50 were replaced by random candidates of the same "
          "pre-T frequency decile; the last column is what a method adds beyond frequency.", ""]
    for setting in ["all", "phrases"]:
        md += summary(r, setting)
    md += ["## Status of every (target, T) pair", "",
           st[["target", "T", "status", "pre_T_target_docs", "post_T_target_docs", "target_tokens",
               "n_candidates_all", "base_rate_all", "n_candidates_phrases", "base_rate_phrases"]]
           .to_markdown(index=False, floatfmt=".3f"), ""]
    for setting in ["all", "phrases"]:
        md += [f"## P@50 per pair, candidates `{setting}` (word2vec: mean of 3 seeds)", ""] + per_pair(r, setting) + [""]
    (L.RESULTS / "backtest_summary.md").write_text("\n".join(md))
    print("\n".join(md))
    for setting in ["all", "phrases"]:
        fig3a(c, setting, L.FIGURES / f"fig3a_{setting}.png")

    if args.hero:
        target, T = args.hero[0], int(args.hero[1])
        h = hero(t, target, T, args.hero_setting)
        (L.RESULTS / "hero_top20.md").write_text("\n".join(h))
        print("\n".join(h))


if __name__ == "__main__":
    main()
