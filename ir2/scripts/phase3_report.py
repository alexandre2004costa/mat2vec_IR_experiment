"""Phase 3d: tables and figures from the backtest csv files.

Usage: ir2/.venv/bin/python ir2/scripts/phase3_report.py [--hero learning_to_rank 2020] [--hero-setting phrases]
                                                         [--variant main|all_versions]
Writes ir2/results/backtest_table{sfx}.md   main table: per pair, P@10/20/50 of word2vec next to random and frequency
       ir2/results/backtest_summary{sfx}.md means over pairs, all methods, pair status
       ir2/results/leakage_comparison.md    main vs all_versions (only when both runs exist)
       ir2/results/hero_top20.md and ir2/figures/fig3a_*{sfx}.png
sfx = "" for the main run, "_all_versions" for the comparison run.
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


def main_table(r, setting):
    """Professor's point 2: for every scored pair, P@10/20/50 of the paper's method next to random and frequency."""
    r = r[r.setting == setting]
    w = r[r.method == "w2v_out"]
    mu = w.groupby(["target", "T"])[["p10", "p20", "p50"]].mean()
    sd = w.groupby(["target", "T"])[["p10", "p20", "p50"]].std()
    other = {m: r[r.method == m].set_index(["target", "T"])[["p10", "p20", "p50"]] for m in ["random", "frequency"]}
    info = r.drop_duplicates(["target", "T"]).set_index(["target", "T"])[["n_candidates", "base_rate"]]
    head = "| target | T | candidates | base rate | " + " | ".join(
        f"P@{k} w2v | P@{k} random | P@{k} freq" for k in (10, 20, 50)) + " |"
    out = [f"### Candidates: `{setting}` ({len(mu)} scored pairs)", "", head,
           "|---|---|--:|--:|" + ":-:|" * 9]
    for key in mu.index:
        cells = []
        for k in ("p10", "p20", "p50"):
            cells += [f"{mu.loc[key, k]:.2f} ± {sd.loc[key, k]:.2f}",
                      f"{other['random'].loc[key, k]:.2f}", f"{other['frequency'].loc[key, k]:.2f}"]
        out.append(f"| {key[0]} | {key[1]} | {info.loc[key, 'n_candidates']:,} | {info.loc[key, 'base_rate']:.2f} | "
                   + " | ".join(cells) + " |")
    # mean row: word2vec = mean over pairs of each seed, then mean ± std over the 3 seeds
    by_seed = w.groupby("seed")[["p10", "p20", "p50"]].mean()
    cells = []
    for k in ("p10", "p20", "p50"):
        cells += [f"**{by_seed[k].mean():.3f} ± {by_seed[k].std():.3f}**",
                  f"**{other['random'][k].mean():.3f}**", f"**{other['frequency'][k].mean():.3f}**"]
    out.append(f"| **mean** | | | {info.base_rate.mean():.2f} | " + " | ".join(cells) + " |")
    # reading rule fixed before the run (tests_backtest.md §9): a pair counts only if all 3 seeds beat the baseline
    worst = w.groupby(["target", "T"]).p50.min()
    wins = lambda m: int((worst > other[m]["p50"].reindex(worst.index)).sum())
    out += ["", f"word2vec P@50 above random for all 3 seeds in {wins('random')}/{len(mu)} pairs, "
                f"above frequency for all 3 seeds in {wins('frequency')}/{len(mu)} pairs.", ""]
    return out


def leakage(variants):
    """Professor's point 3: does the result hold with and without the version filter? Compared on the pairs scored
    in both runs."""
    r = {v: pd.read_csv(L.RESULTS / f"backtest{L.variant_suffix(v)}.csv") for v in variants}
    docs = L.load_docs()[["year", "last_version_year"]]
    out = ["# Leakage check: main run vs all-versions run", "",
           "Main run: pre-T text = papers whose first and latest version are before T. All-versions run: first "
           "version before T only, so abstracts revised in or after T are included.", "",
           "| T | pre-T docs (all versions) | removed by the filter | share |", "|---|--:|--:|--:|"]
    for T in L.CUTOFFS:
        n = int((docs.year < T).sum())
        k = int(((docs.year < T) & (docs.last_version_year < T)).sum())
        out.append(f"| {T} | {n:,} | {n - k:,} | {(n - k) / n:.1%} |")
    for setting in ["all", "phrases"]:
        keys = [set(map(tuple, x[x.setting == setting][["target", "T"]].drop_duplicates().values)) for x in r.values()]
        common = set.intersection(*keys)
        out += ["", f"### Candidates `{setting}`: mean P@k over the {len(common)} pairs scored in both runs", "",
                "| method | " + " | ".join(f"{k} main | {k} all versions" for k in ("P@10", "P@20", "P@50")) + " |",
                "|---|" + ":-:|" * 6]
        for m in METHODS:
            cells = []
            for k in ("p10", "p20", "p50"):
                for v in variants:
                    x = r[v][(r[v].setting == setting) & (r[v].method == m)]
                    x = x[[(t, T) in common for t, T in zip(x.target, x["T"])]]
                    by_seed = x.groupby("seed")[k].mean()
                    cells.append("—" if by_seed.empty else fmt(by_seed.mean(), by_seed.std() if len(by_seed) > 1
                                                                 else np.nan))
            out.append(f"| {LABELS[m]} | " + " | ".join(cells) + " |")
    return out


def per_pair(r, setting):
    r = r[(r.setting == setting)]
    piv = r.groupby(["target", "T", "method"]).p50.mean().unstack("method")
    piv["base_rate"] = r.groupby(["target", "T"]).base_rate.first()
    cols = [m for m in METHODS if m in piv] + ["base_rate"]
    return ["| target | T | " + " | ".join(cols) + " |", "|---|---|" + ":-:|" * len(cols)] + \
        [f"| {t} | {T} | " + " | ".join("—" if np.isnan(v) else f"{v:.2f}" for v in row[cols]) + " |"
         for (t, T), row in piv.iterrows()]


def fig3a(c, setting, path, variant="main"):
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
    run = "" if variant == "main" else ", all-versions run"
    ax.set_title(f"Backtest on cs.IR, cutoffs {min(L.CUTOFFS)}–{max(L.CUTOFFS)}, candidates: {setting}{run}\n"
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
    ap.add_argument("--variant", choices=list(L.VARIANTS), default="main")
    args = ap.parse_args()
    sfx = L.variant_suffix(args.variant)
    r = pd.read_csv(L.RESULTS / f"backtest{sfx}.csv")
    c = pd.read_csv(L.RESULTS / f"backtest_curves{sfx}.csv")
    t = pd.read_csv(L.RESULTS / f"backtest_top{sfx}.csv")
    st = pd.read_csv(L.RESULTS / f"backtest_status{sfx}.csv")
    run = "" if args.variant == "main" else " (all-versions comparison run)"

    tab = [f"# Backtest: precision@k of word2vec (the paper's output·input score) next to random and frequency{run}",
           "", "word2vec: mean ± std over 3 seeds. random: mean of 1,000 random orders of the same candidates. "
           "frequency: candidates sorted by pre-T count. Base rate = share of all candidates found. A pair is scored "
           "only if the concept exists before T (an indicator in ≥ 20 pre-T cs.IR abstracts and in the vocabulary).",
           ""]
    for setting in ["all", "phrases"]:
        tab += main_table(r, setting)
    (L.RESULTS / f"backtest_table{sfx}.md").write_text("\n".join(tab))
    print("\n".join(tab))

    md = [f"# Backtest results: all methods{run}", "",
          "Precision@k = share of the top-k candidates that co-occur with a target indicator in ≥ 3 cs.IR abstracts "
          f"from T to {L.LAST_YEAR}. word2vec rows: mean over scored pairs, ± std across 3 seeds. "
          "\"Frequency-matched\" = expected P@50 if the same top-50 were replaced by random candidates of the same "
          "pre-T frequency decile; the last column is what a method adds beyond frequency.", ""]
    for setting in ["all", "phrases"]:
        md += summary(r, setting)
    md += ["## Status of every (target, T) pair", "",
           "`pre_T_target_docs_ir`: pre-T cs.IR abstracts with an indicator (the validity rule uses this). "
           "`pre_T_target_docs`: the same over all training text (cs.IR ∪ cs.CL).", "",
           st[["target", "T", "status", "pre_T_target_docs_ir", "pre_T_target_docs", "post_T_target_docs",
               "target_tokens",
               "n_candidates_all", "base_rate_all", "n_candidates_phrases", "base_rate_phrases"]]
           .to_markdown(index=False, floatfmt=".3f"), ""]
    for setting in ["all", "phrases"]:
        md += [f"## P@50 per pair, candidates `{setting}` (word2vec: mean of 3 seeds)", ""] + per_pair(r, setting) + [""]
    (L.RESULTS / f"backtest_summary{sfx}.md").write_text("\n".join(md))
    for setting in ["all", "phrases"]:
        fig3a(c, setting, L.FIGURES / f"fig3a_{setting}{sfx}.png", args.variant)

    if all((L.RESULTS / f"backtest{L.variant_suffix(v)}.csv").exists() for v in L.VARIANTS):
        lk = leakage(list(L.VARIANTS))
        (L.RESULTS / "leakage_comparison.md").write_text("\n".join(lk))
        print("\n".join(lk))

    if args.hero and args.variant == "main":
        target, T = args.hero[0], int(args.hero[1])
        h = hero(t, target, T, args.hero_setting)
        (L.RESULTS / "hero_top20.md").write_text("\n".join(h))
        print("\n".join(h))


if __name__ == "__main__":
    main()
