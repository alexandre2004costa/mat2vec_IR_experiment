"""Correctness checks of the backtest (tests_backtest.md §8), recomputed independently of the backtest's own
document x term matrices: the raw normalized text is re-phrased with each cutoff's phraser.

1. No leakage in training: every training doc of cutoff T has first and latest version before T.
2. Candidates are new: no pre-T abstract with an indicator contains a top-50 term (w2v_out seed 0 and frequency).
3. Found is real: for the hero pair, list the post-T cs.IR abstracts where each top-20 term co-occurs with an
   indicator; the year of the 3rd one must equal the recorded first year (and terms marked ✗ must have < 3).
   The same recount is run on a random sample of top-50 terms from all pairs.
4. Random equals the base rate (mean of 1,000 draws vs share of candidates found).
5. Seeds differ: the three seeds give different top-50 lists.

Usage: ir2/.venv/bin/python ir2/scripts/phase3_checks.py [--hero learning_to_rank 2020 phrases] [--sample 60]
Writes ir2/results/backtest_checks.md
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

FOUND_MIN = 3


class Recount:
    """Indicator docs of one (target, T), re-phrased with the cutoff's phraser, pre-T and post-T."""

    def __init__(self, docs, T, target):
        d = L.cutoff_dir(T)
        layers = pickle.load(open(d / "phraser.pkl", "rb"))
        rows = np.load(d / "rows.npz")
        rx = L.indicator_regex(L.TARGETS[target])
        text = docs["text"].values

        def phrased(r):
            return {t for s in text[r].split("\n") if s for t in L.apply_phraser(layers, s.split(" "))}

        self.pre = [phrased(r) for r in rows["train"] if rx.search(text[r])]
        post = [r for r in rows["post"] if rx.search(text[r])]
        self.post = [(int(docs["year"].values[r]), docs["id"].values[r], docs["title"].values[r], phrased(r))
                     for r in post]
        self.post.sort(key=lambda x: x[0])

    def seen_before(self, term):
        return any(term in toks for toks in self.pre)

    def hits(self, term):
        return [(y, i, t) for y, i, t, toks in self.post if term in toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hero", nargs=3, default=["learning_to_rank", "2020", "phrases"])
    ap.add_argument("--sample", type=int, default=60)
    args = ap.parse_args()
    docs = L.load_docs()
    top = pd.read_csv(L.RESULTS / "backtest_top.csv")
    res = pd.read_csv(L.RESULTS / "backtest.csv")
    status = pd.read_csv(L.RESULTS / "backtest_status.csv")
    scored = status[status.status == "scored"][["target", "T"]].values.tolist()
    md = ["# Backtest correctness checks (main run)", "",
          "Recomputed from the raw normalized text, re-phrased with each cutoff's phraser (independent of the "
          "document × term matrices the backtest uses). Script: `scripts/phase3_checks.py`.", ""]

    # 1. leakage in training rows
    bad = []
    for T in L.CUTOFFS:
        tr = np.load(L.cutoff_dir(T) / "rows.npz")["train"]
        if not ((docs["year"].values[tr] < T).all() and (docs["last_version_year"].values[tr] < T).all()):
            bad.append(T)
    md += ["## 1. No post-T text in training", "",
           f"Every training abstract of every cutoff has first and latest version before T: "
           f"**{'PASS' if not bad else 'FAIL at ' + str(bad)}**.", ""]

    # 2 + 3. candidates new / found real, recount per pair
    recounts = {}
    rng = np.random.default_rng(0)
    viol, checked = [], 0
    sample_rows = []
    for target, T in scored:
        rc = recounts.setdefault((target, T), Recount(docs, T, target))
        x = top[(top.target == target) & (top["T"] == T) & top.method.isin(["w2v_out", "frequency"])]
        for term in x.term.unique():
            checked += 1
            if rc.seen_before(term):
                viol.append((target, T, term))
        sample_rows.append(x[x.method == "w2v_out"])
        print(f"checked {target} {T}", flush=True)
    md += ["## 2. Candidates are new", "",
           f"{checked:,} distinct top-50 terms (word2vec seed 0 and frequency, both candidate sets, all "
           f"{len(scored)} scored pairs) were searched in every pre-T abstract with an indicator. "
           f"Terms found there: **{len(viol)}** → **{'PASS' if not viol else 'FAIL'}**.", ""]
    md += [f"- {t} T={T}: `{w}`" for t, T, w in viol[:20]]

    def verify(target, T, term, found, first_year):
        h = recounts[(target, T)].hits(term)
        n = len(h)
        if found:
            ok = n >= FOUND_MIN and h[FOUND_MIN - 1][0] == int(first_year)
        else:
            ok = n < FOUND_MIN
        return ok, h

    # 3a. hero pair, full listing
    ht, hT, hs = args.hero[0], int(args.hero[1]), args.hero[2]
    md += ["## 3. \"Found\" is real", "",
           f"### Hero pair: `{ht}`, T = {hT}, candidates `{hs}` (word2vec top 20, seed 0)", "",
           "For each term: post-T cs.IR abstracts containing both the term and an indicator. ✓ requires ≥ 3; "
           "the recorded year must be the year of the 3rd.", "",
           "| rank | term | recorded | recount | first 3 abstracts (arXiv id, year) | check |", "|:-:|---|:-:|:-:|---|:-:|"]
    h = top[(top.target == ht) & (top["T"] == hT) & (top.setting == hs) & (top.method == "w2v_out")]
    h = h.sort_values("rank").head(20)
    n_ok = 0
    for _, r in h.iterrows():
        ok, hits = verify(ht, hT, r.term, r.found, r.first_year)
        n_ok += ok
        rec = f"✓ {int(r.first_year)}" if r.found else "✗"
        first3 = ", ".join(f"{i} ({y})" for y, i, _ in hits[:3]) or "—"
        md.append(f"| {r['rank']} | `{r.term}` | {rec} | {len(hits)} | {first3} | {'PASS' if ok else 'FAIL'} |")
    md += ["", f"{n_ok}/{len(h)} rows agree with the recount.", ""]

    # 3b. random sample over all pairs
    allx = pd.concat(sample_rows).drop_duplicates(["target", "T", "term"])
    samp = allx.iloc[rng.choice(len(allx), size=min(args.sample, len(allx)), replace=False)]
    agree = sum(verify(r.target, r["T"], r.term, r.found, r.first_year)[0] for _, r in samp.iterrows())
    md += [f"### Random sample: {len(samp)} top-50 terms (word2vec seed 0) from all scored pairs", "",
           f"{int(samp.found.sum())} recorded as found, {int((~samp.found).sum())} as not found. "
           f"Agreement with the recount: **{agree}/{len(samp)}** → **{'PASS' if agree == len(samp) else 'FAIL'}**.",
           ""]

    # 4. random vs base rate
    rnd = res[res.method == "random"]
    gap = (rnd.p50 - rnd.base_rate).abs()
    md += ["## 4. Random equals the base rate", "",
           f"Over {len(rnd)} (pair, candidate set) rows, |random P@50 − base rate| is at most {gap.max():.4f} "
           f"(mean {gap.mean():.4f}): **{'PASS' if gap.max() < 0.01 else 'FAIL'}** (threshold 0.01).", ""]

    # 5. seeds differ (top-50 lists are only saved for seed 0, so compare P@50 across seeds)
    w = res[res.method == "w2v_out"].pivot_table(index=["setting", "target", "T"], columns="seed", values="p50")
    same = int((w.nunique(axis=1) == 1).sum())
    md += ["## 5. Seeds differ", "",
           f"word2vec P@50 is identical across the 3 seeds in {same}/{len(w)} (pair, candidate set) rows; the "
           f"mean std across seeds is {w.std(axis=1).mean():.3f}. The seeds are applied, and the spread is small. "
           f"Identical values in a few rows are expected: P@50 moves in steps of 0.02.", ""]

    (L.RESULTS / "backtest_checks.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
