"""Phase 3c: the historical backtest. For each cutoff T and target, rank candidate terms with every method and
check which candidates co-occur with the target in >= 3 cs.IR abstracts from T onwards.

Usage: ir2/.venv/bin/python ir2/scripts/phase3_backtest.py [--cutoffs ...] [--targets ...] [--no-specter]
Writes ir2/results/backtest.csv           one row per (setting, target, T, method, seed): P@10/20/50, base rate
       ir2/results/backtest_curves.csv    cumulative fraction of the top 50 found within n years
       ir2/results/backtest_top.csv       top-50 lists (seed 0) with found / first year
       ir2/results/backtest_status.csv    why a (target, T) pair was or was not scored
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

FOUND_MIN = 3          # co-occurrences in distinct cs.IR abstracts from T onwards
KS = (10, 20, 50)
N_RANDOM = 1000
MIN_CANDIDATES = 50
SETTINGS = {"all": lambda terms: np.ones(len(terms), bool),
            "phrases": lambda terms: np.array(["_" in t for t in terms])}


class Cutoff:
    """Everything about one cutoff T that does not depend on the target."""

    def __init__(self, T, use_specter):
        d = L.cutoff_dir(T)
        self.T = T
        self.vocab = pd.read_parquet(d / "vocab.parquet")
        self.terms = self.vocab["term"].values
        self.Xtr = sparse.load_npz(d / "X_train.npz").tocsc().astype(np.float32)
        self.Xpo = sparse.load_npz(d / "X_post.npz").tocsr()
        rows = np.load(d / "rows.npz")
        self.train_rows, self.post_rows = rows["train"], rows["post"]
        docs = L.load_docs()
        self.train_text = docs["text"].values[self.train_rows]
        self.post_text = docs["text"].values[self.post_rows]
        self.post_year = docs["year"].values[self.post_rows]
        self.kvs = {}
        for s in L.SEEDS:
            kv = KeyedVectors.load(str(d / f"w2v_s{s}.kv"))
            syn1 = np.load(d / f"w2v_s{s}.syn1neg.npy")
            idx = np.array([kv.key_to_index[t] for t in self.terms])
            W_in = kv.get_normed_vectors()[idx]
            W_out = syn1[idx] / np.linalg.norm(syn1[idx], axis=1, keepdims=True)
            self.kvs[s] = (kv, W_in, W_out)
        # second-order co-occurrence: norms of each term's abstract-level context-count vector X^T X[:, c]
        XtX_norm = np.zeros(len(self.terms), np.float32)
        Xr = self.Xtr.tocsr()
        for i in range(0, len(self.terms), 2000):
            M = (Xr.T @ self.Xtr[:, i:i + 2000]).toarray()
            XtX_norm[i:i + 2000] = np.linalg.norm(M, axis=0)
        self.Xr, self.ctx_norm = Xr, XtX_norm
        # document-encoder term vectors: mean SPECTER2 embedding of the pre-T abstracts containing the term
        self.E_term = None
        if use_specter:
            emb = np.load(L.DATA / "specter2.npy").astype(np.float32)
            pos = {r: i for i, r in enumerate(np.load(L.DATA / "specter2_rows.npy"))}
            self.E_train = emb[[pos[r] for r in self.train_rows]]
            S = self.Xtr.T @ self.E_train
            self.E_term = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-9)


def found_years(cut, ind_post):
    """For each term: first year in which its count of post-T cs.IR abstracts shared with the target reaches 3
    (0 = never found)."""
    first = np.zeros(len(cut.terms), int)
    cum = np.zeros(len(cut.terms), int)
    for y in range(cut.T, L.LAST_YEAR + 1):
        m = ind_post & (cut.post_year == y)
        if m.any():
            cum += np.asarray(cut.Xpo[m].sum(axis=0)).ravel().astype(int)
        newly = (cum >= FOUND_MIN) & (first == 0)
        first[newly] = y
    return first


def rank_metrics(order, found_first, T):
    """order: candidate positions sorted best first. Returns P@k and the found-within-n-years curve of the top 50."""
    hit = found_first[order] > 0
    out = {f"p{k}": float(hit[:k].mean()) for k in KS}
    top = found_first[order[:50]]
    curve = {n: float(((top > 0) & (top < T + n)).mean()) for n in range(1, L.LAST_YEAR - T + 2)}
    return out, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, nargs="+", default=L.CUTOFFS)
    ap.add_argument("--targets", nargs="+", default=list(L.TARGETS))
    ap.add_argument("--no-specter", action="store_true")
    args = ap.parse_args()
    use_specter = not args.no_specter and (L.DATA / "specter2.npy").exists()
    rng = np.random.default_rng(0)
    rows, curves, tops, status = [], [], [], []

    for T in args.cutoffs:
        cut = Cutoff(T, use_specter)
        print(f"T={T}: {len(cut.terms):,} terms, {len(cut.train_rows):,} train docs, {len(cut.post_rows):,} post docs",
              flush=True)
        for name in args.targets:
            ind = L.TARGETS[name]
            rx = L.indicator_regex(ind)
            ind_tr = np.array([bool(rx.search(t)) for t in cut.train_text])
            ind_po = np.array([bool(rx.search(t)) for t in cut.post_text])
            cooc_pre = np.asarray(cut.Xtr[ind_tr].sum(axis=0)).ravel() > 0 if ind_tr.any() \
                else np.zeros(len(cut.terms), bool)
            cand_all = cut.vocab["candidate"].values & ~cooc_pre
            first = found_years(cut, ind_po)
            _, toks = L.target_vector(cut.kvs[0][0], ind)
            st = {"target": name, "T": T, "pre_T_target_docs": int(ind_tr.sum()),
                  "post_T_target_docs": int(ind_po.sum()), "target_tokens": " ".join(toks),
                  "n_candidates_all": int(cand_all.sum())}

            # scores for all terms; candidate masks applied below
            scores = {}
            if toks:
                for s, (kv, W_in, W_out) in cut.kvs.items():
                    tv, _ = L.target_vector(kv, ind)
                    scores[("w2v_in", s)] = W_in @ tv
                    scores[("w2v_out", s)] = W_out @ tv
            if cut.E_term is not None and ind_tr.any():
                tv = cut.E_train[ind_tr].mean(axis=0)
                scores[("specter2", 0)] = cut.E_term @ (tv / np.linalg.norm(tv))
            scores[("frequency", 0)] = cut.vocab["count"].values.astype(float)
            if ind_tr.any():
                g = cut.Xr.T @ ind_tr.astype(np.float32)                      # target context counts
                num = cut.Xr.T @ (cut.Xr @ g)
                scores[("cooc2", 0)] = num / np.maximum(cut.ctx_norm * np.linalg.norm(g), 1e-9)

            for setting, sel in SETTINGS.items():
                cand = np.flatnonzero(cand_all & sel(cut.terms))
                fc = first[cand]
                base = float((fc > 0).mean()) if len(cand) else float("nan")
                st[f"n_candidates_{setting}"], st[f"base_rate_{setting}"] = len(cand), base
                if setting == "all":
                    st["status"] = ("target not in vocabulary" if not toks else
                                    f"fewer than {MIN_CANDIDATES} candidates" if len(cand) < MIN_CANDIDATES
                                    else "scored")
                if len(cand) < MIN_CANDIDATES:
                    continue
                common = {"setting": setting, "target": name, "T": T, "n_candidates": len(cand), "base_rate": base,
                          "w2v_target_in_vocab": bool(toks)}
                # frequency-matched expectation: found rate within pre-T count deciles of the candidates
                logc = np.log(cut.vocab["count"].values[cand])
                bins = np.searchsorted(np.quantile(logc, np.linspace(0, 1, 11)[1:-1]), logc, side="right")
                rate = np.array([(fc[bins == b] > 0).mean() if (bins == b).any() else 0 for b in range(10)])
                for (method, seed), sc in scores.items():
                    order = np.lexsort((np.arange(len(cand)), -sc[cand]))        # ties -> more frequent first
                    m, curve = rank_metrics(order, fc, T)
                    m.update({f"p{k}_freq_matched": float(rate[bins[order[:k]]].mean()) for k in KS})
                    rows.append({**common, "method": method, "seed": seed, **m})
                    curves += [{**common, "method": method, "seed": seed, "n": n, "frac": f} for n, f in curve.items()]
                    if seed == 0:
                        tops += [{"setting": setting, "target": name, "T": T, "method": method, "rank": r + 1,
                                  "term": cut.terms[cand[j]], "score": float(sc[cand[j]]),
                                  "pre_T_count": int(cut.vocab["count"].values[cand[j]]),
                                  "found": bool(fc[j] > 0), "first_year": int(fc[j]) or None}
                                 for r, j in enumerate(order[:50])]
                # random ranking: mean over 1000 draws; its curve is the expectation (= candidate base rate by year)
                draws = [rank_metrics(rng.permutation(len(cand)), fc, T)[0] for _ in range(N_RANDOM)]
                rows.append({**common, "method": "random", "seed": 0,
                             **{f"p{k}": float(np.mean([d[f"p{k}"] for d in draws])) for k in KS}})
                curves += [{**common, "method": "random", "seed": 0, "n": n,
                            "frac": float(((fc > 0) & (fc < T + n)).mean())} for n in range(1, L.LAST_YEAR - T + 2)]
            status.append(st)
            print(f"  {name:32} {st['status']:28} cand(all) {st['n_candidates_all']:6,} "
                  f"base {st.get('base_rate_all', float('nan')):.3f}  tokens: {st['target_tokens']}", flush=True)

    sfx = "" if args.cutoffs == L.CUTOFFS and args.targets == list(L.TARGETS) else "_partial"
    pd.DataFrame(rows).to_csv(L.RESULTS / f"backtest{sfx}.csv", index=False)
    pd.DataFrame(curves).to_csv(L.RESULTS / f"backtest_curves{sfx}.csv", index=False)
    pd.DataFrame(tops).to_csv(L.RESULTS / f"backtest_top{sfx}.csv", index=False)
    pd.DataFrame(status).to_csv(L.RESULTS / f"backtest_status{sfx}.csv", index=False)


if __name__ == "__main__":
    main()
