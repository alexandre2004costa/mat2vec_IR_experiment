"""Live demo: what would word2vec trained on cs.IR (+ cs.CL) text before year T have predicted for a concept?

  ir2/.venv/bin/python ir2/demo.py --target "learning to rank" --year 2018 [--k 15] [--seed 0] [--phrases]

Loads precomputed artefacts only (nothing is trained). Prints side by side the word2vec ranking
(normalized output vector · target input vector, as in the paper) and the frequency baseline, among terms that never
appeared in an abstract with the target before T, and marks each term ✓ (appears with the target in >= 3 cs.IR
abstracts from T on; year it happened) or ✗. A target may be a predefined name (see --list) or any phrase.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir2lib as L  # noqa: E402

FOUND_MIN = 3
GREEN, RED, DIM, BOLD, END = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def indicators_for(target):
    key = target.strip().lower().replace(" ", "_").replace("-", "_")
    if key in L.TARGETS:
        return key, L.TARGETS[key]
    return target, [target]


class State:
    """Everything for one cutoff, loaded once (interactive mode answers each query from memory)."""

    def __init__(self, T, seed):
        from gensim.models import KeyedVectors
        from scipy import sparse
        d = L.cutoff_dir(T)
        self.T = T
        self.vocab = pd.read_parquet(d / "vocab.parquet")
        self.terms = self.vocab["term"].values
        self.count = self.vocab["count"].values
        self.is_phrase = np.array(["_" in t for t in self.terms])
        rows = np.load(d / "rows.npz")
        self.train, self.post = rows["train"], rows["post"]
        docs = pd.read_parquet(L.DATA / "docs.parquet", columns=["year", "text"])
        self.text, self.year = docs["text"].tolist(), docs["year"].values
        self.post_year = self.year[self.post]
        self.kv = KeyedVectors.load(str(d / f"w2v_s{seed}.kv"))
        syn1 = np.load(d / f"w2v_s{seed}.syn1neg.npy")
        idx = np.array([self.kv.key_to_index[t] for t in self.terms])
        self.out = syn1[idx] / np.linalg.norm(syn1[idx], axis=1, keepdims=True)
        self.Xtr = sparse.load_npz(d / "X_train.npz").tocsr()
        self.Xpo = sparse.load_npz(d / "X_post.npz").tocsr()


def query(S, target, k=15, phrases=False):
    t0 = time.time()
    T = S.T
    name, ind = indicators_for(target)
    rx = L.indicator_regex(ind)
    hit_all = np.fromiter((rx.search(t) is not None for t in S.text), bool, len(S.text))    # one pass, all docs
    ind_tr, ind_po = hit_all[S.train], hit_all[S.post]

    print(f"\n{BOLD}Target{END}: {name}   indicators: {' | '.join(ind)}")
    print(f"{BOLD}Cutoff{END}: models trained on cs.IR ∪ cs.CL abstracts published before {T} "
          f"({len(S.train):,} abstracts)")
    print(f"Abstracts mentioning the target: {ind_tr.sum():,} before {T} · {ind_po.sum():,} cs.IR abstracts "
          f"{T}–{L.LAST_YEAR}")
    if hit_all.any():
        print(f"First abstract with the target in the whole corpus: {S.year[hit_all].min()}")

    tv, toks = L.target_vector(S.kv, ind)
    if tv is None:
        print(f"\n{RED}{BOLD}'{target}' is not in the {T} vocabulary.{END} word2vec only knows words seen ≥5 times "
              f"before {T}: it cannot rank anything for a concept that has no word yet.")
        parts = [t for s in L.sentences(target) for t in s if t in S.kv.key_to_index]
        if parts:
            print(f"{DIM}(Its parts exist separately: {', '.join(parts)}. Averaging them would ask about the parts, "
                  f"not the concept.){END}")
    else:
        print(f"word2vec target tokens: {', '.join(toks)}")

    cooc = np.asarray(S.Xtr[ind_tr].sum(0)).ravel() > 0 if ind_tr.any() else np.zeros(len(S.terms), bool)
    cand = S.vocab["candidate"].values & ~cooc
    if phrases:
        cand &= S.is_phrase
    cand = np.flatnonzero(cand)
    first = np.zeros(len(S.terms), int)
    cum = np.zeros(len(S.terms), int)
    for y in range(T, L.LAST_YEAR + 1):
        m = ind_po & (S.post_year == y)
        if m.any():
            cum += np.asarray(S.Xpo[m].sum(0)).ravel().astype(int)
        first[(cum >= FOUND_MIN) & (first == 0)] = y
    base = (first[cand] > 0).mean() if len(cand) else float("nan")
    print(f"Candidates (never with the target before {T}{', phrases only' if phrases else ''}): {len(cand):,} · "
          f"found later: {100 * base:.1f}% (= random ranking's expected precision)\n")

    def mark(j):
        return f"{GREEN}✓ {first[j]}{END}" if first[j] else f"{RED}✗{END}     "

    freq_order = cand[np.argsort(-S.count[cand], kind="stable")][:k]
    w2v_order = cand[np.argsort(-(S.out[cand] @ tv), kind="stable")][:k] if tv is not None else []
    w = max(28, max((len(S.terms[j]) for j in list(freq_order) + list(w2v_order)), default=0) + 2)
    print(f"{BOLD}{'rank':>4}  {'word2vec (paper scoring)':{w}}{'':8}  {'frequency baseline':{w}}{END}")
    for r in range(k):
        left = f"{S.terms[w2v_order[r]]:{w}}{mark(w2v_order[r])}" if r < len(w2v_order) else f"{'—':{w}}{'':6}"
        right = f"{S.terms[freq_order[r]]:{w}}{mark(freq_order[r])}" if r < len(freq_order) else ""
        print(f"{r + 1:>4}  {left}  {right}")
    hits_w = sum(first[j] > 0 for j in w2v_order)
    hits_f = sum(first[j] > 0 for j in freq_order)
    print(f"\n{BOLD}found{END}: word2vec {hits_w}/{len(w2v_order)} · frequency {hits_f}/{len(freq_order)} · "
          f"random {100 * base:.0f}%   {DIM}({time.time() - t0:.1f}s){END}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", help='e.g. "learning to rank", "query_expansion", "knowledge graph"; '
                                     "omit it for interactive mode (load once, type targets)")
    ap.add_argument("--year", type=int, default=2018, help=f"cutoff T, one of {L.CUTOFFS}")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0, choices=L.SEEDS)
    ap.add_argument("--phrases", action="store_true", help="only multiword candidates")
    ap.add_argument("--list", action="store_true", help="list predefined targets")
    a = ap.parse_args()
    if a.list:
        for k, v in L.TARGETS.items():
            print(f"{k:32} {' | '.join(v)}")
        return
    if a.year not in L.CUTOFFS:
        sys.exit(f"no precomputed models for {a.year}: choose one of {L.CUTOFFS}")
    t0 = time.time()
    S = State(a.year, a.seed)
    if a.target:
        query(S, a.target, a.k, a.phrases)
        return
    print(f"loaded cutoff {a.year} in {time.time() - t0:.1f}s. Type a target (empty line or Ctrl-D to quit; "
          f"'year 2020' to switch cutoff).")
    while True:
        try:
            line = input("target> ").strip()
        except EOFError:
            break
        if not line:
            break
        if line.startswith("year "):
            y = int(line.split()[1])
            if y in L.CUTOFFS:
                S = State(y, a.seed)
                print(f"switched to cutoff {y}")
            else:
                print(f"choose one of {L.CUTOFFS}")
            continue
        query(S, line, a.k, a.phrases)


if __name__ == "__main__":
    main()
