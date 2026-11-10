"""Live demo on the shared testbed: the spring baseline (BM25) next to the paper's representation (word2vec trained
with the mat2vec recipe, SIF-averaged), same query, same corpus. Indexes are built beforehand; nothing is trained
or indexed live.

  ir2/.venv/bin/python ir2/testbed_demo.py build                       # once: BM25 indexes (method.py saves ours)
  ir2/.venv/bin/python ir2/testbed_demo.py find --dataset scifact      # per-query wins/losses, to pick hero/failure
  ir2/.venv/bin/python ir2/testbed_demo.py show --dataset scifact --query 1234        # a test query id: ✓ = qrels
  ir2/.venv/bin/python ir2/testbed_demo.py show --dataset scifact --query "free text" # wildcard (no labels)
  ir2/.venv/bin/python ir2/testbed_demo.py show --dataset scifact                     # interactive
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ir2lib as L  # noqa: E402

TESTBED = HERE.parents[1] / "ir2-testbed"
sys.path.insert(0, str(TESTBED))
import testbed  # noqa: E402

BM25_DIR = HERE / "models" / "testbed_bm25"
OURS_DIR = HERE / "models" / "testbed_s0"
GREEN, RED, DIM, BOLD, END = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def bm25_preprocessor():
    """Exactly testbed.bm25_search's preprocessing."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import CountVectorizer
    nltk.download("stopwords", quiet=True)
    sw, porter, analyze = set(stopwords.words("english")), PorterStemmer(), CountVectorizer().build_analyzer()
    return lambda t: [porter.stem(x) for x in analyze(t) if x not in sw]


def cmd_build(args):
    from rank_bm25 import BM25Okapi
    pre = bm25_preprocessor()
    BM25_DIR.mkdir(parents=True, exist_ok=True)
    for name in testbed.DATASETS:
        corpus, _, _ = testbed.load(name)
        bm25 = BM25Okapi([pre(t) for t in testbed.doc_texts(corpus)], k1=1.2, b=0.75)
        pickle.dump({"bm25": bm25, "ids": list(corpus)}, open(BM25_DIR / f"{name}.pkl", "wb"))
        print(f"{name}: BM25 index over {len(corpus):,} documents")


class Systems:
    def __init__(self, name):
        from gensim.models import KeyedVectors
        self.name = name
        self.corpus, self.queries, self.qrels = testbed.load(name)
        b = pickle.load(open(BM25_DIR / f"{name}.pkl", "rb"))
        self.bm25, self.ids = b["bm25"], b["ids"]
        self.pre = bm25_preprocessor()
        o = np.load(OURS_DIR / f"{name}.npz")
        assert list(o["ids"]) == self.ids
        self.D, self.pc = o["D"].astype(np.float32), o["pc"]
        self.kv = KeyedVectors.load(str(OURS_DIR / "w2v.kv"))
        self.W = self.kv.get_normed_vectors()
        self.sif = pickle.load(open(OURS_DIR / "sif.pkl", "rb"))

    def ours(self, text):
        s = self.sif
        toks = [t for sent in L.sentences(text) for t in L.apply_phraser(s["layers"], sent)
                if t in self.kv.key_to_index and t not in L.STOPWORDS and t != L.NUM]
        if not toks:
            return np.zeros(len(self.ids)), toks
        w = np.array([s["a"] / (s["a"] + s["counts"][t] / s["total"]) for t in toks], np.float32)
        q = (w[:, None] * self.W[[self.kv.key_to_index[t] for t in toks]]).sum(0) / w.sum()
        q = q - (q @ self.pc) * self.pc
        return self.D @ (q / max(np.linalg.norm(q), 1e-9)), toks

    def bm(self, text):
        toks = self.pre(text)
        return np.asarray(self.bm25.get_scores(toks)), toks


def title(corpus, d, n=58):
    t = corpus[d]["title"] or corpus[d]["text"]
    t = " ".join(t.split())
    return t if len(t) <= n else t[:n - 1] + "…"


def show(S, query, k):
    t0 = time.time()
    qid = query if query in S.queries else None
    text = S.queries[qid] if qid else query
    rel = {d for d, g in S.qrels.get(qid, {}).items() if g > 0} if qid else set()
    sb, tb = S.bm(text)
    so, to = S.ours(text)
    ids = np.array(S.ids)
    if S.name == "arguana" and qid:                # the query is itself a document (removed, as in scoring)
        sb[ids == qid], so[ids == qid] = -np.inf, -np.inf
    ob, oo = np.argsort(-sb, kind="stable"), np.argsort(-so, kind="stable")
    print(f"\n{BOLD}{S.name}{END} query{f' {qid}' if qid else ' (typed live, no relevance labels)'}: {text}")
    print(f"{DIM}BM25 terms: {' '.join(tb) or '—'}{END}")
    print(f"{DIM}word2vec terms: {' '.join(to) or '— (no word of the query is in the vocabulary)'}{END}\n")
    w = 62
    print(f"{BOLD}{'rank':>4}  {'BM25 (spring baseline)':{w}}  {'word2vec + SIF (paper 12)':{w}}{END}")
    mark = lambda d: f"{GREEN}✓{END}" if d in rel else (" " if not rel else f"{DIM}·{END}")
    for r in range(k):
        a, b = S.ids[ob[r]], S.ids[oo[r]]
        print(f"{r + 1:>4}  {mark(a)} {title(S.corpus, a):{w - 2}}  {mark(b)} {title(S.corpus, b):{w - 2}}")
    if rel:
        rb = {S.ids[j]: i + 1 for i, j in enumerate(ob)}
        ro = {S.ids[j]: i + 1 for i, j in enumerate(oo)}
        print(f"\n{BOLD}relevant documents{END} (rank BM25 → rank word2vec):")
        for d in sorted(rel, key=lambda d: ro[d]):
            col = GREEN if ro[d] < rb[d] else RED if ro[d] > rb[d] else ""
            print(f"  {col}{rb[d]:>5} → {ro[d]:<5}{END} {title(S.corpus, d, 80)}")
    print(f"{DIM}({time.time() - t0:.2f}s){END}\n")


def cmd_show(args):
    S = Systems(args.dataset)
    if args.query:
        show(S, args.query, args.k)
        return
    print(f"{args.dataset} loaded. Type a query id or free text (empty line to quit).")
    while True:
        try:
            line = input("query> ").strip()
        except EOFError:
            break
        if not line:
            break
        show(S, line, args.k)


def cmd_find(args):
    """Per-query nDCG@10 of both systems (the official per-query numbers), sorted by difference."""
    name = args.dataset
    corpus, queries, qrels = testbed.load(name)
    qids, nd, _ = testbed.score_run(json.load(open(HERE / "12-mat2vec" / "runs" / f"{name}.json")), name)
    base = testbed.stored()[name]
    assert base["qids"] == qids
    bm = np.array(base["bm25"]["ndcg10"])
    pre = bm25_preprocessor()
    rows = []
    for q, a, b in zip(qids, nd, bm):
        rel = [d for d, g in qrels[q].items() if g > 0]
        qt = set(pre(queries[q]))
        overlap = np.mean([len(qt & set(pre(testbed.doc_texts({d: corpus[d]})[0]))) / max(len(qt), 1) for d in rel if d in corpus] or [0])
        rows.append((q, a, b, a - b, overlap, queries[q]))
    rows.sort(key=lambda r: -r[3])
    print(f"{name}: word2vec wins {sum(r[3] > 0 for r in rows)}, ties {sum(r[3] == 0 for r in rows)}, "
          f"loses {sum(r[3] < 0 for r in rows)} of {len(rows)} queries (nDCG@10)\n")
    print("Largest wins (ours − BM25), with query-term overlap with the relevant documents:")
    for q, a, b, dlt, ov, t in rows[:args.n]:
        print(f"  {q:>10}  ours {a:.2f}  bm25 {b:.2f}  overlap {ov:.2f}  {t[:110]}")
    print("\nLargest losses:")
    for q, a, b, dlt, ov, t in rows[-args.n:][::-1]:
        print(f"  {q:>10}  ours {a:.2f}  bm25 {b:.2f}  overlap {ov:.2f}  {t[:110]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "find", "show"])
    ap.add_argument("--dataset", default="scifact", choices=testbed.DATASETS)
    ap.add_argument("--query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    {"build": cmd_build, "find": cmd_find, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    main()
