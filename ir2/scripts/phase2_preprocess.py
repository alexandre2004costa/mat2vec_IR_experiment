"""Phase 2: normalize every abstract once, then for each cutoff T learn a phraser on pre-T text only and build
the per-cutoff artefacts used by training, the backtest and the demo.

Usage: ir2/.venv/bin/python ir2/scripts/phase2_preprocess.py [--sample 3000] [--cutoffs 2016 2020] [--jobs 4]

Writes
  ir2/data/docs.parquet                 id, year, is_ir, is_cl, ir_primary, title, abstract, text
                                        (text = normalized unphrased tokens, sentences separated by "\n")
  ir2/models/T{T}/phraser.pkl           list of FrozenPhrases learned on docs with year < T
  ir2/models/T{T}/train.txt             phrased pre-T sentences, one per line (word2vec input)
  ir2/models/T{T}/vocab.parquet         term, count (pre-T), candidate (passes the term filter)
  ir2/models/T{T}/X_train.npz           binary doc x term matrix, pre-T docs (cs.IR ∪ cs.CL)
  ir2/models/T{T}/X_post.npz            binary doc x term matrix, cs.IR docs with year >= T (phrased with the pre-T phraser)
  ir2/models/T{T}/rows.npz              row indices into docs.parquet for X_train and X_post
"""
import argparse
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

MIN_COUNT = 5


def build_docs(sample):
    ir = pd.read_parquet(L.DATA / "csir.parquet")
    cl = pd.read_parquet(L.DATA / "cscl.parquet")
    ir["is_ir"], cl["is_cl"] = True, True
    ir = ir.rename(columns={"primary": "ir_primary"})
    docs = pd.concat([ir, cl[~cl["id"].isin(ir["id"])].drop(columns="primary")], ignore_index=True)
    docs["is_cl"] = docs["categories"].str.split().apply(lambda c: "cs.CL" in c)
    docs["is_ir"] = docs["is_ir"].fillna(False).astype(bool)
    docs["ir_primary"] = docs["ir_primary"].fillna(False).astype(bool)
    if sample:
        docs = docs.sample(n=sample, random_state=0)
    docs = docs.sort_values(["year", "id"]).reset_index(drop=True)
    t0 = time.time()
    with Pool(8) as pool:
        sents = pool.map(L.sentences, [L.doc_text(t, a) for t, a in zip(docs["title"], docs["abstract"])], chunksize=500)
    docs["text"] = ["\n".join(" ".join(s) for s in ss) for ss in sents]
    print(f"normalized {len(docs):,} docs in {time.time() - t0:.0f}s", flush=True)
    return docs[["id", "year", "last_version_year", "is_ir", "is_cl", "ir_primary", "title", "abstract", "categories",
                 "text"]]


def doc_sents(text):
    return [s.split(" ") for s in text.split("\n") if s]


def process_cutoff(args):
    T, out_root = args
    t0 = time.time()
    docs = pd.read_parquet(L.DATA / "docs.parquet", columns=["year", "last_version_year", "is_ir", "text"])
    # pre-T training text: first version AND latest version (whose abstract the snapshot holds) before T
    train_rows = np.flatnonzero((docs["year"].values < T) & (docs["last_version_year"].values < T))
    post_rows = np.flatnonzero((docs["year"].values >= T) & docs["is_ir"].values)
    texts = docs["text"].values
    train_sents = [s for r in train_rows for s in doc_sents(texts[r])]
    layers = L.learn_phraser(train_sents)
    out = out_root / f"T{T}"
    out.mkdir(parents=True, exist_ok=True)
    pickle.dump(layers, open(out / "phraser.pkl", "wb"))

    # phrase pre-T docs, write the word2vec corpus, count
    counts = Counter()
    train_docs = []
    with open(out / "train.txt", "w") as f:
        for r in train_rows:
            ps = [L.apply_phraser(layers, s) for s in doc_sents(texts[r])]
            for s in ps:
                f.write(" ".join(s) + "\n")
                counts.update(s)
            train_docs.append({t for s in ps for t in s})
    vocab = [t for t, c in counts.most_common() if c >= MIN_COUNT and t not in L.STOPWORDS and t != L.NUM]
    index = {t: i for i, t in enumerate(vocab)}
    pd.DataFrame({"term": vocab, "count": [counts[t] for t in vocab],
                  "candidate": [L.is_candidate_term(t) for t in vocab]}).to_parquet(out / "vocab.parquet", index=False)

    def matrix(doc_token_sets):
        indptr, indices = [0], []
        for toks in doc_token_sets:
            indices.extend(sorted(index[t] for t in toks if t in index))
            indptr.append(len(indices))
        return sparse.csr_matrix((np.ones(len(indices), dtype=np.int8), indices, indptr),
                                 shape=(len(doc_token_sets), len(vocab)))

    sparse.save_npz(out / "X_train.npz", matrix(train_docs))
    post_docs = [{t for s in doc_sents(texts[r]) for t in L.apply_phraser(layers, s)} for r in post_rows]
    sparse.save_npz(out / "X_post.npz", matrix(post_docs))
    np.savez(out / "rows.npz", train=train_rows, post=post_rows)
    n_phr = sum(len(p.phrasegrams) for p in layers)
    n_multi = sum("_" in t for t in vocab)
    msg = (f"T={T}: {len(train_rows):,} train docs ({sum(len(s) for s in train_sents):,} tokens), "
           f"{len(post_rows):,} post cs.IR docs, {n_phr:,} phrasegrams, vocab {len(vocab):,} "
           f"({n_multi:,} phrases), {time.time() - t0:.0f}s")
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--cutoffs", type=int, nargs="+", default=L.CUTOFFS)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--skip-docs", action="store_true")
    args = ap.parse_args()
    if not args.skip_docs:
        docs = build_docs(args.sample)
        docs.to_parquet(L.DATA / "docs.parquet", index=False)
    out_root = L.MODELS / "sample" if args.sample else L.MODELS
    with Pool(args.jobs) as pool:
        msgs = pool.map(process_cutoff, [(T, out_root) for T in args.cutoffs])
    if not args.sample:
        (L.RESULTS / "phase2_cutoffs.md").write_text("# Phase 2: per-cutoff preprocessing\n\n" +
                                                     "\n".join(f"- {m}" for m in msgs) + "\n")


if __name__ == "__main__":
    main()
