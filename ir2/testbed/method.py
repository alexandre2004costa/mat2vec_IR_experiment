"""Paper 12 (Tshitoyan et al., Nature 2019) on the IR 2 testbed: the paper's embedding pipeline as a retriever.

The paper does not rank documents, so this row is a transfer test of its *representation*: mat2vec-style preprocessing
(numbers -> <nUm>, two passes of Phrases) and its word2vec hyperparameters (skip-gram, negative 15, window 8,
sample 1e-4, 30 epochs), trained on the testbed corpora text only (never on queries or qrels). A document or query is
the SIF-weighted mean of its word vectors (Arora et al. 2017: a / (a + p(w)), a = 1e-3, first principal component
of the document vectors removed); ranking is by cosine.

Usage: ir2/.venv/bin/python ir2/testbed/method.py [--seed 0] [--testbed PATH] [--runs DIR]
Then:  python <testbed>/testbed.py score ir2/testbed/runs --method "mat2vec word2vec + SIF" --pair 12-mat2vec --against bm25
"""
import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ir2lib as L  # noqa: E402


def find_testbed(arg):
    for p in [arg, HERE.parents[1] if (HERE.parents[1] / "testbed.py").exists() else None,
              HERE.parents[2] / "ir2-testbed"]:
        if p and (Path(p) / "testbed.py").exists():
            return Path(p)
    sys.exit("testbed.py not found: pass --testbed /path/to/ir2-testbed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--testbed", default=None)
    ap.add_argument("--runs", default=str(HERE / "runs"))
    ap.add_argument("--dim", type=int, default=100)
    args = ap.parse_args()
    sys.path.insert(0, str(find_testbed(args.testbed)))
    import testbed

    data = {name: testbed.load(name) for name in testbed.DATASETS}
    # one model on the three corpora together (17.5k documents): document text only, the spring 200-word cut
    doc_sents = {name: [L.sentences(t) for t in testbed.doc_texts(corpus)] for name, (corpus, _, _) in data.items()}
    train = [s for ds in doc_sents.values() for d in ds for s in d]
    layers = L.learn_phraser(train)
    phr = [L.apply_phraser(layers, s) for s in train]
    model = Word2Vec(phr, sg=1, negative=15, window=8, sample=1e-4, min_count=5, alpha=0.01, vector_size=args.dim,
                     epochs=30, workers=9, seed=args.seed)
    counts = Counter(t for s in phr for t in s)
    total = sum(counts.values())
    a = 1e-3
    W = model.wv.get_normed_vectors()

    def embed(sents):
        toks = [t for s in sents for t in L.apply_phraser(layers, s)
                if t in model.wv.key_to_index and t not in L.STOPWORDS and t != L.NUM]
        if not toks:
            return np.zeros(W.shape[1], np.float32)
        w = np.array([a / (a + counts[t] / total) for t in toks], np.float32)
        return (w[:, None] * W[[model.wv.key_to_index[t] for t in toks]]).sum(0) / w.sum()

    out = Path(args.runs)
    # the index used by the live demo (ir2/testbed/testbed_demo.py): model, phraser, SIF weights, document vectors
    index = HERE.parent / "models" / f"testbed_s{args.seed}"
    index.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(index / "w2v.kv"))
    pickle.dump({"layers": layers, "counts": counts, "total": total, "a": a}, open(index / "sif.pkl", "wb"))
    for name, (corpus, queries, _) in data.items():         # qrels are never used
        ids, qids = list(corpus), list(queries)
        D = np.stack([embed(s) for s in doc_sents[name]])
        Q = np.stack([embed(L.sentences(queries[q])) for q in qids])
        pc = np.linalg.svd(D, full_matrices=False)[2][0]      # SIF common component (uncentered), from documents
        D, Q = D - np.outer(D @ pc, pc), Q - np.outer(Q @ pc, pc)
        D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
        Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
        S = Q @ D.T
        top = np.argsort(-S, axis=1)[:, :testbed.K + 1]
        run = {q: {ids[j]: float(S[i, j]) for j in top[i]} for i, q in enumerate(qids)}
        testbed.save_run(run, out / f"{name}.json")
        np.savez_compressed(index / f"{name}.npz", D=D.astype(np.float16), pc=pc, ids=np.array(ids))
        empty = int((np.abs(Q).sum(1) == 0).sum())
        print(f"{name}: {len(ids):,} docs, {len(qids):,} queries ({empty} with no in-vocabulary word)", flush=True)


if __name__ == "__main__":
    main()
