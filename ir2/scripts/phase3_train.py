"""Phase 3a: train word2vec on pre-T text, 3 seeds per cutoff, with the paper's hyperparameters.

Usage: ir2/.venv/bin/python ir2/scripts/phase3_train.py [--cutoffs 2016 ...] [--seeds 0 1 2] [--epochs 30]
Writes ir2/models/T{T}/w2v_s{seed}.kv (input vectors) and w2v_s{seed}.syn1neg.npy (output vectors, same row order).
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

PARAMS = dict(sg=1, negative=15, window=8, sample=1e-4, min_count=5, alpha=0.01, vector_size=100, workers=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoffs", type=int, nargs="+", default=L.CUTOFFS)
    ap.add_argument("--seeds", type=int, nargs="+", default=L.SEEDS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.WARNING)
    for T in args.cutoffs:
        d = L.cutoff_dir(T)
        for seed in args.seeds:
            out = d / f"w2v_s{seed}.kv"
            if out.exists() and not args.force:
                continue
            t0 = time.time()
            # iterable input, not corpus_file: gensim 4.4 + scipy 1.18 print "Exception ignored in our_dot_float" in corpus_file mode
            m = Word2Vec(LineSentence(str(d / "train.txt")), epochs=args.epochs, seed=seed, **PARAMS)
            m.wv.save(str(out))
            np.save(d / f"w2v_s{seed}.syn1neg.npy", m.syn1neg)
            print(f"T={T} seed={seed}: vocab {len(m.wv):,}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
