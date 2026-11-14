# Testbed row: the paper's word2vec as a retriever (Pair 12)

The paper (Tshitoyan et al., Nature 2019) does not rank documents, so this row tests its **representation** as a retriever on the seminar's shared testbed:

- the mat2vec-style preprocessing: lowercase, numbers → `<nUm>`, two passes of gensim `Phrases`;
- the paper's word2vec settings: skip-gram, negative 15, window 8, sample 1e-4, 30 epochs, dimension 100;
- training on the testbed corpora's document text only, never on queries or `qrels`;
- documents and queries as SIF-weighted means of word vectors (a = 1e-3, first singular vector removed), ranked by cosine.

It is compared with BM25 (the closest baseline: also a bag of words) and MiniLM (`all-MiniLM-L6-v2`, a 2021 sentence encoder, as the professor asked). These results were computed before the repository cleanup and recovered from git. The protocol is in `../tests_testbed.md`.

## Results

nDCG@10 and Recall@100 from `testbed.py score`. Our row is the mean ± std over seeds 0, 1 and 2 (sample std). BM25 and MiniLM are the testbed's official baselines.

| Dataset | Queries | nDCG@10 ours | BM25 | MiniLM | R@100 ours | BM25 | MiniLM |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| SciFact | 300 | 0.423 ± 0.006 | 0.676 | 0.646 | 0.799 ± 0.006 | 0.918 | 0.925 |
| NFCorpus | 323 | 0.219 ± 0.003 | 0.321 | 0.316 | 0.209 ± 0.003 | 0.246 | 0.311 |
| ArguAna | 1,406 | 0.387 ± 0.002 | 0.516 | 0.503 | 0.913 ± 0.002 | 0.969 | 0.977 |

**Differences in nDCG@10, seed 0.** p from a paired t-test over queries, CI from bootstrap:

| Dataset | vs BM25: Δ [95% CI] | p | vs MiniLM: Δ [95% CI] | p |
|---|:-:|:-:|:-:|:-:|
| SciFact | −0.260 [−0.306, −0.215] | 2e-24 | −0.229 [−0.276, −0.183] | 5e-19 |
| NFCorpus | −0.102 [−0.124, −0.079] | 8e-17 | −0.097 [−0.119, −0.075] | 7e-16 |
| ArguAna | −0.128 [−0.146, −0.110] | 2e-41 | −0.115 [−0.136, −0.094] | 2e-27 |

The other two seeds against BM25 give the same picture: Δ nDCG@10 between −0.098 (NFCorpus) and −0.250 (SciFact), all with p < 1e-14.

**Reading**
- The 2013-style representation loses to both baselines on every dataset and both metrics. Every difference is significant, and no CI comes near 0.
- The gap to MiniLM is about as large as the gap to BM25. BM25 and MiniLM are close to each other (within 0.03 nDCG@10), so averaged word vectors trained on 17.5k documents are far behind both a lexical and a modern dense method.
- **Per query** (nDCG@10, seed 0, vs BM25, as recorded in the earlier notes):
  - SciFact: word2vec wins 29, ties 117 and loses 154 of 300 queries.
  - NFCorpus: it wins 60, ties 98 and loses 165 of 323.
  - 38 NFCorpus queries contain no in-vocabulary word, so they have an empty vector and score 0.
- The demo cases in `demo.md` show where it helps (an abbreviation learned from context: SciFact 212, "CR", BM25 rank 9 → word2vec rank 1) and where it breaks (an out-of-vocabulary word: NFCorpus "kohlrabi", rank 1 → 2,781).

## Files

| File | What it is |
|---|---|
| `method.py` | trains word2vec on the testbed corpora, writes one run per dataset to `runs/` and the demo index to `ir2/models/testbed_s{seed}` |
| `results.json` | official row: seed 0 vs BM25 |
| `results/testbed_seed{1,2}.json` | seeds 1 and 2 vs BM25 |
| `results/testbed_seed0_vs_minilm.json` | seed 0 vs MiniLM |
| `testbed_demo.py` | live demo: BM25 and word2vec side by side (`build`, `find`, `show`) |
| `demo.md` | hero, failure and wildcard queries |
| `requirements.txt` | `../requirements.txt` plus the testbed's own packages |

## How to run

From the repository root, with `ir2-testbed` cloned next to this repository:

```
ir2/.venv/bin/pip install -r ir2/testbed/requirements.txt
ir2/.venv/bin/python ir2/testbed/method.py --seed 0 --runs ir2/testbed/runs     # also --seed 1/2 with runs_seed1/2
ir2/.venv/bin/python ../ir2-testbed/testbed.py score ir2/testbed/runs --method "mat2vec word2vec + SIF (paper 12)" --pair 12-mat2vec --against bm25 --out ir2/testbed/results.json
ir2/.venv/bin/python ir2/testbed/testbed_demo.py build && ir2/.venv/bin/python ir2/testbed/testbed_demo.py show --dataset scifact --query 212
```

## Not available after the cleanup

- **The run files (`runs*/`) and the demo indexes (`ir2/models/testbed_*`) were never in git and are gone.** Rerunning `method.py` (a few minutes per seed) and `testbed_demo.py build` regenerates them. The demo and the per-query counts need them.
- **Seeds 1 and 2 have not been scored against MiniLM.** After the rerun, use `--against minilm --out ...`.
- **For the pull request,** this folder becomes `ir2-testbed/pairs/12-mat2vec/`. It then needs its own copy of the preprocessing functions from `ir2/ir2lib.py`, as described in `../tests_testbed.md` §2.
