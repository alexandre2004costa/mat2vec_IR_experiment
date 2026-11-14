# Tests for the testbed row: word2vec + SIF vs BM25 and MiniLM

This file is the step-by-step protocol for placing the paper's representation on the seminar's shared table. The work is done in the testbed repository (`ir2-testbed`, folder `pairs/12-mat2vec/`), not in this one. The main result is in `tests_backtest.md`.

The paper does not rank documents, so this row is a **transfer test of its representation**. It answers one question: how far is a 2013-style averaged word2vec retriever from a lexical baseline (BM25) and a 2021 sentence encoder (MiniLM) on the same corpora?

## 0. Starting point

- **Testbed:** SciFact (5,183 documents, 300 queries), NFCorpus (3,633 / 323) and ArguAna (8,674 / 1,406). Documents are title + abstract, first 200 words. Metrics are nDCG@10 and Recall@100 from `pytrec_eval`.
- **Our code and earlier results:** `ir2/testbed/` (see its README). `method.py` imports preprocessing functions from `ir2/ir2lib.py`, so the new folder needs its own copy of them (step 2).
- **Earlier results** (`ir2/testbed/results*`), for reference only. They are used as a check in step 6, not as targets:

  | Dataset | nDCG@10, seeds 0 / 1 / 2 | BM25 | MiniLM |
  |---|:-:|:-:|:-:|
  | SciFact | 0.416 / 0.427 / 0.426 | 0.676 | 0.646 |
  | NFCorpus | 0.219 / 0.223 / 0.216 | 0.321 | 0.316 |
  | ArguAna | 0.388 / 0.385 / 0.387 | 0.516 | 0.503 |

## 1. Set up the testbed and check the baselines

```
cd ir2-testbed
pip install -r requirements.txt
python testbed.py baselines --systems bm25 minilm
```

**Check:** each line ends with `✓ same as baselines/scores.json`. If one does not, stop: our deltas would be measured against a wrong baseline.

## 2. Create the folder

1. Copy `pairs/_template` to `pairs/12-mat2vec/`.
2. Add `pairs/12-mat2vec/w2vlib.py`: a copy of the parts of `ir2/ir2lib.py` that `method.py` uses (`clean`, `sentences`, `STOPWORDS`, `NUM`, `learn_phraser`, `apply_phraser`). No path into this repository, so the folder runs on its own.
3. Add `runs*/` to `.gitignore`: run files stay on our machine, as the testbed README says.

## 3. The method (`method.py`)

**Training text**
- The document text of the three corpora together (about 17,500 documents), cut to 200 words with `testbed.doc_texts`.
- **Never the queries or `qrels`.** They are the test labels; using them would be leakage.

**Preprocessing,** the same as the backtest (mat2vec style):
- lowercase; URLs and LaTeX removed; numbers → `<nUm>`;
- split into sentences;
- two passes of gensim `Phrases` (`min_count` 10, threshold 15), learned on this training text.

**word2vec,** the paper's settings: skip-gram, negative 15, window 8, sample 1e-4, min_count 5, 30 epochs, alpha 0.01, dimension 100. Seed from `--seed`.

**Document and query vectors:** SIF weighting (Arora et al. 2017).
- Each in-vocabulary token, except stopwords and `<nUm>`, gets the weight a / (a + p(w)), with a = 1e-3 and p(w) its frequency in the training text.
- The vector is the weighted mean of the normalized word vectors.
- The first singular vector of the document matrix is computed per dataset and removed from documents and queries, then everything is L2-normalized.
- A query with no in-vocabulary word gets a zero vector. Its ranking is then arbitrary, and the count of such queries is printed.

**Ranking:** cosine; keep the top 101 per query (`testbed.K + 1`, because on ArguAna the query's own document is removed). Save with `testbed.save_run(run, runs/<dataset>.json)`.

**Command line:** `python pairs/12-mat2vec/method.py --seed S --runs pairs/12-mat2vec/runs_seed{S}`, with `--dim` defaulting to 100.

## 4. Runs

1. **Smoke test:** seed 0 on SciFact only, to check that the run file is written and that `testbed.py score` accepts it.
2. **Full runs:** seeds 0, 1 and 2 on the three datasets. Each seed trains one model of a few minutes on the CPU.

## 5. Scoring

Each seed is scored against both baselines. The official row uses seed 0 against BM25; the others go to named files with `--out`.

```
python testbed.py score pairs/12-mat2vec/runs_seed0 --method "mat2vec word2vec + SIF (paper 12)" --pair 12-mat2vec --against bm25 \
       --out pairs/12-mat2vec/results.json
python testbed.py score pairs/12-mat2vec/runs_seed0 --method "mat2vec word2vec + SIF (paper 12)" --pair 12-mat2vec --against minilm \
       --out pairs/12-mat2vec/results_vs_minilm.json
# seeds 1 and 2: the same two commands, --out results_seed{S}_vs_{bm25,minilm}.json
```

**Why BM25 is the official comparison:** `--against` names the closest baseline. A bag of words, which our method also is, is closer to BM25 than to a trained sentence encoder. MiniLM is the professor's point 4: the gap between a 2013 and a 2021 representation.

## 6. Checks

- **Same baselines:** the `baseline.ndcg10` in our results files equals `baselines/scores.json` for BM25 and MiniLM.
- **No label leakage:** `method.py` never reads `qrels`, and queries are only embedded, never used for training.
- **Seeds:** the three seeds give close but not identical scores. An std of exactly 0 means the seed was not applied.
- **Previous version:** the nDCG@10 of each seed is close to the earlier run in section 0. A large difference means the copied preprocessing differs, so find out why before reporting.
- **Out-of-vocabulary queries:** report the number of queries with a zero vector per dataset. It was 38 of 323 on NFCorpus.

## 7. Outputs

1. **Main table,** one row per dataset. nDCG@10 and Recall@100 for BM25, MiniLM and word2vec + SIF (mean ± std over 3 seeds). Then Δ vs BM25 and Δ vs MiniLM for seed 0, each with its p-value (paired t-test over queries) and 95% bootstrap CI.
2. **Per-query wins, ties and losses** vs BM25 on nDCG@10, seed 0. This shows that the mean hides a few queries where word2vec wins.
3. **Reading rules, fixed before the run:**
   - **A difference counts** only if p < 0.05 and the CI excludes 0. A +0.01 with p = 0.4 is noise.
   - **Losing is the expected result and is reported as is.** The point is the size of the gap to BM25 and to MiniLM, not a win.

## 8. Live demo (also in the testbed repository)

- **Script:** a script that loads the precomputed word2vec index (model, phraser, SIF weights, document vectors) and a BM25 index, then shows both rankings side by side for one query, with ✓ for documents labelled relevant. Nothing is trained live; an answer takes under a second.
- **Index check:** the BM25 index must use `testbed.py`'s own preprocessing; check that it reproduces the official nDCG@10.
- **Three queries in `demo.md`:**
  - **hero:** word2vec ranks a relevant document much higher than BM25. Candidate: SciFact 212, where "CR" means caloric restriction.
  - **failure:** the query word is out of vocabulary. Candidate: NFCorpus "kohlrabi".
  - **wildcard:** typed by the audience. Keep a backup query ready.

  These candidates come from the earlier run; check them again on the new runs before writing `demo.md`.
- **Honest frame:** the hand-picked queries are shown next to the aggregate (wins, ties and losses).
- **Fallback:** record a short video of the demo before the session and link it in `demo.md`.

## 9. Deliverables in `pairs/12-mat2vec/`

`README.md` (at most 10 lines), `method.py`, `w2vlib.py`, `results.json`, `results_vs_minilm.json`, `demo.md`, plus `reproduce.ipynb`, `slides.pdf` and `report.pdf` when the course asks for them. The pull request is due by Wednesday 18:00 before the session.
