# Test plan (Topic 12, mat2vec on cs.IR): overview

This page turns `reading_plan.md`, which includes the professor's four comments, into tests. The detailed protocols are in two files:

- **`tests_backtest.md`:** the main result (the historical backtest on arXiv cs.IR) and the embedding timeline. Done in this repository.
- **`tests_testbed.md`:** the testbed row (word2vec + SIF vs BM25 and MiniLM) and the live demo. Code and earlier results in `testbed/`; the pull request goes to the `ir2-testbed` repository.

| Track | Question | Data | Where |
|---|---|---|---|
| **1. Backtest** (main result) | Does word2vec trained before T rank the terms that later co-occur with an IR concept above random and frequency? | arXiv cs.IR (+ cs.CL as training text) | `tests_backtest.md`, Part A |
| **2. Timeline** | How did cs.IR subtopics move between years? | arXiv cs.IR, SPECTER2 vectors | `tests_backtest.md`, Part B |
| **3. Testbed** | How far is a 2013-style word-vector retriever from BM25 and MiniLM? | SciFact, NFCorpus, ArguAna | `tests_testbed.md` |

The tracks are independent: tracks 1 and 2 share the arXiv corpus, while track 3 shares only the preprocessing and the word2vec settings.

**How the professor's comments map to the tests:**

| Comment | Where |
|---|---|
| 1. Only concepts that exist before T | `tests_backtest.md` §2: an indicator in ≥ 20 pre-T cs.IR abstracts |
| 2. Random and frequency next to word2vec | `tests_backtest.md` §7: main table |
| 3. Abstract-version leakage | `tests_backtest.md` §1 and §3: latest version before T, plus an all-versions comparison run |
| 4. MiniLM next to BM25 | `tests_testbed.md` §5 and §7 |

**Order of work:**
1. Backtest: validity rule → main run → all-versions run → report.
2. Timeline.
3. Testbed: rerun `testbed/method.py`, score seeds 1–2 against MiniLM, then the pull request.
4. Update `NOTES.md`, `reproduce.ipynb` and `README.md`.
