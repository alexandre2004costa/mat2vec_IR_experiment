# ir2 decision log

Final pipeline and results for Pair 12 (Tshitoyan et al., Nature 2019). Each entry gives the decision, the reason, and any deviation from the paper.

Scope follows `reading_plan.md` and `test_plan.md`: the historical backtest on arXiv cs.IR (Track 1) and an embedding timeline of the field (Track 2). The testbed row and its live demo (Track 3) are in a separate repository.

## Removed from scope (2026-11-14)

The following were done earlier and deleted from this repository when the plan changed; git history before this date keeps their scripts and tables.

- **Sanity check with the authors' pretrained embeddings:** all three claims held (LiCoO2 neighbours, ferromagnetic − NiFe + IrMn → antiferromagnetic, Bi2Te3 first for "thermoelectric").
- **Thermoelectric backtest on arXiv cond-mat.mtrl-sci:** word2vec output·input P@50 (5 years) 0.220 ± 0.010, 3.3× random (paper: ~8×); frequency 0.340.
- **Testbed row (word2vec + SIF vs BM25 and MiniLM) and its live demo:** moved to the testbed repository.

## Setup

- **Python 3.12 venv in `ir2/.venv`** (Homebrew `python@3.12`). The system Python is 3.15, and gensim 4.4.0 has no wheels past 3.13. The original `mat2vec/` package is untouched; `ir2/` is fresh code.
- **gensim 4.4.0** instead of the paper's 3.x. API renames: `size→vector_size`, `iter→epochs`, `common_terms→connector_words`, `trainables.syn1neg→syn1neg`.
- **Hardware:** MacBook M1 Max, 32 GB RAM, 10 cores. SPECTER2 runs on the MPS GPU; word2vec runs on the CPU (9 threads).
- **Data:** the Kaggle `Cornell-University/arxiv` snapshot (`archive.zip` in the repo root, not in git; an extracted copy `arxiv-metadata-oai-snapshot.json` is also kept locally and gitignored; 5.67 GB JSON, dated 2026-11-08, 3,233,620 records), streamed from inside the zip and never extracted. 2026 is a partial year (about 10 of 12 months).
- **Dating rule: no revised abstracts in pre-T text.**
  - The snapshot holds the abstract of each paper's *latest* version, while the year is taken from its first version.
  - A paper is therefore used as pre-T training text only if **both its first and its latest version are before T**. Otherwise a 2018 paper revised in 2023 would bring 2023 text into a "pre-2019" model (e.g. arXiv 1810.06818 mentions `gpt4ner`).
  - This removes 4.8–6.0% of pre-T abstracts per cutoff. Their original v1 abstracts are not in the snapshot.
  - The "found" check is unaffected: post-T abstracts only decide *when* a hit is counted.
- **word2vec hyperparameters (everywhere):** skip-gram, negative 15, window 8, sample 1e-4, min_count 5, 30 epochs, alpha 0.01 (the mat2vec defaults).
  - **Dimension 100**, because the corpora hold ≤ 9.4M tokens; the paper used 200 on a far larger corpus.
  - **3 seeds per cutoff, reported as mean ± std.** The seed fixes initialization but not thread scheduling, so runs are not bit-reproducible.
- **Known gensim warning:** training prints `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` a few times per run, only with > 1 thread, with both scipy 1.14 and 1.18. At most a handful of updates per run can be affected (out of 10^8–10^9), and the vectors are sane. The lines are filtered from the logs. A caveat, not a fix.

## Backtest on arXiv cs.IR (Track 1) — `scripts/phase1…phase3*.py`, `results/backtest_summary.md`, `figures/fig3a_{all,phrases}.png`

**Corpus** — `results/phase1_counts.md`, `figures/phase1_counts.png`

- **Filter:** keep a paper if `cs.IR` (resp. `cs.CL`) is in its categories; `primary` = listed first.
  - cs.IR: 27,929 papers (16,639 primary).
  - cs.CL: 124,184 papers.
- **cs.IR is small early on** (about 2,200 abstracts before 2015). Training text is therefore **cs.IR ∪ cs.CL** before T. Candidate exclusion uses the same text; "found" uses **cs.IR abstracts only**.

**Preprocessing** — `ir2lib.py`, `results/phase2_cutoffs.md`

- **Text normalization:**
  - NFKD → ASCII; URLs, LaTeX math and LaTeX commands removed; lowercase; numbers → `<nUm>`.
  - Sentences split on `.!?;:`; tokens are runs of `[a-z0-9]`, so hyphens split ("bi-encoder" → `bi encoder`).
- **Phrases:** gensim `Phrases`, 2 passes (up to 4-grams), `min_count=10`, `threshold=15`, gensim's English connector words.
  - Phrases with `<nUm>` are dropped, as in mat2vec.
  - **Deviation:** phrases starting or ending with a stopword or a 1-letter token are also dropped (otherwise `we_propose`, `this_paper`).
- **Leakage control:** one phraser and vocabulary (count ≥ 5) per cutoff, learned on pre-T text only. The same pre-T phraser tokenizes post-T text for the "found" check.
- **Stopwords:** scikit-learn's English list plus `et al eg ie vs etc doi arxiv http https www`. Candidate terms: not a stopword or number, ≥ 3 characters, and phrases do not start or end with a stopword.
- **Indicators are matched on the unphrased text** (whole tokens within a sentence), so detection does not depend on whether the phraser learned a not-yet-existing concept.
- **Sizes:** pre-T training text runs from 5,744 abstracts / 0.85M tokens (T=2016) to 48,486 / 8.2M (T=2023).

**Design**

- **Cutoffs** T = 2016…2023 (each has ≥ 3 future years).
- **Targets and indicators** (`ir2lib.TARGETS`):

  | Target | Indicators |
  |---|---|
  | `query_expansion` | query expansion, query reformulation, query rewriting, relevance feedback |
  | `recommender_systems` | recommender(s), recommendation(s), collaborative filtering |
  | `learning_to_rank` | learning to rank, ranknet, lambdamart, lambdarank, listwise |
  | `dense_retrieval` | dense retrieval, dense retriever(s), dense passage retrieval, dual encoder(s), bi encoder(s) |
  | `conversational_search` | conversational search, conversational recommendation, conversational recommender, clarifying question(s), conversational information seeking |
  | `retrieval_augmented_generation` | retrieval augmented, retrieval enhanced generation, rag |

- **Target vector:** the mean of the normalized input vectors of the indicator tokens in the pre-T vocabulary. If there are none, the pair is "target not in vocabulary": recorded, not scored.
- **Candidates:** pre-T vocabulary terms never in a pre-T abstract with an indicator. Two settings were fixed in advance:
  - `all`: every such term.
  - `phrases`: multiword terms only, the analogue of "materials only".
- **Found:** in ≥ 3 cs.IR abstracts with an indicator from T on (stricter than the paper's ≥ 1, because IR terms are common words). A pair is scored only with ≥ 50 candidates. Ties are broken by frequency.
- **Methods:**
  - `w2v_out`: the paper's score, normalized output vector (`syn1neg`) · target input vector.
  - `w2v_in`: input · input cosine.
  - `frequency`: pre-T count.
  - `random`: 1,000 draws.
  - `cooc2`: the cosine between abstract-level term co-occurrence count vectors.
  - `specter2`: `allenai/specter2_base` without adapters, using the CLS vector of title [SEP] abstract. A term is represented by the mean embedding of its pre-T abstracts. **Caveat:** SPECTER2 was pretrained on papers published after most cutoffs, so it is an upper reference, not a fair backtest.
- **Frequency-matched control (not in the paper):** the expected precision if each top-k term were replaced by a random candidate from the same pre-T frequency decile. `P@k − frequency-matched` measures what a method knows *beyond* frequency.
- Means are over the pairs word2vec can score, so all methods are compared on the same pairs.

**Results**

- 48 (target, T) pairs: **33 scored**. 15 are "not in vocabulary": dense_retrieval T < 2020, conversational_search T < 2021, and **RAG T ≤ 2021**, even though the RAG paper appeared in 2020.
- **Mean P@50, `all` candidates** (base rate 0.149):

  | Method | P@50 | P@50 − frequency-matched |
  |---|:-:|:-:|
  | word2vec output·input (paper) | **0.308 ± 0.007** (2.1× random) | +0.099 ± 0.005 |
  | word2vec input·input | 0.099 (below random) | +0.027 |
  | SPECTER2 | 0.498 | +0.283 |
  | frequency | 0.576 | +0.157 |
  | 2nd-order co-occurrence | **0.785** | +0.375 |

- **`phrases` candidates** (base rate 0.110): word2vec output·input 0.270 ± 0.003 (2.5× random), frequency 0.341, SPECTER2 0.403, cooc2 0.512.
- **Example pair** (`results/hero_top20.md`; picked after seeing all results): learning_to_rank, T=2020, phrases. word2vec finds 7/20 (`personalized_ranking`, `negative_sampling`, `ctr_prediction`), frequency 2/20, random 0.8/20.
- **Failures:**
  - RAG at T=2021: not in the vocabulary.
  - dense_retrieval at T=2020: word2vec 6/10 vs frequency 10/10.
- **Reading:**
  - The effect transfers but weakens. Output·input is ~2× random and above the frequency-matched control.
  - Count baselines beat it everywhere. Plausible reasons: the corpus is small; "found" is a co-occurrence count, which count methods model directly; and IR concepts are generic words, not rare entities.
  - The method cannot predict a concept before it has a word.

## Other caveats

- arXiv category assignments come from the 2026 snapshot (current cross-listings).

## Deliverables

- In `ir2/`: `README.md`, `reading_plan.md` (+ PDF), `test_plan.md`, `NOTES.md`, `reproduce.ipynb` (executed, reads cached artefacts).
- Large artefacts (`ir2/data`, `ir2/models`), `archive.zip` and `arxiv-metadata-oai-snapshot.json` are gitignored.
