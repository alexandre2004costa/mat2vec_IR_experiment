# ir2 decision log

Final pipeline and results for Pair 12 (Tshitoyan et al., Nature 2019). Each entry gives the decision, the reason, and any deviation from the paper.

The project has four parts:

- **A. Sanity check** with the authors' pretrained embeddings.
- **B. Key result:** the thermoelectric backtest (Fig. 3a), small scale, on arXiv materials-science abstracts.
- **C. Transfer to IR:** the same backtest on arXiv cs.IR.
- **D. Testbed row and live demo:** BM25 vs a word2vec retriever.

## Setup

- **Python 3.12 venv in `ir2/.venv`** (Homebrew `python@3.12`). The system Python is 3.15, and gensim 4.4.0 has no wheels past 3.13. The original `mat2vec/` package is untouched; `ir2/` is fresh code.
- **gensim 4.4.0** instead of the paper's 3.x. API renames: `size→vector_size`, `iter→epochs`, `common_terms→connector_words`, `trainables.syn1neg→syn1neg`.
- **Hardware:** MacBook M1 Max, 32 GB RAM, 10 cores. SPECTER2 runs on the MPS GPU; word2vec runs on the CPU (9 threads).
- **Data:** the Kaggle `Cornell-University/arxiv` snapshot (`archive.zip` in the repo root, 5.67 GB JSON, dated 2026-11-08, 3,233,620 records), streamed from inside the zip and never extracted. 2026 is a partial year (about 10 of 12 months).
- **Dating rule (applies to B and C): no revised abstracts in pre-T text.**
  - The snapshot holds the abstract of each paper's *latest* version, while the year is taken from its first version.
  - A paper is therefore used as pre-T training text only if **both its first and its latest version are before T**. Otherwise a 2018 paper revised in 2023 would bring 2023 text into a "pre-2019" model (e.g. arXiv 1810.06818 mentions `gpt4ner`).
  - This removes 4.8–6.0% of pre-T abstracts per cutoff. Their original v1 abstracts are not in the snapshot.
  - The "found" check is unaffected: post-T abstracts only decide *when* a hit is counted.
- **word2vec hyperparameters (everywhere):** skip-gram, negative 15, window 8, sample 1e-4, min_count 5, 30 epochs, alpha 0.01 (the mat2vec defaults).
  - **Dimension 100**, because the corpora hold ≤ 9.4M tokens; the paper used 200 on a far larger corpus.
  - **3 seeds per cutoff, reported as mean ± std.** The seed fixes initialization but not thread scheduling, so runs are not bit-reproducible.
- **Known gensim warning:** training prints `Exception ignored in: 'gensim.models.word2vec_inner.our_dot_float'` a few times per run, only with > 1 thread, with both scipy 1.14 and 1.18. At most a handful of updates per run can be affected (out of 10^8–10^9), and the vectors are sane. The lines are filtered from the logs. A caveat, not a fix.

## A. Sanity check with the pretrained model — `results/phase0_sanity.md`

- The three files (~885 MB) were downloaded from `storage.googleapis.com/mat2vec/` as the README describes. They load in gensim 4.4 without conversion (vocabulary 529,686, dimension 200).
- **Formulas without pymatgen:** a regex stand-in normalizes formulas as mat2vec does, with elements sorted alphabetically (LiCoO2 → CoLiO2). A token is a material if it has ≥ 2 distinct element symbols with optional counts.
- **Results:** all three claims hold.
  - The neighbours of LiCoO2 are other Li-ion cathode materials.
  - ferromagnetic − NiFe + IrMn gives `antiferromagnetic` at rank 1.
  - The output·input ranking for "thermoelectric" puts Bi2Te3 1st, PbTe 4th and CoSb3 8th.
- **No novelty filter here:** the paper's corpus is not distributed, so this checks the embedding, not the prediction.
- **Observation:** output·input favours materials that *appear in the context of* "thermoelectric"; input·input favours materials *used like* known ones (many Bi–Sb–Te stoichiometries).

## B. Key result: thermoelectric backtest — `scripts/te_backtest.py`, `results/te_backtest.md`, `figures/te_fig3a.png`

**Design**

- **Corpus:** arXiv `cond-mat.mtrl-sci`, cross-lists included: 113,695 abstracts, 2,343 of them with thermoelectric/thermoelectrics/thermoelectricity.
- **Tokenization keeps case** so formulas survive:
  - LaTeX subscripts are joined (`Bi$_2$Te$_{3}$` → `Bi2Te3`).
  - Formulas are normalized as in mat2vec (`LiCoO2` → `CoLiO2`).
  - Other words are lowercased, and numbers become `<nUm>`.
  - All-caps acronyms of ≥ 3 one-letter elements without digits are rejected (SCF, BCS). Two-letter ones (BN, CO) are kept, so some false positives remain.
  - No phrase detection: the target and the candidates are single tokens.
- **Cutoffs** 2010, 2012, …, 2020: each has ≥ 5 future years, matching the paper's "within 5 years". Pre-T training text ranges from 2.7M tokens (T=2010) to 9.3M tokens (T=2020).
- **Target vector:** the input vector of the single word `thermoelectric`, as in the paper.
- **Candidates:** formulas seen ≥ 5 times before T and never in a pre-T abstract with a thermoelectric keyword.
- **Found:** in ≥ 1 abstract with a keyword from T on (the paper's criterion).
- **Methods:**
  - `w2v_out`: the paper's score, normalized output vector (`syn1neg`) · target input vector.
  - `w2v_in`: input · input cosine.
  - `frequency`: pre-T count.
  - Co-occurrence count: the cosine between a material's co-occurrence counts with the other materials and the counts of materials in pre-T thermoelectric abstracts.
  - `random`: 1,000 draws.
- **Frequency-matched control (added, not in the paper):** the expected precision if each top-50 material were replaced by a random candidate from the same pre-T frequency decile. `P@50 − frequency-matched` measures what a method knows *beyond* frequency.
- **Metrics:** P@50 within 5 years (headline), P@20, P@50 up to 2026, and Fig. 3a-style curves. No SPECTER2 here, to save compute.

**Results**

- **Per cutoff:** 757–2,456 candidate formulas; 63–405 materials were already thermoelectric before T. The 5-year base rate is 0.075 (T=2010) to 0.051 (T=2020).
- **Mean over the 6 cutoffs:**

  | Method | P@50 (5 y) | × random | P@50 − frequency-matched |
  |---|:-:|:-:|:-:|
  | word2vec output·input (paper) | **0.220 ± 0.010** | **3.3×** | **+0.122 ± 0.006** |
  | word2vec input·input | 0.082 ± 0.008 | 1.2× | +0.047 |
  | co-occurrence count | 0.130 | 2.0× | +0.054 |
  | frequency | **0.340** | **5.1×** | +0.072 |
  | random | 0.067 | 1.0× | — |

- **Per cutoff,** word2vec output·input ranges from 0.087 (T=2010, the smallest corpus) to 0.36 (T=2016, beating frequency's 0.34).
- **Example hits** (T=2016, seed 0, top 20, 11 found by 2026): `HfTe5`, `MnTe`, `Cd3As2`, `NbP`, `LuPtSb`, `LaMnO3`, `Mn5Ge3`. Some entries are tokenization artefacts (`Fe1Ta`) or fragments of doped compositions.
- **Reading:**
  - The paper's key result reproduces qualitatively. Output·input ranks future thermoelectrics well above random, and output vectors matter (input·input is only 1.2×).
  - Of all methods tested, it carries the most information beyond frequency.
  - Quantitatively it is weaker than the paper (3.3× vs ~8×, with a corpus about 30× smaller).
  - A plain frequency ranking, which the paper did not report, has higher raw precision.
- **Fig. 3a caveat:** the bold mean averages only the cutoffs that reach horizon n (≥ 3), so its dip after n = 11 is a change of composition, not a drop.

## C. Transfer to IR: backtest on arXiv cs.IR — `scripts/phase1…phase3*.py`, `results/backtest_summary.md`, `figures/fig3a_{all,phrases}.png`

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
- **Found:** in ≥ 3 cs.IR abstracts with an indicator from T on (stricter than B, because IR terms are common words). A pair is scored only with ≥ 50 candidates. Ties are broken by frequency.
- **Methods:** as in B (`w2v_out`, `w2v_in`, `frequency`, `random`), plus:
  - `cooc2`: the cosine between abstract-level term co-occurrence count vectors.
  - `specter2`: `allenai/specter2_base` without adapters, using the CLS vector of title [SEP] abstract. A term is represented by the mean embedding of its pre-T abstracts. **Caveat:** SPECTER2 was pretrained on papers published after most cutoffs, so it is an upper reference, not a fair backtest.
- The frequency-matched control is as in B. Means are over the pairs word2vec can score, so all methods are compared on the same pairs.

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

## D. Testbed row and live demo — `12-mat2vec/method.py`, `results.json`, `ir2/testbed_demo.py`, `12-mat2vec/demo.md`

- **Row "mat2vec word2vec + SIF".** The paper does not rank documents, so this is a transfer test of its representation:
  - mat2vec-style normalization, two-pass phrases and the word2vec settings above;
  - **trained on the three testbed corpora's document text only** (17,490 documents, the testbed's 200-word cut; never queries or qrels);
  - documents and queries = SIF-weighted mean of word vectors (a = 1e-3, first singular vector of the document matrix removed), cosine ranking.
- **Compared `--against bm25`** (bag-of-words, the closest in spirit); also scored against MiniLM (`results/testbed_seed0_vs_minilm.json`).
- **Results** (`results.json` = seed 0; seeds 1 and 2 in `results/testbed_seed{1,2}.json`). All p < 1e-15.

  | Dataset | nDCG@10 seed 0 / 1 / 2 | BM25 | Δ vs BM25 (seed 0) [95% CI] | MiniLM |
  |---|:-:|:-:|:-:|:-:|
  | SciFact | 0.416 / 0.427 / 0.426 | 0.676 | −0.260 [−0.306, −0.215] | 0.646 |
  | NFCorpus | 0.219 / 0.223 / 0.216 | 0.321 | −0.102 [−0.124, −0.079] | 0.316 |
  | ArguAna | 0.388 / 0.385 / 0.387 | 0.516 | −0.128 [−0.146, −0.110] | 0.503 |

- **Per query (nDCG@10):**
  - SciFact: word2vec wins 29, ties 117 and loses 154 of 300 queries.
  - NFCorpus: it wins 60, ties 98 and loses 165 of 323. 38 NFCorpus queries contain no in-vocabulary word and score 0.
- **Demo:** BM25 (the spring baseline) vs our retriever, side by side, from indexes built beforehand.
  - The BM25 index uses exactly `testbed.py`'s preprocessing and reproduces the official nDCG@10 (SciFact 0.676, NFCorpus 0.321).
  - **Hero:** SciFact 212, "CR …", an abbreviation for caloric restriction. BM25 rank 9 → word2vec rank 1.
  - **Failure:** NFCorpus "kohlrabi", out of vocabulary. Rank 1 → 2,781.
  - **Wildcard backup:** "Eating less slows down biological aging".

## Other caveats

- arXiv category assignments come from the 2026 snapshot (current cross-listings).
- `ir2/demo.py` is an interactive explorer of the cs.IR backtest, kept for making slides; it is not the live demo.

## Deliverables

- `ir2/12-mat2vec/`: `README.md`, `reading_plan.md`, `reproduce.ipynb` (executed, reads cached artefacts), `method.py`, `results.json`, `demo.md`. Copy the folder to `ir2-testbed/pairs/12-mat2vec/` for the pull request, without its `runs*/` folders.
- Large artefacts (`ir2/data`, `ir2/models`) and `archive.zip` are gitignored.
