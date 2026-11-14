# Test plan (Topic 12, mat2vec on cs.IR)

This plan turns `reading_plan.md`, which already includes the professor's four comments, into concrete tests. The tests form three independent tracks:

| Track | Question | Data | Models |
|---|---|---|---|
| **1. Backtest** (main result) | Does word2vec trained before T rank the terms that later co-occur with an IR concept above random and frequency? | arXiv cs.IR (+ cs.CL as training text) | word2vec per cutoff and seed, SPECTER2 |
| **2. Timeline** | How did cs.IR subtopics move between years? | arXiv cs.IR | SPECTER2 (cached) |
| **3. Testbed** (separate repository) | How far is a 2013-style word-vector retriever from BM25 and MiniLM? | SciFact, NFCorpus, ArguAna | word2vec trained on the testbed text, BM25, MiniLM |

Tracks 1 and 2 share the arXiv corpus and the SPECTER2 vectors. Track 3 shares only the preprocessing code (`ir2lib.py`) and the word2vec hyperparameters with them. No result of one track feeds another, so each can be run, reported or dropped on its own. Track 3 and the live demo, which is built on it, live in the testbed repository; this repository holds Tracks 1 and 2 only.

## Shared settings

- **word2vec:** skip-gram, negative 15, window 8, sample 1e-4, min_count 5, 30 epochs, alpha 0.01, dimension 100. Seeds 0, 1 and 2; every word2vec number is reported as mean ± std over the 3 seeds.
- **Preprocessing:** lowercase, URLs and LaTeX removed, numbers → `<nUm>`, two passes of gensim `Phrases` (up to 4-grams), stopword list. Candidate terms are not stopwords or numbers and have ≥ 3 characters.
- **Fixed in advance:** every threshold below is fixed before results are seen and logged in `ir2/NOTES.md`. Any later change is logged as a deviation, with the reason.

## Track 1: historical backtest on cs.IR

### 1.1 Corpus and leakage filter

- **Date** of a paper = year of its first version.
- **Main run:** a paper is used as pre-T text (training, phrase model, vocabulary, candidate exclusion) only if its **latest version is also dated before T**. The snapshot stores the latest abstract, so this guarantees no post-T text before T.
- **Comparison run:** the same pipeline on all papers, without the version filter. Report both runs and the share of pre-T abstracts the filter removes for each T.
- **Training text:** cs.IR ∪ cs.CL before T. cs.IR alone is too small at early cutoffs (575 papers in 2016). **"Found" is checked on cs.IR abstracts only.**
- **Post-T text** is tokenized with the pre-T phrase model of the same cutoff.

### 1.2 Cutoffs, targets and valid pairs

- **Cutoffs:** T = 2016, …, 2023. Each has at least 3 years of future data (the snapshot runs to late 2026).
- **Targets and indicator sets:** indicators are matched as whole tokens on unphrased text.

  | Target | Indicators |
  |---|---|
  | `query_expansion` | query expansion, query reformulation, query rewriting, relevance feedback |
  | `recommender_systems` | recommender(s), recommendation(s), collaborative filtering |
  | `learning_to_rank` | learning to rank, ranknet, lambdamart, lambdarank, listwise |
  | `dense_retrieval` | dense retrieval, dense retriever(s), dense passage retrieval, dual encoder(s), bi encoder(s) |
  | `conversational_search` | conversational search, conversational recommendation, conversational recommender, clarifying question(s), conversational information seeking |
  | `retrieval_augmented_generation` | retrieval augmented, retrieval enhanced generation, rag |

- **Valid (target, T) pair (professor's point 1):** a pair is tested only if both of these hold:
  1. at least one indicator occurs in **≥ 20 pre-T cs.IR abstracts** (after the version filter);
  2. at least one indicator token is in the pre-T word2vec vocabulary, so the target has a vector.

  Any other pair is listed as *dropped*, with its reason and its abstract count. It is not scored.
- **Expected effect, to be confirmed by the recount:** the existing counts are on cs.IR ∪ cs.CL, so they are an upper bound for cs.IR alone. On them, dense retrieval enters at T ≥ 2021, conversational search at T ≥ 2021 and RAG only at T = 2023. Query expansion, recommender systems and learning to rank are valid at every cutoff.
- **Also required:** a valid pair needs ≥ 50 candidates.

### 1.3 Candidates and "found"

- **Candidates for (target, T):** pre-T vocabulary terms (count ≥ 5) never in a pre-T abstract with an indicator. There are two candidate sets, both fixed in advance:
  - `all`: every candidate term;
  - `phrases`: multiword terms only, the analogue of the paper's "materials only".
- **Found:** the candidate co-occurs with an indicator in **≥ 3 cs.IR abstracts from year T on**. The first year this happens is also recorded.

### 1.4 Methods

| Method | Role | Score |
|---|---|---|
| `w2v_out` | **the paper's method** | normalized output vector (`syn1neg`) of the candidate · target input vector (mean of the indicator vectors) |
| `w2v_in` | ablation | input · input cosine |
| `specter2` | contrast | cosine between the mean SPECTER2 vector of the candidate's pre-T abstracts and the mean vector of the target's pre-T abstracts. **Leakage caveat:** the encoder was pretrained on post-T papers, so it is an upper reference, not a fair backtest. |
| `random` | baseline (point 2) | mean precision over 1,000 random rankings of the same candidate set |
| `frequency` | baseline (point 2) | pre-T count, highest first |
| `cooc2` | optional extra baseline | second-order co-occurrence with the target's abstracts |

Ties are broken by frequency. A frequency-matched control is also reported: the expected precision of random candidates drawn from the same pre-T frequency deciles as the method's top-k.

### 1.5 Outputs

1. **Main table** (`results/backtest_table.md`): one row per valid (target, T) pair, with P@10, P@20 and P@50 for `w2v_out` (mean ± std), `random`, `frequency`, `w2v_in` and `specter2` side by side, plus the base rate. Last row: the mean over valid pairs.
2. **Pair table:** kept and dropped pairs, with the pre-T cs.IR abstract counts and the reason for each drop.
3. **Leakage table:** mean P@k of each method, main run vs all-papers run, and the share of abstracts removed per T.
4. **Fig. 3a-style plot** (`figures/fig3a_*.png`): cumulative % of the top-50 found n years after T, one thin line per cutoff, the mean in bold, and random and frequency as references.
5. **Hero list:** the top-20 for one (target, T), each marked ✓ (with the year found) or ✗, next to the frequency top-20. It is picked after seeing all results, which the slide must say.
6. Raw results in `results/backtest.csv`.

### 1.6 Reading rules, fixed before running

- **"word2vec beats random" on a pair:** its mean P@50 is above the random mean, and the 3-seed range does not overlap it.
- **The method "carries knowledge beyond frequency":** `w2v_out` P@50 is above the frequency-matched control. Raw `w2v_out` vs `frequency` is reported as is, whichever wins.
- **The leakage filter "does not matter":** the mean `w2v_out` P@50 changes by less than one seed std between the main and all-papers runs. Otherwise, report the difference.
- **Dropped pairs are a result, not a gap:** for example, "RAG cannot be a target before 2023 because it has no word".

## Track 2: embedding timeline of cs.IR

- **Subtopics:** 6–7 keyword-defined cs.IR subtopics, for example recommender systems, query processing and expansion, learning to rank, neural and dense retrieval, conversational search, evaluation, and RAG/LLMs. Keyword sets are chosen because they are explainable on a slide and match the backtest targets. They are logged in NOTES.md.
- **Data:** cs.IR abstracts with their cached SPECTER2 vectors (no new encoding). An abstract can belong to several subtopics.
- **Centroids:** one per (subtopic, year), computed only where the subtopic has ≥ 20 abstracts that year, with the same threshold as Track 1.
- **Projection:** one PCA fitted on all centroids of all years, so the axes are shared. One trajectory per subtopic, labelled with years.
- **Outputs:** `figures/timeline_specter2.png` and `results/timeline.md` (what moved, the explained variance of the 2 PCA axes, the number of abstracts per point). Caveat: SPECTER2 was trained on recent papers, the same caveat as in Track 1.
- **Optional:** word2vec neighbours of 2–3 concepts across cutoffs, after orthogonal Procrustes alignment on the shared vocabulary.

## Track 3 and the live demo: in the testbed repository

The plan for them is unchanged: word2vec + SIF trained on the testbed corpora (seeds 0–2), scored `--against bm25` (official `results.json`) and `--against minilm`, with one table of all three systems per dataset, and a BM25 vs word2vec demo with a hero, a failure and a wildcard query. All code and outputs for this track live in the testbed repository.

## Status of existing artefacts

| Item | Status |
|---|---|
| cs.IR / cs.CL corpus with first- and last-version years | done (`data/csir.parquet`, `data/cscl.parquet`) |
| Latest-version-before-T filter | done (main run) |
| All-papers comparison run | **to do**: a flag in phase 2/3 scripts, 24 more models (about 45 min) |
| Valid-pair rule (≥ 20 pre-T cs.IR abstracts + in vocabulary) | **to do**: currently "in vocabulary" only, counted on cs.IR ∪ cs.CL |
| Random, frequency, SPECTER2, `w2v_in`, `cooc2` scores | done; re-tabulate in the new main-table layout |
| Fig. 3a, hero list | done; regenerate on valid pairs |
| Timeline (Track 2) | **to do** |
| Testbed row and demo | moved to the testbed repository |
| Thermoelectric backtest, Phase 0 sanity check | removed (summarized in `ir2/NOTES.md`) |

## Order of work

1. Track 1: valid-pair recount → main run tables → all-papers run → leakage table → figures.
2. Track 2: timeline.
3. Update `NOTES.md`, `reproduce.ipynb` and the README.
