# ir2 decision log

Final pipeline and results for Pair 12 (Tshitoyan et al., Nature 2019). Each entry gives the decision, the reason, and any deviation from the paper.

Scope follows `reading_plan.md` and `test_plan.md`: the historical backtest on arXiv cs.IR (Track 1) and an embedding timeline of the field (Track 2). The testbed row and its live demo (Track 3) are in `testbed/` (code and results in `testbed/README.md`) and go to the `ir2-testbed` repository as a pull request.

## Removed from scope (2026-11-14)

The following were done earlier and deleted from this repository when the plan changed; git history before this date keeps their scripts and tables.

- **Sanity check with the authors' pretrained embeddings:** all three claims held (LiCoO2 neighbours, ferromagnetic − NiFe + IrMn → antiferromagnetic, Bi2Te3 first for "thermoelectric").
- **Testbed row (word2vec + SIF vs BM25 and MiniLM) and its live demo:** deleted, then restored from git into `testbed/` (paths fixed, own `requirements.txt`). Run files and demo indexes were not in git and must be regenerated.

## Setup

- **Python 3.12 venv in `ir2/.venv`** (Homebrew `python@3.12`). The system Python is 3.15, and gensim 4.4.0 has no wheels past 3.13. The original `mat2vec/` package is untouched; `ir2/` is fresh code.
- **gensim 4.4.0** instead of the paper's 3.x. API renames: `size→vector_size`, `iter→epochs`, `common_terms→connector_words`, `trainables.syn1neg→syn1neg`.
- **Hardware:** MacBook M1 Max, 32 GB RAM, 10 cores. SPECTER2 runs on the MPS GPU; word2vec runs on the CPU (9 threads).
- **Data:** the Kaggle `Cornell-University/arxiv` snapshot (`archive.zip` in the repo root, not in git; an extracted copy `arxiv-metadata-oai-snapshot.json` is also kept locally and gitignored; 5.67 GB JSON, dated 2026-11-08, 3,233,620 records), streamed from inside the zip and never extracted. 2026 is a partial year (about 10 of 12 months).
- **Dating rule: no revised abstracts in pre-T text.**
  - The snapshot holds the abstract of each paper's *latest* version, while the year is taken from its first version.
  - A paper is therefore used as pre-T training text only if **both its first and its latest version are before T**. Otherwise a 2018 paper revised in 2023 would bring 2023 text into a "pre-2019" model (e.g. arXiv 1810.06818 mentions `gpt4ner`).
  - This removes 4.8–6.0% of pre-T abstracts per cutoff. Their original v1 abstracts are not in the snapshot.
  - **Comparison run without the filter (2026-11-14, professor's point 3):** `phase2_preprocess.py --skip-docs --no-version-filter` writes `models/all_versions/T{T}/` and `results/phase2_cutoffs_all_versions.md`; the main run is untouched. It adds back 298 (T=2016) to 2,424 (T=2023) revised abstracts. Step 2 took 58 s for all 8 cutoffs. Training and backtest on it are still to do.
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
- **Validity rule changed (2026-11-14, professor's point 1):** a (target, T) pair is scored only if an indicator occurs in **≥ 20 pre-T cs.IR abstracts** (counted on cs.IR only, after the version filter) **and** an indicator token is in the pre-T vocabulary. Before, only the vocabulary condition applied, counted on cs.IR ∪ cs.CL. Status order: "fewer than 20 pre-T cs.IR abstracts" → "target not in vocabulary" → "fewer than 50 candidates" → "scored". Example: dense_retrieval at T=2020 was scored before (11 abstracts in cs.IR ∪ cs.CL, token `dual_encoder`) but has only 1 pre-T cs.IR abstract, so it is now dropped.
- **Random baseline seeded per pair (2026-11-14):** one generator per (target, T, candidate set) instead of one shared generator, so the 1,000 draws do not depend on which other pairs are run. Checked on query_expansion T=2020: all other methods give identical scores to the earlier run; random moved by ≤ 0.003.
- **Methods:**
  - `w2v_out`: the paper's score, normalized output vector (`syn1neg`) · target input vector.
  - `w2v_in`: input · input cosine.
  - `frequency`: pre-T count.
  - `random`: 1,000 draws.
  - `cooc2`: the cosine between abstract-level term co-occurrence count vectors.
  - `specter2`: `allenai/specter2_base` without adapters, using the CLS vector of title [SEP] abstract. A term is represented by the mean embedding of its pre-T abstracts. **Caveat:** SPECTER2 was pretrained on papers published after most cutoffs, so it is an upper reference, not a fair backtest.
- **Frequency-matched control (not in the paper):** the expected precision if each top-k term were replaced by a random candidate from the same pre-T frequency decile. `P@k − frequency-matched` measures what a method knows *beyond* frequency.
- Means are over the pairs word2vec can score, so all methods are compared on the same pairs.

**Results** (2026-11-14, new validity rule; `results/backtest_table.md`, `results/backtest_summary.md`, `results/leakage_comparison.md`)

- 48 (target, T) pairs: **28 scored**, 20 dropped, all for "fewer than 20 pre-T cs.IR abstracts". Testable: query_expansion and recommender_systems 2016–2023, learning_to_rank 2017–2023, conversational_search 2021–2023, dense_retrieval 2022–2023. **RAG is never testable**: at most 12 pre-T cs.IR abstracts, even at T=2023. (Old rule: 33 scored.)
- **Main table, mean over the 28 pairs** (word2vec ± std over seeds):

  | Candidates | Method | P@10 | P@20 | P@50 |
  |---|---|:-:|:-:|:-:|
  | `all` (base rate 0.137) | word2vec output·input (paper) | 0.252 ± 0.011 | 0.277 ± 0.010 | 0.279 ± 0.006 |
  | | random | 0.138 | 0.137 | 0.137 |
  | | frequency | 0.457 | 0.475 | 0.513 |
  | `phrases` (base rate 0.102) | word2vec output·input (paper) | 0.298 ± 0.027 | 0.280 ± 0.019 | 0.254 ± 0.002 |
  | | random | 0.103 | 0.102 | 0.102 |
  | | frequency | 0.368 | 0.337 | 0.281 |

  Reading rule fixed before the run (`tests_backtest.md` §9): a pair counts only if **all 3 seeds** beat the baseline. word2vec P@50 beats random in **28/28** pairs (both candidate sets) and frequency in **0/28** (`all`) and **7/28** (`phrases`). (Comparing seed means instead: 2/28 and 10/28.)
- **Other methods, P@50 (`all` / `phrases`):** word2vec input·input 0.104 / 0.155 (below random on `all`); SPECTER2 0.467 / 0.362; 2nd-order co-occurrence 0.760 / 0.475.
- **Beyond frequency** (P@50 − frequency-matched): word2vec output·input +0.101 ± 0.005 (`all`) and +0.133 ± 0.002 (`phrases`); frequency +0.160 / +0.076; SPECTER2 +0.283 / +0.248; cooc2 +0.412 / +0.289.
- **Leakage check (professor's point 3):** without the version filter (4.8–6.0% more abstracts), the same 28 pairs are scored. word2vec P@50 is 0.269 ± 0.001 vs 0.279 ± 0.006 (`all`) and 0.239 ± 0.005 vs 0.254 ± 0.002 (`phrases`). The change is larger than one seed std, so by the rule fixed beforehand **the filter does matter, slightly**. But every method, random included, drops by a similar amount (random 0.137 → 0.130), so the base rate moves rather than the method: word2vec stays at ~2.0× random in both runs. Plausible reason: revised abstracts mention later terms with the concept before T, which removes some soon-to-be-found terms from the candidate pool.
- **Example pair** (`results/hero_top20.md`; picked after seeing all results): learning_to_rank, T=2020, phrases. word2vec finds 7/20 (`personalized_ranking`, `negative_sampling`, `ctr_prediction`), frequency 2/20. Some hits are generic phrases (`outperforms_state_of_the_art`).
- **Correctness checks** (`scripts/phase3_checks.py`, `results/backtest_checks.md`), recomputed from the raw text with each cutoff's phraser, not from the backtest's matrices. All pass:
  - no training abstract has a version dated T or later (8 cutoffs);
  - none of 4,787 top-50 terms (word2vec seed 0 and frequency) appears in a pre-T abstract with an indicator;
  - "found" recount: 20/20 rows of the hero list and a random sample of 60 top-50 terms agree (count ≥ 3 and the year of the 3rd abstract);
  - random P@50 is within 0.005 of the base rate in all 56 rows;
  - the seeds differ (word2vec P@50 identical across seeds in only 3/56 rows; mean std 0.027).
- **Failures:** RAG cannot be a target at all (no word before 2023); frequency beats word2vec in most pairs on `all` candidates.
- **Reading:**
  - The effect transfers but weakens: output·input is ~2× random (paper: ~8×) and above the frequency-matched control.
  - Output vectors matter: input·input is at or below random.
  - Count baselines (frequency, co-occurrence) beat it. Plausible reasons: the corpus is small; "found" is a co-occurrence count, which count methods model directly; and IR concepts are generic words, not rare entities.
  - The method cannot predict a concept before it has a word.

## Thermoelectric backtest in the paper's own domain (restored 2026-11-14) — `scripts/te_backtest.py`, `results/te_backtest*.md`, `figures/te_fig3a*.png`

**What the paper compared against** (full text, open-access copy on eScholarship, item 082091b4): 18 historical corpora with cutoffs 2001–2018, one model each; a material counts as studied once it is "reported in the literature alongside a thermoelectric keyword". The top-50 predictions were "on average eight times more likely to have been studied as thermoelectrics within the next five years as compared to a randomly chosen unstudied material from our corpus at that time" and "three times more likely than a random material with a non-zero DFT bandgap". **No frequency or popularity baseline.** Training: 1.5 M relevant abstracts out of 3.3 M collected; 200 dimensions, window 8, 15 negative samples, 30 epochs. Extended Data Fig. 2: "topological insulator" only got its own embedding in 2011, so it could not be predicted earlier (the same failure as RAG in our cs.IR run).

**Setup `small` (first run, restored from git):** arXiv cond-mat.mtrl-sci, cross-lists included: 113,695 abstracts, 2,343 with thermoelectric/thermoelectrics/thermoelectricity. Case-preserving tokenization; formulas normalized as in mat2vec (`LiCoO2` → `CoLiO2`), LaTeX subscripts joined, numbers → `<nUm>`, all-caps acronyms of ≥ 3 one-letter elements rejected; no phrases. Cutoffs 2010, 2012, …, 2020; 100 dimensions; 3 seeds. Candidates: formulas seen ≥ 5 times before T and never in a pre-T abstract with a keyword. Found: ≥ 1 abstract with a keyword from T on (the paper's criterion). Same version filter as the cs.IR run. Result, mean over 6 cutoffs, P@50 within 5 years: word2vec output·input **0.220 ± 0.010 (3.3× random)**, input·input 0.082 (1.2×), co-occurrence 0.130 (2.0×), **frequency 0.340 (5.1×)**, random 0.067; beyond frequency (P@50 − frequency-matched) +0.122 for word2vec vs +0.072 for frequency.

**Setup `paper` (2026-11-14):** the paper's 200 dimensions and yearly cutoffs 2010–2021 (the paper used 2001–2018; arXiv materials science is too thin before 2010: 2.7 M training tokens at T = 2010, 10.6 M at T = 2021), same corpus and criteria, 3 seeds. 36 models, ~2.5 h on the laptop (including an 18-minute idle-sleep pause). `results/te_backtest_paper.md`, `figures/te_fig3a_paper.png`.

- **Mean over 12 cutoffs, P@50 within 5 years:** word2vec output·input **0.228 ± 0.002 (3.5× random)**; input·input 0.093 (1.4×); co-occurrence 0.125 (1.9×); **frequency 0.335 (5.1×)**; random 0.066. Beyond frequency: word2vec +0.120, frequency +0.073.
- **Per cutoff** word2vec ranges from 1.9× random (T = 2011) to **5.2× (T = 2016: 0.367 vs 0.070)** and 5.1× (T = 2021); it beats frequency only at T = 2016 (0.367 vs 0.34) and ties at T = 2020 (0.22).
- **Reading:** matching the paper's dimension and yearly cutoffs barely moves the result (3.3× → 3.5×). The remaining gap to ~8× is most plausibly corpus size (114 k arXiv abstracts vs 1.5 M) and coverage (arXiv is a biased sample of materials science). The paper's 8× was never compared with frequency; on our corpus, frequency reaches 5.1×, above word2vec at 10 of 12 cutoffs (tied at 1).

## Embedding timeline of cs.IR (Track 2) — `scripts/phase4_timeline.py`, `results/timeline.md`, `figures/timeline_specter2.png`

**Design (2026-11-14; subtopic keyword sets approved by Alexandre on 2026-11-14)**

- **Subtopics = keyword sets**, matched like the indicators (whole tokens, unphrased text) on cs.IR abstracts. Chosen over clustering because they can be explained on a slide and match the backtest targets. An abstract can belong to several. Seven sets: recommender systems; query expansion & reformulation (adds *query suggestion*); learning to rank (adds *pairwise ranking*); dense & neural retrieval (adds *neural ranking, neural retrieval, colbert*); conversational search; IR evaluation (*test collection(s), relevance judgments/judgements/assessments, evaluation metrics/measures, offline evaluation, cranfield*); LLMs & RAG (*large language model(s), llm(s), retrieval augmented, rag*). Exact lists in the script.
- **Points:** mean of the cached SPECTER2 vectors of a subtopic's abstracts per first-version year, 2010–2026, only with ≥ 20 abstracts (same threshold as the backtest). 64 points.
- **One PCA on all centroids**, so axes are shared across years: PC1 44.8%, PC2 20.9% of the variance.
- **Similarities are centered:** raw SPECTER2 centroid cosines are all 0.95–0.99 (all abstracts share a common direction), so tables use cosines after subtracting the mean cs.IR abstract vector. The PCA plot is unaffected.
- **Robustness:** Procrustes disparity between the main 2-D layout and refits without 2026 (0.000, 57 common points) and with ≥ 50 abstracts per point (0.002, 47 points): the layout is stable.
- **Colors, checked with the dataviz skill's validator (Node.js installed 2026-11-14):** the overview `figures/timeline_specter2.png` uses the default categorical palette, slots 1–7, with a legend. It passes the adjacent-pair checks but **fails the all-pairs check**, which applies because any two trajectories can sit side by side in this plot. Green (IR evaluation) vs orange (query expansion) has ΔE 3.2 for protanopes; violet (LLMs & RAG) vs blue (recommender systems) has ΔE 10.5 with normal vision. Three colors are also below 3:1 contrast against the surface; the legend and the tables in `timeline.md` provide the required relief. Following the skill's rule (cut series or facet), **`figures/timeline_specter2_panels.png`** is the accessible version: one panel per subtopic, the focal trajectory in one color over the others in grey, identity from the panel title. Use the panels on slides; the overview is a compact summary only.

**Results** (numbers in `results/timeline.md`)

- Coverage: recommender systems has points from 2012, IR evaluation from 2015 (no point in 2016), learning to rank 2017, query expansion 2018, dense retrieval and conversational search 2020, LLMs & RAG 2022 (29 abstracts) → 2,323 in 2026.
- Centered cosine between subtopics rises over time for several pairs: query expansion ↔ dense retrieval 0.49 (2020) → 0.87 (2026); query expansion ↔ LLMs & RAG 0.45 (2022) → 0.78 (2026); learning to rank ↔ dense retrieval 0.24 (2020) → 0.66 (2026); recommender systems ↔ LLMs & RAG −0.47 (2022) → 0.05 (2026).
- Change within a subtopic (centered cosine first ↔ last year): IR evaluation 0.11 and recommender systems 0.43 moved most; conversational search 0.83 and dense retrieval 0.73 least (but over fewer years).
- word2vec (seed 0, Procrustes-aligned consecutive cutoffs): `recommender` moves more than the median word from 2016 to 2021 (self-similarity 0.81–0.86 vs 0.89–0.90), then less; `query_expansion` and `learning_to_rank` stay near the median. Neighbours of `learning_to_rank` shift from boosting terms (`adaboost`, `bagging`, 2016–2017) to ranking-specific ones (`listwise`, `letor`, `unbiased_learning_to_rank`).

## Other caveats

- arXiv category assignments come from the 2026 snapshot (current cross-listings).

## Deliverables

- In `ir2/`: `README.md`, `reading_plan.md` (+ PDF), `test_plan.md`, `NOTES.md`, `reproduce.ipynb` (executed, reads cached artefacts).
- Large artefacts (`ir2/data`, `ir2/models`), `archive.zip` and `arxiv-metadata-oai-snapshot.json` are gitignored.
