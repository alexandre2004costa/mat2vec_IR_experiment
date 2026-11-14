# Tests for the main result: historical backtest on arXiv cs.IR

This file is the step-by-step protocol for the paper's key result at small scale: does word2vec trained on abstracts before year T rank the terms that later co-occur with an IR concept above a random and a frequency ranking? A second part covers the embedding timeline, which reuses the same corpus. The testbed is in `tests_testbed.md`. Every decision and deviation goes into `NOTES.md`.

All commands run from the repository root with `ir2/.venv/bin/python` (Python 3.12, gensim 4.4.0).

## 0. What exists and what has to change

| Step | Script | State |
|---|---|---|
| Corpus | `scripts/phase1_corpus.py` | done: `data/csir.parquet` (27,929 papers), `data/cscl.parquet` (124,184) |
| Preprocessing per cutoff | `scripts/phase2_preprocess.py` | done for the main run; **add** an option to skip the version filter (step 3) |
| word2vec training | `scripts/phase3_train.py` | done: `models/T2016` … `models/T2023`, seeds 0–2 |
| SPECTER2 encoding | `scripts/phase3_encode.py` | done: `data/specter2.npy` |
| Backtest | `scripts/phase3_backtest.py` | done with the old validity rule; **change** it to the new rule (step 2) |
| Report | `scripts/phase3_report.py` | done; **change** the table layout (step 5) |
| Timeline | none | **to write**: `scripts/phase4_timeline.py` (Part B) |

The existing results in `results/backtest*.csv` used the old rule ("target in vocabulary", counted on cs.IR ∪ cs.CL). They are replaced, not edited, by the new runs.

## Part A: the backtest

### 1. Fixed settings, logged before any new result is seen

| Setting | Value | Why |
|---|---|---|
| Cutoffs T | 2016, 2017, …, 2023 | each has at least 3 future years in the snapshot (to 2026, a partial year of about 10 months) |
| Date of a paper | year of its first version | as in the reading plan |
| Training text | cs.IR ∪ cs.CL abstracts before T | cs.IR alone has 575 papers in 2016 |
| Version filter (main run) | a pre-T paper is kept only if its latest version is also before T | the snapshot stores the latest abstract (professor's point 3) |
| Phrase model and vocabulary | learned per cutoff on pre-T text only; post-T text is tokenized with the same pre-T phrase model | no leakage through tokenization |
| Vocabulary | count ≥ 5 before T; not a stopword or `<nUm>` | as the paper |
| Candidate terms | ≥ 3 characters, not a stopword or number; phrases do not start or end with a stopword | course instructions |
| word2vec | skip-gram, negative 15, window 8, sample 1e-4, min_count 5, 30 epochs, alpha 0.01, dimension 100 | the paper's settings; 100 dimensions because the corpus is small |
| Seeds | 0, 1, 2 | report mean ± std |
| Valid pair | an indicator occurs in ≥ 20 pre-T **cs.IR** abstracts, and at least one indicator token is in the pre-T vocabulary | professor's point 1 |
| Minimum candidates | 50 per pair | below this, P@50 is meaningless |
| Found | the candidate co-occurs with an indicator in ≥ 3 cs.IR abstracts from year T on | IR terms are common words, so 1 co-occurrence is too easy |
| Candidate sets | `all` (every candidate) and `phrases` (multiword terms only) | `phrases` is the analogue of "materials only" |

Targets and indicators are in `ir2lib.TARGETS`:

| Target | Indicators |
|---|---|
| `query_expansion` | query expansion, query reformulation, query rewriting, relevance feedback |
| `recommender_systems` | recommender(s), recommendation(s), collaborative filtering |
| `learning_to_rank` | learning to rank, ranknet, lambdamart, lambdarank, listwise |
| `dense_retrieval` | dense retrieval, dense retriever(s), dense passage retrieval, dual encoder(s), bi encoder(s) |
| `conversational_search` | conversational search, conversational recommendation, conversational recommender, clarifying question(s), conversational information seeking |
| `retrieval_augmented_generation` | retrieval augmented, retrieval enhanced generation, rag |

### 2. Change the validity rule (`phase3_backtest.py`)

1. Add `MIN_TARGET_DOCS = 20` next to `FOUND_MIN`.
2. Load `is_ir` with the docs and count the pre-T abstracts with an indicator **among cs.IR rows only**: `pre_T_target_docs_ir = (ind_tr & is_ir[train_rows]).sum()`. Keep the cs.IR ∪ cs.CL count as a second column for reference.
3. Set `status` in this order: `fewer than 20 pre-T cs.IR abstracts`, then `target not in vocabulary`, then `fewer than 50 candidates`, else `scored`. Only `scored` pairs produce rows in `backtest.csv`.
4. Keep everything else unchanged: candidate exclusion still uses all pre-T training text (cs.IR ∪ cs.CL), because a term seen with the concept in a cs.CL abstract is not a new discovery.

**Check:** `results/backtest_status.csv` lists all 48 pairs (6 targets × 8 cutoffs), each with its status and both counts. The existing counts (cs.IR ∪ cs.CL) are an upper bound for the cs.IR count, so expect dense retrieval to start at T ≥ 2021 at the earliest and RAG at T = 2023 at the earliest.

### 3. Add the all-papers comparison run

1. `phase2_preprocess.py`: add `--no-version-filter`. With it, `train_rows` is `year < T` only, and the output goes to `models/all_versions/T{T}/`, so the main run is never overwritten. Log the number of pre-T abstracts with and without the filter for each T, and the share removed.
2. `phase3_train.py` and `phase3_backtest.py`: add `--variant {main,all_versions}`, which picks the model folder and appends `_all_versions` to every output file name.
3. `L.cutoff_dir(T)` takes the variant as an argument instead of a global path.

**Check:** for each T, the all-versions run has more training abstracts than the main run, by about 5–6% according to the earlier measurement. If the gap is 0, the flag is not applied.

### 4. Run order: smoke test first, then the full run

1. **Smoke test** (a few minutes): one cutoff, one target, both variants.
   ```
   ir2/.venv/bin/python ir2/scripts/phase3_backtest.py --cutoffs 2016 --targets query_expansion
   ```
   With partial arguments the script writes `*_partial.csv`, so the full results are not touched. Look at the printed status and the top-10 of `backtest_top_partial.csv` before going on.
2. **Main run:** `phase3_backtest.py` (word2vec models and SPECTER2 vectors are reused, nothing is retrained).
3. **All-versions run:**
   ```
   ir2/.venv/bin/python ir2/scripts/phase2_preprocess.py --skip-docs --no-version-filter
   ir2/.venv/bin/python ir2/scripts/phase3_train.py --variant all_versions          # 24 models, about 45 min
   ir2/.venv/bin/python ir2/scripts/phase3_backtest.py --variant all_versions
   ```
4. **Report:** `phase3_report.py`, then `make_notebook.py`.

Keep the logs (`results/phase3_*.log`). If a step fails, keep its output and write it in `NOTES.md`; do not rerun silently with other settings.

### 5. Methods compared in each pair

| Method | Role | Score of a candidate |
|---|---|---|
| `w2v_out` | **the paper's method** | its normalized output vector (`syn1neg`) · the target vector (mean of the normalized input vectors of the in-vocabulary indicator tokens) |
| `w2v_in` | ablation: does the output vector matter? | input · input cosine with the target vector |
| `specter2` | document-encoder contrast | cosine between the mean SPECTER2 vector of its pre-T abstracts and the mean vector of the pre-T abstracts with an indicator |
| `random` | baseline | mean P@k over 1,000 random orders of the same candidates |
| `frequency` | baseline | pre-T count, highest first |
| `cooc2` | extra baseline, already computed | cosine between its abstract co-occurrence counts and the target's |

Ties are broken by frequency. For every method the script also computes the **frequency-matched control**: the expected P@k if each term in the top k were replaced by a random candidate from the same pre-T frequency decile.

**SPECTER2 caveat, to say in the talk:** the encoder was pretrained on papers published after most cutoffs, so `specter2` is an upper reference and not a fair backtest.

### 6. Metrics

For every valid pair, candidate set, method and seed: P@10, P@20 and P@50 (the share of the top k found), and the base rate (the share of all candidates found). Precision counts hits up to the end of the snapshot. The Fig. 3a curve uses the year each hit was found.

### 7. Outputs (`phase3_report.py`)

1. **Main table** (`results/backtest_table.md`), one per candidate set. One row per valid pair: target, T, number of candidates, base rate, then P@10 / P@20 / P@50 for `w2v_out` (mean ± std over seeds), `random`, `frequency`, `w2v_in`, `specter2` and `cooc2`. The last row is the mean over valid pairs, with word2vec as mean ± std of the per-seed means. This is the table the professor asked for (point 2).
2. **Beyond-frequency table:** `P@50 − frequency-matched` per method, mean over pairs.
3. **Pair table:** all 48 pairs, kept or dropped, with the reason and the pre-T cs.IR abstract count. Dropped pairs are reported as a result (for example, "RAG has no word before 2023").
4. **Leakage table:** for each method, mean P@50 in the main run vs the all-versions run, and the share of abstracts removed per T.
5. **Fig. 3a plot** (`figures/fig3a_all.png`, `figures/fig3a_phrases.png`): x = years after T, y = cumulative % of the `w2v_out` top 50 found. One thin line per cutoff, the mean in bold, `random` and `frequency` as dashed references. The bold mean at horizon n only averages the cutoffs that reach n: say so in the caption.
6. **Hero list** (`results/hero_top20.md`): for one (target, T), the `w2v_out` top 20 with ✓ and the year found or ✗, next to the `frequency` top 20. It is chosen after seeing all results, and the slide says so.
7. Raw files: `results/backtest.csv`, `backtest_curves.csv`, `backtest_top.csv`, `backtest_status.csv`, and the same with `_all_versions`.

### 8. Correctness checks, run once after the main run

- **No leakage in training:** for each T, every row in `rows.npz["train"]` has `year < T` and `last_version_year < T`.
- **Candidates are new:** for a sample of candidates of each pair, no pre-T abstract contains both the candidate and an indicator.
- **Found is consistent:** for 5 hits of the hero list, open the cs.IR abstracts where they co-occur with an indicator and confirm the year and the count ≥ 3.
- **Random matches the base rate:** the random P@50 is within 0.01 of the base rate in every pair.
- **Seeds differ:** the three seeds give different top-50 lists, with a small std. A std of 0 means the seeds were not applied.

### 9. Reading rules, fixed before the run

- **word2vec beats random in a pair** if its P@50 is above random for all 3 seeds.
- **word2vec carries knowledge beyond frequency** if the mean `w2v_out` P@50 is above its frequency-matched control.
- **The raw comparison with `frequency` is reported as it comes out**, whichever method wins. If frequency wins, that is the failure case for the talk.
- **The version filter does not matter** if the mean `w2v_out` P@50 changes by less than one seed std between the two runs. Otherwise the difference is the result.
- **No threshold changes after seeing results.** If one is needed, rerun with both values and report both in `NOTES.md`.

## Part B: embedding timeline of cs.IR

### 10. Procedure (`scripts/phase4_timeline.py`, to write)

1. **Subtopics:** 6–7 keyword sets, matched like the indicators on unphrased text: recommender systems, query processing and expansion, learning to rank, neural and dense retrieval, conversational search, evaluation, and RAG/LLMs. Keyword sets are chosen because they can be explained on a slide and match the backtest targets. Log them in `NOTES.md` before plotting.
2. **Documents:** cs.IR abstracts with their cached SPECTER2 vectors (no new encoding). An abstract can belong to several subtopics.
3. **Centroids:** the mean vector for each (subtopic, year), only where the subtopic has ≥ 20 abstracts that year (the same threshold as Part A). Years 2010–2026.
4. **Projection:** one PCA fitted on all centroids together, so the axes are shared across years. Report the explained variance of the 2 axes.
5. **Plot** (`figures/timeline_specter2.png`): one line per subtopic, points labelled with years, the first and last year marked.
6. **Summary** (`results/timeline.md`): number of abstracts per point, explained variance, and 3–5 sentences on what moved, for example whether dense retrieval and RAG move towards each other after 2020.
7. **Optional, word2vec version:** align the cutoff spaces with orthogonal Procrustes on the shared vocabulary (as in HistWords) and list the 10 nearest neighbours of 2–3 concepts per cutoff.

### 11. Checks

- **Robustness:** refit the PCA without 2026 (partial year), and with ≥ 50 abstracts per point. The trajectories should keep their shape; if not, say so.
- **Caveat:** SPECTER2 was trained on recent papers, the same caveat as in Part A.
