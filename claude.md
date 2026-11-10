# Context

I'm presenting Tshitoyan et al., "Unsupervised word embeddings capture latent knowledge from materials science literature" (Nature 2019) in an M2 Information Retrieval seminar (IR 2, Université Paris-Saclay, paper #12). This repo is my fork of the paper's code (mat2vec).

The paper's key result is a **historical backtest**:
1. Train word2vec on abstracts published before year T.
2. Rank materials by similarity to "thermoelectric", keeping only materials that have never appeared in an abstract with a thermoelectric keyword before T.
3. Check how many of the top 50 appear with a thermoelectric keyword after T.

The paper's top-50 picks were about 8× more likely to be "found" within 5 years than random candidates. Its ranking score is the dot product of the material's normalized *output* embedding with the *word* embedding of the target (input vectors).

The course asks me to:
- reproduce this backtest at small scale on a dated corpus of IR papers, with word2vec and, for contrast, a document encoder;
- plot subtopic centroid trajectories over the years (an "embedding timeline" of the field);
- run the method on the course's shared testbed;
- prepare a 10-minute live demo with three queries: **hero** (method clearly works), **failure** (method collapses), and a **wildcard** typed live by the audience.

Compute: my laptop (CPU, 16 GB RAM). Everything must run on it in reasonable time.

# How I want you to work

- **Start by exploring.** Read this repo (README, `mat2vec/processing`, `mat2vec/training`). Then clone https://github.com/kimgerdes/ir2-testbed into a sibling folder and read its README, `testbed.py`, the `results.json` schema and the folder conventions.
- **Propose a short plan before writing code**, then work phase by phase. Stop after each phase and show me what you got.
- **Don't fight legacy code.** mat2vec targets old Python and gensim versions. Reuse its ideas, such as phrase detection and number normalization, but write fresh code in a new folder `ir2/` with a current Python, a current gensim and a `requirements.txt`. Don't break the original package.
- **Ask me before** anything that needs my credentials (Kaggle API) or downloads more than about 1 GB.
- **Test on a small sample first**, then run the full thing.
- Fix random seeds. Word2vec is nondeterministic, so train 3 seeds per cutoff and report mean ± std.
- **Never invent or "expected" numbers.** If something fails, say so and keep the failure. A documented failed reproduction is a valid result for the course.
- Keep a `ir2/NOTES.md` log of every decision and its reason: corpus filters, thresholds, targets, anything that deviates from the paper. I'll need it for my reading plan and my report.

# Phase 0: sanity check with the paper's pretrained model (optional, quick)

If the README explains how to get the pretrained mat2vec embeddings and the download is reasonable:
- load them;
- reproduce three claims of the paper: the nearest neighbours of `LiCoO2`, the analogy `ferromagnetic − NiFe + IrMn ≈ antiferromagnetic`, and the top of the ranking for `thermoelectric`.

Save the output to `ir2/results/phase0_sanity.md`. This gives me a "their claims hold" slide. If it's painful, skip it and tell me.

# Phase 1: build a dated corpus

- **Source:** the Kaggle arXiv metadata snapshot (`Cornell-University/arxiv`, a JSON-lines file of several GB). Tell me what to do for the Kaggle credentials, or let me download the file manually and give you the path.
- **Stream** the file; don't load it into memory. Keep papers whose categories include `cs.IR`.
- **Date** = year of the first version (`versions[0].created`).
- **Save** `ir2/data/csir.parquet` with columns id, year, title, abstract, categories, plus whether cs.IR is the primary category.
- **Also build an optional `cs.CL` set** (same format). The cs.IR field is small (roughly 500 papers in 2015 and a few thousand recently), so I may need cs.CL as extra *training* text. Evaluation stays on cs.IR abstracts.
- **Report** the number of papers per year for both sets, as a table and a bar chart.

# Phase 2: preprocessing (in the spirit of mat2vec)

- Lowercase and tokenize. Replace numbers with `<nUm>`, and remove LaTeX and URLs.
- Learn **multiword phrases** with gensim `Phrases`, two passes so that up to 4-grams are possible. IR concepts are multiword ("query expansion", "learning to rank").
- **Avoid leakage.** For a cutoff year T, the phrase model and the vocabulary must be learned **only from text before T**. Then apply that same pre-T phrase model to post-T text when checking whether a prediction came true, so tokenization is consistent.
- Use a stopword list. Candidate terms must not be stopwords, numbers or tokens shorter than 3 characters.

# Phase 3: the backtest (main reproduction)

**Setup**
- **Cutoff years:** pick several T, for example 2015–2022, so that each has at least 3 years of future data. Justify the choice from the corpus sizes in Phase 1. If pre-T cs.IR text is too small, train on cs.IR + cs.CL and record that in NOTES.md.
- **Targets:** 4–6 IR concepts, each with an indicator set of synonyms, like the paper's thermoelectric keywords. For example:
  - `query_expansion` (query expansion, query reformulation, pseudo relevance feedback);
  - `recommender_systems` (recommendation, recommender);
  - `learning_to_rank`;
  - `dense_retrieval` (dense retrieval, dual encoder, bi-encoder);
  - `retrieval_augmented_generation`, which did not exist before about 2020 and is a likely failure case.

  Propose the final list and the indicator sets to me before running.
- **Candidates for (target, T):** terms seen at least 5 times before T that never appear in the same abstract as any indicator of the target before T.
- **"Found":** the candidate appears in the same abstract as an indicator in at least 3 abstracts from year T onwards. Also record the first year it was found.

**Ranking methods**
1. **word2vec, input × input** cosine with the target. Use skip-gram, negative sampling 15, window 8, sample 1e-4, min_count 5, 30 epochs, as in the paper. Use dimension 100–200, choosing 100 if the corpus is small, and record the choice.
2. **word2vec, output × input**: the normalized output vector of the candidate (`syn1neg`) dotted with the target's input vector. This is the paper's actual scoring.
3. **Document-encoder contrast.** Encode every abstract once with `allenai/specter2_base`, or a sentence-transformers model if adapters are a problem; tell me which. Represent a term as the mean embedding of the pre-T abstracts containing it, and the target as the mean over abstracts containing its indicators. Rank candidates by cosine. Note in NOTES.md that the pretrained encoder has seen post-T literature, which is a leakage caveat I must mention in the talk.
4. **Baselines:**
   - random ranking, averaged over 1,000 draws;
   - frequency ranking by pre-T count, because frequent terms end up near everything;
   - optionally, a second-order co-occurrence baseline that counts shared context words with the target.

**Metrics and outputs**
- precision@10, @20 and @50 per (target, T, method), plus the mean over targets and cutoffs;
- a Fig. 3a-style plot: the cumulative % of the top-50 found n years after T, one thin line per cutoff, the mean in bold, and the random and frequency baselines;
- for one illustrative (target, T): the top-20 list with ✓/✗ and the year each hit was found, which becomes my hero slide;
- results in `ir2/results/backtest.csv` and figures in `ir2/figures/`.

# Phase 4: embedding timeline of the field

- Define 5–8 subtopics of cs.IR, for example by keyword sets, or by clustering the document embeddings and naming the clusters. Propose which approach and justify it.
- **Document encoder:** for each subtopic and year, compute the centroid of its abstracts' embeddings. Project all centroids with one shared PCA, fitted on all years together so the axes are comparable. Plot one trajectory per subtopic, with year labels.
- **word2vec equivalent:** word2vec spaces from different years are not aligned. Align them with orthogonal Procrustes on shared vocabulary (as in HistWords, Hamilton et al. 2016) before comparing positions. Then show how the neighbours of a few concepts change over the years.
- Save the figures and a short markdown summary of what moved.

# Phase 5: shared testbed run

- Follow the ir2-testbed README for the expected run format, `method.py`, and `python testbed.py score` producing `results.json`.
- The paper is not a ranking method, so propose the most honest testbed row and ask me before implementing. My default suggestion: a word2vec-averaging retriever (mean or SIF-weighted word vectors of query and document, cosine) trained on the testbed corpora, or on the cs.IR text. Score it on SciFact, NFCorpus and ArguAna against the closest baseline, with the paired t-test and bootstrap CI the script computes. This places the 2019 method on the seminar's shared table, even if it loses.

# Phase 6: live demo

- A script `ir2/demo.py --target "<concept>" --year T [--k 15]`. It loads **precomputed** models (nothing is trained live), prints side by side the word2vec ranking and the frequency baseline at year T, and marks each candidate ✓ found later (with the year) or ✗. It must answer in a couple of seconds for any target the audience types, including a target missing from the vocabulary: in that case it prints a clear message, and this is itself a teaching moment.
- Propose candidates for the **hero** query (a target and year where predictions clearly came true) and the **failure** query (for example RAG before it was named, or a case where the frequency baseline wins). Write them to `demo.md`.

# Deliverables (follow the ir2-testbed folder conventions)

- `README.md`: how to run, in 10 lines or fewer
- `reproduce.ipynb`: Phases 1–4 end to end, reading from cached artefacts
- `method.py`: the testbed run
- `results.json`: from `testbed.py score`
- `demo.md`: the three queries
- `ir2/NOTES.md`: the decision log
- figures and csv results

Start with exploration and the plan.