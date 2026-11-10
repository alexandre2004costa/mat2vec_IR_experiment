# Reading plan — Pair 12: Tshitoyan et al., "Unsupervised word embeddings capture latent knowledge from materials science literature" (Nature, 2019)

**The paper in 5 sentences.** The authors train word2vec on 3.3 million materials-science abstracts, with no labels or chemical knowledge. The embeddings encode real scientific relations: related compounds end up close together, and analogies like "ferromagnetic − NiFe + IrMn ≈ antiferromagnetic" hold. Ranking materials by the similarity between their *output* embedding and the word "thermoelectric" yields candidates for thermoelectric applications. In a historical backtest, models trained only on abstracts published before a given year rank materials never yet linked to "thermoelectric". Their top 50 picks were about 8× more likely than random candidates to be studied as thermoelectrics within five years, so latent knowledge in the literature anticipates discoveries.

**What we will reproduce.**

- **Key result, at small scale, transferred to one subfield: arXiv cs.IR, year by year.**
  - **Corpus:** cs.IR abstracts from the Kaggle arXiv metadata, dated by the year of their first version. Early cs.IR is small, so cs.CL abstracts can be added as extra *training* text only.
  - **Backtest:** for cutoff years T = 2016–2023, we train word2vec on pre-T abstracts only. For each target concept X (query expansion, recommender systems, learning to rank, dense retrieval, conversational search, retrieval-augmented generation), we rank the terms that have never appeared in an abstract with X before T. A term counts as "found" if it co-occurs with X in cs.IR abstracts after T.
  - **Metric:** precision@10, @20 and @50 of "concepts near X in year T", plus, as in the paper, the cumulative share of the top 50 found 1, 2, 3… years after T, compared with the random and frequency baselines. word2vec is trained with 3 seeds per cutoff and reported as mean ± std.
  - **Methods:** the paper's output·input score, input·input cosine, and averaged document vectors for contrast. For the document vectors, each abstract is encoded with SPECTER2, a term is the mean vector of the pre-T abstracts containing it, and X is the mean over abstracts mentioning it. The baselines are random and frequency rankings.
- **Apply: an embedding timeline of the field.**
  - **Subtopics:** 6–7 cs.IR subtopics defined by keyword sets.
  - **Trajectories:** for each subtopic and year, the centroid of its abstracts' SPECTER2 vectors. All centroids are projected with one PCA fitted on every year, so positions are comparable across years.
  - **word2vec version:** the yearly word2vec spaces are aligned with orthogonal Procrustes (the alignment method of Hamilton et al., 2016) to show how the neighbours of a few concepts move.
  - **Why not the testbed:** its corpora have no dates, so the timeline uses the dated cs.IR corpus.

**Testbed.** The paper does not rank documents, so our row will test its representation as a retriever. We use the mat2vec preprocessing and word2vec settings, trained on the testbed corpora's text only, and represent documents and queries as SIF-weighted averages of word vectors. It will be scored on SciFact, NFCorpus and ArguAna against BM25. The live demo will show BM25 and our method side by side on a hero query, a failure query and a wildcard query.

**Compute plan.**

- **Hardware:** one laptop (M1 Max, 32 GB).
- **Data:** the arXiv metadata file (5.7 GB) is streamed, never loaded whole.
- **word2vec:** 8 cutoffs × 3 seeds = 24 models on at most about 8M tokens each, a few minutes per model on CPU.
- **SPECTER2:** every abstract is encoded once on the laptop GPU and cached. Both the backtest and the timeline reuse these vectors, so the timeline (centroids, PCA, Procrustes) takes minutes.
- **Demo:** uses indexes built beforehand.

**Risks.**

- **Corpus size:** cs.IR is tiny before 2015 (about 2,200 abstracts), far from the paper's 3.3M, so the effect may be much weaker.
- **Concepts without a word:** a concept cannot be ranked before it has a name. Retrieval-augmented generation and dense retrieval are likely out of vocabulary for early cutoffs.
- **Frequency confound:** IR concepts are common words, and frequent terms co-occur with everything, so the frequency baseline may win.
- **Hidden leakage:**
  - The metadata stores each paper's latest abstract, which may hold post-T text.
  - SPECTER2 was pretrained on papers published after most cutoffs, so it is an upper reference, not a fair backtest.
- **Timeline artefacts:** centroid shifts may reflect the field's growth or the PCA rather than real change. Small subtopic-years give noisy centroids. Procrustes alignment needs enough shared vocabulary between years.
- **Testbed vocabulary:** a word2vec model trained on small corpora will have no vector for rare query words.
