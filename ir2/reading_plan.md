# Reading plan (Topic 12): Tshitoyan et al., "Unsupervised word embeddings capture latent knowledge from materials science literature" (Nature, 2019)

**Summary**

The paper investigates whether unsupervised word embeddings can capture scientific knowledge from literature related to materials without using manually labelled data. The authors use the skip-gram Word2Vec model to learn word representations from approximately 1.5 million abstracts that focus on inorganic materials. The study claims that these embeddings capture meaningful relationships between materials, chemical properties, and applications, despite that no chemical knowledge being explicitly provided during the training. Furthermore, when the study ranks materials by their similarity to application terms, such as "thermoelectric", it is found that the model yields candidates whose potential for these applications had not yet been explored. Finally, through a historical backtest, the authors mention that their top 50 predicted thermoelectric materials were approximately eight times more likely to be investigated within the next five years than randomly selected candidates, which suggests that latent knowledge in the literature anticipates discoveries.

**What we will reproduce**

- **Key result, at small scale, on arXiv cs.IR year by year:** for cutoff years T = 2016-2023, we train word2vec on cs.IR abstracts published before T (dated by first version; cs.CL might be added as extra *training* text only). For each target concept X (query expansion, recommender systems, learning to rank, dense retrieval, conversational search, retrieval-augmented generation), we rank the terms never seen with X before T, using the paper's score (output vector · input vector). A term is "found" if it co-occurs with X in cs.IR abstracts after T, and we report precision@10, @20 and @50.
- **Only valid (concept, T) pairs:** a pair is tested only if the concept already exists before T, i.e. its indicator terms occur in at least 20 pre-T cs.IR abstracts (default threshold, logged in our notes). Dense retrieval and retrieval-augmented generation barely occur before 2020, so they are not targets for T = 2016-2019 and only enter at the later cutoffs; the list of kept and dropped pairs will be reported.
- **Baselines in the same table:** every precision@k row for word2vec sits next to a random ranking (mean over 1,000 draws from the same candidate set) and a frequency ranking (candidates sorted by pre-T count). Since IR concepts are built from common words, word2vec only counts as working where it beats the frequency ranking, not just random.
- **Abstract-version leakage:** the arXiv metadata holds the *latest* version of each abstract, so a paper first posted before T may carry text revised after T. A paper is therefore used as pre-T text only if its latest version is also dated before T, so its abstract cannot contain post-T text. This gives the same guarantee as keeping single-version papers while losing fewer abstracts. We also rerun on all papers and report both, with the share of abstracts the filter removes.
- **Contrast:** averaged SPECTER2 document vectors, where a term is the mean vector of the pre-T abstracts containing it. If we have time, other methods (like input-input).
- **Apply, an embedding timeline of the field:** for 6-7 keyword-defined cs.IR subtopics, the yearly centroid of their abstracts' SPECTER2 vectors, projected with one PCA fitted on all years. The testbed has no dates, so the timeline uses cs.IR.

**Testbed**

The paper does not rank documents, so our row will test its representation as a retriever. We use the mat2vec preprocessing and word2vec settings, trained on the testbed corpora's text only, and represent documents and queries as SIF-weighted averages of word vectors. It will be scored on SciFact, NFCorpus and ArguAna against two references on the same corpora: BM25 (lexical) and MiniLM (`all-MiniLM-L6-v2`, a 2021 sentence encoder). The MiniLM row shows how far a 2013-style averaged word2vec representation is from a modern dense one, beyond its gap to BM25. The live demo will show BM25 and our method side by side on a hero query, a failure query and a wildcard query.

**Compute plan**

A laptop (M1 Max, 32 GB) will be used to train Word2Vec models per cutoff, with 3 seeds each (reported as mean ± std). Hence, a total of 24 models (8 cutoffs × 3 seeds) will be trained on text from the arXiv metadata file (5.7 GB), which is streamed instead of loaded whole; the all-papers comparison run doubles this to 48. MiniLM encoding of the three testbed corpora is small and is done once and cached. For SPECTER2, every abstract is encoded once on the laptop GPU and cached. Both the backtest and the timeline reuse these vectors. Lastly, the demo will use precomputed indexes to retrieve documents, so that the presentation may not stop for reprocessing.

**Risks**

- The cs.IR corpus is considerably small, especially before 2015, hence the performance may be weaker.
- Word2Vec cannot rank concepts that do not appear in its training vocabulary, which hinders identification of emerging topics. Besides, Word2Vec struggles with rare query words when trained on small corpora.
- IR concepts often include common words that co-occur with many other terms. Thus, high precision may reflect term frequency rather than meaningful relationships learned by the model. The frequency baseline in the same table is our check for this.
- The metadata holds the latest version of each abstract, which leaks post-T text into pre-T training. Keeping only papers whose latest version predates T removes this, but drops about 5-6% of pre-T abstracts from an already small corpus.
- Restricting to valid (concept, T) pairs leaves fewer pairs at early cutoffs, so the mean over targets is computed on an unbalanced set.
- SPECTER2 may not be suitable for historical backtest since it was pretrained on more recent papers.