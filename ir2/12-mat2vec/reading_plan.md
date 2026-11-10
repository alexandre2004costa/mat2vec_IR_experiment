# Reading plan — Pair 12: Tshitoyan et al., "Unsupervised word embeddings capture latent knowledge from materials science literature" (Nature, 2019)

**The paper in 5 sentences.** The authors train word2vec on 3.3 million materials-science abstracts, with no labels or chemical knowledge. The embeddings encode real scientific relations: related compounds end up close together, and analogies like "ferromagnetic − NiFe + IrMn ≈ antiferromagnetic" hold. Ranking materials by the similarity between their *output* embedding and the word "thermoelectric" yields candidates for thermoelectric applications. In a historical backtest, models trained only on abstracts published before a given year rank materials never yet linked to "thermoelectric". Their top 50 picks were about 8× more likely than random candidates to be studied as thermoelectrics within five years, so latent knowledge in the literature anticipates discoveries.

**What we will reproduce.**

- **Sanity check:** three claims with the authors' pretrained embeddings (neighbours of LiCoO2, the magnetic analogy, the thermoelectric ranking).
- **Key result, small scale, in the paper's own domain:** the thermoelectric backtest (Fig. 3a) on arXiv materials-science abstracts (cond-mat.mtrl-sci, Kaggle metadata). We will use several cutoff years, chemical formulas as candidates and the paper's success criterion. We will compare the paper's output·input score with input·input cosine, random and frequency baselines, report the share of the top 50 "found" within 5 years, and train 3 seeds per cutoff.
- **Transfer to IR:** the same protocol on arXiv cs.IR abstracts, with IR concepts as targets (learning to rank, dense retrieval, RAG…) and terms as candidates. This tests whether the effect survives outside chemistry.

**Testbed.** The paper does not rank documents, so our row will be a transfer test of its representation: the mat2vec preprocessing and word2vec settings, trained on the testbed corpora's text only, with documents and queries as SIF-weighted averages of word vectors. It will be scored on SciFact, NFCorpus and ArguAna against BM25. The live demo will show BM25 and our method side by side on one hero query, one failure query and one wildcard query.

**Compute plan.** Everything will run on one laptop (M1 Max, 32 GB). The arXiv file will be streamed, never loaded whole. Each word2vec model should take minutes on CPU. The demo will use indexes built beforehand.

**Risks.**

- **Corpus size:** arXiv holds far fewer abstracts than the paper's 3.3M, so the models will be weaker.
- **Material names:** detecting chemical formulas with a simple pattern will miss or mangle some.
- **Frequency confound:** co-occurrence-based success favours frequent terms, so frequency baselines are essential.
- **Concepts with no word yet:** a concept absent from the vocabulary before its cutoff cannot be predicted at all.
- **Hidden leakage:** arXiv metadata may hold revised abstracts.
- **Software:** the original code targets old Python and gensim, so we will write fresh code.
