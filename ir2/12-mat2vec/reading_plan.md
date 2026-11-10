# Reading plan — Pair 12: Tshitoyan et al., "Unsupervised word embeddings capture latent knowledge from materials science literature" (Nature, 2019)

**The paper in 5 sentences.** The authors train word2vec on 3.3 million materials-science abstracts, with no labels or chemical knowledge. The embeddings encode real scientific relations: related compounds end up close together, and analogies like "ferromagnetic − NiFe + IrMn ≈ antiferromagnetic" hold. Ranking materials by the similarity between their *output* embedding and the word "thermoelectric" yields candidates for thermoelectric applications. In a historical backtest, models trained only on abstracts published before a given year rank materials never yet linked to "thermoelectric". Their top 50 picks were about 8× more likely than random candidates to be studied as thermoelectrics within five years, so latent knowledge in the literature anticipates discoveries.

**What we will reproduce.**

- **Sanity check:** three claims with the authors' pretrained embeddings (neighbours of LiCoO2, the magnetic analogy, the thermoelectric ranking).
- **Key result, at small scale:** the thermoelectric backtest (Fig. 3a) on arXiv materials-science abstracts (cond-mat.mtrl-sci, Kaggle metadata). For several cutoff years, we will train word2vec on earlier abstracts only, rank chemical formulas never yet linked to "thermoelectric", and measure how many of the top 50 are studied as thermoelectrics within 5 years. We will compare the paper's output·input score with input·input cosine and with random and frequency rankings, using 3 seeds per cutoff.

**Testbed.** The paper does not rank documents, so our row will test its representation as a retriever: the mat2vec preprocessing and word2vec settings, trained on the testbed corpora's text only, with documents and queries as SIF-weighted averages of word vectors. It will be scored on SciFact, NFCorpus and ArguAna against BM25. The live demo will show BM25 and our method side by side on a hero query, a failure query and a wildcard query.

**Compute plan.** Everything will run on one laptop (M1 Max, 32 GB). The arXiv file will be streamed, never loaded whole. Each word2vec model should take a few minutes on CPU. The demo will use indexes built beforehand.

**Risks.**

- **Corpus size:** arXiv holds far fewer materials abstracts than the paper's 3.3M, so the effect may be weaker.
- **Material names:** detecting chemical formulas with a simple pattern will miss or mangle some.
- **Frequency confound:** frequently mentioned materials are more likely to be "found", so a frequency baseline is essential.
- **Hidden leakage:** arXiv metadata may hold revised abstracts, which could bring future text into the training data.
- **Testbed vocabulary:** a word2vec model trained on small corpora will have no vector for rare query words.
- **Software:** the original code targets old Python and gensim, so we will write fresh code.
