# Unsupervised word embeddings capture latent knowledge from materials science literature (Tshitoyan et al., Nature 2019)
Pair: **Alexandre Costa** and **Valentina Cadime**. 

Method: first, Word2Vec is trained on the testbed documents without the queries or relevance labels, then, these documents and the desired queries will be represented via vectors that are obtained through the SIF weighted average of their word vectors, then, the documents are ranked by cosine similarity, and finally, the retrieval performance is compared with BM25 and MiniLM.

From the repository root, run:

```powershell
python -m pip install -r pairs/12-mat2vec/requirements.txt
python pairs/12-mat2vec/method.py
python testbed.py score pairs/12-mat2vec/runs --method "mat2vec word2vec + SIF (paper 12)" --pair 12-mat2vec --against bm25
python testbed.py score pairs/12-mat2vec/runs --method "mat2vec word2vec + SIF (paper 12)" --pair 12-mat2vec --against minilm --out pairs/12-mat2vec/results_vs_minilm.json
```

## Historical backtest and timeline

The paper's own experiment, a historical backtest adapted to arXiv cs.IR (plus the thermoelectric backtest in the paper's domain and an embedding timeline of the field), is in [reproduction/](reproduction/README.md). [reproduce.ipynb](reproduce.ipynb) shows its saved results. Reading plan is in [docs/](docs/).

## Warning
The paper method was adapted to rank documents. It was not it's main goal. Full results can be seen on [results/](results/), the demo website, or in the slides.