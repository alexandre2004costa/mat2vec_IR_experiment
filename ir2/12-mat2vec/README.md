# Unsupervised word embeddings capture latent knowledge from materials science literature (Tshitoyan et al., Nature 2019)
Pair 12. Method: the paper's word2vec pipeline (mat2vec preprocessing, skip-gram, output·input scoring) as a historical backtest on cs.IR and, for the testbed, as a SIF-averaged word2vec retriever trained on the corpus text. It loses to BM25 on all three datasets. How to run, from the mat2vec repository root (Python 3.12, `pip install -r ir2/requirements.txt`; Kaggle arXiv zip as `archive.zip`):

```
for s in phase1_corpus phase2_preprocess phase3_train phase3_encode phase3_backtest phase3_report; do ir2/.venv/bin/python ir2/scripts/$s.py; done
ir2/.venv/bin/python ir2/12-mat2vec/method.py                     # testbed runs -> ir2/12-mat2vec/runs
python testbed.py score <path>/12-mat2vec/runs --method "mat2vec word2vec + SIF (paper 12)" --pair 12-mat2vec --against bm25
ir2/.venv/bin/python ir2/scripts/te_backtest.py all                 # thermoelectric backtest (key result)
ir2/.venv/bin/python ir2/testbed_demo.py build && ir2/.venv/bin/python ir2/testbed_demo.py show --query 212   # demo
```
