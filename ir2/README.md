# Tshitoyan et al. (Nature 2019) on arXiv cs.IR: historical backtest
Pair 12. The paper's word2vec pipeline (mat2vec preprocessing, skip-gram, output·input scoring) is run as a historical backtest on cs.IR, plus an embedding timeline of the field. The testbed row (code and results) is in `testbed/`; see `testbed/README.md`. Plans: `reading_plan.md`, `test_plan.md` (overview), `tests_backtest.md`, `tests_testbed.md`. Decision log: `NOTES.md`.

From the repository root (Python 3.12, `pip install -r ir2/requirements.txt`, Kaggle arXiv zip as `archive.zip`):

```
for s in phase1_corpus phase2_preprocess phase3_train phase3_encode phase3_backtest phase3_report phase3_checks phase4_timeline; do ir2/.venv/bin/python ir2/scripts/$s.py; done
# leakage comparison: phase2_preprocess --skip-docs --no-version-filter, then phase3_train/backtest/report --variant all_versions
ir2/.venv/bin/python ir2/scripts/make_notebook.py      # rebuilds ir2/reproduce.ipynb from the cached results
```

Layout of `ir2/`: `scripts/` (pipeline), `ir2lib.py` (shared code), `results/` (tables), `figures/`, `data/` and `models/` (cached artefacts, not in git). The rest of the repository is the original mat2vec code, left untouched.
