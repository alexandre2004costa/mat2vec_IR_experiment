"""Builds and executes ir2/reproduce.ipynb (Phases 1-3 from cached artefacts)."""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Reproduction: Tshitoyan et al. (2019) on cs.IR, a historical backtest\n\n"
       "Paper 12, IR 2 seminar. This notebook **reads cached artefacts** produced by the scripts in `ir2/scripts/`; "
       "each section names the command that regenerates them. Decisions and deviations are in `ir2/NOTES.md`."),
    code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, Markdown, display\n"
         "IR2 = Path('.').resolve()\nR, F = IR2 / 'results', IR2 / 'figures'\n"
         "show = lambda p: display(Markdown((R / p).read_text()))"),
    md("## Phase 1: dated corpus (Kaggle arXiv snapshot, "
       "2026-11-08)\n`python ir2/scripts/phase1_corpus.py`"),
    code("counts = pd.read_csv(R / 'phase1_counts.csv', index_col='year')\ndisplay(counts.loc[2008:])\n"
         "display(Image(F / 'phase1_counts.png'))"),
    md("## Phase 2: preprocessing with a phraser learned before each cutoff\n"
       "`python ir2/scripts/phase2_preprocess.py`"),
    code("show('phase2_cutoffs.md')"),
    md("## Phase 3: the backtest\n`python ir2/scripts/phase3_train.py && python ir2/scripts/phase3_encode.py && "
       "python ir2/scripts/phase3_backtest.py && python ir2/scripts/phase3_report.py --hero recommender_systems 2017`\n\n"
       "Mean precision over the scored (target, T) pairs; word2vec is averaged over 3 seeds."),
    code("bt = pd.read_csv(R / 'backtest.csv')\nbt = bt[bt.w2v_target_in_vocab]\n"
         "(bt.groupby(['setting', 'method'])[['p10', 'p20', 'p50', 'p50_freq_matched']].mean()\n"
         "   .assign(beyond_frequency=lambda d: d.p50 - d.p50_freq_matched).round(3))"),
    code("display(Image(F / 'fig3a_all.png'))\ndisplay(Image(F / 'fig3a_phrases.png'))"),
    md("Per-seed spread of the paper's scoring (mean P@50 over pairs):"),
    code("bt[bt.method.str.startswith('w2v')].groupby(['setting', 'method', 'seed']).p50.mean().unstack('seed').round(3)"),
    md("Hero list (chosen after seeing every pair):"),
    code("show('hero_top20.md')"),
    md("Every (target, T) pair and why it was or was not scored:"),
    code("pd.read_csv(R / 'backtest_status.csv')[['target', 'T', 'status', 'pre_T_target_docs', "
         "'post_T_target_docs', 'n_candidates_all', 'base_rate_all']]"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "reproduce.ipynb")
print("wrote", HERE / "reproduce.ipynb")
