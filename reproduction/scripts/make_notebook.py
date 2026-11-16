"""Builds and executes reproduce.ipynb (Phases 1-4 from cached artefacts)."""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parents[2]  # repository root (pairs/12-mat2vec in ir2-testbed)
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Reproduction: Tshitoyan et al. (2019) on cs.IR, a historical backtest\n\n"
       "Paper 12, IR 2 seminar. This notebook **reads cached artefacts** produced by the scripts in `reproduction/scripts/`; "
       "each section names the command that regenerates them. Decisions and deviations are in `reproduction/NOTES.md`."),
    code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, Markdown, display\n"
         "HERE = Path('.').resolve()\nif not (HERE / 'reproduction').is_dir():\n    HERE = HERE / 'pairs' / '12-mat2vec'\n"
         "IR2 = HERE / 'reproduction'\nR, F = IR2 / 'results', IR2 / 'figures'\n"
         "show = lambda p: display(Markdown((R / p).read_text()))"),
    md("## Phase 1: dated corpus (Kaggle arXiv snapshot, "
       "2026-11-08)\n`python reproduction/scripts/phase1_corpus.py`"),
    code("counts = pd.read_csv(R / 'phase1_counts.csv', index_col='year')\ndisplay(counts.loc[2008:])\n"
         "display(Image(F / 'phase1_counts.png'))"),
    md("## Phase 2: preprocessing with a phraser learned before each cutoff\n"
       "`python reproduction/scripts/phase2_preprocess.py`"),
    code("show('phase2_cutoffs.md')"),
    md("## Phase 3: the backtest\n"
       "`python reproduction/scripts/phase3_train.py && python reproduction/scripts/phase3_encode.py && "
       "python reproduction/scripts/phase3_backtest.py && python reproduction/scripts/phase3_report.py`\n\n"
       "A (target, T) pair is scored only if the concept exists before T: an indicator in ≥ 20 pre-T cs.IR "
       "abstracts and in the word2vec vocabulary. Which pairs were scored, and why the others were not:"),
    code("st = pd.read_csv(R / 'backtest_status.csv')\n"
         "st.pivot(index='target', columns='T', values='pre_T_target_docs_ir')"),
    code("st.pivot(index='target', columns='T', values='status')"),
    md("### Main table: word2vec next to random and frequency (professor's point 2)"),
    code("show('backtest_table.md')"),
    md("### All methods, and what each adds beyond frequency"),
    code("print((R / 'backtest_summary.md').read_text().split('## Status')[0])"),
    code("display(Image(F / 'fig3a_all.png'))\ndisplay(Image(F / 'fig3a_phrases.png'))"),
    md("### Leakage check: with and without the version filter (professor's point 3)\n"
       "`python reproduction/scripts/phase2_preprocess.py --skip-docs --no-version-filter && "
       "python reproduction/scripts/phase3_train.py --variant all_versions && "
       "python reproduction/scripts/phase3_backtest.py --variant all_versions && "
       "python reproduction/scripts/phase3_report.py --variant all_versions`"),
    code("show('leakage_comparison.md')"),
    md("### Hero list (chosen after seeing every pair)"),
    code("show('hero_top20.md')"),
    md("### Correctness checks\n`python reproduction/scripts/phase3_checks.py`"),
    code("show('backtest_checks.md')"),
    md("## Phase 4: embedding timeline of cs.IR\n`python reproduction/scripts/phase4_timeline.py`"),
    code("display(Image(F / 'timeline_specter2_panels.png'))\ndisplay(Image(F / 'timeline_specter2.png'))\n"
         "show('timeline.md')"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "reproduce.ipynb")
print("wrote", HERE / "reproduce.ipynb")
