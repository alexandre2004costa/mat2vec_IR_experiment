"""Builds and executes reproduce.ipynb (all results, read from cached artefacts).

Usage: python reproduction/scripts/make_notebook.py
The numbers quoted in the markdown conclusions come from reproduction/results/ (leakage_comparison.md,
backtest_summary.md, te_backtest_paper.md); update them here if those files change.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parents[2]  # repository root (pairs/12-mat2vec in ir2-testbed)
S = "python reproduction/scripts/"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Reproduction: Tshitoyan et al. (2019), historical backtests and an embedding timeline\n\n"
       "Paper 12, IR 2 seminar. The paper's historical backtest, run on arXiv cs.IR and, for comparison, on arXiv "
       "materials science (the paper's own domain), plus an embedding timeline of cs.IR.\n\n"
       "This notebook **reads cached artefacts** produced by the scripts in `reproduction/scripts/`; each section "
       "names the command that regenerates them. Decisions and deviations are in `reproduction/NOTES.md`."),
    code("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, Markdown, display\n"
         "HERE = Path('.').resolve()\nif not (HERE / 'reproduction').is_dir():\n    HERE = HERE / 'pairs' / '12-mat2vec'\n"
         "IR2 = HERE / 'reproduction'\nR, F = IR2 / 'results', IR2 / 'figures'\n"
         "show = lambda p: display(Markdown((R / p).read_text()))"),

    md("## Phase 1: dated corpus (Kaggle arXiv snapshot, 2026-11-08)\n"
       f"`{S}phase1_corpus.py`"),
    code("counts = pd.read_csv(R / 'phase1_counts.csv', index_col='year')\ndisplay(counts.loc[2008:])\n"
         "display(Image(F / 'phase1_counts.png'))"),

    md("## Phase 2: preprocessing with a phraser learned before each cutoff\n"
       f"`{S}phase2_preprocess.py`"),
    code("show('phase2_cutoffs.md')"),

    md("## Phase 3: the backtest on cs.IR\n"
       f"`{S}phase3_train.py && {S}phase3_encode.py && {S}phase3_backtest.py && {S}phase3_report.py`\n\n"
       "A (target, T) pair is scored only if the concept exists before T: an indicator in ≥ 20 pre-T cs.IR "
       "abstracts and in the word2vec vocabulary. Which pairs were scored, and why the others were not:"),
    code("st = pd.read_csv(R / 'backtest_status.csv')\n"
         "st.pivot(index='target', columns='T', values='pre_T_target_docs_ir')"),
    code("st.pivot(index='target', columns='T', values='status')"),

    md("### Main table: word2vec next to random and frequency"),
    code("show('backtest_table.md')"),

    md("### All methods, and what each adds beyond frequency"),
    code("text = (R / 'backtest_summary.md').read_text().split('## Status')[0]\n"
         "display(Markdown(text.split('\\n', 1)[1]))  # drop the file's own title"),
    code("display(Image(F / 'fig3a_all.png'))\ndisplay(Image(F / 'fig3a_phrases.png'))"),
    md("**What this shows**\n\n"
       "- The paper's score (word2vec output·input) beats random: P@50 0.279 against 0.137 on all terms (≈ 2×), "
       "and 0.254 against 0.102 on multiword phrases (≈ 2.5×). The paper reported ≈ 8× on thermoelectrics.\n"
       "- Ranking by frequency alone beats it on all terms (0.513), and roughly ties it on phrases (0.281).\n"
       "- Beyond frequency, word2vec still adds something: +0.10 (all terms) and +0.13 (phrases) over random "
       "candidates of the same frequency.\n"
       "- Plain cosine (input·input) is below random on all terms (0.104): the paper's output·input score matters.\n"
       "- Second-order co-occurrence is the strongest method (0.760). The signal the paper attributes to "
       "embeddings is already present in raw counts on this small corpus. SPECTER2 (0.467) was pretrained on "
       "papers written after most cutoffs, so it is an upper reference, not a fair backtest."),

    md("### Leakage check: with and without the version filter\n"
       "arXiv metadata stores only the latest version of each abstract, so a paper first posted before T but "
       "revised later can carry post-T text. The main run keeps a paper as pre-T text only if its latest version "
       "is also before T. This check reruns everything without that filter.\n\n"
       f"`{S}phase2_preprocess.py --skip-docs --no-version-filter && {S}phase3_train.py --variant all_versions && "
       f"{S}phase3_backtest.py --variant all_versions && {S}phase3_report.py --variant all_versions`"),
    code("show('leakage_comparison.md')"),
    md("**What this shows**\n\n"
       "- The filter removes 4.8–6.0 % of pre-T abstracts per cutoff. The same 28 pairs are scored in both runs.\n"
       "- Without the filter, every method loses about 0.01 P@50 (all terms): word2vec 0.279 → 0.269, "
       "frequency 0.513 → 0.504, co-occurrence 0.760 → 0.747, SPECTER2 0.467 → 0.442, and **random too**, "
       "0.137 → 0.130.\n"
       "- Because random drops as well, the change is in the candidate pool, not in what word2vec learned. "
       "Relative to random, word2vec is unchanged: 2.04× with the filter, 2.07× without (2.49× and 2.44× on phrases).\n"
       "- One would expect leaked text to help. It does not here, plausibly because a revised abstract that "
       "mentions a later term next to the concept makes that term count as already known, so it leaves the "
       "candidate list just before it would have been found. The snapshot holds no original versions, so this "
       "mechanism cannot be checked directly.\n"
       "- By the rule fixed before running (a change larger than one seed's std), the filter does matter, "
       "slightly: 0.010 against a std of 0.006. **No conclusion changes**; the filtered run is the main result."),

    md("### Hero list (chosen after seeing every pair)"),
    code("show('hero_top20.md')"),

    md("### Correctness checks\n"
       f"`{S}phase3_checks.py`"),
    code("show('backtest_checks.md')"),

    md("## Phase 3b: the thermoelectric backtest, in the paper's own domain\n"
       f"`{S}te_backtest.py all`\n\n"
       "The same backtest on arXiv materials science (`cond-mat.mtrl-sci`, 113,695 abstracts), with the paper's "
       "settings: target word `thermoelectric`, chemical formulas as candidates, 200 dimensions, yearly cutoffs "
       "2010–2021, found = one abstract with a thermoelectric keyword within 5 years. The goal is to see how close "
       "a small corpus gets to the paper's ≈ 8× random, and to add the frequency baseline the paper did not report."),
    code("show('te_backtest_paper.md')"),
    code("display(Image(F / 'te_fig3a_paper.png'))"),
    md("**What this shows**\n\n"
       "- word2vec reaches **3.5× random** (P@50 within 5 years: 0.228 against 0.066), not the paper's ≈ 8×. "
       "Per cutoff it ranges from 2.0× (T = 2010 and 2011) to 5.2× (T = 2016).\n"
       "- **Frequency reaches 5.1×** (0.335) and is above word2vec at 10 of 12 cutoffs; word2vec wins at "
       "T = 2016 (0.367 against 0.34) and ties at T = 2020 (0.22). The paper compared only against random.\n"
       "- Beyond frequency, word2vec adds +0.120 over random candidates of the same frequency, more than "
       "frequency itself (+0.073), so the embedding carries real information.\n"
       "- Here, unlike cs.IR, co-occurrence with known thermoelectrics is weak (0.125), and plain cosine "
       "(input·input) is low again (0.093).\n"
       "- The most plausible cause of the gap with the paper is data: 114 k arXiv abstracts against the paper's "
       "1.5 M, and arXiv is a biased sample of materials science."),

    md("## Phase 4: embedding timeline of cs.IR\n"
       f"`{S}phase4_timeline.py`"),
    code("display(Image(F / 'timeline_specter2.png'))\ndisplay(Image(F / 'timeline_specter2_panels.png'))\n"
         "show('timeline.md')"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "reproduce.ipynb")
print("wrote", HERE / "reproduce.ipynb")
