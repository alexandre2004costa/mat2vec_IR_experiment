"""Builds and executes reproduce.ipynb, the report of the reproduction (all outputs read from saved result files).

Usage: python reproduction/scripts/make_notebook.py
The numbers quoted in the markdown cells come from reproduction/results/ (backtest_table.md, backtest_summary.md,
leakage_comparison.md, hero_top20.md, backtest_checks.md, te_backtest_paper.md, timeline.md); update the text here if
those files change.
"""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parents[2]  # repository root
S = "python reproduction/scripts/"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def run(cmd):
    return f"*Regenerate with:* `{cmd}`"


cells = [
    # ------------------------------------------------------------------ introduction
    md("""# Reproducing the historical backtest of Tshitoyan et al. (2019) on arXiv cs.IR

Tshitoyan et al. (2019, *Nature*) trained word2vec on materials-science abstracts and showed that materials ranked
close to the word *thermoelectric* in an embedding trained only on abstracts up to year T were, years later,
studied as thermoelectrics far more often than random materials (about 8× random for the top 50). This notebook
reports three experiments:

1. **The backtest transferred to Information Retrieval (Sections 1–3).** Can a word2vec model trained on cs.IR and
   cs.CL abstracts written before year T anticipate which terms will start appearing next to an IR concept
   (query expansion, learning to rank, …) after T? We compare the paper's score with stronger and simpler baselines
   that the paper did not report.
2. **The thermoelectric backtest in the paper's own domain (Section 4)**, on arXiv materials science, to see how
   close a small public corpus gets to the original result.
3. **An embedding timeline of cs.IR (Section 5)**: how subtopics of the field moved relative to each other between
   2012 and 2026.

All outputs below are read from saved result files in `reproduction/results/` and `reproduction/figures/`; each
section names the script that regenerates them (see `reproduction/README.md` for the full pipeline)."""),
    md("""### Experimental setup at a glance

| | cs.IR backtest |
|---|---|
| Data | Kaggle arXiv metadata snapshot of 2026-11-08: titles and abstracts of cs.IR and cs.CL papers |
| Cutoffs T | 2016, 2017, …, 2023 (every cutoff has at least 3 full years of future, up to 2026) |
| Training text | cs.IR ∪ cs.CL abstracts whose first **and latest** version are before T (no revised text from after T) |
| Concepts (targets) | query expansion, recommender systems, learning to rank, dense retrieval, conversational search, retrieval-augmented generation; each defined by a list of indicator terms |
| A (concept, T) pair is scored if | the concept appears in ≥ 20 pre-T cs.IR abstracts, an indicator is in the word2vec vocabulary, and there are ≥ 50 candidates |
| Candidates | terms seen ≥ 5 times before T that **never** appear in a pre-T abstract with the concept; two settings: *all* terms and multiword *phrases* only |
| "Found" | the candidate appears together with the concept in ≥ 3 cs.IR abstracts from T to 2026 |
| Metric | precision@k (k = 10, 20, 50): share of the top-k ranked candidates that are later found |
| word2vec | skip-gram, 100 dimensions, window 8, 15 negative samples, sub-sampling 1e-4, learning rate 0.01, 30 epochs, 3 seeds |

**Ranking methods compared**

| method | how a candidate is scored |
|---|---|
| word2vec output·input (**the paper's score**) | dot product of the candidate's normalized *output* vector with the concept's normalized *input* vector |
| word2vec input·input | the usual cosine between input vectors (to test whether the paper's choice matters) |
| frequency | number of pre-T occurrences of the candidate (popular terms first) |
| random | mean of 1,000 random orderings of the candidates (equals the base rate in expectation) |
| 2nd-order co-occurrence | cosine between the candidate's and the concept's abstract-level co-occurrence profiles: the candidate never appears *with* the concept, but it may appear with the same words |
| SPECTER2 abstract means | cosine between the mean SPECTER2 embedding of the candidate's pre-T abstracts and that of the concept's pre-T abstracts |

SPECTER2 was pretrained on papers written after most cutoffs, so it is not a temporally clean method; it is
reported as an upper reference, not as a fair competitor."""),
    code("""from pathlib import Path
import re
import pandas as pd
from IPython.display import Image, Markdown, display

HERE = Path('.').resolve()
if not (HERE / 'reproduction').is_dir():
    HERE = HERE / 'pairs' / '12-mat2vec'
R, F = HERE / 'reproduction' / 'results', HERE / 'reproduction' / 'figures'


def show_md(name, start=None, stop=None):
    \"\"\"Display a saved markdown result, without its own title, optionally only between two headings.\"\"\"
    text = (R / name).read_text()
    if start:
        text = text[text.index(start):]
    if stop and stop in text:
        text = text[:text.index(stop)]
    if text.startswith('# '):
        text = text.split('\\n', 1)[1]
    display(Markdown(text))"""),

    # ------------------------------------------------------------------ 1. corpus
    md(f"""## 1. Corpus

cs.IR and cs.CL papers per year, dated by their **first** version. cs.CL is added to the training text because cs.IR
alone is small (a few hundred papers per year before 2017); "found" is always counted in cs.IR only.
2026 is a partial year (the snapshot ends on 2026-11-08).

{run(S + 'phase1_corpus.py --zip <path to archive.zip>')}"""),
    code("""counts = pd.read_csv(R / 'phase1_counts.csv', index_col='year')
display(counts.loc[2010:])
display(Image(F / 'phase1_counts.png'))"""),
    md("""**Observations**

- cs.IR grows from 180 papers in 2010 to 4,596 in 2026, with the steepest growth after 2017; cs.CL grows even
  faster (86 → 24,911), driven by neural NLP and LLMs.
- This sets the scale of the experiment: the first cutoff (T = 2016) has fewer than 6,000 training abstracts, and
  even the last (T = 2023) has about 48,000, against the 1.5 million abstracts of the original paper."""),

    # ------------------------------------------------------------------ 2. preprocessing
    md(f"""## 2. Preprocessing per cutoff

For each cutoff T the text is lower-cased, numbers are normalized, and multiword phrases (e.g. `learning_to_rank`)
are learned in two passes **on the pre-T text only**, so the vocabulary cannot contain phrases coined after T.
The table gives, per cutoff, the training set, the cs.IR abstracts used to decide what is "found", and the vocabulary.

{run(S + 'phase2_preprocess.py')}"""),
    code("""rows = []
for line in (R / 'phase2_cutoffs.md').read_text().splitlines():
    m = re.match(r'- T=(\\d+): ([\\d,]+) train docs \\(([\\d,]+) tokens\\), ([\\d,]+) post cs.IR docs, ([\\d,]+) phrasegrams, '
                 r'vocab ([\\d,]+) \\(([\\d,]+) phrases\\)', line)
    if m:
        rows.append([int(x.replace(',', '')) for x in m.groups()])
cut = pd.DataFrame(rows, columns=['T', 'training abstracts', 'training tokens', 'cs.IR abstracts from T on',
                                  'phrases learned', 'vocabulary (count ≥ 5)', 'of which phrases']).set_index('T')
cut.style.format('{:,}')"""),
    md("""**Observations**

- Training text grows tenfold across cutoffs (0.85 M → 8.2 M tokens), while the future window shrinks
  (25,267 → 14,987 cs.IR abstracts). Later cutoffs therefore have better models but less time for candidates to be
  found, which lowers their base rate (Section 3)."""),

    # ------------------------------------------------------------------ 3. backtest
    md(f"""## 3. The backtest on cs.IR

{run(S + 'phase3_train.py && ' + S + 'phase3_encode.py && ' + S + 'phase3_backtest.py && ' + S + 'phase3_report.py')}

### 3.1 Which (concept, T) pairs can be scored

A concept must already exist before T, otherwise "terms close to it" is meaningless. Each cell below is the number
of pre-T cs.IR abstracts that mention the concept; ✓ marks the pairs that pass all three rules and are scored."""),
    code("""st = pd.read_csv(R / 'backtest_status.csv')
st['cell'] = st.pre_T_target_docs_ir.astype(str) + st.status.eq('scored').map({True: ' ✓', False: ''})
display(st.pivot(index='target', columns='T', values='cell'))
print(f"{st.status.eq('scored').sum()} of {len(st)} pairs scored")"""),
    md("""**Observations**

- 28 of 48 pairs are scored. Recommender systems and query expansion exist from the first cutoff; learning to rank
  from 2017; conversational search from 2021; dense retrieval only from 2022.
- Retrieval-augmented generation is **never** scored: even before 2023 it appears in only 12 cs.IR abstracts. Its
  rise happened after the last cutoff that still leaves three years to observe it, so the backtest cannot test it."""),

    md("""### 3.2 Main result: the paper's score against random and frequency

**How to read:** one row per scored pair. *base rate* = share of all candidates that are later found (what a random
ranking achieves); word2vec values are mean ± std over 3 seeds. The first table uses all candidate terms, the second
only multiword phrases."""),
    code("show_md('backtest_table.md')"),
    md("""**Observations**

- **The paper's score works, but weakly.** Over the 28 pairs, word2vec's P@50 is 0.279 on all terms against 0.137
  for random (≈ 2×), and 0.254 against 0.102 on phrases (≈ 2.5×). It is above random for all three seeds in 27/28
  and 28/28 pairs. The paper reported ≈ 8× random for thermoelectrics.
- **Ranking by frequency is better.** Simply putting the most common terms first gives P@50 = 0.513 on all terms:
  word2vec is above frequency for all three seeds in **0 of 28** pairs. On phrases the two are close (0.254 vs
  0.281) and word2vec wins in 7 of 28 pairs.
- The seed spread is small (std of the mean P@50 ≤ 0.006), so these differences are not noise from training."""),
    md("""The ratio to random depends strongly on the base rate. The next cell computes, from `backtest.csv`, the mean
P@50 per concept and per cutoff (word2vec averaged over seeds), with word2vec and frequency as multiples of random."""),
    code("""b = pd.read_csv(R / 'backtest.csv')
p50 = b.groupby(['setting', 'target', 'T', 'method']).p50.mean().unstack()[['w2v_out', 'frequency', 'random']]
for setting in ['all', 'phrases']:
    for by in ['target', 'T']:
        t = p50.loc[setting].groupby(level=by).mean()
        t['w2v × random'] = t.w2v_out / t.random
        t['frequency × random'] = t.frequency / t.random
        display(Markdown(f'**Candidates `{setting}`, mean P@50 per {by}**'))
        display(t.round(3))"""),
    md("""**Observations**

- **Concepts:** recommender systems has the lowest ratio (1.4× on all terms) because it is a huge topic: about 30 %
  of all candidates end up co-occurring with it anyway, which leaves little room above random. Narrower concepts
  (conversational search, dense retrieval, learning to rank) reach 3–5× random.
- **Cutoffs:** the ratio of word2vec to random rises from 1.2× (T = 2016) to 3.9× (T = 2023) on all terms, but the
  ratio of frequency rises faster (2.2× → 11×). At late cutoffs few candidates have time to be found, and those
  that are found are mostly common words, which favours frequency."""),

    md("""### 3.3 All methods, and what each adds beyond popularity

**How to read:** besides P@k, two columns separate *popularity* from *relatedness*. *P@50 frequency-matched* is the
P@50 one would expect if each of a method's top 50 terms were replaced by a random candidate from the same pre-T
frequency decile. *P@50 − freq-matched* is therefore what the method adds beyond picking terms of the right frequency."""),
    code("show_md('backtest_summary.md', stop='## Status')"),
    md("""**Observations**

- **The paper's choice of output·input matters.** The usual cosine (input·input) is *below* random on all terms
  (0.104 vs 0.137), and on phrases it reaches only 0.155 against the paper's 0.254. Input·input finds words used
  in the same way as the concept (synonyms, variants), which plausibly rarely appear next to it. The output·input score instead measures how likely a word is to appear *in the
  context* of the concept, which is what the backtest rewards.
- **word2vec does carry information beyond popularity:** +0.101 on all terms and +0.133 on phrases. On phrases this is
  more than frequency itself adds (+0.076), i.e. word2vec picks phrases that are more likely to be found than other
  phrases of the same frequency. Its absolute P@50 is still lower than frequency on all terms because it picks
  rarer terms (frequency-matched expectation 0.178 vs 0.353).
- **2nd-order co-occurrence is the best method by far** (P@50 0.760 on all terms, 0.475 on phrases, and the best
  or tied-best method in 27 of 28 pairs on all terms). Terms that share many neighbours with a concept tend to meet
  it later. On a corpus of this size, the signal the paper attributes to embeddings is already present, and more
  directly usable, in raw co-occurrence counts.
- SPECTER2 (0.467) is above word2vec but below co-occurrence, despite having seen post-cutoff papers in pretraining."""),
    md("""**Cumulative view.** For the top 50 of each method, the share already found *n* years after T (n = 1 means
found in year T itself). Every curve rises because more time means more chances to co-occur; each point averages the
cutoffs that have that many years of follow-up, so the right end of each curve comes from the early cutoffs only.
Thin lines are word2vec per cutoff."""),
    code("display(Image(F / 'fig3a_all.png'))\ndisplay(Image(F / 'fig3a_phrases.png'))"),
    md("""**Observations**

- The ordering of the methods is the same at every horizon: co-occurrence > SPECTER2 ≈ frequency > word2vec
  output·input > random > word2vec input·input (all terms). On phrases word2vec and frequency overlap at every
  horizon.
- The gap between word2vec and random grows with time, as in the paper's Fig. 3a: the predictions are not only
  found sooner, more of them are found eventually."""),

    md(f"""### 3.4 Leakage check: with and without the version filter

arXiv metadata keeps only the **latest** version of each abstract. A paper first posted before T but revised after T
can therefore carry post-T text. The main run excludes such papers from training. This check reruns the whole
pipeline without that filter, to measure how much the leak would change the results.

{run(S + 'phase2_preprocess.py --skip-docs --no-version-filter && ' + S + 'phase3_train.py --variant all_versions && ' + S + 'phase3_backtest.py --variant all_versions && ' + S + 'phase3_report.py --variant all_versions')}"""),
    code("show_md('leakage_comparison.md')"),
    md("""**Observations**

- The filter removes 4.8–6.0 % of the pre-T abstracts per cutoff, and the same 28 pairs are scored in both runs.
- Without the filter every method loses about 0.01 P@50 on all terms: word2vec 0.279 → 0.269, frequency
  0.513 → 0.504, co-occurrence 0.760 → 0.747, SPECTER2 0.467 → 0.442, and **random as well**, 0.137 → 0.130.
- Since random also drops, the change comes from the candidate pool, not from what word2vec learned. A revised
  abstract that already mentions a later term next to the concept makes that term "already known", so it leaves the
  candidate list just before it would have been found. Relative to random, word2vec is unchanged: 2.04× with the
  filter, 2.07× without (2.49× and 2.44× on phrases). The snapshot holds no original versions, so this explanation
  cannot be checked directly.
- The difference (0.010) is larger than one seed's std (0.006), so the filter matters slightly, but **no conclusion
  changes**. The filtered run is the main result because it is the only one free of post-T text."""),

    md("""### 3.5 An example ranking

**How to read:** the top 20 phrases for learning to rank at T = 2020, from word2vec (seed 0) and from frequency.
✓ = later found, with the year of the third co-occurring abstract. This pair was chosen *after* all pairs were scored,
as a readable illustration; it is not a typical pair."""),
    code("show_md('hero_top20.md')"),
    md("""**Observations**

- word2vec's list is clearly about ranking and recommendation (`personalized_ranking`, `negative_sampling`,
  `ctr_prediction`, `bandit_feedback`), while frequency returns generic NLP phrases (`sentiment_analysis`,
  `neural_machine_translation`) that almost never meet learning to rank later.
- The example favours word2vec at the top: 7/20 found for seed 0, but 5/20 for the other two seeds. At P@50 the two
  methods tie (word2vec 0.27 averaged over seeds, frequency 0.28).
- Several misses are plausible but not found under the strict rule (`multi_armed_bandit` co-occurs in 2 abstracts,
  one short of the threshold)."""),

    # ------------------------------------------------------------------ 4. thermoelectric
    md(f"""## 4. The thermoelectric backtest, in the paper's own domain

The same procedure on arXiv materials science (`cond-mat.mtrl-sci`, 113,695 abstracts), with the paper's own
setup: target word *thermoelectric*, chemical formulas as candidates, 200 dimensions, yearly cutoffs 2010–2021
(arXiv materials science is too thin before 2010), and the paper's criterion for "found": at least one abstract
with a thermoelectric keyword. The question is how close a small public corpus gets to the paper's ≈ 8× random,
and how the frequency baseline, which the paper did not report, compares.

{run(S + 'te_backtest.py all')}"""),
    code("show_md('te_backtest_paper.md')"),
    code("display(Image(F / 'te_fig3a_paper.png'))"),
    md("""**Observations**

- word2vec reaches **3.5× random** (P@50 within 5 years: 0.228 against 0.066), below the paper's ≈ 8×. Per cutoff
  it ranges from 2.0× (T = 2010 and 2011) to 5.2× (T = 2016).
- **Frequency reaches 5.1×** (0.335) and is above word2vec at 10 of 12 cutoffs; word2vec wins at T = 2016
  (0.367 vs 0.34) and ties at T = 2020 (0.22).
- Beyond frequency, word2vec adds +0.120 over materials of the same frequency, more than frequency itself (+0.073):
  the embedding does carry information about which materials are related to thermoelectrics.
- Unlike in cs.IR, co-occurrence with known thermoelectrics is weak here (0.125), and input·input is again low
  (0.093). The paper's output·input score is the best of the non-trivial methods in its own domain.
- The most plausible reason for the gap with the paper is data: 114 k arXiv abstracts against the paper's 1.5 M
  journal abstracts, and arXiv is a biased sample of materials science."""),

    # ------------------------------------------------------------------ 5. timeline
    md(f"""## 5. Embedding timeline of cs.IR

Seven subtopics are defined by keyword lists (table below). For each subtopic and year, the abstracts that match its
keywords are embedded with SPECTER2 and averaged into one *centroid*; a centroid is kept only if it has ≥ 20
abstracts. One PCA is fitted on all centroids, so positions are comparable across years.

Similarities are cosines between centroids **after subtracting the mean cs.IR abstract vector**. Without centering,
all centroids have cosine 0.95–0.99 with each other, because all abstracts share a common direction. Centering does
not change the PCA plot.

{run(S + 'phase4_timeline.py')}"""),
    code("display(Image(F / 'timeline_specter2.png'))\ndisplay(Image(F / 'timeline_specter2_panels.png'))"),
    md("""**How to read:** each line is one subtopic, each dot is its centroid in one year, from the hollow dot (first
year with ≥ 20 abstracts) to the arrow (2026). The panels show one subtopic at a time with the others in grey. The
plane keeps 66 % of the variance, so distances in it are approximate; the tables below use the full space."""),
    code("show_md('timeline.md', start='| subtopic | keywords', stop='### Points per subtopic')"),
    code("""pts = pd.read_csv(R / 'timeline_centroids.csv').pivot(index='subtopic', columns='year', values='n')
display(Markdown('**Abstracts per subtopic and year** (— = fewer than 20, no point drawn)'))
display(pts.map(lambda v: '—' if pd.isna(v) else f'{int(v):,}'))"""),
    md("""**Observations**

- **Query expansion, dense retrieval and LLMs/RAG converge.** Query expansion ↔ dense retrieval goes from 0.49
  (2020) to 0.87 (2026), and query expansion ↔ LLMs & RAG from 0.45 (2022) to 0.78. This matches the recent history of
  the field, in which query rewriting is increasingly done with neural and LLM-based models.
- **Recommender systems moves towards LLM work** (−0.47 → 0.05: from opposed to neutral), and **learning to rank
  moves towards dense retrieval** (0.24 → 0.66), as ranking models became neural.
- **Conversational search** changes least (0.83 between its 2020 and 2026 centroids). It looks isolated in the
  plane, but in the full space it moves closer to recommender systems (0.03 → 0.48), partly because *conversational
  recommendation* is one of its keywords.
- **IR evaluation** changes most (0.11 between 2015 and 2026), and recommender systems also changes a lot (0.43
  over 2012–2026), consistent with a topic whose methods were replaced over the period.
- The first points of most subtopics rest on few abstracts (20–35), so year-to-year wiggles, such as dense retrieval ↔ LLMs & RAG dropping
  to 0.48 in 2023, should not be over-read. SPECTER2 was pretrained on recent papers, which may also make recent
  years look more coherent."""),
    md("""### 5.1 The same question with the backtest's word2vec models

**How to read:** each cutoff has its own word2vec model (seed 0), and the spaces are aligned between consecutive
cutoffs with orthogonal Procrustes (as in HistWords). *Self-similarity* is the cosine between a word's vector at one
cutoff and at the next: lower means its usage changed more. The median over all shared words is the reference.
The neighbour lists show the 8 closest terms at each cutoff."""),
    code("show_md('timeline.md', start='| concept |')"),
    md("""**Observations**

- `recommender` moves more than the median word up to 2020→2021 (0.81–0.86 against 0.89–0.90), then settles
  (0.91 and 0.94, at or above the median). This is consistent with the SPECTER2 result that recommender systems is one of the subtopics that
  changed most.
- The neighbours track the field without any supervision. `learning_to_rank` starts among generic ensemble methods
  (`bagging`, `adaboost`, 2016) and becomes surrounded by its own vocabulary (`listwise`, `letor`,
  `unbiased_learning_to_rank`) from 2020. `query_expansion` gains `pseudo_relevance_feedback` from 2019 and neural
  methods (`ceqe`, `deepimpact`) in 2022–2023. `retrieval` gains `dense_passage` (2022) and
  `dense_passage_retrieval` (2023)."""),

    # ------------------------------------------------------------------ conclusions
    md("""## 6. Conclusions

1. **The paper's method transfers to IR, but with a much weaker effect.** The output·input word2vec score ranks
   later co-occurring terms about 2× (all terms) to 2.5× (phrases) above random, compared with ≈ 8× in the paper.
2. **The comparison with random hides strong simple baselines.** Ranking by frequency beats word2vec on all terms in
   every pair, and second-order co-occurrence beats every method, including SPECTER2. On this corpus, what word2vec
   captures is mostly already in the co-occurrence counts.
3. **The embedding still adds real information beyond popularity**: +0.10 (all terms) and +0.13 (phrases) over
   terms of the same frequency, and +0.12 for thermoelectric materials. On phrases and on thermoelectrics this is
   more than frequency itself adds. The paper's choice of the output·input score is essential: the usual cosine is
   below random on all terms and well below the paper's score everywhere.
4. **In the paper's own domain, a small public corpus reaches 3.5× random, not 8×**, and frequency again does
   better on precision. The result is reproducible in direction but not in size; corpus size is the most likely
   cause.
5. **The results are robust:** all correctness checks pass, three seeds agree closely, and removing the
   version-leakage filter changes no conclusion.
6. **The timeline shows the expected convergence of query reformulation, dense retrieval and LLM/RAG work**, with
   recommender systems and learning to rank moving towards neural retrieval; word2vec neighbours across cutoffs tell
   the same story.

### Limitations

- The concepts are defined by hand-picked indicator lists; another choice of indicators changes which abstracts
  count as mentioning a concept.
- "Found" is a co-occurrence count (≥ 3 abstracts), not a judgement that the association is meaningful; generic
  words (`outperforms_state_of_the_art`) can be "found".
- The last cutoffs have only 3–5 years of follow-up (2026 is a partial year), so their precision is a lower bound.
- The training corpus (≤ 8.2 M tokens) is small for word2vec, which likely favours count-based methods.
- SPECTER2 is not temporally clean, both as a backtest method and for the timeline."""),

    # ------------------------------------------------------------------ appendix
    md(f"""## Appendix: correctness checks of the cs.IR backtest

Independent recomputations from the raw text (not from the matrices the backtest uses), to confirm that the
backtest has no access to the future and that its metric counts what it claims. The result of each check is read
from `reproduction/results/backtest_checks.md`, which also lists the full term-by-term recount.

{run(S + 'phase3_checks.py')}"""),
    code("""text = (R / 'backtest_checks.md').read_text()
sections = re.split(r'^## ', text, flags=re.M)[1:]
guards = {'1': 'the model has seen text from after the cutoff',
          '2': 'a "prediction" was already associated with the concept before the cutoff',
          '3': 'the "found" metric counts co-occurrences wrongly',
          '4': 'the random baseline is computed wrongly',
          '5': 'the three seeds are in fact the same model'}
rows = []
for s in sections:
    title = s.splitlines()[0]
    num, name = title.split('. ', 1)
    result = 'FAIL' if 'FAIL' in s else 'PASS' if 'PASS' in s else 'see file'
    rows.append({'check': name, 'rules out': guards.get(num, ''), 'result': result})
pd.DataFrame(rows).style.hide(axis='index')"""),
    md("""**Details:** no training abstract of any cutoff has a first or latest version after the cutoff; none of the
4,787 distinct top-50 terms appears in a pre-cutoff abstract with its concept; a recount from the raw text confirms
the recorded "found" status and year of 80 of 80 terms (the 20 of the example list and a random sample of 60);
random P@50 differs from the base rate by at most 0.0049; and the three seeds give different models (identical P@50
in 3 of 56 rows, mean std 0.027)."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(HERE)}})
nbf.write(nb, HERE / "reproduce.ipynb")
print("wrote", HERE / "reproduce.ipynb")
