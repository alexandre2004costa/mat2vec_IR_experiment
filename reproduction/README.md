# Reproducing the backtest and timeline


## Backtest
The reproduction adapts the paper's historical backtest to IR concepts: can terms ranked near a concept before a given year anticipate associations that appear later in the literature?

The notebook, [reproduce.ipynb](../reproduce.ipynb), displays the saved results and figures.


### Instruction to reproduce

1. Install dependencies

```powershell
python -m pip install -r pairs/12-mat2vec/requirements.txt
```

2. Download data

Dowload the Cornell-University/arxiv metadata archive from Kaggle. Our results use the snapshot dated 2026-11-08. Of course, using a newer snapshot will result in differences in the corpus. 

3. Replace the archive path in the first command with your own, and run every command.

```powershell
python pairs/12-mat2vec/reproduction/scripts/phase1_corpus.py --zip "C:/path/to/archive.zip"
python pairs/12-mat2vec/reproduction/scripts/phase2_preprocess.py
python pairs/12-mat2vec/reproduction/scripts/phase3_train.py
python pairs/12-mat2vec/reproduction/scripts/phase3_encode.py
python pairs/12-mat2vec/reproduction/scripts/phase3_backtest.py
python pairs/12-mat2vec/reproduction/scripts/phase3_report.py
python pairs/12-mat2vec/reproduction/scripts/phase3_checks.py
python pairs/12-mat2vec/reproduction/scripts/phase4_timeline.py
```

> Note: the encoding downloads `allenai/specter2_base` and can take substantial time on CPU.

### Cutoff information
For each cutoff from 2016 to 2023, training uses ``cs.IR`` and ``cs.CL`` abstracts published before that year. Furthermore, we also require the latest version to precede the cutoff. Otherwise, an old paper could bring later revisions into the training text. Future associations are counted only in ``cs.IR``.

### Preprocessing information
Preprocessing lowercases the text, removes accents, URLs and LaTeX, normalizes numbers, and learns phrases in two passes on the pre-cutoff text. Phrases containing *numbers* or starting or ending with *stopwords* or *tokens with one letter* are excluded.

### Skip-gram information
Word2Vec uses Skip-gram with: 
- 100 dimensions
- window of 8
- 15 negative samples
- minimum count of 5
- subsampling at 1e-4
- initial learning rate of 0.01
- 30 epochs

Results use three seeds.

### Concept information
A concept is only evaluated if it:
- occurs in at least 20 pre-cutoff ``cs.IR`` abstracts
- has an indicator token in the vocabulary
- leaves at least 50 candidates

### Candidate information
Candidates are terms seen before the cutoff that never shared a training abstract with the concept. A candidate counts as found if it appears with the concept in at least three ``cs.IR`` abstracts from the cutoff onward. We evaluate both all eligible terms and multiword terms alone.

### Ranking information
The candidates are ranked using the paper’s score: the dot product between a candidate’s normalized output vector and the concept’s input vector. 

We measure precision@10, @20 and @50 and compare the results with frequency, random rankings, input-vector cosine, second-order co-occurrence and SPECTER2. The random baseline averages 1000 draws.

### Timeline information
To build the timeline, we group abstracts by subtopic and year, and average their SPECTER2 vectors. We only include a group if it has at least 20 abstracts. Afterwards, each average becomes a point on the plot, and we use one PCA projection for all years so we can follow how the subtopics move over time.

Furthemore, keywords determine which subtopics an abstract belongs to, so an abstract can appear in more than one group.

Additionally, when calculating similarities, we subtract the average vector of all `cs.IR` abstracts to reduce what they have in common.

Of course, one limitations may arise, because SPECTER2 was pretrained on recent papers, so it is not a temporally clean baseline.



## Thermoelectric backtest (the paper's own domain)

The same backtest on arXiv materials science (`cond-mat.mtrl-sci`), with the paper's settings (200 dimensions, cutoffs 2010–2021, found = one abstract with a thermoelectric keyword within 5 years). It reads the same `archive.zip`, which must be in the repository root:

```powershell
python pairs/12-mat2vec/reproduction/scripts/te_backtest.py all
```

## Rebuilding the notebook

```powershell
python pairs/12-mat2vec/reproduction/scripts/make_notebook.py
```

## Instruction to reproduce backtest with updated abstracts

To understand how does the ranking change when the papers before the cutoff contain updated abstracts, run:

```powershell
python pairs/12-mat2vec/reproduction/scripts/phase2_preprocess.py --skip-docs --no-version-filter
python pairs/12-mat2vec/reproduction/scripts/phase3_train.py --variant all_versions
python pairs/12-mat2vec/reproduction/scripts/phase3_backtest.py --variant all_versions
python pairs/12-mat2vec/reproduction/scripts/phase3_report.py --variant all_versions
```
