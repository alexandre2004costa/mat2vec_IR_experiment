"""Shared code for the ir2 reproduction: text normalization, phrases, targets, per-cutoff artefacts.

Ideas reused from mat2vec (rewritten for gensim 4 / Python 3.12):
- numbers -> <nUm> (mat2vec.processing.process.MaterialsTextProcessor.process)
- two passes of gensim Phrases with connector words, phrases containing <nUm> removed
  (mat2vec.training.phrase2vec.wordgrams / exclude_words)
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np

IR2 = Path(__file__).resolve().parent
DATA, MODELS, RESULTS, FIGURES = IR2 / "data", IR2 / "models", IR2 / "results", IR2 / "figures"
NUM = "<nUm>"

CUTOFFS = list(range(2016, 2024))          # every T has >= 3 full future years (T .. 2026)
LAST_YEAR = 2026
SEEDS = [0, 1, 2]

# ---------------------------------------------------------------- text normalization
_URL = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|org|net|io)/\S*)")
_MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?")
_NUMBER = re.compile(r"(?<![a-z0-9])[-+]?\d+(?:[.,]\d+)*(?:\s?%)?(?![a-z0-9])")
_SENT = re.compile(r"[.!?;:]+(?:\s|$)")
_TOKEN = re.compile(r"<nUm>|[a-z0-9]+")


def clean(text):
    """Lowercase, strip accents, URLs, LaTeX math and commands; numbers -> <nUm>."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _URL.sub(" ", text)
    text = _MATH.sub(" ", text)
    text = _LATEX_CMD.sub(" ", text)
    text = text.replace("{", " ").replace("}", " ").lower()
    text = _NUMBER.sub(" <nUm> ", text)
    return text


def sentences(text):
    """Cleaned text -> list of token lists. Hyphens and other punctuation split tokens."""
    out = []
    for s in _SENT.split(clean(text)):
        toks = [NUM if t.isdigit() else t for t in _TOKEN.findall(s)]
        if toks:
            out.append(toks)
    return out


def doc_text(title, abstract):
    return f"{title}. {abstract}"


# ---------------------------------------------------------------- stopwords and candidate filter
def _stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    extra = {"nUm", NUM, "et", "al", "eg", "ie", "vs", "etc", "doi", "arxiv", "http", "https", "www"}
    return frozenset(ENGLISH_STOP_WORDS) | extra


STOPWORDS = _stopwords()


def is_candidate_term(tok):
    """Paper-analogue filter: no stopwords, numbers or tokens shorter than 3 characters.
    For a phrase, every part must pass except connector words in the middle."""
    if NUM in tok or len(tok) < 3:
        return False
    parts = tok.split("_")
    if parts[0] in STOPWORDS or parts[-1] in STOPWORDS:
        return False
    return not any(p.isdigit() for p in parts)


from gensim.models.phrases import ENGLISH_CONNECTOR_WORDS  # noqa: E402

CONNECTORS = ENGLISH_CONNECTOR_WORDS


# ---------------------------------------------------------------- phrases
def _keep_phrase(gram):
    """Drop phrases with <nUm> (as mat2vec) and phrases starting or ending with a stopword ("we_propose", "e_g")."""
    parts = gram.split("_")
    return NUM not in gram and parts[0] not in STOPWORDS and parts[-1] not in STOPWORDS and len(parts[0]) > 1 \
        and len(parts[-1]) > 1


def learn_phraser(sents, passes=2, min_count=10, threshold=15.0):
    """Two passes of Phrases (bigrams, then up to 4-grams), as in mat2vec (min_count 10, threshold 15).
    Returns a list of FrozenPhrases, applied in order. Phrases containing <nUm> are dropped (mat2vec exclude_words)."""
    from gensim.models.phrases import Phrases
    layers = []
    cur = sents
    for _ in range(passes):
        p = Phrases(cur, min_count=min_count, threshold=threshold, connector_words=CONNECTORS,
                    delimiter="_").freeze()
        p.phrasegrams = {k: v for k, v in p.phrasegrams.items() if _keep_phrase(k)}
        layers.append(p)
        cur = [p[s] for s in cur]
    return layers


def apply_phraser(layers, sent):
    for p in layers:
        sent = p[sent]
    return sent


# ---------------------------------------------------------------- targets
# Indicator sets (like the paper's thermoelectric keywords). Matched on the *unphrased* normalized text with
# word boundaries, so detection does not depend on whether the pre-T phraser learned the phrase.
TARGETS = {
    "query_expansion": ["query expansion", "query reformulation", "query rewriting", "relevance feedback"],
    "recommender_systems": ["recommender", "recommenders", "recommendation", "recommendations",
                            "collaborative filtering"],
    "learning_to_rank": ["learning to rank", "ranknet", "lambdamart", "lambdarank", "listwise"],
    "dense_retrieval": ["dense retrieval", "dense retriever", "dense retrievers", "dense passage retrieval",
                        "dual encoder", "dual encoders", "bi encoder", "bi encoders"],
    "conversational_search": ["conversational search", "conversational recommendation",
                              "conversational recommender", "clarifying question", "clarifying questions",
                              "conversational information seeking"],
    "retrieval_augmented_generation": ["retrieval augmented", "retrieval enhanced generation", "rag"],
}


class IndicatorMatcher:
    """Does a normalized document contain one of the indicators as whole tokens?
    Documents are sentences joined by "\\n", tokens by " ", tokens only [a-z0-9<>], so \\b on the alphanumeric
    indicator edges equals a token boundary, and an indicator never spans two sentences (no "\\n" inside it).
    A plain substring test runs first (C speed); the regex only confirms the token boundaries."""

    def __init__(self, indicators):
        self.literals = [" ".join(s) for i in indicators for s in sentences(i)[:1]]
        alts = sorted((re.escape(x) for x in self.literals), key=len, reverse=True)
        self.rx = re.compile(r"\b(?:" + "|".join(alts) + r")\b")

    def search(self, text):
        if any(x in text for x in self.literals) and self.rx.search(text):
            return True
        return None


def indicator_regex(indicators):
    return IndicatorMatcher(indicators)


def indicator_tokens(indicators):
    """Possible single tokens for each indicator in a phrased vocabulary: 'query expansion' -> 'query_expansion'."""
    return ["_".join(sentences(i)[0]) for i in indicators]


def target_vector(kv, indicators, out=None):
    """Mean of the normalized input vectors of the indicator tokens present in the vocabulary.
    Returns (vector or None, list of tokens used)."""
    toks = [t for t in indicator_tokens(indicators) if t in kv.key_to_index]
    if not toks:
        return None, []
    v = np.mean([kv.get_vector(t, norm=True) for t in toks], axis=0)
    return v / np.linalg.norm(v), toks


# ---------------------------------------------------------------- per-cutoff artefacts
def cutoff_dir(T):
    return MODELS / f"T{T}"


@lru_cache(maxsize=None)
def load_docs():
    """All documents (cs.IR ∪ cs.CL) with normalized unphrased text. Built by phase2_preprocess.py."""
    import pandas as pd
    return pd.read_parquet(DATA / "docs.parquet")
