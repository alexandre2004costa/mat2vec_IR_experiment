"""Small-scale reproduction of the paper's key result in its own domain: the thermoelectric backtest (Fig. 3a) on
arXiv cond-mat.mtrl-sci abstracts.

For each cutoff T: train word2vec on abstracts before T, rank chemical formulas never mentioned in an abstract with a
thermoelectric keyword before T, and check which were mentioned with one from T on (the paper's criterion: at least
one abstract), within 5 years and up to the snapshot.

Usage: ir2/.venv/bin/python ir2/scripts/te_backtest.py corpus|prep|train|eval|report|all [--setup small|paper]
  --setup small (default): 100 dimensions, cutoffs 2010, 2012, ..., 2020 (the first run)
  --setup paper: the paper's 200 dimensions and yearly cutoffs 2010-2021 (the paper used 2001-2018 yearly;
                 arXiv materials science is too thin before 2010); results get the suffix "_paper"
Writes ir2/data/te_docs.parquet, ir2/models/te/T{T}/*, ir2/results/te_*{_paper}.{csv,md},
       ir2/figures/te_fig3a{_paper}.png
"""
import argparse
import io
import json
import pickle
import re
import sys
import time
import unicodedata
import zipfile
from collections import Counter
from email.utils import parsedate_to_datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

CATEGORY = "cond-mat.mtrl-sci"
OUT = L.MODELS / "te"
CUTOFFS = [2010, 2012, 2014, 2016, 2018, 2020]     # each has >= 5 future years in the snapshot (to 2026)
SEEDS = [0, 1, 2]
HORIZON = 5                                         # the paper's "within 5 years"
INDICATORS = {"thermoelectric", "thermoelectrics", "thermoelectricity"}
TARGET = "thermoelectric"                           # the paper scores against the single word
PARAMS = dict(sg=1, negative=15, window=8, sample=1e-4, min_count=5, alpha=0.01, workers=9)
SETUPS = {"small": {"cutoffs": CUTOFFS, "dim": 100, "tag": ""},
          "paper": {"cutoffs": list(range(2010, 2022)), "dim": 200, "tag": "_paper"}}   # T=2021: 2021-2025 window

# ---------------------------------------------------------------- formulas (as in phase0_sanity.py)
ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm").split()
EL = "|".join(sorted(ELEMENTS, key=len, reverse=True))
FORMULA = re.compile(rf"^(?:(?:{EL})(?:\d+(?:\.\d+)?)?){{2,}}$")
PART = re.compile(rf"({EL})(\d+(?:\.\d+)?)?")
SINGLE = {e for e in ELEMENTS if len(e) == 1}


def normalized_formula(tok):
    """Simple formula -> elements sorted alphabetically with their counts (mat2vec: LiCoO2 -> CoLiO2), else None.
    Rejects repeated elements and all-caps acronyms made of one-letter elements without digits (SCF, BCS, CPU)."""
    if not FORMULA.match(tok):
        return None
    parts = PART.findall(tok)
    els = [e for e, _ in parts]
    if len(set(els)) != len(els):
        return None
    if all(e in SINGLE for e in els) and not any(n for _, n in parts) and len(els) >= 3:
        return None
    return "".join(e + n for e, n in sorted(parts))


# ---------------------------------------------------------------- text
_LATEX_SUB = re.compile(r"\$?_\{?(\d+(?:\.\d+)?)\}?\$?")       # Bi$_2$Te$_{3}$ -> Bi2Te3
_URL = re.compile(r"https?://\S+|www\.\S+")
_SENT = re.compile(r"(?<=[a-z0-9)\]])[.!?;:]\s+|\n+")
_TOK = re.compile(r"[A-Za-z0-9]+(?:\.\d+[A-Za-z0-9]*)*")
_NUM = re.compile(r"^\d+(?:\.\d+)?$")


def te_sentences(text):
    """Case-preserving tokenization: formulas normalized, numbers -> <nUm>, other words lowercased."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _URL.sub(" ", text)
    text = _LATEX_SUB.sub(r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+|[${}\\]", " ", text)
    out = []
    for s in _SENT.split(text):
        toks = []
        for t in _TOK.findall(s):
            if _NUM.match(t):
                toks.append(L.NUM)
                continue
            f = normalized_formula(t)
            toks.append(f if f else t.lower())
        if toks:
            out.append(toks)
    return out


# ---------------------------------------------------------------- steps
def step_corpus(args):
    rows = []
    with zipfile.ZipFile(L.IR2.parent / "archive.zip") as z, z.open("arxiv-metadata-oai-snapshot.json") as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            if CATEGORY not in line:
                continue
            d = json.loads(line)
            if CATEGORY not in d["categories"].split():
                continue
            rows.append({"id": d["id"], "year": parsedate_to_datetime(d["versions"][0]["created"]).year,
                         "last_version_year": parsedate_to_datetime(d["versions"][-1]["created"]).year,
                         "primary": d["categories"].split()[0] == CATEGORY,
                         "text": " ".join((d["title"] + ". " + d["abstract"]).split())})
    df = pd.DataFrame(rows).sort_values(["year", "id"]).reset_index(drop=True)
    with Pool(8) as pool:
        sents = pool.map(te_sentences, df["text"].tolist(), chunksize=500)
    df["tokens"] = ["\n".join(" ".join(s) for s in ss) for ss in sents]
    df["te"] = [any(t in INDICATORS for s in ss for t in s) for ss in sents]
    df.to_parquet(L.DATA / "te_docs.parquet", index=False)
    t = df.groupby("year").agg(papers=("id", "size"), thermoelectric=("te", "sum"))
    t.to_csv(L.RESULTS / "te_counts.csv")
    print(f"{len(df):,} {CATEGORY} papers; {int(df.te.sum()):,} with a thermoelectric keyword")
    print(t.loc[2000:].to_string())


def doc_sents(text):
    return [s.split(" ") for s in text.split("\n") if s]


def prep_cutoff(T):
    df = pd.read_parquet(L.DATA / "te_docs.parquet", columns=["year", "last_version_year", "tokens", "te"])
    train = np.flatnonzero((df.year.values < T) & (df.last_version_year.values < T))    # no revised-abstract leak
    post = np.flatnonzero(df.year.values >= T)
    toks = df.tokens.values
    d = OUT / f"T{T}"
    d.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with open(d / "train.txt", "w") as f:
        for r in train:
            for s in doc_sents(toks[r]):
                f.write(" ".join(s) + "\n")
                counts.update(s)
    # candidate materials: formula tokens (they keep their capitals), seen >= 5 times before T
    mats = sorted((t for t, c in counts.items() if c >= 5 and t != t.lower() and normalized_formula(t)),
                  key=lambda t: -counts[t])
    index = {t: i for i, t in enumerate(mats)}

    def matrix(rows):
        indptr, indices = [0], []
        for r in rows:
            indices.extend(sorted({index[t] for s in doc_sents(toks[r]) for t in s if t in index}))
            indptr.append(len(indices))
        return sparse.csr_matrix((np.ones(len(indices), np.int8), indices, indptr), shape=(len(rows), len(mats)))

    sparse.save_npz(d / "M_train.npz", matrix(train))
    sparse.save_npz(d / "M_post.npz", matrix(post))
    pd.DataFrame({"material": mats, "count": [counts[t] for t in mats]}).to_parquet(d / "materials.parquet")
    np.savez(d / "rows.npz", train=train, post=post)
    n_tok = sum(counts.values())
    return f"T={T}: {len(train):,} train abstracts ({n_tok:,} tokens), {len(post):,} post, {len(mats):,} materials"


def step_prep(args):
    with Pool(3) as pool:
        for m in pool.map(prep_cutoff, args.cutoffs):
            print(m, flush=True)


def step_train(args):
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    for T in args.cutoffs:
        for s in SEEDS:
            d = OUT / f"T{T}"
            name = f"w2v{args.tag}_s{s}"
            if (d / f"{name}.kv").exists():
                continue
            t0 = time.time()
            m = Word2Vec(LineSentence(str(d / "train.txt")), epochs=30, seed=s, vector_size=args.dim, **PARAMS)
            m.wv.save(str(d / f"{name}.kv"))
            np.save(d / f"{name}.syn1neg.npy", m.syn1neg)
            print(f"T={T} seed={s} dim={args.dim}: {time.time() - t0:.0f}s", flush=True)


def step_eval(args):
    from gensim.models import KeyedVectors
    df = pd.read_parquet(L.DATA / "te_docs.parquet", columns=["year", "te"])
    rows, curves, tops = [], [], []
    rng = np.random.default_rng(0)
    for T in args.cutoffs:
        d = OUT / f"T{T}"
        mats = pd.read_parquet(d / "materials.parquet")
        r = np.load(d / "rows.npz")
        Mtr, Mpo = sparse.load_npz(d / "M_train.npz"), sparse.load_npz(d / "M_post.npz").tocsr()
        te_tr, te_po = df.te.values[r["train"]], df.te.values[r["post"]]
        py = df.year.values[r["post"]]
        known = np.asarray(Mtr[te_tr].sum(0)).ravel() > 0
        cand = np.flatnonzero(~known)
        first = np.zeros(len(mats), int)
        for y in range(T, L.LAST_YEAR + 1):
            m = te_po & (py == y)
            if m.any():
                hit = (np.asarray(Mpo[m].sum(0)).ravel() > 0) & (first == 0)
                first[hit] = y
        fc = first[cand]
        found5 = (fc > 0) & (fc < T + HORIZON)
        common = {"T": T, "n_materials": len(mats), "n_known": int(known.sum()), "n_candidates": len(cand),
                  "base_rate_5y": float(found5.mean()), "base_rate_all": float((fc > 0).mean())}
        # count baseline (2nd order, abstract level): cosine between a material's co-occurrence counts with the other
        # materials and the counts of materials appearing in pre-T thermoelectric abstracts
        Mtr_f = Mtr.astype(np.float32).tocsc()
        g = Mtr_f.T @ te_tr.astype(np.float32)                        # target context: materials in TE abstracts
        num = Mtr_f.T @ (Mtr_f @ g)
        norms = np.sqrt(np.asarray((Mtr_f.T @ Mtr_f).multiply(Mtr_f.T @ Mtr_f).sum(0)).ravel())
        scores = {("frequency", 0): mats["count"].values.astype(float),
                  ("cooc2", 0): num / np.maximum(norms * np.linalg.norm(g), 1e-9)}
        for s in SEEDS:
            kv = KeyedVectors.load(str(d / f"w2v{args.tag}_s{s}.kv"))
            syn1 = np.load(d / f"w2v{args.tag}_s{s}.syn1neg.npy")
            idx = np.array([kv.key_to_index[t] for t in mats.material])
            tv = kv.get_vector(TARGET, norm=True)
            out = syn1[idx] / np.linalg.norm(syn1[idx], axis=1, keepdims=True)
            scores[("w2v_out", s)] = out @ tv
            scores[("w2v_in", s)] = kv.get_normed_vectors()[idx] @ tv
        # frequency-matched expectation: 5-year found rate within pre-T count deciles of the candidates
        logc = np.log(mats["count"].values[cand])
        bins = np.searchsorted(np.quantile(logc, np.linspace(0, 1, 11)[1:-1]), logc, side="right")
        rate = np.array([found5[bins == b].mean() if (bins == b).any() else 0 for b in range(10)])
        for (method, s), sc in scores.items():
            order = np.lexsort((np.arange(len(cand)), -sc[cand]))
            top = order[:50]
            rows.append({**common, "method": method, "seed": s, "p50_5y": float(found5[top].mean()),
                         "p50_5y_freq_matched": float(rate[bins[top]].mean()),
                         "p50_all": float((fc[top] > 0).mean()), "p20_5y": float(found5[order[:20]].mean())})
            curves += [{"T": T, "method": method, "seed": s, "n": n,
                        "frac": float(((fc[top] > 0) & (fc[top] < T + n)).mean())} for n in range(1, L.LAST_YEAR - T + 2)]
            if s == 0:
                tops += [{"T": T, "method": method, "rank": k + 1, "material": mats.material.values[cand[j]],
                          "pre_T_count": int(mats["count"].values[cand[j]]), "first_year": int(fc[j]) or None}
                         for k, j in enumerate(order[:50])]
        draws = [found5[rng.permutation(len(cand))[:50]].mean() for _ in range(1000)]
        rows.append({**common, "method": "random", "seed": 0, "p50_5y": float(np.mean(draws)),
                     "p50_all": float((fc > 0).mean()), "p20_5y": float(found5.mean())})
        curves += [{"T": T, "method": "random", "seed": 0, "n": n, "frac": float(((fc > 0) & (fc < T + n)).mean())}
                   for n in range(1, L.LAST_YEAR - T + 2)]
        print(f"T={T}: {len(mats):,} materials, {int(known.sum()):,} already thermoelectric, {len(cand):,} candidates, "
              f"base rate (5 y) {found5.mean():.3f}", flush=True)
    pd.DataFrame(rows).to_csv(L.RESULTS / f"te_backtest{args.tag}.csv", index=False)
    pd.DataFrame(curves).to_csv(L.RESULTS / f"te_curves{args.tag}.csv", index=False)
    pd.DataFrame(tops).to_csv(L.RESULTS / f"te_top{args.tag}.csv", index=False)


def step_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = pd.read_csv(L.RESULTS / f"te_backtest{args.tag}.csv")
    c = pd.read_csv(L.RESULTS / f"te_curves{args.tag}.csv")
    labels = {"w2v_out": "word2vec output·input (paper)", "w2v_in": "word2vec input·input",
              "cooc2": "co-occurrence with known thermoelectrics", "frequency": "frequency",
              "random": "random (1000 draws)"}
    md = [f"# Thermoelectric backtest on arXiv cond-mat.mtrl-sci (small-scale reproduction of Fig. 3a), "
          f"setup `{args.setup}`: {args.dim} dimensions, cutoffs {', '.join(map(str, args.cutoffs))}", "",
          "Candidates: chemical formulas seen ≥ 5 times before T and never in an abstract with "
          "thermoelectric/thermoelectrics/thermoelectricity before T. Found: in ≥ 1 abstract with such a keyword "
          f"from T on (the paper's criterion). P@50 (5 y) = share of the top 50 found within {HORIZON} years "
          "(T … T+4). word2vec: mean ± std over 3 seeds.", "",
          "| T | materials | already TE | candidates | base rate (5 y) |", "|---|:-:|:-:|:-:|:-:|"]
    for T, g in r.groupby("T"):
        x = g.iloc[0]
        md.append(f"| {T} | {x.n_materials:,} | {x.n_known:,} | {x.n_candidates:,} | {x.base_rate_5y:.3f} |")
    md += ["", "| method | P@20 (5 y) | P@50 (5 y) | × random (P@50, 5 y) | P@50 frequency-matched | P@50 − freq-matched | P@50 (to 2026) |",
           "|---|:-:|:-:|:-:|:-:|:-:|:-:|"]
    rand = r[r.method == "random"].p50_5y.mean()
    for m in ["w2v_out", "w2v_in", "cooc2", "frequency", "random"]:
        x = r[r.method == m].groupby("seed")[["p20_5y", "p50_5y", "p50_all", "p50_5y_freq_matched"]].mean()
        x["lift"] = x["p50_5y"] - x["p50_5y_freq_matched"]
        mu, sd = x.mean(), x.std()
        f = (lambda k: f"{mu[k]:.3f} ± {sd[k]:.3f}") if len(x) > 1 else (lambda k: f"{mu[k]:.3f}")
        fm = (lambda k: "—") if m == "random" else f
        md.append(f"| {labels[m]} | {f('p20_5y')} | {f('p50_5y')} | {mu['p50_5y'] / rand:.1f}× | "
                  f"{fm('p50_5y_freq_matched')} | {fm('lift')} | {f('p50_all')} |")
    md += ["", "Per cutoff, P@50 within 5 years (word2vec: mean of 3 seeds):", "",
           r.groupby(["T", "method"]).p50_5y.mean().unstack("method").round(3).to_markdown(), ""]
    t = pd.read_csv(L.RESULTS / f"te_top{args.tag}.csv")
    for T in r["T"].unique()[-3:-2]:
        x = t[(t["T"] == T) & (t.method == "w2v_out")].head(20)
        md += [f"Top 20 of the paper's scoring at T = {T} (seed 0): " + ", ".join(
            f"`{a.material}`" + (f" ✓{int(a.first_year)}" if a.first_year == a.first_year else " ✗")
            for a in x.itertuples()), ""]
    (L.RESULTS / f"te_backtest{args.tag}.md").write_text("\n".join(md))
    print("\n".join(md))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    w = c[c.method == "w2v_out"].groupby(["T", "n"]).frac.mean().reset_index()
    for T, g in w.groupby("T"):
        ax.plot(g.n, 100 * g.frac, color="#1f4e9c", lw=0.8, alpha=0.45)
        ax.annotate(str(T), (g.n.iloc[-1], 100 * g.frac.iloc[-1]), fontsize=7, color="#1f4e9c",
                    xytext=(2, 0), textcoords="offset points", va="center")
    cols = {"w2v_out": "#1f4e9c", "w2v_in": "#6f9fd8", "cooc2": "#e76f51", "frequency": "#7a7a7a", "random": "#b5b5b5"}
    for m, col in cols.items():
        g = c[c.method == m].groupby(["T", "n"]).frac.mean().groupby("n").agg(["mean", "count"])
        g = g[g["count"] >= 3]
        ax.plot(g.index, 100 * g["mean"], color=col, lw=3 if m == "w2v_out" else 1.8,
                ls="-" if m.startswith("w2v") else "--", label=labels[m] + (" (mean)" if m == "w2v_out" else ""))
    ax.set_xlabel("years after cutoff T (n = 1 means found in year T)")
    ax.set_ylabel("% of top-50 materials studied as thermoelectrics")
    ax.set_title(f"Thermoelectric backtest, arXiv {CATEGORY}, cutoffs {min(args.cutoffs)}–{max(args.cutoffs)}\n"
                 f"thin lines: word2vec output·input per cutoff (mean of 3 seeds), {args.dim} dimensions", fontsize=10)
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(L.FIGURES / f"te_fig3a{args.tag}.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["corpus", "prep", "train", "eval", "report", "all"])
    ap.add_argument("--setup", choices=list(SETUPS), default="small")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=None)
    args = ap.parse_args()
    setup = SETUPS[args.setup]
    args.cutoffs = args.cutoffs or setup["cutoffs"]
    args.dim, args.tag = setup["dim"], setup["tag"]
    steps = ["corpus", "prep", "train", "eval", "report"] if args.step == "all" else [args.step]
    for s in steps:
        globals()[f"step_{s}"](args)


if __name__ == "__main__":
    main()
