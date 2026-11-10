"""Phase 0: check three claims of Tshitoyan et al. (2019) with the authors' pretrained mat2vec embeddings.

Usage: ir2/.venv/bin/python ir2/scripts/phase0_sanity.py
Writes ir2/results/phase0_sanity.md. Nothing here is trained; it only loads and queries.
"""
import re
import sys
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

IR2 = Path(__file__).resolve().parents[1]
MODEL = IR2 / "models" / "mat2vec_pretrained" / "pretrained_embeddings"
OUT = IR2 / "results" / "phase0_sanity.md"

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm").split()
# A "simple formula" token: >= 2 element symbols, each with an optional count, nothing else.
# This replaces mat2vec's pymatgen-based is_simple_formula (pymatgen/chemdataextractor are not installed).
EL = "|".join(sorted(ELEMENTS, key=len, reverse=True))
FORMULA = re.compile(rf"^(?:(?:{EL})(?:\d+(?:\.\d+)?)?){{2,}}$")
PART = re.compile(rf"({EL})(\d+(?:\.\d+)?)?")


def is_formula(tok):
    if not FORMULA.match(tok):
        return False
    els = [m.group(1) for m in PART.finditer(tok)]
    return len(set(els)) == len(els) >= 2      # distinct elements, so "CoCo" or plain words are rejected


def normalized(formula):
    """mat2vec normalizes formulae by sorting elements alphabetically (README: LiCoO2 -> CoLiO2)."""
    parts = sorted(PART.findall(formula))
    return "".join(e + n for e, n in parts)


def table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def main():
    if not MODEL.exists():
        sys.exit(f"missing {MODEL}: download it first (see NOTES.md, Phase 0)")
    m = Word2Vec.load(str(MODEL))
    wv = m.wv
    out = [f"# Phase 0: sanity check with the pretrained mat2vec embeddings",
           "",
           f"Model: `pretrained_embeddings` from storage.googleapis.com/mat2vec, loaded with gensim "
           f"(vocabulary {len(wv.index_to_key):,} tokens, dimension {wv.vector_size}). "
           f"Formulae are normalized as in mat2vec (elements sorted alphabetically) before lookup.",
           ""]

    # 1. Nearest neighbours of LiCoO2
    q = normalized("LiCoO2")
    out += [f"## 1. Nearest neighbours of LiCoO2 (normalized `{q}`)", ""]
    if q in wv:
        nn = wv.most_similar(q, topn=15)
        out.append(table([(i + 1, f"`{w}`", f"{s:.3f}") for i, (w, s) in enumerate(nn)], ["rank", "token", "cosine"]))
    else:
        out.append(f"`{q}` is not in the vocabulary.")
    out.append("")

    # 2. Analogy ferromagnetic - NiFe + IrMn
    a, b = normalized("NiFe"), normalized("IrMn")
    out += [f"## 2. Analogy: ferromagnetic − NiFe + IrMn (normalized `{a}`, `{b}`)", ""]
    missing = [t for t in ("ferromagnetic", a, b) if t not in wv]
    if missing:
        out.append(f"Missing from the vocabulary: {missing}.")
    else:
        res = wv.most_similar(positive=["ferromagnetic", b], negative=[a], topn=10)
        out.append(table([(i + 1, f"`{w}`", f"{s:.3f}") for i, (w, s) in enumerate(res)], ["rank", "token", "cosine"]))
        hit = [i + 1 for i, (w, _) in enumerate(res) if w == "antiferromagnetic"]
        out += ["", f"`antiferromagnetic` rank in top 10: {hit[0] if hit else 'not in top 10'}."]
    out.append("")

    # 3. Ranking of materials for "thermoelectric"
    out += ["## 3. Materials ranked for `thermoelectric`", "",
            "Candidates: every vocabulary token that parses as a simple formula of ≥2 distinct elements "
            "(a regex stand-in for mat2vec's pymatgen check). Two scores are shown: input·input cosine "
            "(`most_similar`), and the paper's score, the normalized **output** embedding (`syn1neg`) of the "
            "material dotted with the normalized **input** embedding of `thermoelectric`.", "",
            "Caveat: the paper also removes materials that already co-occur with thermoelectric keywords in the "
            "corpus. The corpus is not distributed, so this filter cannot be applied here: this list mixes known "
            "and predicted thermoelectrics.", ""]
    keys = wv.index_to_key
    mats = [i for i, w in enumerate(keys) if is_formula(w)]
    t = wv.get_vector("thermoelectric", norm=True)
    W_in = wv.get_normed_vectors()[mats]
    syn1 = m.syn1neg[mats]
    W_out = syn1 / np.linalg.norm(syn1, axis=1, keepdims=True)
    s_in, s_out = W_in @ t, W_out @ t
    top_in, top_out = np.argsort(-s_in)[:20], np.argsort(-s_out)[:20]
    rows = [(r + 1, f"`{keys[mats[i]]}`", f"{s_in[i]:.3f}", f"`{keys[mats[j]]}`", f"{s_out[j]:.3f}")
            for r, (i, j) in enumerate(zip(top_in, top_out))]
    out += [f"{len(mats):,} formula candidates in the vocabulary.", "",
            table(rows, ["rank", "input·input", "cos", "output·input (paper)", "score"]), ""]
    known = ["Bi2Te3", "PbTe", "SnSe", "Sb2Te3", "CoSb3", "Mg2Si"]
    ranks_out = {keys[mats[j]]: r + 1 for r, j in enumerate(np.argsort(-s_out))}
    out += ["Rank (output·input) of a few textbook thermoelectrics: " +
            ", ".join(f"{k} → {ranks_out.get(normalized(k), 'not in vocab')}" for k in known) + ".", ""]

    OUT.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
