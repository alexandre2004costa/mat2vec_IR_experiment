"""Phase 1: dated cs.IR and cs.CL corpora from the Kaggle arXiv metadata snapshot.

Streams arxiv-metadata-oai-snapshot.json from inside archive.zip line by line (never loaded or extracted).
Usage: ir2/.venv/bin/python ir2/scripts/phase1_corpus.py [--zip archive.zip] [--limit N]
Writes ir2/data/csir.parquet, ir2/data/cscl.parquet, ir2/results/phase1_counts.{csv,md}, ir2/figures/phase1_counts.png
"""
import argparse
import io
import json
import zipfile
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd

IR2 = Path(__file__).resolve().parents[1]
ROOT = IR2.parent
SETS = {"csir": "cs.IR", "cscl": "cs.CL"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "archive.zip"))
    ap.add_argument("--limit", type=int, default=0, help="stop after N lines (test on a sample)")
    args = ap.parse_args()

    rows = {k: [] for k in SETS}
    n = 0
    with zipfile.ZipFile(args.zip) as z, z.open("arxiv-metadata-oai-snapshot.json") as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            n += 1
            if args.limit and n > args.limit:
                break
            if '"cs.IR' not in line and '"cs.CL' not in line and " cs.IR" not in line and " cs.CL" not in line:
                continue                                   # fast pre-filter before json parsing
            d = json.loads(line)
            cats = d["categories"].split()
            for key, cat in SETS.items():
                if cat in cats:
                    year = parsedate_to_datetime(d["versions"][0]["created"]).year
                    # the snapshot stores the abstract of the LATEST version: keep its year to avoid leakage
                    last_year = parsedate_to_datetime(d["versions"][-1]["created"]).year
                    rows[key].append({"id": d["id"], "year": year, "last_version_year": last_year,
                                      "title": " ".join(d["title"].split()),
                                      "abstract": " ".join(d["abstract"].split()),
                                      "categories": d["categories"],
                                      "primary": cats[0] == cat})
            if n % 500000 == 0:
                print(f"{n:,} lines, cs.IR {len(rows['csir']):,}, cs.CL {len(rows['cscl']):,}", flush=True)
    print(f"done: {n:,} lines")

    (IR2 / "data").mkdir(exist_ok=True)
    suffix = "_sample" if args.limit else ""
    dfs = {}
    for key in SETS:
        df = pd.DataFrame(rows[key]).sort_values(["year", "id"]).reset_index(drop=True)
        df.to_parquet(IR2 / "data" / f"{key}{suffix}.parquet", index=False)
        dfs[key] = df
        print(key, len(df), "papers;", int(df["primary"].sum()), "primary")
    if args.limit:
        return

    # counts per year
    ir, cl = dfs["csir"], dfs["cscl"]
    both = set(ir["id"]) & set(cl["id"])
    t = pd.DataFrame({
        "cs.IR (any)": ir.groupby("year").size(),
        "cs.IR (primary)": ir[ir["primary"]].groupby("year").size(),
        "cs.CL (any)": cl.groupby("year").size(),
        "cs.CL not cs.IR": cl[~cl["id"].isin(both)].groupby("year").size(),
    }).fillna(0).astype(int)
    t.index.name = "year"
    (IR2 / "results").mkdir(exist_ok=True)
    t.to_csv(IR2 / "results" / "phase1_counts.csv")
    md = ["# Phase 1: papers per year (year of first version)", "",
          f"Snapshot: Kaggle `Cornell-University/arxiv`, {n:,} records. cs.IR total {len(ir):,} "
          f"({int(ir['primary'].sum()):,} primary); cs.CL total {len(cl):,}; {len(both):,} papers in both.", "",
          t.to_markdown()]
    (IR2 / "results" / "phase1_counts.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (col, pcol, title) in zip(axes, [("cs.IR (any)", "cs.IR (primary)", "cs.IR"),
                                             ("cs.CL (any)", None, "cs.CL")]):
        ax.bar(t.index, t[col], color="#4C78A8", label="any category")
        if pcol:
            ax.bar(t.index, t[pcol], color="#F58518", label="primary category")
            ax.legend(frameon=False)
        ax.set_title(f"{title} papers per year (first version)")
        ax.set_xlabel("year")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (IR2 / "figures").mkdir(exist_ok=True)
    fig.savefig(IR2 / "figures" / "phase1_counts.png", dpi=150)


if __name__ == "__main__":
    main()
