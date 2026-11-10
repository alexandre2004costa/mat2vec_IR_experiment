"""Phase 3b: encode abstracts once with allenai/specter2_base (no adapter), as the model card prescribes:
title + [SEP] + abstract, CLS token of the last layer, max 512 tokens.

Encoded: every cs.IR abstract (all years, also used by Phase 4) and every cs.CL abstract before 2023
(the largest cutoff), i.e. all text any cutoff trains on.
Usage: ir2/.venv/bin/python ir2/scripts/phase3_encode.py [--limit 256] [--batch 32] [--device mps]
Writes ir2/data/specter2.npy (float16, L2-normalized) and ir2/data/specter2_rows.npy (row indices into docs.parquet).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ir2lib as L  # noqa: E402

MODEL = "allenai/specter2_base"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    docs = L.load_docs()
    rows = np.flatnonzero(docs["is_ir"].values | (docs["year"].values < max(L.CUTOFFS)))
    if args.limit:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(args.device).eval()
    texts = [t + tok.sep_token + a for t, a in zip(docs["title"].values[rows], docs["abstract"].values[rows])]
    # sort by length so batches have similar padding, then restore the order
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(rows), model.config.hidden_size), dtype=np.float16)
    t0 = time.time()
    with torch.inference_mode():
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512,
                      return_tensors="pt", return_token_type_ids=False).to(args.device)
            cls = model(**enc).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            out[idx] = cls.float().cpu().numpy().astype(np.float16)
            if (i // args.batch) % 100 == 0:
                done = i + len(idx)
                print(f"{done:,}/{len(rows):,} {done / (time.time() - t0):.0f} docs/s", flush=True)
    print(f"encoded {len(rows):,} docs in {time.time() - t0:.0f}s on {args.device}")
    if not args.limit:
        np.save(L.DATA / "specter2.npy", out)
        np.save(L.DATA / "specter2_rows.npy", rows)


if __name__ == "__main__":
    main()
