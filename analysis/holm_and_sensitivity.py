"""Additional statistics reported in the paper (no model calls).

1. Paired comparisons with exact McNemar tests and 95 % bootstrap CIs of the accuracy difference,
   Holm correction within each family (data type x split: 6 models x 5 configuration pairs = 30 tests).
   Output: runs/final/paired_holm.csv
2. Sensitivity analysis on the 66 test examples NOT used in the pilots. The 54 pilot examples are
   reconstructed exactly as in the pilot: 9 per calculator, pandas sample with random_state 2026,
   from ../verified/datasets/test_data.csv.

python analysis/holm_and_sensitivity.py --verified ../verified
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as D  # noqa: E402

RUN = os.path.join("runs", "final")
PAIRS = [("C1", "C0"), ("C3", "C1"), ("C3", "C0"), ("C2", "C0"), ("B4", "C0")]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(p):
    order = np.argsort(p)
    adj, cur = np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        cur = max(cur, min(1.0, (len(p) - i) * p[j]))
        adj[j] = cur
    return adj


def paired(df, data_tag, split):
    w = df.pivot_table(index=["model", "item"], columns="config", values="in_limits", aggfunc="first")
    w = w.astype(float).fillna(0.0).astype(bool)
    rows = []
    for m, g in w.groupby(level=0):
        for a, b in PAIRS:
            A, B = g[a], g[b]
            diff = A.astype(float).values - B.astype(float).values
            bs = np.random.default_rng(2026).choice(diff, (10000, len(diff))).mean(axis=1)
            rows.append({"data": data_tag, "split": split, "model": m, "cmp": f"{a} vs {b}", "n": len(g),
                         "diff": 100 * diff.mean(), "lo": 100 * np.quantile(bs, .025), "hi": 100 * np.quantile(bs, .975),
                         "p": mcnemar(int((A & ~B).sum()), int((~A & B).sum()))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=os.path.join("..", "verified"))
    a = ap.parse_args()
    rows = []
    for fname, tag in (("cases.csv", "natural"), ("cases_inject.csv", "inject")):
        df = pd.read_csv(os.path.join(RUN, fname))
        df = df[df["repeat"] == 0]
        for split in ("test", "train"):
            rows += paired(df[df["split"] == split], tag, split)
    r = pd.DataFrame(rows)
    r["p_holm"] = r.groupby(["data", "split"])["p"].transform(lambda s: holm(s.values))
    out = os.path.join(RUN, "paired_holm.csv")
    r.to_csv(out, index=False)
    print(f"écrit : {out}")
    print(r[r["split"] == "test"].round(4).to_string(index=False))

    D.set_impl_path(os.path.abspath(os.path.join(a.verified, "calculator_implementations")))
    pilot = {str(it["row"]) for it in D.load_items(os.path.join(a.verified, "datasets", "test_data.csv"),
                                                    n_per_calc=9, seed=2026)}
    print(f"\nexemples pilote reconstitués : {len(pilot)}")
    for fname in ("cases.csv", "cases_inject.csv"):
        df = pd.read_csv(os.path.join(RUN, fname))
        df = df[(df["split"] == "test") & (df["repeat"] == 0)]
        df["item"] = df["item"].astype(str)
        s = df[~df["item"].isin(pilot)]
        acc = 100 * s.pivot_table(index="model", columns="config", values="in_limits",
                                  aggfunc=lambda x: x.fillna(False).astype(float).mean())
        print(f"\n--- {fname} : précision (%) sur les exemples test hors pilote ---")
        print(acc.round(1).to_string())
        w = s.pivot_table(index=["model", "item"], columns="config", values="in_limits", aggfunc="first")
        w = w.astype(float).fillna(0.0).astype(bool)
        for m, g in w.groupby(level=0):
            A, B = g["C3"], g["C1"]
            print(f"   {m}: n={len(g)}  C3 - C1 = {100 * (A.mean() - B.mean()):+.1f} pp, "
                  f"p = {mcnemar(int((A & ~B).sum()), int((~A & B).sum())):.4f}")


if __name__ == "__main__":
    main()
