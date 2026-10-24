"""Constitue l'ensemble d'évaluation final / Формирует итоговое оценочное множество.

- test : les 120 exemples des 6 calculateurs (dont les 54 du pilote, déclarés dans l'article) ;
- train : 20 exemples par calculateur, tirés (graine 2026) parmi ceux qui passent le filtre
  annoncé à l'avance : valeurs de référence convertibles, toutes dans les couloirs de plausibilité,
  réponse de référence dans le couloir de sortie, note absente du split test.

python prepare_items.py --verified ../verified --out items_main.csv
"""
import argparse
import os

import pandas as pd

import config as C
import data as D


def passes_filter(it):
    x = D.canonical_vector(it["cid"], it["gold_entities"])
    if x is None:
        return False
    for k, v in x.items():
        lo, hi = C.PARAM_KINDS[C.CALCULATORS[it["cid"]]["params"][k][0]]["range"]
        if not lo <= v <= hi:
            return False
    lo, hi = C.CALCULATORS[it["cid"]]["out_range"]
    return lo <= it["gt"] <= hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    D.set_impl_path(os.path.abspath(os.path.join(a.verified, "calculator_implementations")))
    te = pd.read_csv(os.path.join(a.verified, "datasets", "test_data.csv"))
    tr = pd.read_csv(os.path.join(a.verified, "datasets", "train_data.csv"))
    ids = list(C.CALCULATORS)
    te = te[te["Calculator ID"].isin(ids)].assign(Split="test")
    tr = tr[tr["Calculator ID"].isin(ids)]
    test_notes = set(te["Patient Note"])
    items = {int(it["row"]): it for it in D.load_items(os.path.join(a.verified, "datasets", "train_data.csv"), ids)}
    keep = [rn for rn, it in items.items() if passes_filter(it) and it["note"] not in test_notes]
    tr = tr[tr["Row Number"].isin(keep)]
    parts = [g.sample(min(a.n_train, len(g)), random_state=a.seed) for _, g in tr.groupby("Calculator ID")]
    tr = pd.concat(parts).assign(Split="train")
    out = pd.concat([te, tr])
    assert out["Row Number"].is_unique
    out.to_csv(a.out, index=False)
    print(out.groupby(["Split", "Calculator ID"]).size().unstack(fill_value=0))
    print(f"{len(out)} exemples -> {a.out}")


if __name__ == "__main__":
    main()
