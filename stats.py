"""Statistiques finales / Итоговая статистика (hors ligne, aucun appel aux modèles).

- précision (part dans les limites du benchmark) par modèle × configuration, avec IC 95 % bootstrap
  (rééchantillonnage des exemples, 10 000 tirages, graine 2026) ;
- comparaisons appariées sur les mêmes exemples : C1 vs C0, C3 vs C1, C3 vs C0, C2 vs C0, B4 vs C0,
  test exact de McNemar (bilatéral) ;
- pour l'injection : précision par type d'erreur et part des erreurs détectées par le couloir.

python stats.py --cases runs/final/cases.csv  [--cases runs/final/cases_inject.csv]
"""
import argparse
import math
import os

import numpy as np
import pandas as pd

PAIRS = [("C1", "C0"), ("C3", "C1"), ("C3", "C0"), ("C2", "C0"), ("B4", "C0")]


def boot_ci(x, n=10000, seed=2026):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)))


def mcnemar_exact(b, c):
    """b = A juste / B faux ; c = A faux / B juste. p bilatéral exact."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def acc_table(df):
    rows = []
    for (m, cfg), g in df.groupby(["model", "config"]):
        y = g["in_limits"].fillna(False).astype(float)      # échec = réponse fausse
        lo, hi = boot_ci(y)
        nfail = int((g["ok"] == False).sum()) if "ok" in g else 0
        rows.append({"model": m, "config": cfg, "n": len(y), "n_fail": nfail,
                     "acc": y.mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def paired(df):
    rows = []
    wide = df.pivot_table(index=["model", "item"], columns="config", values="in_limits", aggfunc="first")
    wide = wide.astype("float").fillna(0.0)
    for m, g in wide.groupby(level=0):
        for a, b in PAIRS:
            if a not in g or b not in g:
                continue
            A, B = g[a].astype(bool), g[b].astype(bool)
            nb, nc = int((A & ~B).sum()), int((~A & B).sum())
            rows.append({"model": m, "comparison": f"{a} vs {b}", "n": len(g),
                         "diff_pp": 100 * (A.mean() - B.mean()), "only_first_right": nb, "only_second_right": nc,
                         "p_mcnemar": mcnemar_exact(nb, nc)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", action="append", required=True)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 500)
    for path in a.cases:
        df = pd.read_csv(path)
        df = df[df["repeat"] == 0]
        base = os.path.splitext(path)[0]
        print(f"\n===== {path} =====")
        for split in sorted(df["split"].dropna().unique()) if "split" in df else ["all"]:
            sub = df[df["split"] == split] if split != "all" else df
            t = acc_table(sub)
            p = paired(sub)
            t.to_csv(f"{base}_acc_{split}.csv", index=False)
            p.to_csv(f"{base}_paired_{split}.csv", index=False)
            print(f"\n--- split = {split} : précision [IC 95 %] ---")
            t["acc_ci"] = t.apply(lambda r: f"{100*r.acc:5.1f} [{100*r.ci_low:4.1f}; {100*r.ci_high:4.1f}]", axis=1)
            print(t.pivot(index="model", columns="config", values="acc_ci").to_string())
            bad = t[t["n_fail"] > 0]
            if len(bad):
                print("\nRéponses inexploitables du modèle (JSON illisible, unité inconnue, réponse tronquée), comptées comme fausses.\n"
                      "Les erreurs d'API sont contrôlées séparément par check_pilot.py / Неразборчивые ответы модели, засчитаны как ошибки:")
                print(bad.pivot(index="model", columns="config", values="n_fail").fillna(0).astype(int).to_string())
            print(f"\n--- split = {split} : comparaisons appariées (McNemar exact) ---")
            print(p.round(4).to_string(index=False))
        if "inj_type" in df:
            print("\n--- injection : précision par type d'erreur ---")
            print((100 * df.pivot_table(index=["inj_type", "model"], columns="config", values="in_limits",
                                        aggfunc=lambda s: s.fillna(False).astype(float).mean())).round(1).to_string())
            det = df[df["config"] == "C3"].groupby(["inj_type", "model"])["requeried"].mean()
            print("\n--- injection : part des erreurs détectées par le couloir (re-demande déclenchée) ---")
            print((100 * det).round(1).unstack().to_string())


if __name__ == "__main__":
    main()
