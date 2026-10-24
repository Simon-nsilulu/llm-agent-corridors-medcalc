"""Expérience complémentaire : le couloir aide-t-il par l'INFORMATION qu'il donne ou seulement parce qu'on redemande ?
Дополнительный эксперимент: помогает ли информация коридора или только сам повторный запрос?

À lancer APRÈS run_inject.py, sur le même dossier --out (mêmes erreurs injectées, mêmes modèles).
Configurations comparées sur les MÊMES exemples injectés :
  C1   : projection sur le couloir (déjà calculé, repris du cache) ;
  C3   : re-demande AVEC la valeur fautive et la plage plausible, seulement si hors couloir (repris du cache) ;
  C3g  : re-demande GÉNÉRIQUE (« vérifiez vos valeurs »), SANS valeur ni plage, seulement si hors couloir ;
  C3all: re-demande GÉNÉRIQUE pour TOUS les exemples injectés (aucune détection par le couloir).
Dans tous les cas la réponse corrigée est projetée sur le couloir, puis l'agent 2 calcule.

Sortie : ablation_cases{suffix}.csv et un résumé (précision, McNemar exact, correction de Holm).
python run_ablation.py --dataset items_main.csv --impl ../verified/calculator_implementations --models models.final.json --out runs/final
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

import data as D
import prompts as P
from llm import Logger, Model, MockBehaviour, QuotaExhausted, ModelUnavailable
from run_inject import fake_response
from run_pilot import project_inputs, display_entities, violations


def num(rec):
    p = rec.get("parsed") if rec else None
    try:
        v = float(p["answer"])
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = np.argsort(ps)
    adj, cur = np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        cur = max(cur, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = cur
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--suffix", default="", help="même suffixe que run_inject.py (graine d'injection)")
    ap.add_argument("--analyze-only", action="store_true", help="ne fait aucun appel, analyse les fichiers existants")
    a = ap.parse_args()
    D.set_impl_path(os.path.abspath(a.impl))
    items = {f"{it['row']}": it for it in D.load_items(a.dataset)}
    specs = json.load(open(a.models, encoding="utf-8"))
    if a.only:
        specs = [s for s in specs if s["name"] == a.only]
    if not a.analyze_only:
        try:
            for mi, spec in enumerate(specs):
                run_model(a, spec, items, MockBehaviour(seed=2026 + mi))
        except QuotaExhausted as e:
            print("\n*** ARRÊT : quota épuisé / clé refusée ***\n   ", str(e)[:200])
            raise SystemExit(2)
    summarize(a)


def run_model(a, spec, items, mock):
    log = Logger(os.path.join(a.out, f"raw_{spec['name']}.jsonl"))
    model = Model(spec, log)
    path = os.path.join(a.out, f"trace_inject{a.suffix}_{spec['name']}.jsonl")
    if not os.path.exists(path):
        print(f"   {spec['name']} : pas de {path}, lancer d'abord run_inject.py")
        return
    rows = []
    traces = [json.loads(l) for l in open(path, encoding="utf-8")]
    try:
        for n, t in enumerate(traces, 1):
            if n % 20 == 0:
                print(f"   {spec['name']}: {n}/{len(traces)}", flush=True)
            it = items[t["item"]]
            cid = t["cid"]
            q = D.question_text(cid)
            xinj = t["xhat"]
            flagged = bool(violations(cid, xinj))
            prev = fake_response(cid, xinj)

            def agent2(xin):
                xp = project_inputs(cid, xin)
                r2 = model.call("A2", t["item"], 0, P.calculation(it, q, display_entities(cid, xp)),
                                lambda f=D.f_from_canonical(cid, xp): mock.calculation(f))
                y = num(r2)
                return y is not None and it["lo"] <= y <= it["hi"]

            def generic():
                rr = model.call("A1G", t["item"], 0, P.requery_generic(it, prev),
                                lambda: mock.extraction(it["gold_entities"]))
                cand = D.canonical_vector(cid, rr["parsed"]) if isinstance(rr.get("parsed"), dict) else None
                return cand if cand is not None else xinj

            res = {"C1": agent2(xinj), "C3": agent2(t.get("xhat_r") or xinj)}
            xg = generic() if flagged else xinj
            res["C3g"] = agent2(xg)
            res["C3all"] = agent2(generic())
            for cfg, ok in res.items():
                rows.append({"model": spec["name"], "item": t["item"], "split": it.get("split", "test"),
                             "inj_type": t["injection"]["type"], "flagged": flagged, "config": cfg, "in_limits": ok})
    except ModelUnavailable as e:
        print(f"*** {spec['name']} : modèle saturé, relancer plus tard ***\n   {str(e)[:200]}")
    out = os.path.join(a.out, f"ablation_cases{a.suffix}_{spec['name']}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"terminé : {spec['name']} -> {out}")


def summarize(a):
    files = glob.glob(os.path.join(a.out, f"ablation_cases{a.suffix}_*.csv"))
    if not files:
        print("aucun fichier ablation_cases")
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    pd.set_option("display.width", 200)
    for split in sorted(df["split"].unique()):
        s = df[df["split"] == split]
        acc = (100 * s.pivot_table(index="model", columns="config", values="in_limits", aggfunc="mean")).round(1)
        print(f"\n--- {split} : précision (%) sur les exemples injectés ---")
        print(acc[["C1", "C3", "C3g", "C3all"]].to_string())
        w = s.pivot_table(index=["model", "item"], columns="config", values="in_limits", aggfunc="first").astype(bool)
        rows = []
        for m, g in w.groupby(level=0):
            for x, y in (("C3", "C3g"), ("C3g", "C1"), ("C3", "C3all")):
                A, B = g[x], g[y]
                rows.append({"model": m, "comparison": f"{x} vs {y}", "n": len(g),
                             "diff_pp": round(100 * (A.mean() - B.mean()), 1),
                             "p": mcnemar(int((A & ~B).sum()), int((~A & B).sum()))})
        r = pd.DataFrame(rows)
        r["p_holm"] = holm(r["p"].values)
        print(f"\n--- {split} : comparaisons appariées (McNemar exact, Holm sur {len(r)} tests) ---")
        print(r.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
