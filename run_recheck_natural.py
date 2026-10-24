"""Contrôle indispensable : la re-demande générique systématique (C3all) sur des extractions NON corrompues.
Обязательная проверка: систематический повторный запрос на НЕискажённых извлечениях.

Sur données injectées, C3all (redemander toujours) bat C3. Mais redemander une extraction déjà correcte
peut la dégrader. Ici : sorties réelles de l'agent 1 (run final, aucune injection), re-demande générique
pour TOUS les exemples, projection, puis agent 2. On compare à C1 (même chaîne sans re-demande) et on compte
les cas où la re-demande rend faux un exemple juste (« dégradations ») et l'inverse (« réparations »).

python run_recheck_natural.py --dataset items_main.csv --impl ../verified/calculator_implementations --models models.final.json --out runs/final [--only NOM] [--analyze-only]
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
from run_pilot import project_inputs, display_entities
from run_ablation import num, mcnemar, holm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--analyze-only", action="store_true")
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
    path = os.path.join(a.out, f"trace_main_{spec['name']}.jsonl")
    if not os.path.exists(path):
        print(f"   {spec['name']} : pas de {path}")
        return
    rows = []
    traces = [json.loads(l) for l in open(path, encoding="utf-8")]
    traces = [t for t in traces if t.get("repeat", 0) == 0 and t.get("xhat") is not None]
    try:
        for n, t in enumerate(traces, 1):
            if n % 20 == 0:
                print(f"   {spec['name']}: {n}/{len(traces)}", flush=True)
            it = items[t["item"]]
            cid = t["cid"]
            q = D.question_text(cid)
            a1 = log.get(t["A1"]) or {}
            prev = a1.get("response") or ""

            def agent2(xin):
                xp = project_inputs(cid, xin)
                r2 = model.call("A2", t["item"], 0, P.calculation(it, q, display_entities(cid, xp)),
                                lambda f=D.f_from_canonical(cid, xp): mock.calculation(f))
                y = num(r2)
                return y is not None and it["lo"] <= y <= it["hi"]

            rr = model.call("A1G", t["item"], 0, P.requery_generic(it, prev),
                            lambda: mock.extraction(it["gold_entities"]))
            cand = D.canonical_vector(cid, rr["parsed"]) if isinstance(rr.get("parsed"), dict) else None
            changed = cand is not None and any(abs(math.log(cand[k] / t["xhat"][k])) > 1e-6 for k in cand
                                               if cand[k] > 0 and t["xhat"].get(k, 0) > 0)
            ok_c1 = agent2(t["xhat"])
            ok_all = agent2(cand if cand is not None else t["xhat"])
            rows.append({"model": spec["name"], "item": t["item"], "split": it.get("split", "test"),
                         "C1": ok_c1, "C3all_nat": ok_all, "extraction_changed": changed,
                         "requery_parsed": cand is not None})
    except ModelUnavailable as e:
        print(f"*** {spec['name']} : modèle saturé, relancer plus tard ***\n   {str(e)[:200]}")
    out = os.path.join(a.out, f"recheck_natural_{spec['name']}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"terminé : {spec['name']} -> {out}")


def summarize(a):
    files = glob.glob(os.path.join(a.out, "recheck_natural_*.csv"))
    if not files:
        print("aucun fichier recheck_natural")
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    pd.set_option("display.width", 200)
    for split in sorted(df["split"].unique()):
        s = df[df["split"] == split]
        rows = []
        for m, g in s.groupby("model"):
            A, B = g["C3all_nat"].astype(bool), g["C1"].astype(bool)
            rows.append({"model": m, "n": len(g), "acc_C1": round(100 * B.mean(), 1), "acc_C3all": round(100 * A.mean(), 1),
                         "degraded(juste->faux)": int((B & ~A).sum()), "repaired(faux->juste)": int((A & ~B).sum()),
                         "extraction_changed_%": round(100 * g["extraction_changed"].mean(), 1),
                         "p": mcnemar(int((A & ~B).sum()), int((~A & B).sum()))})
        r = pd.DataFrame(rows)
        r["p_holm"] = holm(r["p"].values)
        print(f"\n--- {split} : re-demande générique systématique sur extractions réelles (sans injection) ---")
        print(r.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
