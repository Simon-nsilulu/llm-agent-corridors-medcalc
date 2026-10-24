"""Запуск конвейера / Exécution du pipeline.

Пример / Exemple :
  python run_pilot.py --dataset ../verified/datasets/test_data.csv \
      --impl ../verified/calculator_implementations --models models.json \
      --n-per-calc 9 --repeats 3 --out runs/pilot
"""
import argparse
import json
import os

import config as C
import data as D
import prompts as P
from llm import Logger, Model, MockBehaviour, QuotaExhausted, ModelUnavailable


def clip(x, lo, hi):
    return min(max(x, lo), hi)


def project_inputs(cid, xc):
    out = {}
    for k, v in xc.items():
        lo, hi = C.PARAM_KINDS[C.CALCULATORS[cid]["params"][k][0]]["range"]
        out[k] = clip(v, lo, hi)
    return out


def display_entities(cid, xc):
    """Вход агента 2 в канонических единицах / entrée de l'agent 2 en unités canoniques."""
    disp = {}
    for k, v in xc.items():
        kind = C.CALCULATORS[cid]["params"][k][0]
        if kind == "platelets":
            disp[k] = [round(v / 1e9, 6), "10^9/L"]
        elif kind == "height" and cid == 60:          # формула Мостеллера: рост в см
            disp[k] = [round(v * 100, 4), "cm"]
        else:
            disp[k] = [round(v, 6), D.canonical_to_entity(kind, v)[1]]
    return disp


def violations(cid, xc):
    """Параметры вне коридора / paramètres hors couloir : {clé: (valeur, lo, hi, unité)}."""
    out = {}
    for k, v in xc.items():
        kind = C.CALCULATORS[cid]["params"][k][0]
        lo, hi = C.PARAM_KINDS[kind]["range"]
        if not lo <= v <= hi:
            out[k] = (v, lo, hi, C.PARAM_KINDS[kind]["unit"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--n-per-calc", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="nom d'un seul modèle (pour lancer les modèles en parallèle)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    D.set_impl_path(os.path.abspath(a.impl))
    items = D.load_items(a.dataset, n_per_calc=a.n_per_calc, seed=a.seed)
    specs = json.load(open(a.models, encoding="utf-8"))
    if a.only:
        specs = [s for s in specs if s["name"] == a.only]
        if not specs:
            raise SystemExit(f"modèle inconnu : {a.only}")
    print(f"{len(items)} примеров / exemples ; {len(specs)} модел(и/ей)")

    trace_path = os.path.join(a.out, f"trace_main_{a.only}.jsonl" if a.only else "trace.jsonl")
    try:
        _run(a, items, specs, trace_path)
    except QuotaExhausted as e:
        print("\n*** ARRÊT : quota épuisé chez le fournisseur / квота исчерпана ***")
        print("   ", str(e)[:200])
        print("   Les appels réussis sont gardés. Relancez la même commande plus tard (ou après activation de la facturation).")
        raise SystemExit(2)


def _run(a, items, specs, trace_path):
    with open(trace_path, "w", encoding="utf-8") as tr:
        for mi, spec in enumerate(specs):
            log = Logger(os.path.join(a.out, f"raw_{spec['name']}.jsonl"))
            model = Model(spec, log)
            mock = MockBehaviour(seed=a.seed + mi)
            try:
                _run_model(a, items, spec, model, mock, tr)
            except ModelUnavailable as e:
                print(f"\n*** {spec['name']} : modèle saturé chez le fournisseur, on passe au suivant ***")
                print("   ", str(e)[:250])
                print("   Relancez la même commande plus tard : les appels réussis sont gardés.\n")


def _run_model(a, items, spec, model, mock, tr):
        for n_it, it in enumerate(items, 1):
            if n_it % 10 == 0:
                print(f"   {spec['name']}: {n_it}/{len(items)}", flush=True)
            cid, iid = it["cid"], f"{it['row']}"
            gold_c = D.canonical_vector(cid, it["gold_entities"])
            q = D.question_text(cid)
            for rep in range(a.repeats):
                t = {"model": spec["name"], "item": iid, "cid": cid, "repeat": rep}
                r = model.call("B0", iid, rep, P.end_to_end(it), lambda: mock.end_to_end(it["gt"]))
                t["B0"] = r["key"]
                r1 = model.call("A1", iid, rep, P.extraction(it),
                                lambda: mock.extraction(it["gold_entities"]))
                t["A1"] = r1["key"]
                xhat = D.canonical_vector(cid, r1["parsed"]) if isinstance(r1.get("parsed"), dict) else None
                t["xhat"] = xhat
                if xhat is not None and gold_c is not None:
                    x1 = project_inputs(cid, xhat)
                    x1s = project_inputs(cid, gold_c)
                    # C3: встречный запрос при выходе за коридор / C3 : re-demande si hors couloir
                    viol = violations(cid, xhat)
                    xhat_r = xhat
                    if viol:
                        rr = model.call("A1R", iid, rep, P.requery(it, r1.get("response"), viol),
                                        lambda: mock.extraction(it["gold_entities"]))
                        t["A1R"] = rr["key"]
                        cand = D.canonical_vector(cid, rr["parsed"]) if isinstance(rr.get("parsed"), dict) else None
                        if cand is not None:
                            xhat_r = cand
                    t["xhat_r"] = xhat_r
                    t["requeried"] = bool(viol)
                    x1r = project_inputs(cid, xhat_r)
                    for tag, xin in (("A2_raw", xhat), ("A2_proj", x1), ("A2_cf", x1s), ("A2_c3", x1r)):
                        ftrue = D.f_from_canonical(cid, xin)
                        r2 = model.call("A2", iid, rep, P.calculation(it, q, display_entities(cid, xin)),
                                        lambda f=ftrue: mock.calculation(f))
                        t[tag] = r2["key"]
                tr.write(json.dumps(t, ensure_ascii=False) + "\n")
        print(f"готово / terminé : {spec['name']}")


if __name__ == "__main__":
    main()
