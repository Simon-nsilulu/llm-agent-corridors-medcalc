"""Injection contrôlée d'erreurs dans la sortie de l'agent 1 / Контролируемое внесение ошибок.

À lancer APRÈS run_pilot.py, sur le même dossier --out : la sortie réelle de l'agent 1
(déjà enregistrée) est reprise, UN paramètre est corrompu, puis l'orchestrateur et l'agent 2
travaillent sur cette sortie corrompue (configurations C0, C1, C2, C3, B4).

Plan d'injection (fixé à l'avance, identique pour tous les modèles) :
  - paramètre corrompu : tiré au hasard parmi ceux du calculateur (graine liée à l'exemple) ;
  - type d'erreur : cycle sur l'ordre des exemples -> "decimal" (×10 ou ÷10), "scale" (×1000 ou ÷1000),
    "unit" (confusion d'unité propre au paramètre, voir UNIT_ERRORS).
Le résultat est écrit dans trace_inject.jsonl ; analyser avec analyze.py --trace trace_inject.jsonl --tag inject

python run_inject.py --dataset items_main.csv --impl ../verified/calculator_implementations --models models.final.json --out runs/final
"""
import argparse
import glob
import hashlib
import json
import os
import random

import config as C
import data as D
import prompts as P
from llm import Logger, Model, MockBehaviour, QuotaExhausted, ModelUnavailable
from run_pilot import project_inputs, display_entities, violations

# Confusions d'unité typiques (facteur appliqué à la valeur canonique) / типичные ошибки единиц
UNIT_ERRORS = {
    "height": (100.0, "cm lu comme m"),
    "weight": (2.20462, "lb lu comme kg"),
    "glucose": (1 / 18.016, "mmol/L lu comme mg/dL"),
    "insulin": (6.0, "pmol/L lu comme µIU/mL"),
    "platelets": (1000.0, "count/µL lu comme 10^9/L"),
    "age": (12.0, "mois lus comme années"),
    "qt": (0.001, "ms lues comme s"),
}
TYPES = ("decimal", "scale", "unit")


def plan_for(item_index, item, seed):
    keys = sorted(C.CALCULATORS[item["cid"]]["params"])
    h = int(hashlib.sha256(f"{seed}|{item['row']}".encode()).hexdigest(), 16)
    rng = random.Random(h)
    key = rng.choice(keys)
    kind = C.CALCULATORS[item["cid"]]["params"][key][0]
    typ = TYPES[item_index % len(TYPES)]
    up = rng.random() < 0.5
    if typ == "unit" and kind not in UNIT_ERRORS:
        typ = "decimal"                      # pas de confusion d'unité naturelle -> décimale
    if typ == "decimal":
        factor, label = (10.0, "×10") if up else (0.1, "÷10")
    elif typ == "scale":
        factor, label = (1000.0, "×1000") if up else (0.001, "÷1000")
    else:
        factor, label = UNIT_ERRORS[kind]
    return {"key": key, "kind": kind, "type": typ, "factor": factor, "label": label}


def fake_response(cid, xc):
    """Sortie JSON de l'agent 1 telle que l'orchestrateur la reçoit (valeurs corrompues)."""
    disp = display_entities(cid, xc)
    return json.dumps(disp, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", default=None, help="nom d'un seul modèle")
    ap.add_argument("--suffix", default="", help="suffixe du fichier trace (ex. _s2027) pour une autre graine")
    a = ap.parse_args()
    D.set_impl_path(os.path.abspath(a.impl))
    items = D.load_items(a.dataset)
    items.sort(key=lambda it: it["row"])
    specs = json.load(open(a.models, encoding="utf-8"))
    if a.only:
        specs = [s for s in specs if s["name"] == a.only]
    main_trace = {}
    for path in glob.glob(os.path.join(a.out, "trace.jsonl")) + glob.glob(os.path.join(a.out, "trace_main_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)
                if t["repeat"] == 0:
                    main_trace[(t["model"], t["item"])] = t
    print(f"{len(items)} exemples ; {len(specs)} modèles ; injection d'une erreur par exemple")
    out_path = os.path.join(a.out, f"trace_inject{a.suffix}_{a.only}.jsonl" if a.only else f"trace_inject{a.suffix}.jsonl")
    try:
        with open(out_path, "w", encoding="utf-8") as tr:
            for mi, spec in enumerate(specs):
                model = Model(spec, Logger(os.path.join(a.out, f"raw_{spec['name']}.jsonl")))
                mock = MockBehaviour(seed=a.seed + mi)
                try:
                    for n_it, it in enumerate(items):
                        if (n_it + 1) % 20 == 0:
                            print(f"   {spec['name']}: {n_it + 1}/{len(items)}", flush=True)
                        cid, iid = it["cid"], f"{it['row']}"
                        base = main_trace.get((spec["name"], iid))
                        if not base or base.get("xhat") is None:
                            continue                  # sortie réelle absente : rien à corrompre
                        gold = D.canonical_vector(cid, it["gold_entities"])
                        plan = plan_for(n_it, it, a.seed)
                        xinj = dict(base["xhat"])
                        xinj[plan["key"]] = xinj[plan["key"]] * plan["factor"]
                        t = {"model": spec["name"], "item": iid, "cid": cid, "repeat": 0,
                             "B0": base.get("B0"), "A1": base.get("A1"), "xhat": xinj,
                             "injection": plan}
                        q = D.question_text(cid)
                        x1 = project_inputs(cid, xinj)
                        x1s = project_inputs(cid, gold)
                        viol = violations(cid, xinj)
                        xhat_r = xinj
                        if viol:
                            prev = fake_response(cid, xinj)
                            rr = model.call("A1R", iid, 0, P.requery(it, prev, viol),
                                            lambda: mock.extraction(it["gold_entities"]))
                            t["A1R"] = rr["key"]
                            cand = D.canonical_vector(cid, rr["parsed"]) if isinstance(rr.get("parsed"), dict) else None
                            if cand is not None:
                                xhat_r = cand
                        t["xhat_r"] = xhat_r
                        t["requeried"] = bool(viol)
                        x1r = project_inputs(cid, xhat_r)
                        for tag, xin in (("A2_raw", xinj), ("A2_proj", x1), ("A2_cf", x1s), ("A2_c3", x1r)):
                            ftrue = D.f_from_canonical(cid, xin)
                            r2 = model.call("A2", iid, 0, P.calculation(it, q, display_entities(cid, xin)),
                                            lambda f=ftrue: mock.calculation(f))
                            t[tag] = r2["key"]
                        tr.write(json.dumps(t, ensure_ascii=False) + "\n")
                    print(f"terminé : {spec['name']}")
                except ModelUnavailable as e:
                    print(f"\n*** {spec['name']} : modèle saturé, on passe au suivant ***\n   {str(e)[:250]}\n")
    except QuotaExhausted as e:
        print("\n*** ARRÊT : quota épuisé ***\n   ", str(e)[:200])
        raise SystemExit(2)


if __name__ == "__main__":
    main()
