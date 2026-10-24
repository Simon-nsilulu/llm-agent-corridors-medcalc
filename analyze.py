"""Анализ (без вызовов моделей) / Analyse (sans appel aux modèles).

python analyze.py --run runs/pilot --dataset ... --impl ...
Выход / Sortie : cases.csv, summary.csv, summary.json, checksums.txt
"""
import argparse
import glob
import hashlib
import itertools
import json
import math
import os

import numpy as np
import pandas as pd

import config as C
import data as D
from run_pilot import project_inputs, clip

S = C.SIGMA


def z(x):
    return math.log(x) / S


def num(rec):
    p = rec.get("parsed") if rec else None
    try:
        v = float(p["answer"])
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None


def vec_dist(a, b):
    return max(abs(z(a[k]) - z(b[k])) for k in a)


def analyse_case(it, t, recs, cid):
    M = C.lipschitz_M(cid)
    gold = D.canonical_vector(cid, it["gold_entities"])
    rows = []
    base = {"model": t["model"], "item": t["item"], "cid": cid, "repeat": t["repeat"],
            "split": it.get("split", "test")}
    inj = t.get("injection")
    if inj:
        base.update(inj_type=inj["type"], inj_key=inj["key"], inj_label=inj["label"])
    # B0: одна модель от начала до конца / modèle seul de bout en bout
    y0 = num(recs.get(t.get("B0")))
    rows.append({**base, "config": "B0", "ok": y0 is not None,
                 "y": y0, "in_limits": (y0 is not None and it["lo"] <= y0 <= it["hi"]),
                 "rel_err": (abs(y0 / it["gt"] - 1) if y0 is not None else np.nan)})
    xhat = t.get("xhat")
    if xhat is None or gold is None:
        for cfg in ("C0", "C1", "C2", "C3", "B4"):
            rows.append({**base, "config": cfg, "ok": False, "fail": "extraction"})
        return rows
    x1 = project_inputs(cid, xhat)
    x1s = project_inputs(cid, gold)
    e1 = vec_dist(xhat, gold)                       # собственная ошибка агента 1
    widths = [math.log(C.PARAM_KINDS[C.CALCULATORS[cid]["params"][k][0]]["range"][1] /
                       C.PARAM_KINDS[C.CALCULATORS[cid]["params"][k][0]]["range"][0]) / S for k in x1]
    l1 = max(widths)
    d1 = vec_dist(x1, x1s)
    f_x1, f_x1s, f_xhat, f_gold = (D.f_from_canonical(cid, v) for v in (x1, x1s, xhat, gold))
    lo_o, hi_o = C.CALCULATORS[cid]["out_range"]
    s_raw, s_proj, s_cf, s_c3 = (num(recs.get(t.get(k))) for k in ("A2_raw", "A2_proj", "A2_cf", "A2_c3"))
    xhat_r = t.get("xhat_r") or xhat
    x1r = project_inputs(cid, xhat_r)
    e1r = vec_dist(xhat_r, gold)
    d1r = vec_dist(x1r, x1s)
    f_x1r = D.f_from_canonical(cid, x1r)

    def J2(cfg, f_in):
        if cfg == "C0":
            return (0.0, math.inf), math.inf, 0.0
        if cfg in ("C1", "C3"):
            return (lo_o, hi_o), math.log(hi_o / lo_o) / S, 0.0
        return (f_in * math.exp(-C.TAU), f_in * math.exp(C.TAU)), 2 * C.TAU / S, M

    for cfg in ("C0", "C1", "C2", "C3"):
        if "A2_c3" not in t and cfg == "C3":
            continue
        if cfg == "C0":
            xin, s2, d_in, c1, xref, f_in, f_ref = xhat, s_raw, e1, e1, gold, f_xhat, f_gold
        elif cfg == "C3":
            xin, s2, d_in, c1, xref, f_in, f_ref = x1r, s_c3, d1r, min(e1r, l1), x1s, f_x1r, f_x1s
        else:
            xin, s2, d_in, c1, xref, f_in, f_ref = x1, s_proj, d1, min(e1, l1), x1s, f_x1, f_x1s
        row = {**base, "config": cfg, "e1": (e1r if cfg == "C3" else e1), "delta1": d_in, "c1": c1, "M": M,
               "requeried": bool(t.get("requeried"))}
        if s2 is None:
            rows.append({**row, "ok": False, "fail": "calculation"})
            continue
        (a, b), l2, L2 = J2(cfg, f_in)
        (as_, bs), _, _ = J2(cfg, f_ref)
        y = clip(s2, a, b)
        x2s = clip(f_ref, as_, bs)
        e2 = abs(z(s2) - z(f_in))
        d2 = abs(z(y) - z(x2s))
        bound_a = min(e2, l2) + (M + L2) * d_in
        bound_b = max(e2 + M * d_in, L2 * d_in)
        bound = min(bound_a, bound_b)
        # следствие 2 при фиксированной e2 (гарантировано) / corollaire 2 à e2 fixée (garanti)
        y_cf_struct = clip(f_ref * math.exp(S * (z(s2) - z(f_in))), as_, bs)
        attr_struct = abs(z(y) - z(y_cf_struct))
        attr_bound = c1 * (M + L2)
        # контрфактический вызов БЯМ (эмпирически) / contrefactuel LLM (empirique)
        attr_llm = abs(z(y) - z(clip(s_cf, as_, bs))) if s_cf is not None else np.nan
        tol = 1e-2
        row.update(ok=True, y=y, e2=e2, l2=l2, L2=L2, delta2=d2, bound=bound,
                   viol_a=d2 > bound_a + tol, viol_b=d2 > bound_b + tol,
                   viol_delta1=(cfg != "C0" and d_in > c1 + tol),
                   ratio=(bound / d2 if d2 > tol else np.nan),
                   attr_struct=attr_struct, attr_bound=attr_bound,
                   viol_attr=attr_struct > attr_bound + tol,
                   attr_llm=attr_llm,
                   attr_llm_exceeds=(attr_llm > attr_bound + tol) if attr_llm == attr_llm else np.nan,
                   in_limits=it["lo"] <= y <= it["hi"], rel_err=abs(y / it["gt"] - 1))
        rows.append(row)
    yb4 = f_x1                                       # агент 2 заменён кодом / agent 2 = code
    d4 = abs(z(yb4) - z(f_x1s))
    rows.append({**base, "config": "B4", "ok": True, "y": yb4, "delta2": d4, "bound": M * d1,
                 "viol_a": d4 > M * d1 + 1e-2, "in_limits": it["lo"] <= yb4 <= it["hi"],
                 "rel_err": abs(yb4 / it["gt"] - 1), "e1": e1, "delta1": d1})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--impl", required=True)
    ap.add_argument("--trace", default="trace.jsonl")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    D.set_impl_path(os.path.abspath(a.impl))
    items = {str(it["row"]): it for it in D.load_items(a.dataset)}
    recs = {}
    for p in glob.glob(os.path.join(a.run, "raw_*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("error") is None:
                    recs[r["key"]] = r
    rows = []
    trace_files = sorted({p for pat in a.trace.split(",") for p in glob.glob(os.path.join(a.run, pat))})
    if not trace_files:
        raise SystemExit(f"aucun fichier trace : {a.trace}")
    print("traces :", ", ".join(os.path.basename(p) for p in trace_files))
    traces = []
    for path in trace_files:
        with open(path, encoding="utf-8") as fh:
            traces += [json.loads(line) for line in fh]
    for t in traces:
        rows += analyse_case(items[t["item"]], t, recs, t["cid"])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.run, f"cases{sfx}.csv"), index=False)

    main_rep = df[df["repeat"] == 0]
    g = main_rep.groupby(["model", "config"])
    summ = pd.DataFrame({
        "n": g.size(),
        "fail_rate": g["ok"].apply(lambda s: 1 - s.mean()),
        "acc_in_limits": g["in_limits"].apply(lambda s: s.dropna().astype(float).mean()),
        "median_rel_err": g["rel_err"].median() if "rel_err" in df else np.nan,
        "share_rel_err_gt_50pct": g["rel_err"].apply(lambda s: (s.dropna() > 0.5).mean()),
        "violations_bound": g.apply(lambda x: int(x.get("viol_a", pd.Series(dtype=bool)).fillna(False).sum()
                                                 + x.get("viol_b", pd.Series(dtype=bool)).fillna(False).sum())),
        "violations_attr": g.apply(lambda x: int(x.get("viol_attr", pd.Series(dtype=bool)).fillna(False).sum())),
        "median_bound_over_delta": g["ratio"].median(),
        "llm_cf_exceeds_bound": g["attr_llm_exceeds"].apply(lambda s: s.dropna().astype(float).mean()),
    }).reset_index()
    summ.to_csv(os.path.join(a.run, f"summary{sfx}.csv"), index=False)

    # П5: корреляция ошибок извлечения между моделями / corrélation des erreurs d'extraction
    corr = {}
    tr = pd.DataFrame(traces)
    tr = tr[(tr["repeat"] == 0) & tr["xhat"].notna()]
    signed = {}
    for _, t in tr.iterrows():
        it = items[str(t["item"])]
        gold = D.canonical_vector(t["cid"], it["gold_entities"])
        for k, v in t["xhat"].items():
            signed.setdefault(t["model"], {})[(t["item"], k)] = z(v) - z(gold[k])
    if "injection" in tr.columns:
        signed = {}          # erreurs injectées identiques entre modèles : П5 sans objet ici
    for m1, m2 in itertools.combinations(sorted(signed), 2):
        keys = sorted(set(signed[m1]) & set(signed[m2]))
        u = np.array([signed[m1][k] for k in keys]); w = np.array([signed[m2][k] for k in keys])
        eu, ew = np.abs(u) > 0.5, np.abs(w) > 0.5        # ошибка > 0,5 % / erreur > 0,5 %
        both, either = eu & ew, eu | ew
        corr[f"{m1}|{m2}"] = {"n_params": len(keys), "n_either": int(either.sum()), "n_both": int(both.sum()),
                              "co_error_rate": (float(both.sum() / either.sum()) if either.any() else None),
                              "sign_agreement_when_both": (float(np.mean(np.sign(u[both]) == np.sign(w[both])))
                                                           if both.any() else None)}
    # повторяемость агента 1 / reproductibilité de l'agent 1
    rep = df[df["config"] == "C1"].groupby(["model", "item"])["e1"].nunique()
    sums = {}
    for p in sorted(glob.glob(os.path.join(a.run, "raw_*.jsonl"))):
        sums[os.path.basename(p)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    with open(os.path.join(a.run, "checksums.txt"), "w") as fh:
        for k, v in sums.items():
            fh.write(f"{v}  {k}\n")
    out = {"summary": summ.to_dict(orient="records"), "error_sign_agreement": corr,
           "extraction_identical_across_repeats": (float((rep == 1).mean()) if len(rep) and df["repeat"].max() > 0 else None),
           "sha256": sums}
    json.dump(out, open(os.path.join(a.run, f"summary{sfx}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    pd.set_option("display.width", 200)
    print(summ.to_string(index=False))
    print(json.dumps({k: out[k] for k in ("error_sign_agreement", "extraction_identical_across_repeats")}, indent=1))


if __name__ == "__main__":
    main()
