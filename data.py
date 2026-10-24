"""Данные, единицы, официальные формулы / Données, unités, formules officielles.

Формула f берётся из официальных реализаций MedCalc-Bench-Verified, чтобы эталон
совпадал с Ground Truth Answer (проверено: 120/120 в test, 1729/1729 в train).
La formule f vient des implémentations officielles de MedCalc-Bench-Verified.
"""
import ast
import importlib
import json
import math
import sys

import pandas as pd

from config import CALCULATORS, PARAM_KINDS, ALLOWED_UNITS

_IMPL = None
_MAP = None


def set_impl_path(path):
    """Путь к calculator_implementations / chemin vers calculator_implementations."""
    global _IMPL, _MAP
    _IMPL = path
    if path not in sys.path:
        sys.path.insert(0, path)
    with open(f"{path}/name_to_python.json", encoding="utf-8") as fh:
        _MAP = json.load(fh)


def question_text(cid):
    return _MAP[str(cid)]["question"]


def official_f(cid, entities):
    """Официальная формула на словаре в формате Relevant Entities."""
    spec = _MAP[str(cid)]
    mod = importlib.import_module(spec["file path"][:-3])
    fn = getattr(mod, spec["explanation function"])
    params = {spec.get(k, k): v for k, v in entities.items()}
    return float(fn(params)["Answer"])


def _norm_unit(u):
    u = str(u).strip()
    # в данных встречаются испорченные символы µ / caractères µ corrompus dans les données
    if u.endswith("IU/mL"):
        return "µIU/mL"
    for bad in ["u", "μ", "mc"]:
        if u == bad + "L":
            return "µL"
    return u


def to_canonical(kind, value_unit):
    """[значение, единица] -> число в канонических единицах; None при ошибке."""
    import weight_conversion, height_conversion, unit_converter_new
    try:
        v, u = float(value_unit[0]), _norm_unit(value_unit[1])
    except Exception:
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    try:
        if kind == "weight":
            if u not in ALLOWED_UNITS["weight"]:
                return None
            return float(weight_conversion.weight_conversion_explanation([v, u])[1])
        if kind == "height":
            if u not in ALLOWED_UNITS["height"]:
                return None
            return float(height_conversion.height_conversion_explanation([v, u])[1])
        if kind == "age":
            return v if "year" in u else (v / 12.0 if "month" in u else None)
        if kind in ("heart_rate", "qt", "enzyme", "bmi"):
            return v if u in ALLOWED_UNITS[kind] else None
        if kind == "glucose":
            if u not in ALLOWED_UNITS["glucose"]:
                return None
            return float(unit_converter_new.conversion_explanation(v, "glucose", 180.16, None, u, "mg/dL")[1])
        if kind == "insulin":
            # та же конвенция, что в официальной реализации homa_ir.py [TO VERIFY: ×6 для pmol/L]
            factor = {"µIU/mL": 1.0, "pmol/L": 6.0, "ng/mL": 24.8}.get(u)
            return None if factor is None else v * factor
        if kind == "platelets":
            if u in ("10^9/L", "10^3/µL", "10^3/uL", "x10^9/L", "×10^9/L"):
                return v * 1e9                     # 10^3/µL = 10^9/L
            if u not in ALLOWED_UNITS["platelets"]:
                return None
            return float(unit_converter_new.convert_to_units_per_liter_explanation(v, u, "platelets", "L")[1])
    except Exception:
        return None
    return None


def canonical_to_entity(kind, x):
    """Каноническое число -> [значение, единица] в формате Relevant Entities."""
    unit = {"weight": "kg", "height": "m", "age": "years", "heart_rate": "beats per minute",
            "qt": "msec", "glucose": "mg/dL", "insulin": "µIU/mL", "enzyme": "U/L",
            "platelets": "L", "bmi": "kg/m^2"}[kind]
    return [x, unit]


def canonical_vector(cid, entities):
    """Словарь сущностей -> {ключ: каноническое значение}; None, если чего-то нет."""
    out = {}
    for key, (kind, _) in CALCULATORS[cid]["params"].items():
        if key not in entities:
            return None
        x = to_canonical(kind, entities[key])
        if x is None:
            return None
        out[key] = x
    return out


def f_from_canonical(cid, xc):
    ents = {k: canonical_to_entity(CALCULATORS[cid]["params"][k][0], v) for k, v in xc.items()}
    return official_f(cid, ents)


def load_items(csv_path, calc_ids=None, n_per_calc=None, seed=2026):
    d = pd.read_csv(csv_path)
    calc_ids = calc_ids or list(CALCULATORS)
    d = d[d["Calculator ID"].isin(calc_ids)]
    if n_per_calc:
        parts = [g.sample(min(n_per_calc, len(g)), random_state=seed)
                 for _, g in d.groupby("Calculator ID")]
        d = pd.concat(parts)
    items = []
    for _, r in d.iterrows():
        items.append({
            "row": int(r["Row Number"]), "cid": int(r["Calculator ID"]),
            "note_id": str(r["Note ID"]), "note": r["Patient Note"], "question": r["Question"],
            "gold_entities": ast.literal_eval(r["Relevant Entities"]),
            "gt": float(r["Ground Truth Answer"]),
            "lo": float(r["Lower Limit"]), "hi": float(r["Upper Limit"]),
            "split": str(r["Split"]) if "Split" in r and isinstance(r["Split"], str) else "test",
        })
    return items
