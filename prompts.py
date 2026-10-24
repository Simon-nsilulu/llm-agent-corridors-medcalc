"""Промпты агентов (английский — язык набора данных) / Prompts des agents.

Тексты промптов сохраняются в журнале каждого вызова и публикуются в приложении статьи.
"""
import json

from config import CALCULATORS, ALLOWED_UNITS

SYSTEM = "You are a careful clinical data assistant. Reply with a single JSON object and nothing else."
SYSTEM_REASON = ("You are a careful clinical calculation assistant. Work step by step, briefly. "
                 "End your reply with a single JSON object on the last line.")


def extraction(item):
    spec = CALCULATORS[item["cid"]]["params"]
    fields = "\n".join(f'- "{k}": [value, unit], unit one of {ALLOWED_UNITS[kind]}'
                       for k, (kind, _) in spec.items())
    user = (f"Patient note:\n{item['note']}\n\n"
            f"Extract the following parameters from the note. Report each value as stated in the note, "
            f"with its unit. If the note uses another unit, convert to one of the allowed units. "
            f"For counts, give the full count for the stated unit (e.g. 250000 with unit µL, or 250 with unit 10^9/L).\n"
            f"{fields}\n\nReturn JSON with exactly these keys.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def calculation(item, question, entities):
    user = (f"Task: {question}\n\n"
            f"Use only these patient parameters:\n{json.dumps(entities, ensure_ascii=False)}\n\n"
            'Compute the result step by step, then give the final line as JSON: {"answer": <number>}.')
    return [{"role": "system", "content": SYSTEM_REASON}, {"role": "user", "content": user}]


def end_to_end(item):
    user = (f"Patient note:\n{item['note']}\n\nQuestion: {item['question']}\n\n"
            'Work step by step, then give the final line as JSON: {"answer": <number>}.')
    return [{"role": "system", "content": SYSTEM_REASON}, {"role": "user", "content": user}]


def requery(item, previous_response, viol):
    """Встречный запрос оркестратора / re-demande de l'orchestrateur (C3)."""
    lines = "\n".join(f'- "{k}": your value converts to {v:.4g} {u}, outside the plausible range [{lo:.4g}, {hi:.4g}] {u}.'
                       for k, (v, lo, hi, u) in viol.items())
    msgs = extraction(item)
    msgs.append({"role": "assistant", "content": previous_response or ""})
    msgs.append({"role": "user", "content":
                 "The orchestrator flagged these values as implausible:\n" + lines +
                 "\nRe-read the note and check the value and its unit. Return the corrected JSON with all keys."})
    return msgs


def requery_generic(item, previous_response):
    """Абляция C3g: повторный запрос БЕЗ значения и диапазона / re-demande SANS valeur ni plage."""
    msgs = extraction(item)
    msgs.append({"role": "assistant", "content": previous_response or ""})
    msgs.append({"role": "user", "content":
                 "The orchestrator asks you to double-check your extraction. "
                 "Re-read the note and check each value and its unit. Return the corrected JSON with all keys."})
    return msgs
