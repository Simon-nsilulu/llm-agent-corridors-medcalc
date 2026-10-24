"""Вызовы моделей с журналом и кэшем / Appels aux modèles avec journal et cache.

Любой OpenAI-совместимый сервер: OpenAI, OpenRouter, vLLM (Kaggle/Colab),
Gemini через OpenAI-совместимую точку доступа [TO VERIFY: URL в документации Google].
Ключи читаются только из переменных окружения / clés lues uniquement depuis l'environnement.
"""
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import time


class ModelUnavailable(Exception):
    """Модель недоступна (перегрузка у поставщика) / modèle saturé chez le fournisseur."""


class QuotaExhausted(Exception):
    """Квота исчерпана: продолжать бессмысленно / quota épuisé : inutile de continuer."""


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]


class Logger:
    """Сырые ответы в JSONL; файл только дописывается / JSONL brut, en ajout seulement."""

    def __init__(self, path):
        self.path = path
        self.done = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if self._usable(rec):
                        self.done[rec["key"]] = rec

    def get(self, key):
        return self.done.get(key)

    def write(self, rec):
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self._usable(rec):
            self.done[rec["key"]] = rec

    @staticmethod
    def _usable(rec):
        """Ответ можно переиспользовать / réponse réutilisable : pas d'erreur, pas tronquée."""
        return rec.get("error") is None and rec.get("finish_reason") not in ("length", "error")


def parse_json(text):
    """Последний JSON-объект в ответе (после рассуждения) / dernier objet JSON de la réponse."""
    if text is None:
        return None
    dec = json.JSONDecoder()
    found = None
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    found = obj
            except Exception:
                pass
    return found


class Model:
    def __init__(self, spec, logger):
        self.spec = spec            # {"name","provider","model","base_url","api_key_env","max_tokens"}
        self.log = logger
        self.client = None
        if spec["provider"] != "mock":
            from openai import OpenAI
            # délai max par appel 180 s, pas de reprise interne (nos reprises sont gérées plus bas)
            self.client = OpenAI(base_url=spec.get("base_url"),
                                 api_key=os.environ.get(spec.get("api_key_env", ""), "EMPTY"),
                                 timeout=float(spec.get("timeout_s", 180)), max_retries=0)

    def _create_with_retry(self, kw):
        """Пауза между вызовами и повтор при 429/5xx / pause entre appels et reprise sur 429/5xx.

        Ключи в models.json: "min_interval_s" (пауза, по умолчанию 0), "max_retries" (по умолчанию 6).
        """
        gap = float(self.spec.get("min_interval_s", 0))
        wait = gap - (time.time() - getattr(self, "_last", 0.0))
        if wait > 0:
            time.sleep(wait)
        delay = 10.0
        for attempt in range(int(self.spec.get("max_retries", 6)) + 1):
            try:
                self._last = time.time()
                return self.client.chat.completions.create(**kw)
            except Exception as e:
                code = getattr(e, "status_code", None)
                msg = str(e)
                if code in (401, 402, 403):
                    # clé invalide/supprimée ou crédit épuisé : inutile de continuer
                    raise QuotaExhausted(f"[{code}] clé API refusée ou crédit insuffisant : {msg[:200]}")
                if code is None and type(e).__name__ in ("APITimeoutError", "APIConnectionError"):
                    code = 504                               # délai dépassé / coupure : on réessaie
                if code not in (429, 500, 502, 503, 504) or attempt == self.spec.get("max_retries", 6):
                    raise
                low = msg.lower()
                if code == 429 and ("per day" in low or "perday" in low or "per-day" in low
                                    or "exceeded your current quota" in low or "insufficient_quota" in low):
                    raise QuotaExhausted(msg[:300])   # ждать бесполезно / attendre ne sert à rien
                m = re.search(r"retry in ([0-9.]+)s|retryDelay['\"]?:\s*['\"]?([0-9.]+)s", msg)
                pause = float(m.group(1) or m.group(2)) + 1 if m else delay
                print(f"   [{code}] pause {pause:.0f} s (tentative {attempt + 1})", flush=True)
                time.sleep(pause)
                delay = min(delay * 2, 120.0)

    def call(self, stage, item_id, repeat, messages, mock_fn=None):
        key = _key(self.spec["name"], stage, item_id, repeat, messages)
        cached = self.log.get(key)
        if cached:
            return cached
        rec = {"key": key, "stage": stage, "item": item_id, "repeat": repeat,
               "model_name": self.spec["name"], "provider": self.spec["provider"],
               "model": self.spec.get("model"), "base_url": self.spec.get("base_url"),
               "temperature": 0, "messages": messages,
               "ts": dt.datetime.now(dt.timezone.utc).isoformat(), "error": None}
        t0 = time.time()
        try:
            if self.spec["provider"] == "mock":
                text, usage, served = mock_fn(), None, "mock"
            else:
                kw = dict(model=self.spec["model"], messages=messages)
                if not self.spec.get("no_temperature"):   # некоторые модели рассуждения не принимают temperature
                    kw["temperature"] = 0
                # модели рассуждения OpenAI: "max_completion_tokens" / modèles de raisonnement OpenAI
                kw[self.spec.get("token_param", "max_tokens")] = self.spec.get("max_tokens", 400)
                if self.spec.get("reasoning_effort"):     # ограничить «размышление» / limiter la réflexion
                    kw["reasoning_effort"] = self.spec["reasoning_effort"]
                if self.spec.get("send_seed"):            # не все серверы принимают seed
                    kw["seed"] = 2026 + repeat
                r = self._create_with_retry(kw)
                text = r.choices[0].message.content
                served = getattr(r, "model", None)
                usage = r.usage.model_dump() if r.usage else None
                rec["finish_reason"] = r.choices[0].finish_reason   # "length" = ответ обрезан / réponse tronquée
            rec.update(response=text, parsed=parse_json(text), usage=usage, served_model=served)
        except Exception as e:  # ошибка сохраняется, при перезапуске вызов повторится
            rec.update(response=None, parsed=None, usage=None, error=repr(e)[:500])
            rec["latency_s"] = round(time.time() - t0, 3)
            self.log.write(rec)
            if isinstance(e, QuotaExhausted):
                raise
            code = getattr(e, "status_code", None)
            if code in (429, 500, 502, 503, 504):
                self._fails = getattr(self, "_fails", 0) + 1
                if self._fails >= int(self.spec.get("skip_after_failures", 2)):
                    raise ModelUnavailable(repr(e)[:300])
            return rec
        rec["latency_s"] = round(time.time() - t0, 3)
        self.log.write(rec)
        self._fails = 0
        return rec


# ---------- Имитация для сухого прогона / simulation pour essai à blanc ----------
class MockBehaviour:
    """Не модель БЯМ! Только для проверки кода / Pas un LLM ! Seulement pour tester le code."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def extraction(self, gold):
        out = {}
        for k, (v, u) in ((k, (vv[0], vv[1])) for k, vv in gold.items() if isinstance(vv, list)):
            r = self.rng.random()
            if r < 0.08:
                v = v * self.rng.choice([10, 0.1, 2.2])      # грубая ошибка / erreur grossière
            elif r < 0.6:
                v = v * math.exp(self.rng.gauss(0, 0.02))
            out[k] = [round(v, 4), u]
        return json.dumps(out, ensure_ascii=False)

    def calculation(self, f_true):
        r = self.rng.random()
        y = f_true * (self.rng.choice([10, 0.5]) if r < 0.05 else math.exp(self.rng.gauss(0, 0.01)))
        return json.dumps({"answer": y})

    def end_to_end(self, gt):
        return json.dumps({"answer": gt * math.exp(self.rng.gauss(0, 0.15))})
