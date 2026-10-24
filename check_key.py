"""Vérifie la clé OpenRouter avec un appel minuscule (coût ≈ 0) / Проверка ключа.
python check_key.py
"""
import os
import sys

from openai import OpenAI

key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    sys.exit("OPENROUTER_API_KEY absente dans ce terminal.")
print("Clé utilisée : début", key[:10], "... longueur", len(key))
cli = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=60, max_retries=0)
try:
    r = cli.chat.completions.create(model="meta-llama/llama-3.2-3b-instruct",
                                    messages=[{"role": "user", "content": "Say OK"}], max_tokens=3)
    print("OK : la clé fonctionne. Réponse :", (r.choices[0].message.content or "").strip())
except Exception as e:
    print("ÉCHEC :", repr(e)[:300])
    sys.exit(1)
