"""Contrôle rapide d'un run réel / Быстрая проверка запуска.
python check_pilot.py runs/pilot

Chaque appel est compté une fois : s'il a fini par réussir, les anciennes erreurs sont ignorées.
"""
import collections, glob, json, os, sys

run = sys.argv[1]
for p in sorted(glob.glob(os.path.join(run, "raw_*.jsonl"))):
    last = {}
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        prev = last.get(r["key"])
        if prev is None or prev.get("error") is not None or prev.get("finish_reason") in ("length", "error"):
            last[r["key"]] = r
    st = collections.Counter(); errs = collections.Counter(); tok_in = tok_out = n = 0
    for r in last.values():
        if r.get("error"):
            st["error"] += 1; errs[r["error"][:320]] += 1; continue
        n += 1
        st["finish=" + str(r.get("finish_reason"))] += 1
        st["json_ok" if isinstance(r.get("parsed"), dict) else "json_FAIL"] += 1
        u = r.get("usage") or {}
        tok_in += u.get("prompt_tokens") or 0; tok_out += u.get("completion_tokens") or 0
    print(os.path.basename(p), f"appels distincts: {len(last)}", dict(st))
    if n:
        print(f"   tokens moyens / appel : entrée {tok_in/n:.0f}, sortie {tok_out/n:.0f}")
    for e, c in errs.most_common(3):
        print(f"   erreur restante x{c}: {e}")
    if not st.get("error"):
        print("   OK : aucun appel en échec.")
