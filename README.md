# Error propagation in technologically linked LLM agents: corridors, projection and re-querying

Code, prompts, selected examples, raw model responses and result tables for the article

> С.Н. Нсилулу и И.Л. Кашириной. *Оценка распространения непреднамеренных ошибок в технологически связанных подзадачах агентов на основе больших языковых моделей* (Voronezh State University, 2026, in preparation).
>
> S.N. Nsilulu, I.L. Kashirina. *Estimating the propagation of unintentional errors in technologically linked subtasks of large language model agents.*

The repository contains everything needed to **re-compute every number of the experimental section without calling any model**, and to re-run the experiment with your own OpenRouter key.

## What the experiment does

A two-agent pipeline on six formula calculators of [MedCalc-Bench-Verified](https://github.com/nikhilk7153/MedCalc-Bench-Verified) (BMI, body surface area, QTc Bazett, HOMA-IR, FIB-4, target weight):

1. **Agent 1** extracts the parameters (value + unit, JSON) from the patient note.
2. The **orchestrator** checks them against plausibility corridors.
3. **Agent 2** computes the score from the extracted values only.

| Configuration | Meaning |
| --- | --- |
| B0 | one model, end to end, from the note |
| C0 | pipeline, no corridor |
| C1 | projection of parameters and result onto the corridors |
| C2 | output corridor ±5 % around the formula value (orchestrator knows the formula) |
| C3 | counter-request: if a parameter is out of corridor, agent 1 is told the value and the plausible range and asked to re-read the note |
| C3g | same trigger, generic request without value or range |
| C3all | generic request for every example |
| B4 | agent 2 replaced by code |

Six open models via OpenRouter, temperature 0, one call per prompt: `meta-llama/llama-3.2-3b-instruct`, `meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-3.3-70b-instruct`, `mistralai/mistral-small-24b-instruct-2501`, `qwen/qwen3-30b-a3b-instruct-2507`, `google/gemma-4-31b-it`.

## Repository layout

| Path | Content |
| --- | --- |
| `config.py`, `data.py`, `prompts.py`, `llm.py` | calculators, exact Lipschitz constants (log metric), corridors, unit conversion, prompts, API calls with append-only JSONL cache |
| `prepare_items.py` → `items_main.csv` | the 240 evaluation examples (120 test + 120 filtered train, seed 2026) |
| `run_pilot.py` | main run: B0, agent 1, agent 2 on raw / projected / counter-requested / reference inputs |
| `run_inject.py` | controlled injection of one error into agent 1's real output (decimal ×10/÷10, scale ×1000/÷1000, unit confusion); `--seed`, `--suffix` |
| `run_ablation.py` | C1 vs C3 vs C3g vs C3all on the same injected errors |
| `run_recheck_natural.py` | C3all on real, non-corrupted extractions (repairs vs degradations) |
| `analyze.py` | offline analysis: accuracy, bounds (3.1), (3.2), (3.4), counterfactuals → `cases*.csv`, `summary*` |
| `stats.py` | bootstrap CIs, exact McNemar tests |
| `analysis/holm_and_sensitivity.py` | Holm correction, CIs of differences, analysis without the 54 pilot examples |
| `analysis/verify_checksums.py` | checks every published SHA-256 |
| `simulation/graph_simulation.py` | section 6: random agent graphs (seed 2026, ~2 min), fixed vs input-dependent corridors; output `simulation/results.json`, `results.txt` |
| `run_final.ps1`, `run_complement.ps1`, `run_recheck.ps1` | the exact Windows launch scripts used (23–24.10.2026) |
| `runs/final/raw_*.jsonl` | every model call: prompt, model actually served, raw response, finish reason, tokens, latency |
| `runs/final/trace_*.jsonl` | links each example to its calls (main run, injection seeds 2026 and 2027) |
| `runs/final/*.csv`, `*.txt`, `*.json` | per-case results and printed statistics |
| `docs/PROTOCOL.md` | protocol frozen before the final run (22.10.2026) |
| `docs/NOTES_pilot_fr_ru.md` | working notes from the pilot phase (FR/RU) |

Not included: pilot runs, process logs, local backups, API keys.

## Reproduce the numbers without any API call

```bash
git clone https://github.com/<user>/<this-repo>.git
git clone https://github.com/nikhilk7153/MedCalc-Bench-Verified.git verified
cd verified && git checkout 801592132bfd833f049b517a4e013e00a5fc40fa && cd ..
cd <this-repo>
pip install -r requirements.txt

python analysis/verify_checksums.py
python analyze.py --run runs/final --dataset items_main.csv --impl ../verified/calculator_implementations --trace "trace_main_*.jsonl"
python analyze.py --run runs/final --dataset items_main.csv --impl ../verified/calculator_implementations --trace "trace_inject_*.jsonl" --tag inject
python stats.py --cases runs/final/cases.csv --cases runs/final/cases_inject.csv
python analysis/holm_and_sensitivity.py --verified ../verified
python simulation/graph_simulation.py
python run_ablation.py --dataset items_main.csv --impl ../verified/calculator_implementations --models models.final.json --out runs/final --analyze-only
python run_recheck_natural.py --dataset items_main.csv --impl ../verified/calculator_implementations --models models.final.json --out runs/final --analyze-only
```

Note: `analyze.py` rewrites `runs/final/checksums.txt`; the reference checksums are the `checksums_*.txt` files.

Regenerating from the current logs reproduces `cases.csv` exactly. In `cases_inject.csv` two rows differ (Llama 3.1 8B, examples 99 and 2441, configuration C0): these calls were truncated on 23.10 and the identical prompt was answered completely during the seed-2027 run on 24.10; the cache now returns the complete answer. Both answers are outside the tolerance interval, so all accuracies and tests are unchanged. To reproduce the 23.10 state byte for byte, keep only the first N lines of each `raw_*.jsonl` (N is given in `checksums_23oct.txt`).

Because every call is cached in `raw_*.jsonl` under a hash of (model, stage, example, prompt), re-running `run_pilot.py`, `run_inject.py`, `run_ablation.py` or `run_recheck_natural.py` with these files present makes **no new call** and reproduces the traces exactly. To re-query the models, start from an empty output folder and set `OPENROUTER_API_KEY` in your environment (never in a file). The PowerShell scripts call `c:\python314\python.exe`; edit `$PY` for your machine. `models.mock.json` runs the whole pipeline with a simulator to test the code (simulator numbers are not results).

## Checksums

The raw logs are append-only. Each stage recorded its own checksums:

| File | Stage |
| --- | --- |
| `checksums_23oct.txt` | final run, 23.10.2026 (hash of the first N lines) |
| `checksums_complement.txt` | + ablation and second injection seed, 24.10.2026 |
| `checksums_recheck.txt` | + re-check on clean extractions, 24.10.2026 (= current files) |

`analysis/verify_checksums.py` checks each hash against the whole file or its line prefix.

## Declared deviations and limitations

- 54 of the 120 test examples were used in the pilots (prompt tuning); results without them are computed by `analysis/holm_and_sensitivity.py`.
- Train examples were selected with reference values inside the corridors (favourable to corridors); main conclusions use the test split.
- C3 was added after the pilots, before the final run; C3g, C3all, the second injection seed and the clean-data re-check were added after the final run.
- One run at temperature 0; corridors are illustrative, not clinical; no physician evaluation; injected errors are artificial.
- Labels of the original MedCalc-Bench were found partly erroneous by independent audits; the Verified version is used, and its official implementations reproduce the reference answers on all selected examples.

## Licences

- Code (`*.py`, `*.ps1`): MIT, see `LICENSE`.
- `items_main.csv` and everything in `runs/` contain patient notes and labels from MedCalc-Bench-Verified and are distributed under **CC BY-SA 4.0**, like the source dataset. See `DATA_LICENSE.md`.

## Кратко по-русски

Репозиторий содержит код, промпты, отобранные примеры, сырые ответы моделей с контрольными суммами SHA-256 и таблицы результатов к статье С.Н. Нсилулу и И.Л. Кашириной. Все числа экспериментального раздела воспроизводятся без обращения к моделям (раздел «Reproduce the numbers…»). Код распространяется по лицензии MIT, данные и результаты — по лицензии CC BY-SA 4.0, как исходный набор MedCalc-Bench-Verified.
