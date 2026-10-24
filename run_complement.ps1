# Expériences complémentaires demandées par la revue critique (24/10/2026).
# Дополнительные эксперименты по итогам критического анализа.
#
#  A. Ablation : re-demande générique (sans valeur ni plage) vs re-demande avec plage (C3) vs projection (C1),
#     sur les MÊMES erreurs injectées que le run final (graine 2026).
#  B. Robustesse : nouvelle injection avec une autre graine (2027) -> C0, C1, C2, C3, B4 + ablation.
#
# Avant, dans le même terminal :
#   $env:PYTHONUTF8 = "1"
#   $env:OPENROUTER_API_KEY = "VOTRE_CLE"      (ne jamais la coller dans une conversation)
# Puis :
#   powershell -ExecutionPolicy Bypass -File .\run_complement.ps1
# Relancer reprend là où cela s'est arrêté (les appels réussis sont en cache).

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PY = "c:\python314\python.exe"
$VER = "..\verified"
$IMPL = "$VER\calculator_implementations"
$ITEMS = "items_main.csv"
$OUT = "runs\final"
$MODELS = "models.final.json"
$NAMES = @("llama32_3b", "llama31_8b", "llama33_70b", "mistral_small_2501", "qwen3_30b_instr", "gemma4_31b_paid")
$MAXPAR = 3

if (-not $env:OPENROUTER_API_KEY) { Write-Host "Clé OPENROUTER_API_KEY absente."; exit 1 }
if (-not (Test-Path "$OUT\trace_inject_llama33_70b.jsonl")) { Write-Host "Le run final ($OUT) est introuvable : lancer ce script dans le dossier code."; exit 1 }

# Sauvegarde intacte du run final du 23/10 (les fichiers raw_*.jsonl vont recevoir de nouveaux appels)
if (-not (Test-Path "runs\final_23oct_backup")) {
    Write-Host "== Sauvegarde de $OUT -> runs\final_23oct_backup"
    Copy-Item $OUT "runs\final_23oct_backup" -Recurse
}

Write-Host "== Vérification de la clé"
& $PY check_key.py
if ($LASTEXITCODE -ne 0) { Write-Host "Clé refusée."; exit 1 }

function Run-Parallel($script, $phase, $extra) {
    for ($i = 0; $i -lt $NAMES.Count; $i += $MAXPAR) {
        $batch = $NAMES[$i..([Math]::Min($i + $MAXPAR, $NAMES.Count) - 1)]
        $procs = @()
        foreach ($n in $batch) {
            $argList = @($script, "--dataset", $ITEMS, "--impl", $IMPL, "--models", $MODELS, "--out", $OUT, "--only", $n) + $extra
            $procs += Start-Process -FilePath $PY -ArgumentList $argList -NoNewWindow -PassThru `
                -RedirectStandardOutput "$OUT\logs\${phase}_$n.log" -RedirectStandardError "$OUT\logs\${phase}_${n}_err.log"
        }
        Write-Host "   lancés : $($batch -join ', ')"
        $procs | Wait-Process
    }
    $stop = Select-String -Path "$OUT\logs\${phase}_*.log" -Pattern "ARR.{1,2}T" -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($stop) { Write-Host "*** Arrêt (clé, crédit ou quota). Corriger puis relancer. ***"; exit 2 }
    $crash = Select-String -Path "$OUT\logs\${phase}_*_err.log" -Pattern "Traceback|MemoryError" -ErrorAction SilentlyContinue
    if ($crash) { Write-Host "*** Un processus a planté : voir $OUT\logs\${phase}_*_err.log ***"; exit 3 }
}

Write-Host "== A. Ablation sur les erreurs injectées du run final (graine 2026)"
Run-Parallel "run_ablation.py" "ablation" @()

Write-Host "== B1. Nouvelle injection, graine 2027"
Run-Parallel "run_inject.py" "inject2027" @("--seed", "2027", "--suffix=-s2027")
Write-Host "== B2. Ablation, graine 2027"
Run-Parallel "run_ablation.py" "ablation2027" @("--suffix=-s2027")

Write-Host "== C. Contrôle et analyse (aucun appel aux modèles)"
& $PY check_pilot.py $OUT
& $PY analyze.py --run $OUT --dataset $ITEMS --impl $IMPL --trace "trace_inject-s2027_*.jsonl" --tag inject-s2027
& $PY stats.py --cases "$OUT\cases_inject-s2027.csv" | Tee-Object -FilePath "$OUT\stats_s2027.txt"
& $PY run_ablation.py --dataset $ITEMS --impl $IMPL --models $MODELS --out $OUT --analyze-only | Tee-Object -FilePath "$OUT\ablation_2026.txt"
& $PY run_ablation.py --dataset $ITEMS --impl $IMPL --models $MODELS --out $OUT --analyze-only --suffix=-s2027 | Tee-Object -FilePath "$OUT\ablation_2027.txt"
Get-FileHash "$OUT\raw_*.jsonl", "$OUT\ablation_cases*.csv", "$OUT\cases_inject-s2027.csv" -Algorithm SHA256 | `
    ForEach-Object { "$($_.Hash.ToLower())  $(Split-Path $_.Path -Leaf)" } | Out-File "$OUT\checksums_complement.txt" -Encoding utf8
Write-Host "== Terminé. Envoyer : stats_s2027.txt, ablation_2026.txt, ablation_2027.txt, checksums_complement.txt, cases_inject-s2027.csv, ablation_cases*.csv, et le solde OpenRouter."
