# Expérience finale — lance les 6 modèles EN PARALLÈLE, puis l'injection, puis l'analyse.
# Финальный эксперимент: 6 моделей параллельно, затем внесение ошибок, затем анализ.
#
# Avant : dans le même terminal
#   $env:PYTHONUTF8 = "1"
#   $env:OPENROUTER_API_KEY = "VOTRE_CLE"
# Puis :
#   powershell -ExecutionPolicy Bypass -File .\run_final.ps1
# Relancer le même script reprend là où il s'est arrêté (les appels réussis ne sont pas refaits).

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

if (-not $env:OPENROUTER_API_KEY) { Write-Host "Clé OPENROUTER_API_KEY absente : faites d'abord `$env:OPENROUTER_API_KEY = ""...""" ; exit 1 }
New-Item -ItemType Directory -Force -Path $OUT, "$OUT\logs" | Out-Null

if (-not (Test-Path $ITEMS)) {
    Write-Host "== 0. Préparation des exemples (test 120 + train filtré 120)"
    & $PY prepare_items.py --verified $VER --out $ITEMS
}

$MAXPAR = 3   # nombre de modèles lancés en même temps (mémoire limitée) / одновременно

function Run-Parallel($script, $phase, $extra) {
    for ($i = 0; $i -lt $NAMES.Count; $i += $MAXPAR) {
        $batch = $NAMES[$i..([Math]::Min($i + $MAXPAR, $NAMES.Count) - 1)]
        $procs = @()
        foreach ($n in $batch) {
            $argList = @($script, "--dataset", $ITEMS, "--impl", $IMPL, "--models", $MODELS, "--out", $OUT, "--only", $n) + $extra
            $procs += Start-Process -FilePath $PY -ArgumentList $argList -NoNewWindow -PassThru `
                -RedirectStandardOutput "$OUT\logs\${phase}_$n.log" -RedirectStandardError "$OUT\logs\${phase}_${n}_err.log"
        }
        Write-Host "   lancés : $($batch -join ', ') ; suivi : $OUT\logs\${phase}_*.log"
        $procs | Wait-Process
    }
    foreach ($n in $NAMES) {
        $tail = Get-Content "$OUT\logs\${phase}_$n.log" -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
        Write-Host "   $n : $tail"
    }
    $stop = Select-String -Path "$OUT\logs\${phase}_*.log" -Pattern "ARR.{1,2}T" -Encoding UTF8 -ErrorAction SilentlyContinue
    if ($stop) {
        Write-Host "`n*** Un ou plusieurs modèles se sont arrêtés (clé refusée, crédit ou quota). Corrigez puis relancez ce script. ***"
        $stop | ForEach-Object { Write-Host "   $($_.Filename) : $($_.Line)" }
        exit 2
    }
}

Write-Host "== Vérification de la clé"
& $PY check_key.py
if ($LASTEXITCODE -ne 0) { Write-Host "Clé refusée : corrigez `$env:OPENROUTER_API_KEY puis relancez."; exit 1 }

# crash (mémoire, etc.) : un journal d'erreurs avec "Traceback" ou "error" bloque la suite
function Check-Crash($phase) {
    $crash = Select-String -Path "$OUT\logs\${phase}_*_err.log" -Pattern "Traceback|MemoryError|OpenBLAS error" -ErrorAction SilentlyContinue
    if ($crash) {
        Write-Host "`n*** Un processus a planté (voir ci-dessous). Fermez les autres programmes puis relancez ce script. ***"
        $crash | Select-Object -First 3 | ForEach-Object { Write-Host "   $($_.Filename) : $($_.Line)" }
        exit 3
    }
}

Write-Host "== 1. Passage principal (par groupes de $MAXPAR modèles)"
Run-Parallel "run_pilot.py" "main" @("--repeats", "1")
Check-Crash "main"

Write-Host "== 2. Injection contrôlée d'erreurs (par groupes de $MAXPAR modèles)"
Run-Parallel "run_inject.py" "inject" @()
Check-Crash "inject"

Write-Host "== 3. Contrôle"
& $PY check_pilot.py $OUT

Write-Host "== 4. Analyse"
& $PY analyze.py --run $OUT --dataset $ITEMS --impl $IMPL --trace "trace_main_*.jsonl"
& $PY analyze.py --run $OUT --dataset $ITEMS --impl $IMPL --trace "trace_inject_*.jsonl" --tag inject
& $PY stats.py --cases "$OUT\cases.csv" --cases "$OUT\cases_inject.csv" | Tee-Object -FilePath "$OUT\stats.txt"
Write-Host "== Terminé. Envoyer : $OUT\stats.txt, cases.csv, cases_inject.csv, checksums.txt, et le solde OpenRouter."
