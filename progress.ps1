# Avancement du run final / прогресс: nombre d'appels enregistrés par modèle
$OUT = "runs\final"
Get-ChildItem "$OUT\raw_*.jsonl" | ForEach-Object {
    $n = (Get-Content $_.FullName | Measure-Object -Line).Lines
    "{0,-32} {1,6} appels" -f $_.Name, $n
}
Get-ChildItem "$OUT\logs\*.log" | Where-Object { $_.Length -gt 0 } | ForEach-Object {
    "--- $($_.Name) : " + (Get-Content $_.FullName -Tail 1)
}
