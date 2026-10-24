# Protocole figé de l'expérience finale — 22 octobre 2026
# Зафиксированный протокол итогового эксперимента — 22 октября 2026

Ce protocole est fixé AVANT le lancement de l'expérience finale. Toute modification ultérieure
sera datée et signalée dans l'article. / Протокол зафиксирован до запуска; любые изменения датируются и указываются в статье.

## 1. Données
- MedCalc-Bench-Verified, dépôt GitHub nikhilk7153/MedCalc-Bench-Verified, commit 801592132bfd833f049b517a4e013e00a5fc40fa (8 sept. 2026), licence CC-BY-SA 4.0.
- 6 calculateurs à formule « produit de puissances » : IMC (6), surface corporelle (60), QTc Bazett (11), HOMA-IR (31), FIB-4 (19), poids cible (61).
- **Test** : les 120 exemples de ces calculateurs (20 chacun). 54 d'entre eux ont servi aux pilotes (déclaré).
- **Train filtré** : 20 exemples par calculateur, tirés avec la graine 2026 parmi ceux qui passent le filtre :
  valeurs de référence convertibles, toutes dans les couloirs de plausibilité, réponse de référence dans le couloir
  de sortie, note absente du test (`prepare_items.py`). Aucun apprentissage n'est fait sur ces exemples.
- Référence de la formule : implémentations officielles du benchmark (reproduisent la réponse de référence pour 100 % des exemples de ces calculateurs).

## 2. Modèles (OpenRouter, modèles ouverts, température 0, 1 passage)
llama-3.2-3b-instruct, llama-3.1-8b-instruct, llama-3.3-70b-instruct (Meta) ; mistral-small-24b-instruct-2501 (Mistral) ;
qwen3-30b-a3b-instruct-2507 (Qwen) ; gemma-4-31b-it (Google). Le modèle effectivement servi est enregistré à chaque appel.

## 3. Prompts (fichier prompts.py, inchangés depuis la v8)
- Agent 1 : extraction en JSON, sans raisonnement.
- Agent 2 et B0 : raisonnement bref autorisé, dernier objet JSON lu.
- Re-demande (C3) : la sortie de l'agent 1, les valeurs hors couloir et la plage sont renvoyées à l'agent 1.

## 4. Configurations
B0 modèle seul ; C0 sans couloir ; C1 projection sur les couloirs (entrées et sortie) ; C2 couloir ±5 % autour de la
formule (orchestrateur connaissant la formule) ; C3 re-demande à l'agent 1 si hors couloir, puis projection ;
B4 calcul par le code sur la sortie projetée de l'agent 1. Couloirs : config.py (plages illustratives fixées avant les pilotes, non modifiées).

## 5. Injection contrôlée (run_inject.py)
Sur la sortie RÉELLE de l'agent 1 de chaque modèle, un paramètre (tiré par graine liée à l'exemple, identique pour tous
les modèles) est corrompu : décimale (×10 / ÷10), échelle (×1000 / ÷1000) ou confusion d'unité propre au paramètre,
selon un cycle fixé. Les erreurs injectées sont présentées comme telles dans l'article.

## 6. Critères et analyses (analyze.py, stats.py)
- Critère principal : réponse dans les limites du benchmark (Lower/Upper Limit).
- Vérification des bornes (а), (б) de l'утверждение 2 et du corollaire 2 : nombre de violations (attendu : 0).
- Comparaisons appariées pré-déclarées : C1 vs C0, C3 vs C1, C3 vs C0, C2 vs C0, B4 vs C0 ; test exact de McNemar ;
  IC 95 % bootstrap (10 000 tirages, graine 2026). Résultats rapportés séparément pour test et train, sans correction
  multiple déclarée comme confirmatoire : les comparaisons sont exploratoires et présentées comme telles.
- Injection : précision par type d'erreur ; part des erreurs détectées par le couloir.
- П5 (co-occurrence des erreurs d'extraction entre modèles) : rapporté seulement si au moins 10 paires d'erreurs communes.

## 7. Traçabilité
Réponses brutes en JSONL jamais modifiées ; empreintes SHA-256 (checksums.txt) ; coût total rapporté.
