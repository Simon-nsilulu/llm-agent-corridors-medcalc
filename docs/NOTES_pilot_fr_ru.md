# Pilote MedCalc : pipeline extraction → calcul avec couloirs
# Пилот MedCalc: конвейер «извлечение → расчёт» с коридорами

Protocole : section 8 de l'article « Оценка распространения непреднамеренных ошибок в технологически связанных подзадачах агентов… ».
Протокол: раздел 8 статьи.

## 1. Ce que fait le code / Что делает код

| Fichier | Rôle |
| --- | --- |
| `config.py` | 6 calculateurs à formule produit de puissances (BMI, BSA, QTc Bazett, HOMA-IR, FIB-4, Target weight) ; constante M_f exacte en métrique log ; plages de plausibilité (illustratives, non cliniques) ; τ du couloir C2 |
| `data.py` | Chargement de MedCalc-Bench-Verified ; conversion des unités ; formule f = implémentation officielle du benchmark |
| `prompts.py` | Prompts : agent 1 (extraction JSON), agent 2 (calcul), B0 (modèle seul) |
| `llm.py` | Appels OpenAI-compatibles (OpenAI, OpenRouter, vLLM, Gemini) ; journal JSONL brut, reprise automatique ; simulateur `mock` pour tester le code |
| `run_pilot.py` | Exécute B0, A1 et A2 (entrée brute, entrée après couloir, entrée de référence = contrefactuel) |
| `analyze.py` | Calcule hors ligne C0/C1/C2/B4, vérifie les bornes (а), (б), le corollaire 2, П5, et écrit `cases.csv`, `summary.csv`, `summary.json`, `checksums.txt` |

Vérifié ici : les implémentations officielles reproduisent exactement le Ground Truth Answer pour ces calculateurs (test 120/120, train 1729/1729). Un essai à blanc avec le simulateur donne 0 violation des bornes garanties. **Les chiffres du simulateur ne sont pas des résultats et ne doivent jamais être rapportés.**

## 2. Préparation (Kaggle ou Colab)

```bash
git clone --depth 1 https://github.com/nikhilk7153/MedCalc-Bench-Verified.git verified
cd verified && git rev-parse HEAD   # noter le commit dans le journal (ici : 8015921…, 8 sept. 2026)
cd ..
pip install openai pandas numpy
```

Clés API : **uniquement** dans les Secrets Kaggle/Colab, exportées en variables d'environnement (`OPENAI_API_KEY`, `GEMINI_API_KEY`…). Ne jamais les écrire dans un fichier ni les envoyer dans la conversation.

Modèle ouvert sur GPU Kaggle (vLLM) :

```bash
pip install vllm
nohup vllm serve <identifiant-du-modèle-HF> --max-model-len 8192 > vllm.log 2>&1 &
# serveur OpenAI-compatible sur http://localhost:8000/v1
```

Copier `models.example.json` en `models.json` et remplir les identifiants exacts des modèles choisis. Pour chaque modèle, noter la version servie (champ `served_model` du journal).

## 3. Pilote (50 exemples ≈ 9 par calculateur)

```bash
python run_pilot.py --dataset verified/datasets/test_data.csv \
  --impl verified/calculator_implementations --models models.json \
  --n-per-calc 9 --repeats 3 --out runs/pilot
python analyze.py --run runs/pilot --dataset verified/datasets/test_data.csv \
  --impl verified/calculator_implementations
```

Appels par exemple, modèle et répétition : B0 + A1 + jusqu'à 3 × A2. Les appels A2 dont l'entrée est identique sont réutilisés depuis le cache. Si l'exécution est interrompue, relancer la même commande : les appels déjà réussis ne sont pas refaits.

**À regarder après le pilote :**
- `fail_rate` : taux de réponses non analysables. S'il est élevé, il faut corriger les prompts avant le run complet.
- Nombre de tokens réel (champ `usage`) : il sert à recalculer le budget.
- `violations_bound` et `violations_attr` : ils doivent valoir 0. Sinon, c'est une erreur d'implémentation ou de constante. Envoyez-moi les fichiers.

## 4. Run complet

Le test ne contient que **20 exemples par calculateur, soit 120**, ce qui est peu pour des statistiques. Proposition :
- garder le test complet comme jeu principal ;
- ajouter un échantillon stratifié du split train comme jeu d'évaluation supplémentaire (par exemple `--n-per-calc 100` sur `train_data.csv`) ;
- aucun apprentissage n'est fait, mais il faut le déclarer dans l'article.

## 5. Ce que mesure chaque configuration

| Config | Couloir 1 (entrées) | Couloir 2 (sortie) | L₂ |
| --- | --- | --- | --- |
| B0 | — (un seul modèle, note → réponse) | — | — |
| C0 | non | non | 0 |
| C1 | plausibilité | plausibilité de la sortie | 0 |
| C2 | plausibilité | [f(x₁)e^(−τ), f(x₁)e^(τ)] | M_f |
| B4 | plausibilité | agent 2 remplacé par le code (e₂ = 0) | — |

**Deux contrefactuels sont calculés :**
- `attr_struct` : e₂ est fixée. C'est le cas couvert par le corollaire 2, et la borne est garantie.
- `attr_llm` : l'agent 2 est réellement rappelé sur les valeurs de référence. C'est une mesure empirique, car l'erreur de l'agent 2 change avec son entrée. `llm_cf_exceeds_bound` mesure à quelle fréquence cet effet dépasse la borne. C'est un résultat intéressant en soi, à rapporter honnêtement.

## 6. Points à vérifier / Что проверить

- URL Gemini vérifiée le 15/10/2026 : `https://generativelanguage.googleapis.com/v1beta/openai/`. La réflexion ne peut pas être coupée sur les modèles Gemini 3 : on la limite avec `reasoning_effort: "low"` et on prévoit `max_tokens` large (2000), sinon la réponse est tronquée (`finish_reason = length`).
- Après tout run réel : `python check_pilot.py runs/pilot` (erreurs, JSON valides, réponses tronquées, tokens moyens).
- [TO VERIFY] Conversion de l'insuline pmol/L : l'implémentation officielle multiplie par 6. On garde la même convention pour rester cohérent avec le benchmark, mais le facteur semble inversé ; à signaler aux auteurs du benchmark si c'est confirmé.
- Le Cockcroft–Gault du benchmark utilise un poids ajusté qui change de formule selon l'IMC. Il est donc **discontinu** et non lipschitzien près des seuils, et il est exclu de ce pilote. C'est un exemple à citer dans l'article (§ 6).
- П4 (métrique absolue vs logarithmique) n'est pas encore implémenté : il sera ajouté après le pilote.
