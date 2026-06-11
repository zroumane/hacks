# Brief — État de la Situation (pour agent suivant)

## Contexte hackathon

**Hackathon** : HAKS Airbus × IBM × AWS 2026 — compétition Kaggle.  
**Sujet** : Prédire le risque de corrosion d'avions à des dates d'inspection données.  
**Métrique officielle** : **Brier Score** `BS = mean((P_i − Y_i)²)` — lower is better.  
**Évaluation** : sur **20 % des avions** non fournis en entraînement (hold-out par avion, pas par date).  
**Temps restant** : ~3h.

---

## Données disponibles

| Fichier | Lignes | Description |
|---|---|---|
| `input/corrosions_training.csv` | 790 | Pour chaque avion : `aircraft_id`, `observation_date` (date de *détection*), `aircraft_delivery_year/month` |
| `input/environment_training.csv` | 63 524 | Historique mensuel de 758 avions × 36 features environnementales |
| `input/environment_test.csv` | 14 303 | Historique mensuel de 142 avions (mêmes 36 features) |
| `input/sample_submission.csv` | **164** | `id` (`<aircraft_id>_<year_month>`) + `corrosion_risk` à remplir |

**36 features environnementales** : météo METAR (température, humidité, rosée, vent, précipitations), aérosols de sel marin (3 tailles), poussière (3 tailles), matière organique (hydrophile/hydrophobe), carbone noir (hydrophile/hydrophobe), sulphates, SO₂, HNO₃, ozone, NOₓ, OH, H₂O₂, formaldéhyde, nitrates organiques, CO, éthane, propane, isoprène, humidité spécifique, température atmosphérique, `total_parking_minutes`.

---

## Compréhension du problème (vérifiée)

### Ce qu'on a en entraînement

- **Pas de `Y_i` explicites**. `corrosions_training.csv` donne seulement **une date de *détection*** par avion (lors d'un C-CHECK). Tous les 758 avions ont corrodé — **aucun avion sain** dans le train.
- La date d'observation = date où la corrosion a été *constatée*, **pas** où elle est *apparue*. La corrosion est apparue quelque part dans `]livraison, date_détection]` → **censure par intervalle**.

### Ce qu'on prédit

- Pour chaque des **164 couples (avion_test × date_inspection)** du `sample_submission.csv` : `P_i = P(corrosion détectée à cette date)`, probabilité ∈ [0,1].
- Les 142 avions du test sont **entièrement disjoints** du train (vérification faite sur les données).

### Construction des labels d'entraînement

Puisqu'il n'y a pas de `Y_i` directs, on les fabrique temporellement par avion :
- `Y_i = 1` à la date de détection (seul ancrage positif certain)
- `Y_i = 0` sur les dates antérieures (avion pas encore corrodé)
- **Zone grise** : les mois juste avant la détection sont incertains (corrosion peut-être déjà présente, pas encore inspectée) → à exclure avec un buffer de ~6 mois

---

## Code actuel — `src/algo/train.py` + `src/utils/ml.py`

Le code est **fonctionnel mais comporte plusieurs erreurs méthodologiques importantes** :

### ✅ Ce qui est correct
- Jointure `environment_training ← corrosions_training` sur `aircraft_id`
- Filtrage anti-leakage : on ne garde que les mois ≤ `observation_date`
- Feature `aircraft_age_months` calculée depuis la date de livraison
- Sélection correcte des features environnementales (subset cohérent)
- Pipeline train → save → predict → submission opérationnel

### ❌ Bugs / erreurs méthodologiques à corriger

**1. Mauvaise cible — régression continue au lieu de classification probabiliste**
```python
# CODE ACTUEL (incorrect)
merged["corrosion_risk"] = 1 / (1 + merged["months_until"])
model = XGBRegressor(...)
```
- Le modèle apprend à minimiser le MSE sur une cible continue `1/(1+months)`.
- Le Brier Score officiel évalue `(P_i − Y_i)²` où `Y_i ∈ {0,1}` — la cible continue ne reproduit pas ce signal.
- Les Brier Scores calculés en CV (`brier_score(y_continu, pred)`) **ne sont pas comparables** au score Kaggle qui utilisera des `Y_i` binaires.

**2. Mauvaise validation — TimeSeriesSplit au lieu de GroupKFold**
```python
# CODE ACTUEL (incorrect)
tscv = TimeSeriesSplit(n_splits=3)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
```
- `TimeSeriesSplit` splitte sur **l'index de ligne** (trié par avion×date) → des mois du même avion se retrouvent à la fois en train et en val.
- Or le scoring officiel évalue sur des **avions entiers jamais vus**. La CV actuelle est trop optimiste et ne simule pas le vrai hold-out.
- **Correct** : `GroupKFold(groups=aircraft_id)` + tri chronologique à l'intérieur de chaque fold.

**3. Pas de calibration des probabilités**
- `XGBRegressor.predict()` retourne des valeurs réelles, pas des probabilités calibrées.
- Le Brier Score pénalise la **sur-confiance** ; un modèle non calibré peut être excellent en discrimination mais mauvais en Brier.
- À corriger : passer à `XGBClassifier(objective='binary:logistic')` + `predict_proba()[:,1]`, ou appliquer Venn-Abers/isotonic regression.

**4. Features manquantes**
- Pas de features d'**exposition cumulée** (somme des aérosols × parking depuis la livraison) : la corrosion est un phénomène cumulatif.
- Pas de **moyennes mobiles** (3/6/12 mois) pour capturer la tendance récente.
- Pas d'**interactions** clés : `humidity × sea_salt`, `temperature × humidity`.
- La feature `ground_to_flight_ratio` mentionnée dans le brief Airbus (slide 2) n'est qu'approximée par `total_parking_minutes` — pas de normalisation par le temps de vol.

**5. Approximation grossière de l'âge des avions du test**
```python
# CODE ACTUEL : utilise le premier mois dispo dans env_test comme proxy
first_month = env_test.groupby("aircraft_id")["year_month"].min()
```
- Valable seulement si l'historique test commence à la livraison, ce qui n'est pas garanti.
- `corrosions_training.csv` contient les vraies dates de livraison pour les avions train → à utiliser dans une feature globale (distribution des âges), même si on ne peut pas joindre directement sur test.

---

## Ce que le code produit actuellement

En l'état, si on lance `train.py` puis `predict.py` :
1. Un `XGBRegressor` entraîné sur la cible `1/(1+months_until)` (régression, pas proba)
2. Des CV Brier Scores calculés sur cette cible continue (pas représentatifs du scoring Kaggle)
3. Une `submission.csv` de 164 lignes avec des valeurs dans [0,1] — **format correct**, mais prédictions non calibrées, non probabilistes

---

## Stratégie recommandée (priorisée par impact/temps)

### Priorité 1 — Corriger les bugs (impact maximum, ~45 min)

```python
# a. Cible binaire avec buffer sur la zone grise
BUFFER_MONTHS = 6
merged["Y"] = 0
merged.loc[merged["months_until"] == 0, "Y"] = 1
merged = merged[(merged["months_until"] == 0) | (merged["months_until"] > BUFFER_MONTHS)]

# b. Classifier, pas régresseur
from xgboost import XGBClassifier
model = XGBClassifier(objective='binary:logistic', n_estimators=500, ...)
model.fit(X, y_binary)
proba = np.clip(model.predict_proba(X_test)[:, 1], 0.02, 0.98)

# c. GroupKFold par avion
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(X, y, groups=aircraft_ids):
    ...
    score = brier_score_loss(y[va], proba_va)
```

### Priorité 2 — Modèles bien calibrés sans tuning (~30 min)

- **CatBoost** (`CatBoostClassifier(loss_function='Logloss')`) : calibration native, peu de tuning
- **TabPFN v2.5** (`pip install tabpfn`) : SOTA sur < 10k échantillons, zéro tuning, ~secondes

```python
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

### Priorité 3 — Features cumulatives (~30 min, fort impact théorique)

```python
# Tri par avion + date, puis cumul
df = df.sort_values(["aircraft_id", "year_month"])
for col in ["sea_salt_total", "sulphate_aerosol_mixing_ratio", "metar_relative_humidity"]:
    df[f"cum_{col}"] = df.groupby("aircraft_id")[col].cumsum()
    df[f"roll3_{col}"] = df.groupby("aircraft_id")[col].transform(lambda x: x.rolling(3).mean())
# Interactions
df["humidity_x_salt"] = df["metar_relative_humidity"] * df["sea_salt_total"]
```

### Priorité 4 — XGBoost AFT (censure par intervalle, plus rigoureux, ~45 min)

Modélise directement l'incertitude sur l'apparition (pas juste la détection) :
```python
dtrain = xgb.DMatrix(X)
dtrain.set_float_info('label_lower_bound', y_lower)  # 0 ou âge dernier C-check négatif
dtrain.set_float_info('label_upper_bound', y_upper)  # âge à la détection
params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal'}
```

### Priorité 5 — Calibration post-hoc et ensemble (~30 min)

- **Venn-Abers** (`pip install venn-abers`) sur XGBoost/LightGBM
- Stacking : méta-modèle `LogisticRegression` sur les probas out-of-fold de 3 modèles
- Clip final : `np.clip(proba, 0.02, 0.98)`

---

## Stack technique

- **Environnement** : Python 3.13, `uv` (gestionnaire de paquets)
- **Installé** : `xgboost`, `scikit-learn`, `pandas`, `numpy`, `lightgbm` (dans uv.lock), `ibm-watsonx-ai`, `docling`, `streamlit`
- **À installer** : `catboost`, `tabpfn`, `venn-abers`, `scikit-survival`
- **Lancement** : `uv run src/algo/train.py` puis `uv run src/algo/predict.py output/models/xgb_corrosion.pkl`
- **Output** : `output/submission.csv` (164 lignes, colonnes `id` + `corrosion_risk`)

---

## Fichiers clés

```
input/
├── corrosions_training.csv     # 790 lignes, dates de détection par avion
├── environment_training.csv    # 63 524 lignes, 36 features, 758 avions
├── environment_test.csv        # 14 303 lignes, 36 features, 142 avions
└── sample_submission.csv       # 164 lignes, format de sortie attendu

src/
├── algo/train.py               # Pipeline principal (à corriger)
├── algo/predict.py             # Génération soumission (format OK)
└── utils/ml.py                 # Utilitaires XGBoost + CV (à corriger)

Docs/
├── Analyse.md                  # Formalisme complet du problème
├── IdeesModeles.md             # Récap modèles + snippets (CatBoost, TabPFN, AFT, Venn-Abers)
├── Implementation.md           # Pipeline technique détaillé
└── UseCase.md                  # Use case Airbus (source PDF)
```
