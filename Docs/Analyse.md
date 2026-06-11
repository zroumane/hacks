# Analyse - Hackathon HAKS Airbus x IBM x AWS 2026

## Objectif

Développer un modèle de Machine Learning pour **estimer la probabilité de corrosion d'un avion à une date d'inspection C-CHECK donnée**, en analysant :
- l'âge de l'avion
- l'exposition environnementale cumulée
- le ratio ground-to-flight (temps au sol vs en vol) avant cette date

**Type de problème** : Classification probabiliste sur séries temporelles (données mensuelles par avion). Le modèle produit `P_i ∈ [0, 1]` = probabilité de corrosion détectée au C-CHECK.

## Formalisme du Problème (compréhension vérifiée)

Cette section fixe **précisément** ce qu'on a en entrée et ce qu'on doit prédire — c'est le point le plus subtil du sujet.

### Ce qu'on a vraiment

- **En entraînement, on n'a pas de `Y_i` explicites.** `corrosions_training.csv` donne seulement, pour chaque avion, **une date à laquelle une corrosion a été *détectée***. Vérifié sur les données : les **758 avions** de `environment_training` ont **tous** corrodé une fois → il n'y a **aucun avion sain** ni aucun C-CHECK négatif documenté dans le train.
- **En soumission**, on reçoit pour chaque avion une **date d'inspection**, et on doit prédire la **probabilité `P_i = P(Y_i = 1)`** qu'à cette date une corrosion soit observée (`Y_i ∈ {0,1}`). Les avions du test (142) sont **disjoints** du train → on ne connaît jamais leur date de corrosion.

### Le piège : censure par intervalle

La date connue est celle de la **détection** (lors d'un C-CHECK), **pas** celle de l'**apparition** de la corrosion. La corrosion est apparue *quelque part* dans `]livraison , date_détection]`, sans qu'on sache quand :

```
livraison        apparition réelle        détectée au C-CHECK
(avion sain)     (inconnue ❓)            (connue ✅)
   |──────────────────?──────────────────────|
                                        date_observation
```

C'est une **censure par intervalle** (problème d'analyse de survie). Conséquence directe sur la construction des labels :

| Date d'inspection hypothétique | Label `Y_i` |
|---|---|
| Avion jeune, loin avant la détection | `0` fiable (quasi sûrement sain) |
| Quelques mois avant la détection | **incertain** (peut-être déjà corrodé, pas encore inspecté) → zone grise |
| À la date de détection | `1` certain (seul label vraiment sûr) |

### Comment construire la cible

On transforme « 1 avion = 1 date » en « 1 avion = plusieurs `(date, Y_i)` » :
- `Y_i = 1` à la date de détection (**unique ancrage positif**) ;
- `Y_i = 0` aux dates suffisamment **antérieures** (avion jeune) ;
- on **exclut la zone grise** juste avant la détection (buffer de quelques mois) pour ne pas injecter de bruit ;
- on ne labellise **jamais** `Y_i = 1` *après* la détection (réparation → l'avion redevient sain).

Le modèle apprend ainsi une probabilité qui **croît avec l'âge et l'exposition cumulée** — exactement la courbe sigmoïde du slide 2 (~2 % à la livraison → ~98 % à 36 mois). Cadre plus rigoureux possible : modèle de survie à censure par intervalle (cf. [IdeesModeles.md](IdeesModeles.md)).

## Structure des Données

| Dataset | Lignes | Description |
|---------|--------|-------------|
| `corrosions_training.csv` | 790 | Observations de corrosion réelles |
| `environment_training.csv` | 63 524 | Conditions environnementales mensuelles (train) |
| `environment_test.csv` | 14 303 | Conditions environnementales mensuelles (test) |

### `corrosions_training.csv`
- `aircraft_id` : identifiant de l'avion
- `observation_date` : date d'observation de la corrosion
- `aircraft_delivery_year` / `aircraft_delivery_month` : date de livraison

**Note** : ce fichier ne contient pas de mesure quantitative — la variable cible doit être construite.

### `environment_training.csv` / `environment_test.csv` (36 features)

**Identifiants** : `aircraft_id`, `year_month`, `month_start_date`

**Opérationnel** : `total_parking_minutes`

**Météo (METAR)** : `metar_temperature_c`, `metar_relative_humidity`, `metar_dew_point_c`, `metar_wind_speed_kn`, `metar_visibility_mi`, `metar_hour_precipitation`

**Aérosols** :
- `sea_salt_aerosol_*` : sel marin (3 tailles : 0.03–0.5, 0.5–5, 5–20 µm)
- `dust_aerosol_*` : poussière (3 tailles)
- `hydrophilic/hydrophobic_organic_matter_aerosol_mixing_ratio`
- `hydrophilic/hydrophobic_black_carbon_aerosol_mixing_ratio`
- `sulphate_aerosol_mixing_ratio`

**Composés chimiques** : `ethane`, `c3h8`, `isoprene`, `carbon_monoxide_mass_mixing_ratio`, `ozone_mass_mixing_ratio`, `h2o2`, `formaldehyde`, `hno3`, `nitrogen_monoxide/dioxide_mass_mixing_ratio`, `oh`, `organic_nitrates`, `sulphur_dioxide_mass_mixing_ratio`, `specific_humidity`, `temperature`

### Format de soumission (`sample_submission.csv`)
**164 lignes × 2 colonnes** (vérifié) : `id` (= `<aircraft_id>_<year_month>`) et `corrosion_risk ∈ [0,1]`.
On ne prédit **pas tous les mois** : seulement **164 couples (avion × date d'inspection)** sur les 142 avions du test (≈ 1,15 date par avion = les C-CHECKs à scorer).

## Facteurs de Corrosion

**Critiques** :
1. **Sel marin** (`sea_salt_aerosol_*`) : corrosion galvanique, surtout en proximité océan
2. **Humidité** (`metar_relative_humidity`, `specific_humidity`) : accélérateur majeur (>60% critique)
3. **Composés soufrés** (`sulphate_aerosol`, `sulphur_dioxide`) : corrosion acide
4. **Température** : influence la vitesse des réactions chimiques
5. **Temps de stationnement** : exposition prolongée sans protection

**Interactions importantes** :
- Humidité × Sel → corrosion électrochimique accélérée
- Température × Humidité → condensation, réactions plus rapides
- Aérosols acides (SO₂, HNO₃) + Humidité → attaque chimique des surfaces

## Approche

### 1. Construction de la Variable Cible

> Détail complet (censure par intervalle, zone grise, ancrage positif) dans la section **Formalisme du Problème** ci-dessus.

En bref : on génère des couples `(date, Y_i)` par avion — `Y_i = 1` à la date de détection, `Y_i = 0` sur les dates bien antérieures, zone grise exclue (buffer). Le modèle prédit `P_i = P(Y_i = 1)`, probabilité croissante avec l'âge et l'exposition (courbe sigmoïde du slide 2 : ~2 % à la livraison → ~98 % à 36 mois).

### 2. Feature Engineering

**Features de base** :
- Âge de l'avion (`aircraft_age_months`)
- **Ratio ground-to-flight** : `total_parking_minutes / total_flight_minutes` — plus l'avion reste au sol, plus l'exposition est longue
- Agrégations d'aérosols : `total_sea_salt`, `total_dust`, `total_black_carbon`
- Indices de corrosivité composite et d'agressivité chimique
- Features cycliques : `month_sin`, `month_cos`

**Features d'interaction** :
- `humidity_salt = metar_relative_humidity × total_sea_salt`
- `temp_humidity = metar_temperature_c × metar_relative_humidity`
- `condensation_risk = metar_relative_humidity / (metar_temperature_c + 1)`
- `cumulative_salt_exposure`, `cumulative_humidity_exposure` (par avion, cumulé jusqu'à la date d'inspection)

**Features temporelles** :
- Moyennes mobiles sur 3, 6, 12 mois
- Lag features : valeurs à 1, 3, 6 mois
- Différences (deltas) entre périodes

### 3. Modèles

| Modèle | Avantages | Usage |
|--------|-----------|-------|
| **XGBoost / LightGBM** | Performance élevée, features tabulaires | Modèle principal |
| **Random Forest** | Robuste, interprétable | Baseline solide |
| **Granite TimeSeries** (IBM) | Pré-entraîné sur séries temporelles | Patterns temporels |
| **Modèle de survie** (Cox) | Modélise directement le "temps avant corrosion" | Optionnel |
| **Ensemble (Stacking)** | Combine les forces de chaque modèle | Prédiction finale |

### 4. Validation

**Split avions** (conforme au scoring officiel) : **20% des avions sont réservés pour l'évaluation** et ne sont pas fournis à l'entraînement. Utiliser un Group K-Fold par `aircraft_id` pour simuler ce hold-out.

**Time-series split** en complément pour respecter l'ordre temporel et éviter le leakage :
- Train sur C-CHECKs passés, validation sur C-CHECKs futurs

**Métrique officielle** : **Brier Score** (plus bas = meilleur)

```
BS = (1/N) × Σ (P_i − Y_i)²
```
- `P_i` : probabilité prédite de corrosion
- `Y_i ∈ {0, 1}` : résultat réel au C-CHECK
- Score parfait = 0, pire cas = 1

## Technologies IBM

| Technologie | Usage |
|-------------|-------|
| **watsonx.ai** | Entraînement GPU, déploiement, cycle de vie ML |
| **Granite TimeSeries** | Modèle foundation pour séries temporelles, fine-tuning sur nos données |
| **Docling** | Extraction de rapports de maintenance (si PDFs disponibles) |

- [Granite TimeSeries — HuggingFace](https://huggingface.co/collections/ibm-granite/granite-time-series)
- [Granite TimeSeries Cookbook](https://github.com/ibm-granite-community/granite-timeseries-workshop)

## Planning

| Phase | Durée | Contenu |
|-------|-------|---------|
| 1 — Exploration | 2 j | EDA, identification des facteurs, stratégie cible |
| 2 — Préparation | 3 j | Variable cible, feature engineering, pipeline validation |
| 3 — Modélisation | 4 j | Baseline → ML classiques → Granite → Ensemble |
| 4 — Optimisation | 2 j | Hyperparameter tuning (Optuna), feature selection |
| 5 — Finalisation | 2 j | Prédictions test, soumission, dashboard, documentation |

**Durée totale** : 13 jours

## Objectifs de Performance

| Métrique | Objectif | Notes |
|----------|----------|-------|
| **Brier Score** | < 0.10 | Métrique officielle, lower is better |
| Kaggle Rank | Top 20% | Évalué sur 20% d'avions hold-out |

*Référence slide : un modèle qui prédit toujours 70% de risque obtient ~70% des cas corrects — le Brier Score pénalise la sur-confiance.*

## Indicateurs de Suivi

| Indicateur | Fréquence | Objectif |
|------------|-----------|----------|
| Brier Score validation | Quotidien | Tendance décroissante |
| Calibration (reliability diagram) | Hebdomadaire | P_i bien calibré vs fréquence réelle |
| Feature importance | Hebdomadaire | Cohérence avec la physique |
| Score Kaggle | À chaque soumission | Amélioration continue |

## Défis et Mitigations

| Défi | Mitigation |
|------|------------|
| Pas de mesure quantitative de corrosion | Approche "temps avant corrosion" |
| Leakage temporel | Time-series split strict |
| Déséquilibre des données | Pondération, SMOTE si nécessaire |
| Complexité des interactions (36 features) | Feature engineering guidé par la physique |
