# Use Case — Prédiction du Risque de Corrosion (Airbus x IBM)

## Objectif

Construire un modèle prédictif qui estime le **risque de corrosion** d'un avion à une date d'inspection donnée, en analysant :

- L'**âge** de l'avion (date de livraison)
- L'**exposition environnementale cumulée** (conditions météo et aérosols mois par mois)
- Le **ratio sol/vol** avant cette date (`total_parking_minutes` — plus un avion stationne, plus il est exposé)

## Métrique de Compétition

**Brier Score** calculé sur 20 % des avions non fournis en entraînement.

Le Brier Score mesure l'erreur d'une prédiction probabiliste :

```
BS = moyenne( (corrosion_risk_prédit - corrosion_réelle)² )
```

- `0.0` → prédictions parfaites
- `0.25` → pire cas (prédire 0.5 systématiquement)
- **Plus bas = meilleur**

## Données

### `corrosions_training.csv` — Cible (790 lignes)

| Colonne | Description |
|---|---|
| `aircraft_id` | Identifiant anonymisé de l'avion |
| `observation_date` | Date à laquelle la corrosion a été constatée |
| `aircraft_delivery_year` | Année de livraison de l'avion |
| `aircraft_delivery_month` | Mois de livraison de l'avion |

> Chaque ligne = un avion pour lequel une corrosion a été **observée à une date précise**. Il n'y a pas de score ici — c'est un événement binaire (la corrosion a eu lieu).

### `environment_training.csv` / `environment_test.csv` — Features (36 colonnes)

Données **mensuelles** par avion couvrant 2014–2026. Regroupées en 4 familles :

**Météo locale (METAR)**
- `metar_temperature_c`, `metar_relative_humidity`, `metar_dew_point_c`
- `metar_wind_speed_kn`, `metar_visibility_mi`, `metar_hour_precipitation`

**Aérosols marins** — favorisent la corrosion saline
- `sea_salt_aerosol_003_05_mixing_ratio`, `_05_5`, `_5_20`

**Aérosols de poussière**
- `dust_aerosol_003_055_mixing_ratio`, `_055_09`, `_09_20`

**Aérosols carbonés et organiques**
- `hydrophilic/hydrophobic_organic_matter_aerosol_mixing_ratio`
- `hydrophilic/hydrophobic_black_carbon_aerosol_mixing_ratio`

**Composés chimiques corrosifs**
- `sulphate_aerosol_mixing_ratio`, `sulphur_dioxide_mass_mixing_ratio`
- `hno3` (acide nitrique), `ozone_mass_mixing_ratio`
- `nitrogen_monoxide/dioxide_mass_mixing_ratio`
- `formaldehyde`, `h2o2`, `oh`, `organic_nitrates`

**Gaz traceurs**
- `ethane`, `c3h8` (propane), `isoprene`, `carbon_monoxide_mass_mixing_ratio`

**Opérationnel**
- `total_parking_minutes` — temps de stationnement mensuel (proxy du ratio sol/vol)
- `specific_humidity`, `temperature` (modèle atmosphérique)

### `sample_submission.csv` — Ce qu'on doit prédire (164 lignes)

Format de l'`id` : `<aircraft_id>_<year_month>` (ex: `894378_2018-08`)

On prédit un `corrosion_risk` ∈ [0, 1] pour chaque combinaison avion × mois.

## Interprétation du Problème

### Construction de la cible

La variable cible n'est pas dans les données — elle doit être **construite** à partir de `corrosions_training.csv`. Attention : la date connue est celle de la **détection**, pas de l'**apparition** → **censure par intervalle**.

Approche retenue (détail dans [Analyse.md](Analyse.md) § *Formalisme du Problème*) : on génère des couples `(date, Y_i)` avec `Y_i ∈ {0,1}` — `1` à la date de détection, `0` sur les dates bien antérieures, zone grise exclue. Le modèle prédit `P(Y_i = 1)`, scoré au Brier Score.

> ⚠️ Ne pas utiliser un score continu type `1/(1+months)` : il ne correspond pas à la cible binaire réellement scorée.

### Facteurs clés attendus

D'après la physique de la corrosion aéronautique et les features disponibles :

| Facteur | Variables | Impact |
|---|---|---|
| Sel marin | `sea_salt_aerosol_*` | Très élevé — accélère la corrosion galvanique |
| Humidité | `metar_relative_humidity`, `metar_dew_point_c` | Élevé — favorise l'électrolyse |
| Composés acides | `hno3`, `sulphur_dioxide`, `sulphate` | Élevé — attaque les alliages aluminium |
| Temps sol | `total_parking_minutes` | Élevé — exposition prolongée sans protection vol |
| Âge | `aircraft_delivery_year/month` | Modéré — dégradation cumulative |
| Température | `metar_temperature_c` | Modéré — accélère les réactions chimiques |

### Structure de la prédiction

```
aircraft_id × year_month
        ↓
Historique environnemental (tous les mois précédents)
        +
Âge de l'avion à cette date
        +
Ratio sol/vol cumulé
        ↓
corrosion_risk ∈ [0, 1]
```

## Contraintes Méthodologiques

- **Pas de data leakage** : pour prédire le risque à un mois M, on ne peut utiliser que les données antérieures à M
- **Validation temporelle** : le split train/test doit respecter l'ordre chronologique
- **Déséquilibre** : tous les avions du training ont eu une corrosion — les avions sains ne sont pas représentés, ce qui biaise la distribution du risque

## Pages Non Extractibles du PDF

2 des 4 slides sont des images (pas de couche texte). Elles contiennent vraisemblablement :
- Un schéma du pipeline de données
- Un exemple de visualisation ou de résultat attendu

> Pour les analyser, activer Docling Vision avec un modèle multimodal sur watsonx.
