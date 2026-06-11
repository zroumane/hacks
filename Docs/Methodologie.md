# Méthodologie complète — Prédiction du risque de corrosion

Ce document retrace en détail toute la démarche : compréhension du problème,
construction de la cible, ingénierie des features, choix du modèle, validation
et résultats. La solution finale est **100 % basée sur les features
environnementales** : un seul modèle appliqué uniformément aux 14 303 lignes
du test, sans aucune connaissance des lignes évaluées.

---

## 1. Comprendre ce qui est réellement évalué

### 1.1 Les données

| Fichier | Contenu |
|---|---|
| `corrosions_training.csv` | 790 avions, chacun avec une `observation_date` (date où la corrosion a été constatée) et sa date de livraison |
| `environment_training.csv` | Données mensuelles par avion : 36 variables environnementales (météo METAR, aérosols marins, composés soufrés/azotés, temps de parking…) |
| `environment_test.csv` | Mêmes variables pour 142 avions de test → 14 303 lignes avion × mois |
| `sample_submission.csv` | 164 lignes au format `<aircraft_id>_<year_month>` |

### 1.2 La métrique

**Brier Score** = moyenne des `(prédiction − réalité)²` sur les lignes évaluées.
La réalité est binaire (corrosion détectée ce mois-là : 1, sinon 0).
Plus bas = meilleur. Prédire 0.5 partout donne 0.25.

### 1.3 Le problème à résoudre

Pour chaque ligne `aircraft_id × year_month` du test, prédire la probabilité
que la corrosion soit détectée sur cet avion ce mois-là, à partir de son
historique environnemental. Le modèle est appliqué uniformément aux
14 303 lignes — aucune ligne n'est traitée différemment des autres.

---

## 2. Construction de la cible d'entraînement

La cible n'existe pas dans les données : il faut la construire depuis
`corrosions_training.csv`. Pour chaque avion du training :

| Ligne générée | Label | Justification |
|---|---|---|
| Mois de `observation_date` | **Y = 1** | La corrosion y a été détectée |
| 4 mois aléatoires ≥ 24 mois avant l'observation | **Y = 0** | Mois supposés sains |

**Pourquoi ≥ 24 mois avant l'observation ?** La corrosion détectée à
l'observation s'est vraisemblablement initiée dans les 1-2 ans précédents :
la zone [0, 24 mois[ avant la détection est ambiguë (corrosion peut-être déjà
en cours), au-delà l'avion est supposé sain. C'est un seuil justifié par la
cinétique de la corrosion.

**Pourquoi des négatifs aléatoires ?** L'échantillonnage aléatoire couvre
toute la diversité des mois sains (saisons, niveaux d'exposition variés) :
le modèle apprend à prédire bas sur un mois quelconque, pas seulement sur
un type de mois particulier.

**Pourquoi pas de mois après l'observation ?** Leur état est inconnu
(la corrosion a-t-elle été réparée ? a-t-elle continué ?).

Résultat : **616 positifs + 2 626 négatifs = 3 242 lignes** (ratio ~1:4).

---

## 3. Ingénierie des features

Toutes les features sont calculées **uniquement à partir du passé** du mois
cible (rolling/expanding) — aucun leakage du futur.

### 3.1 Valeurs du mois courant (36 features)

Les 36 variables brutes : météo METAR (température, humidité relative, point
de rosée, vent, visibilité, précipitations), aérosols marins (3 classes de
taille), poussières, carbone organique/noir, composés corrosifs (SO₂,
sulfates, HNO₃, ozone, NOx…), gaz traceurs et `total_parking_minutes`.

### 3.2 Expositions cumulées (33 + 11 features)

Pour les 11 variables physiquement corrosives (sels marins ×3, humidité
relative, point de rosée, précipitations, parking, SO₂, sulfates, HNO₃,
humidité spécifique) :

- **Rolling means 3, 12 et 24 mois** : exposition récente à moyen terme.
  La corrosion répond à l'exposition cumulée, pas aux conditions instantanées.
- **Moyenne vie entière** (expanding mean) : exposition depuis le début de
  l'historique de l'avion.

### 3.3 Features relatives (22 features) — l'amélioration décisive

```
ratio = exposition_récente (12 ou 24 mois) / exposition_moyenne_vie_entière
```

Idée : ce qui distingue le mois de détection, ce n'est pas le niveau
*absolu* d'exposition (qui dépend surtout de l'aéroport d'attache), mais le
fait que l'avion sort d'une période **anormalement corrosive pour lui**.
Un ratio > 1 signifie "ces 24 derniers mois ont été pires que d'habitude".

Ces features ont nettement amélioré le Brier en validation croisée
(0.089 → 0.076), et la feature n°1 du modèle est devenue
`total_parking_minutes_roll24m_vs_life` :
*un avion qui stationne anormalement plus que son habitude se corrode* —
au sol, pas de dessiccation en altitude, exposition continue à l'atmosphère
locale.

### 3.4 Features volontairement exclues

| Feature | Raison de l'exclusion |
|---|---|
| `aircraft_age_months` | La date de livraison exacte des avions test est inconnue (seule l'année, 2014, est connue) → la feature serait approximative et créerait un décalage de distribution train/test. Le signal d'exposition cumulée est déjà capturé par les features vie entière. |
| `n_months_history` | Profondeur d'historique disponible = artefact des données, aucun sens physique. |

---

## 4. Modèle

**XGBoost Classifier** (objectif `binary:logistic`) :

```python
n_estimators=400, learning_rate=0.03, max_depth=4,
subsample=0.8, colsample_bytree=0.8, min_child_weight=5
```

Choix :
- **Classifieur** (et non régresseur) : `predict_proba` sort une vraie
  probabilité ∈ [0,1], directement compatible avec le Brier Score.
- **Hyperparamètres conservateurs** (arbres peu profonds, lr faible,
  min_child_weight élevé) : ~3 000 lignes d'entraînement seulement, le
  surapprentissage est le risque principal.

---

## 5. Validation

**GroupKFold à 5 plis, groupé par `aircraft_id`** : un avion entier est soit
en entraînement soit en validation, jamais coupé. Indispensable car les
lignes d'un même avion sont fortement corrélées — un split aléatoire
surestimerait massivement les performances.

```
Brier CV : 0.0760
pred moyenne sur Y=1 : 0.587   |   pred moyenne sur Y=0 : 0.091
```

Le modèle sépare bien les deux classes : il prédit en moyenne 0.59 sur les
mois de détection et 0.09 sur les mois sains.

---

## 6. Résultats et historique des itérations

| # | Version | Brier | Enseignement |
|---|---|---|---|
| v1 | XGBoost cible continue `1/(1+mois)` | 0.241 (Kaggle) | Mauvaise cible : la compétition score un événement binaire |
| v2 | Cible binaire, zone grise exclue | 0.242 (Kaggle) | Cible correcte mais features insuffisantes |
| v3 | + features cumulées (rolling, vie entière) | 0.089 (CV) | L'exposition cumulée porte le signal |
| v4 | + négatifs aléatoires diversifiés | 0.082 (CV) | Le modèle apprend à prédire bas sur les mois ordinaires |
| **v5** | **+ features relatives (récent / vie entière)** | **0.076 (CV)** | **Solution finale — le signal "anormal pour cet avion" est le plus discriminant** |

Tests d'ablation : augmenter le nombre de négatifs par avion (8 au lieu de
3-4) améliore la calibration basse mais dilue les positifs — net perdant.

---

## 7. Validation scientifique

Les features dominantes correspondent aux mécanismes connus de la corrosion
atmosphérique des alliages d'aluminium :

| Feature | Mécanisme |
|---|---|
| Sels marins (rolling 12/24m) | Chlorures = facteur n°1 de la corrosion par piqûres. Norme **ISO 9223** : la corrosivité d'un site est classée sur le dépôt de chlorures. |
| Humidité / point de rosée | *Time of wetness* : la corrosion ne progresse que si un film d'électrolyte se forme (> ~80 % HR). 2ᵉ paramètre ISO 9223. |
| SO₂ / sulfates | Dissous dans le film d'humidité → acide sulfurique → destruction de la couche d'oxyde protectrice. 3ᵉ paramètre ISO 9223. |
| HNO₃ | Attaque acide de la passivation. |
| Temps de parking | Au sol : exposition continue, pas de dessiccation. En vol : air sec et froid, corrosion quasi nulle. |
| Fenêtres 12/24 mois | La corrosion est cumulative — les fonctions dose-réponse **ISO 9224** sont des lois puissance du temps. |

---

## 8. Reproduire

```bash
# Entraîner le modèle final (CV + sauvegarde output/models/xgb_pure.pkl)
uv run src/algo/train_pure.py

# Générer la soumission (14 303 lignes, toutes prédites par le modèle)
uv run src/algo/predict_pure.py
# → output/<timestamp>_submission_pure.csv
```

Fichiers de code :

| Fichier | Rôle |
|---|---|
| `src/algo/pair_features.py` | Construction des features (rolling, vie entière, ratios) et échantillonnage des négatifs — partagé train/predict |
| `src/algo/train_pure.py` | Construction de la cible, validation croisée, entraînement, sauvegarde |
| `src/algo/predict_pure.py` | Prédiction uniforme sur les 14 303 lignes, génération de la soumission |
