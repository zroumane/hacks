# Résumé Exécutif - Hackathon HAKS 2026

## 🎯 Objectif du Projet

Développer un modèle de Machine Learning pour **prédire le risque de corrosion des avions** en fonction de leur exposition environnementale, dans le cadre du hackathon Kaggle "HAKS Airbus x IBM x AWS 2026".

## 📊 Problématique

### Contexte
La corrosion des avions est un enjeu majeur pour l'industrie aéronautique :
- **Coûts de maintenance** élevés
- **Sécurité** des appareils
- **Durée de vie** des flottes

### Challenge
Prédire un score de risque de corrosion (`corrosion_risk` entre 0 et 1) pour chaque avion et chaque mois futur, en utilisant :
- 36 variables environnementales (température, humidité, aérosols, composés chimiques)
- Historique d'observations de corrosion
- Données mensuelles sur plusieurs années

## 🔍 Analyse des Données

### Données Disponibles

| Dataset | Lignes | Description |
|---------|--------|-------------|
| `corrosions_training.csv` | 790 | Observations de corrosion réelles |
| `environment_training.csv` | 63,524 | Conditions environnementales mensuelles (train) |
| `environment_test.csv` | 14,303 | Conditions environnementales mensuelles (test) |

### Variables Clés Identifiées

**Facteurs de corrosion critiques** :
1. **Aérosols de sel marin** : Corrosion galvanique (proximité océan)
2. **Humidité relative** : Accélérateur de corrosion (>60% critique)
3. **Composés soufrés** (SO₂, sulphates) : Corrosion acide
4. **Température** : Influence la vitesse des réactions chimiques
5. **Temps de stationnement** : Exposition prolongée sans protection

**Interactions importantes** :
- Humidité × Sel marin → Corrosion électrochimique accélérée
- Température × Humidité → Condensation et réactions plus rapides
- Aérosols acides + Humidité → Attaque chimique des surfaces

## 🎯 Approche Proposée

### 1. Construction de la Variable Cible

**Défi** : Le dataset de corrosion ne contient pas de mesure quantitative, seulement des dates d'observation.

**Solution** : Approche "Temps avant Corrosion"
```
corrosion_risk = 1 / (months_until_corrosion + 1)
```

**Logique** :
- Corrosion dans 1 mois → risk = 1.0 (risque imminent)
- Corrosion dans 12 mois → risk = 0.08 (risque faible)
- Pas de corrosion observée → risk = 0.0

### 2. Feature Engineering

**Features de base** :
- Âge de l'avion (mois depuis livraison)
- Agrégations d'aérosols (total sel, total poussière)
- Indices de corrosivité composite

**Features d'interaction** :
- `humidity × salt` : Synergie corrosive
- `temperature × humidity` : Risque de condensation
- `cumulative_exposure` : Exposition cumulée au fil du temps

**Features temporelles** :
- Moyennes mobiles (3, 6, 12 mois)
- Lag features (valeurs passées)
- Tendances et variations

### 3. Modélisation

**Stratégie multi-modèles** :

| Modèle | Avantages | Usage |
|--------|-----------|-------|
| **XGBoost / LightGBM** | Performance élevée, gestion features complexes | Modèle principal |
| **Granite TimeSeries** (IBM) | Pré-entraîné sur séries temporelles | Capture patterns temporels |
| **Random Forest** | Robuste, interprétable | Baseline solide |
| **Ensemble (Stacking)** | Combine forces de chaque modèle | Prédiction finale |

### 4. Validation

**Time-Series Split** :
- Respecter l'ordre chronologique (pas de leakage temporel)
- Validation sur périodes futures (2023-2024)
- Métrique : RMSE (Root Mean Squared Error)

## 📈 Résultats Attendus

### Objectifs de Performance

| Métrique | Objectif | Justification |
|----------|----------|---------------|
| **RMSE** | < 0.15 | Erreur acceptable pour un score 0-1 |
| **R²** | > 0.70 | Bon pouvoir prédictif |
| **Kaggle Rank** | Top 20% | Position compétitive |

### Livrables

1. **Modèle ML entraîné** : Prêt pour prédictions
2. **Fichier de soumission Kaggle** : Format attendu
3. **Dashboard Streamlit** : Visualisation interactive des prédictions
4. **Documentation complète** : Architecture, méthodologie, résultats
5. **Code reproductible** : Scripts modulaires et commentés

## 🛠️ Technologies IBM Utilisées

### watsonx.ai
- Plateforme d'entraînement avec accès GPU
- Déploiement de modèles en production
- Gestion du cycle de vie ML

### Granite TimeSeries
- Modèle foundation IBM pour séries temporelles
- Fine-tuning sur nos données de corrosion
- [Documentation](https://huggingface.co/collections/ibm-granite/granite-time-series)

### Docling (si applicable)
- Extraction de documentation technique
- Analyse de rapports de maintenance

## 📅 Planning

### Phase 1 : Exploration (2 jours)
- ✅ Analyse exploratoire des données
- ✅ Identification des facteurs de corrosion
- ✅ Stratégie de construction de la cible

### Phase 2 : Préparation (3 jours)
- Construction de la variable cible
- Feature engineering (base + interactions + temporelles)
- Pipeline de validation

### Phase 3 : Modélisation (4 jours)
- Baseline (moyenne, régression simple)
- Modèles ML classiques (XGBoost, LightGBM)
- Granite TimeSeries (IBM)
- Ensemble de modèles

### Phase 4 : Optimisation (2 jours)
- Hyperparameter tuning (Optuna)
- Feature selection
- Validation croisée

### Phase 5 : Finalisation (2 jours)
- Génération prédictions test set
- Création fichier soumission
- Dashboard Streamlit
- Documentation

**Durée totale** : 13 jours

## 💡 Insights Clés

### Points Forts de l'Approche

1. **Approche scientifique** : Basée sur la physico-chimie de la corrosion
2. **Feature engineering intelligent** : Capture les interactions réelles
3. **Validation rigoureuse** : Time-series split pour éviter le leakage
4. **Technologies IBM** : Granite TimeSeries pour patterns temporels
5. **Ensemble de modèles** : Combine forces de différentes approches

### Défis Identifiés

1. **Construction de la cible** : Pas de mesure quantitative directe
   - *Mitigation* : Approche "temps avant corrosion" validée
   
2. **Leakage temporel** : Risque d'utiliser des données futures
   - *Mitigation* : Time-series split strict
   
3. **Déséquilibre des données** : Peu d'observations de corrosion
   - *Mitigation* : Techniques de pondération, SMOTE si nécessaire
   
4. **Complexité des interactions** : 36 features environnementales
   - *Mitigation* : Feature engineering guidé par la physique

## 🎯 Facteurs de Succès

1. **Compréhension du domaine** : Corrosion aéronautique
2. **Qualité des features** : Engineering intelligent
3. **Validation robuste** : Pas d'overfitting
4. **Technologies IBM** : Granite TimeSeries
5. **Itération rapide** : Commencer simple, complexifier

## 📊 Indicateurs de Suivi

| Indicateur | Fréquence | Objectif |
|------------|-----------|----------|
| RMSE validation | Quotidien | Tendance décroissante |
| Feature importance | Hebdomadaire | Cohérence avec physique |
| Temps d'entraînement | Par modèle | < 30 min |
| Score Kaggle | À chaque soumission | Amélioration continue |

## 🚀 Prochaines Actions

### Immédiat (Aujourd'hui)
1. ✅ Analyse exploratoire complète
2. ✅ Documentation du plan
3. Validation de l'approche avec l'équipe

### Court terme (Cette semaine)
1. Construction de la variable cible
2. Feature engineering de base
3. Premier modèle baseline

### Moyen terme (Semaine prochaine)
1. Modèles ML avancés
2. Granite TimeSeries
3. Optimisation

## 📞 Contact et Support

- **Documentation** : [`Docs/`](../Docs/)
- **Code** : [`src/algo/`](../src/algo/)
- **Dashboard** : [`src/streamlit/`](../src/streamlit/)
- **Ressources IBM** : [`Docs/Ressources.md`](Ressources.md)

---

**Préparé par** : Bob (Mode Plan)  
**Date** : 11 juin 2026  
**Version** : 1.0