# Architecture - Hackathon HAKS 2026

## Vue d'Ensemble du Projet

Ce projet vise à prédire le risque de corrosion des avions en fonction de données environnementales mensuelles, dans le cadre du hackathon Kaggle "HAKS Airbus x IBM x AWS 2026".

## Flux de Données Principal

```mermaid
flowchart TB
    subgraph Input["📥 Données d'Entrée"]
        A1[corrosions_training.csv<br/>790 observations]
        A2[environment_training.csv<br/>63,524 lignes]
        A3[environment_test.csv<br/>14,303 lignes]
    end
    
    subgraph Processing["⚙️ Traitement (src/algo)"]
        B1[01-02: EDA & Quality]
        B2[03: Target Engineering]
        B3[04-06: Feature Engineering]
        B4[07: Validation Strategy]
        B5[08-12: Modélisation]
        B6[13-14: Optimisation]
        B7[15-16: Prédiction & Soumission]
    end
    
    subgraph Output["📤 Sorties"]
        C1[Modèles entraînés]
        C2[Prédictions]
        C3[submission.csv<br/>Kaggle]
        C4[Rapports & Métriques]
    end
    
    subgraph UI["🖥️ Interface (src/streamlit)"]
        D1[Dashboard Visualisation]
        D2[Analyse Prédictions]
        D3[Feature Importance]
    end
    
    A1 --> B2
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> B5
    B5 --> B6
    B6 --> B7
    B7 --> C1
    B7 --> C2
    C2 --> C3
    B5 --> C4
    
    C1 --> D1
    C2 --> D2
    C4 --> D3
```

## Pipeline de Modélisation

```mermaid
flowchart LR
    subgraph Data["Données"]
        D1[Environment<br/>36 features]
        D2[Corrosions<br/>Observations]
    end
    
    subgraph Target["Construction Cible"]
        T1[Calcul<br/>months_until_corrosion]
        T2[Formule<br/>corrosion_risk]
    end
    
    subgraph Features["Feature Engineering"]
        F1[Features Base<br/>âge, agrégations]
        F2[Interactions<br/>humidité×sel]
        F3[Temporelles<br/>rolling, lag]
    end
    
    subgraph Models["Modèles"]
        M1[Baseline<br/>Moyenne]
        M2[XGBoost<br/>LightGBM]
        M3[Granite<br/>TimeSeries]
        M4[Ensemble<br/>Stacking]
    end
    
    subgraph Validation["Validation"]
        V1[Time-Series<br/>Split]
        V2[RMSE<br/>Métrique]
    end
    
    D1 --> T1
    D2 --> T1
    T1 --> T2
    T2 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> M1
    F3 --> M2
    F3 --> M3
    M1 --> M4
    M2 --> M4
    M3 --> M4
    M4 --> V1
    V1 --> V2
```

## Workflow Détaillé

```mermaid
sequenceDiagram
    actor User as Data Scientist
    participant EDA as 01-02_EDA
    participant Target as 03_Target
    participant FE as 04-06_Features
    participant Model as 08-12_Models
    participant Optim as 13-14_Optim
    participant Pred as 15-16_Predict
    participant Kaggle as Kaggle Platform
    participant UI as Streamlit Dashboard

    User->>EDA: Analyser les données
    EDA->>EDA: Distributions, corrélations
    EDA->>User: Rapport EDA
    
    User->>Target: Construire variable cible
    Target->>Target: Calculer corrosion_risk
    Target->>User: Dataset labellisé
    
    User->>FE: Feature engineering
    FE->>FE: Base + Interactions + Temporelles
    FE->>User: Dataset enrichi
    
    User->>Model: Entraîner modèles
    Model->>Model: Baseline → ML → Granite
    Model->>User: Scores validation
    
    alt Score insuffisant
        User->>Optim: Optimiser
        Optim->>Optim: Hyperparams + Features
        Optim->>Model: Réentraîner
    end
    
    User->>Pred: Générer prédictions
    Pred->>Pred: Prédire test set
    Pred->>Kaggle: Soumettre submission.csv
    Kaggle->>User: Score public leaderboard
    
    User->>UI: Visualiser résultats
    UI->>User: Dashboard interactif
```

## Architecture Technique

### Structure des Scripts

```
src/algo/
├── 01_eda.py                          # Exploration des données
├── 02_data_quality.py                 # Analyse qualité
├── 03_target_engineering.py           # Construction cible
├── 04_feature_engineering_base.py     # Features de base
├── 05_feature_engineering_interactions.py  # Interactions
├── 06_feature_engineering_temporal.py # Features temporelles
├── 07_validation_strategy.py          # Stratégie validation
├── 08_baseline.py                     # Modèle baseline
├── 09_ml_models.py                    # XGBoost, LightGBM, RF
├── 10_granite_timeseries.py           # Modèle Granite
├── 11_survival_model.py               # Modèle de survie (optionnel)
├── 12_ensemble.py                     # Ensemble de modèles
├── 13_hyperparameter_tuning.py        # Optimisation hyperparams
├── 14_feature_selection.py            # Sélection features
├── 15_generate_predictions.py         # Prédictions test
└── 16_create_submission.py            # Fichier soumission
```

### Technologies Utilisées

```mermaid
mindmap
  root((HAKS 2026))
    Data Processing
      pandas
      numpy
      docling
    ML Classique
      XGBoost
      LightGBM
      scikit-learn
    IBM Technologies
      watsonx.ai
      Granite TimeSeries
      IBM Cloud
    Optimisation
      Optuna
      SHAP
    Visualisation
      Streamlit
      Plotly
      Seaborn
    Séries Temporelles
      statsmodels
      prophet
```

## Facteurs Clés de Corrosion

```mermaid
mindmap
  root((Corrosion<br/>Aéronautique))
    Environnement Marin
      Sel marin
      Aérosols salins
      Proximité océan
    Conditions Météo
      Humidité élevée
      Température
      Précipitations
      Condensation
    Composés Chimiques
      SO₂ Dioxyde de soufre
      HNO₃ Acide nitrique
      Sulphates
      Ozone
    Facteurs Opérationnels
      Temps de stationnement
      Âge de l'avion
      Historique exposition
    Interactions
      Humidité × Sel
      Température × Humidité
      Aérosols acides
```

## Stratégie de Validation

```mermaid
flowchart TD
    A[Dataset Complet] --> B{Time-Series Split}
    B --> C[Train: 2015-2022]
    B --> D[Val: 2023-H1]
    B --> E[Val: 2023-H2]
    B --> F[Val: 2024-H1]
    
    C --> G[Entraînement Modèle]
    D --> H[Validation 1]
    E --> I[Validation 2]
    F --> J[Validation 3]
    
    G --> H
    G --> I
    G --> J
    
    H --> K[RMSE 1]
    I --> L[RMSE 2]
    J --> M[RMSE 3]
    
    K --> N[RMSE Moyen]
    L --> N
    M --> N
    
    N --> O{Score OK?}
    O -->|Non| P[Ajuster Modèle]
    O -->|Oui| Q[Prédire Test Set]
    P --> G
```

## Déploiement et Utilisation

### Exécution des Scripts

```bash
# 1. Exploration des données
uv run src/algo/01_eda.py

# 2. Construction de la cible
uv run src/algo/03_target_engineering.py

# 3. Feature engineering
uv run src/algo/04_feature_engineering_base.py

# 4. Entraînement modèles
uv run src/algo/09_ml_models.py

# 5. Génération prédictions
uv run src/algo/15_generate_predictions.py

# 6. Création soumission
uv run src/algo/16_create_submission.py
```

### Interface Streamlit

```bash
# Lancer le dashboard
uv run streamlit run src/streamlit/dashboard.py
```

## Métriques de Performance

| Métrique | Objectif | Description |
|----------|----------|-------------|
| **RMSE** | < 0.15 | Erreur quadratique moyenne sur corrosion_risk |
| **MAE** | < 0.10 | Erreur absolue moyenne |
| **R²** | > 0.70 | Coefficient de détermination |
| **Kaggle Score** | Top 20% | Position sur le leaderboard public |

## Risques et Mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| Leakage temporel | Élevé | Time-series split strict |
| Overfitting | Élevé | Validation croisée, régularisation |
| Features manquantes | Moyen | Imputation intelligente |
| Déséquilibre cible | Moyen | Pondération, SMOTE |
| Complexité modèle | Faible | Commencer simple, itérer |