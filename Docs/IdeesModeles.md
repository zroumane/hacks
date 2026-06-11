# Récap Modèles — Stratégie pour le meilleur Brier Score

> Contexte : ~3h30 restantes. Objectif = minimiser le **Brier Score** sur 20 % d'avions hold-out.
> Problème = **classification probabiliste** sur **petit dataset** (758 avions train, 142 test),
> avec **censure par intervalle** (on connaît la date de *détection* de corrosion, pas d'*apparition*).

## 1. Ce qui guide le choix des modèles

| Contrainte | Implication sur le choix de modèle |
|---|---|
| Métrique = **Brier Score** | La **calibration** des probabilités prime sur la pure discrimination |
| **Petit dataset** (758 avions) | Modèles « foundation » pré-entraînés (TabPFN) et boosting brillent ; deep learning lourd à éviter |
| **Censure par intervalle** | Les modèles de **survie** (AFT, Cox) sont conceptuellement les plus justes |
| Données **tabulaires** + agrégats temporels | Gradient boosting = référence sur tabulaire |
| Hold-out **par avion** | Validation impérative en `GroupKFold(groups=aircraft_id)` |

**Classement « calibration » (étude empirique Manokhin)** :
- 🦅 *Eagles* (bonne calibration **et** discrimination) : **CatBoost, TabPFN**, EBM, Random Forest
- 🐂 *Bulls* (forte discrimination, calibration **médiocre mais corrigible**) : **XGBoost, LightGBM, HistGB** → à coupler avec **Venn-Abers**

## 2. Modèles recommandés (priorisés)

### Tier 1 — à faire absolument (rapides, fort ROI)

| Modèle | Pourquoi | Mise en œuvre |
|---|---|---|
| **CatBoost** (classif binaire) | Calibration native excellente, peu de tuning, robuste | `CatBoostClassifier(loss_function='Logloss')` |
| **LightGBM / XGBoost** (binaire) | Référence tabulaire, très rapide | + calibration **Venn-Abers** ou isotonic obligatoire |
| **TabPFN v2.5** | SOTA sur < 10k échantillons, **zéro tuning**, probas calibrées en ~secondes | `pip install tabpfn` → `TabPFNClassifier()` |

### Tier 2 — gère la censure par intervalle (le vrai problème du sujet)

| Modèle | Pourquoi | Mise en œuvre |
|---|---|---|
| **XGBoost AFT** | Gère nativement la **censure par intervalle** via `label_lower_bound`/`label_upper_bound` — colle exactement à « corrosion apparue entre livraison et détection » | `objective='survival:aft'` |
| **scikit-survival** : `GradientBoostingSurvivalAnalysis`, `RandomSurvivalForest`, `CoxnetSurvivalAnalysis` | Risque cumulé monotone = courbe sigmoïde du slide 2 | `loss='coxph'` ou `'ipcwls'` |
| **lifelines** : Weibull AFT, Cox | Baseline survie paramétrique interprétable | `WeibullAFTFitter` (supporte l'intervalle-censure) |

> Conversion survie → proba : `P(corrodé à la date d'inspection) = 1 − S(âge | exposition)` où `S` est la fonction de survie prédite. C'est directement scorable au Brier.

### Tier 3 — si du temps reste (gains marginaux)

| Modèle / technique | Pourquoi |
|---|---|
| **AutoGluon** (`presets='best_quality'`, `eval_metric='brier'`) | AutoML : teste + ensemble + calibre automatiquement ; lancer en tâche de fond ~1h |
| **EBM** (Explainable Boosting Machine) | Bien calibré + interprétable (bon pour le pitch « facteurs de corrosion ») |
| **Stacking** des Tier 1/2 | Méta-modèle `LogisticRegression` sur les probas out-of-fold |

## 3. La calibration — là où se gagne le Brier Score

Le Brier = **calibration + raffinement**. Étapes systématiques :

1. **Venn-Abers** (`venn-abers` sur PyPI) — calibration la plus fiable, un seul fit, garanties théoriques. À mettre sur XGBoost/LightGBM.
2. Sinon `CalibratedClassifierCV(method='isotonic')` (assez de données) ou `'sigmoid'` (Platt, si peu de positifs).
3. **Clip final** des probas dans `[0.02, 0.98]` : une proba extrême fausse coûte très cher au Brier.
4. **Toujours mesurer le Brier en CV** (`brier_score_loss`), pas l'accuracy/AUC.

## 4. Plan d'attaque sur 3h30

```
0:00–0:30  Construction cible + features (anti-leakage), GroupKFold par avion
           → labels (âge, Yi) avec buffer sur la zone grise (censure)
0:30–1:00  Baselines Tier 1 : CatBoost + LightGBM(+Venn-Abers), Brier en CV
1:00–1:30  TabPFN v2.5 (quasi plug-and-play) — souvent le meilleur seul
1:30–2:15  XGBoost AFT (censure par intervalle) + scikit-survival
           → comparer au cadre classif binaire
2:15–2:45  (option) lancer AutoGluon best_quality en tâche de fond
2:45–3:15  Stacking/ensemble des meilleurs + calibration finale (Venn-Abers)
3:15–3:30  Clip [0.02,0.98], génération submission.csv, sanity-check format
```

**Baseline de sécurité** : prédire la prévalence de `Yi=1` observée aux dates équivalentes. Tout modèle doit la battre en Brier.

## 5. Snippets clés

### XGBoost AFT — censure par intervalle
```python
import xgboost as xgb
# Pour chaque avion : la corrosion est apparue dans ]livraison, date_detection]
# borne basse = 0 (ou âge dernier C-check négatif), borne haute = âge à la détection
dtrain = xgb.DMatrix(X)
dtrain.set_float_info('label_lower_bound', y_lower)   # ex. âge dernier négatif
dtrain.set_float_info('label_upper_bound', y_upper)   # âge à la détection
params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.0,
          'tree_method': 'hist', 'learning_rate': 0.05}
bst = xgb.train(params, dtrain, num_boost_round=500)
# Prédiction = temps de survie médian → convertir en P(corrodé à la date cible)
```

### TabPFN — plug-and-play
```python
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier()          # pré-entraîné, aucun tuning
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]
```

### Venn-Abers sur un booster
```python
from venn_abers import VennAbersCalibrator
va = VennAbersCalibrator(estimator=lgbm, inductive=True, cal_size=0.2)
va.fit(X_train, y_train)
proba = va.predict_proba(X_test)[:, 1]
```

### Validation correcte
```python
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss
gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(X, y, groups=aircraft_id):
    ...  # score = brier_score_loss(y[va], proba_va)
```

## 6. Note techno IBM (pour le pitch)

- **Granite TimeSeries** : moins adapté ici (le sujet est de la survie tabulaire, pas du forecasting de série), mais utilisable pour générer des features d'embedding temporel de l'exposition.
- **watsonx.ai** : entraînement GPU pour AutoGluon / TabPFN, déploiement du modèle final.

---

## Sources

- [TabPFN — Nature 2024](https://www.nature.com/articles/s41586-024-08328-6) · [TabPFN v2.5](https://rehoyt.medium.com/tabpfn-v2-5-an-even-better-algorithm-68b72c6be5d5) · [GitHub PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)
- [XGBoost AFT — doc officielle](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html) · [Survival regression with AFT in XGBoost (arXiv)](https://arxiv.org/pdf/2006.04920)
- [scikit-survival — Gradient Boosted Models](https://scikit-survival.readthedocs.io/en/stable/user_guide/boosting.html)
- [Boosting methods for interval-censored data (arXiv 2026)](https://arxiv.org/pdf/2601.17973)
- [Manokhin Probability Matrix — calibration des classifieurs (arXiv 2026)](https://arxiv.org/html/2605.03816)
- [Venn-Abers calibration — V. Manokhin](https://valeman.medium.com/how-to-calibrate-your-classifier-in-an-intelligent-way-a996a2faf718)
