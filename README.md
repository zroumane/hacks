# Project

## Architecture

Voir [Docs/Architecture.md](Docs/Architecture.md) pour les diagrammes Mermaid.

## Project Structure

```
.
├── Docs/               # Project documentation
│   ├── Architecture.md     # Diagrammes Mermaid (flux + workflow)
│   └── Ressources.md       # Liens et outils du projet
├── input/              # Input documents
├── output/             # Timestamped output results
├── scripts/            # Scripts Bash
├── src/                # Code Python
│   ├── algo/           # Scripts de traitement
│   ├── streamlit/      # Scripts d'interface Streamlit
│   └── utils/          # Utilitaires partagés
│       └── io.py       # Chargement inputs / sauvegarde outputs
├── .gitignore
├── pyproject.toml      # Dépendances et config du projet (uv)
├── uv.lock             # Lockfile uv
└── README.md
```

## Setup

```bash
# Installer les dépendances et créer l'environnement virtuel
uv sync
```

## Dépendances python

```bash
# Ajouter une dépendance
uv add <package>

# Supprimer une dépendance
uv remove <package>

```

## Utils I/O

`utils/io.py` est un utilitaire partagé à importer dans tous les scripts. Il gère le chargement des inputs et la sauvegarde des outputs.

### Import

Depuis un script dans `src/algo/` ou `src/streamlit/`, ajouter `src/` au path avant l'import :

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input, save_output
```

### Fonctions disponibles

```python
# Parse et retourne le contenu de /input/<filename>
# Le format est détecté via l'extension (.txt supporté, autres en TODO)
content = load_input("data.txt")  # -> str | ...

# Sérialise data dans /output/<timestamp>_<filename>
# Le format est déterminé par l'extension du filename fourni
output_path = save_output(data, "result.txt")  # -> Path
```

Formats supportés : `.txt`, `.csv`, `.json`. Les autres (`.pdf` via Docling, `.docx`, `.pptx`...) sont aussi supportés en lecture.

## Pipeline de modélisation

### 1. Entraînement

Construit la cible, valide par cross-validation temporelle, entraîne XGBoost et sauvegarde le modèle.

```bash
uv run src/algo/train.py
# → output/models/xgb_corrosion.pkl
```

### 2. Prédiction

Génère le fichier de soumission Kaggle à partir d'un modèle `.pkl`.

```bash
uv run src/algo/predict.py output/models/xgb_corrosion.pkl
# → output/<timestamp>_submission.csv
```

## Usage général

### Script Python classique

```bash
uv run src/algo/<script.py>
```

### App Streamlit

Les apps Streamlit nécessitent le CLI.

```bash
uv run streamlit run src/streamlit/<app.py>
```
