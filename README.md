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

Formats supportés : `.txt`. Les autres (`.csv`, `.json`, `.pdf`) sont en TODO dans `utils/io.py`.

## Usage

### Script Python classique

```bash
# Le script s'exécute directement via l'interpréteur Python
uv run src/<script.py>
```

### App Streamlit

Les apps Streamlit nécessitent le CLI.

```bash
uv run streamlit run src/<app.py>
```
