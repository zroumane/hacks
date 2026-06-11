import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
import os

load_dotenv()

api_key    = os.getenv("WATSONX_API_KEY")
project_id = os.getenv("WATSONX_PROJECT_ID")
url        = os.getenv("WATSONX_URL")

if not all([api_key, project_id, url]):
    print("Erreur : variables manquantes dans .env (WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL)")
    sys.exit(1)

from utils.io import load_input, save_output
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# --- Chargement du PDF via Docling ---
if len(sys.argv) < 2:
    print("Usage : uv run src/algo/test_watsonx.py <fichier>")
    print("Exemple : uv run src/algo/test_watsonx.py article.pdf")
    sys.exit(1)

filename = sys.argv[1]
print(f"Chargement de {filename}...")
text = load_input(filename)
print(f"Extraction OK ({len(text)} caractères)")

# --- Connexion watsonx ---
print(f"Connexion à {url}...")
client = APIClient(Credentials(url=url, api_key=api_key))
client.set.default_project(project_id)

model = ModelInference(
    model_id="ibm/granite-4-h-small",
    api_client=client,
)

# --- Résumé ---
messages = [
    {
        "role": "system",
        "content": "Tu es un assistant spécialisé en analyse de documents. Réponds uniquement à partir du texte fourni.",
    },
    {
        "role": "user",
        "content": f"Voici le contenu d'un document :\n\n{text}\n\nFais un résumé concis de ce document.",
    },
]

print("Génération du résumé...")
response = model.chat(messages=messages)
summary = response["choices"][0]["message"]["content"]

print(f"\n--- Résumé ---\n{summary}")

# --- Sauvegarde ---
output_path = save_output(summary, "resume.txt")
print(f"\nRésumé sauvegardé : {output_path}")
