import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input, save_output
from utils.watsonx import get_llm

if len(sys.argv) < 2:
    print("Usage : uv run src/algo/test_watsonx.py <fichier>")
    print("Exemple : uv run src/algo/test_watsonx.py article.pdf")
    sys.exit(1)

filename = sys.argv[1]
print(f"Chargement de {filename}...")
text = load_input(filename)
print(f"Extraction OK ({len(text)} caractères)")

model = get_llm("ibm/granite-4-h-small")

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

output_path = save_output(summary, "resume.txt")
print(f"\nRésumé sauvegardé : {output_path}")
