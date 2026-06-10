import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input, save_output

# --- Test load_input ---
print("=== load_input() ===")
try:
    content = load_input("test.txt")
    print(content)
except FileNotFoundError:
    print("  (aucun fichier test.txt dans /input)")

# --- Test save_output ---
print("\n=== save_output() ===")
dummy_data = "données de test \nligne 2\nligne 3"
output_path = save_output(dummy_data, "test.txt")
print(f"  Fichier créé : {output_path}")
