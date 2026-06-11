import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.io import load_input

datasets = [
    "corrosions_training.csv",
    "environment_test.csv",
    "environment_training.csv",
    "sample_submission.csv",
]

for filename in datasets:
    df = load_input(filename)
    print(f"\n{'='*50}")
    print(f"{filename}  ({df.shape[0]} lignes x {df.shape[1]} colonnes)")
    print(f"{'='*50}")
    for col in df.columns:
        print(f"  {col}  [{df[col].dtype}]")
