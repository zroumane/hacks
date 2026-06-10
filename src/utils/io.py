from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def _parse_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8")
    # TODO: implémenter .csv  → retourner un pandas DataFrame
    # TODO: implémenter .json → retourner un dict
    # TODO: implémenter .pdf  → retourner le texte extrait
    raise NotImplementedError(f"Format non supporté : {ext}")


def _serialize_file(data, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".txt":
        path.write_text(str(data), encoding="utf-8")
        return
    # TODO: implémenter .csv  → attendre un pandas DataFrame, appeler .to_csv()
    # TODO: implémenter .json → attendre un dict, appeler json.dump()
    # TODO: implémenter .pdf  → générer un PDF à partir de data
    raise NotImplementedError(f"Format non supporté : {ext}")


def load_input(filename: str):
    """Parse et retourne le contenu de /input/<filename>."""
    path = INPUT_DIR / filename
    return _parse_file(path)


def ensure_output_dir() -> Path:
    """Crée /output s'il n'existe pas et retourne son chemin."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_output(data, filename: str) -> Path:
    """Sérialise data dans /output/<timestamp>_<filename> et retourne le chemin."""
    ensure_output_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}_{filename}"
    _serialize_file(data, output_path)
    return output_path
