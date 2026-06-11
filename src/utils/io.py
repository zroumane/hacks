import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

DOCLING_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".asciidoc"}

_converter = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False
        pdf_options.do_table_structure = False
        _converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pdf_options)}
        )
    return _converter


def _parse_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8")
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if ext in DOCLING_FORMATS:
        result = _get_converter().convert(str(path))
        return result.document.export_to_markdown()
    if ext == ".csv":
        return pd.read_csv(path)
    raise NotImplementedError(f"Format non supporté : {ext}")


def _serialize_file(data, path: Path) -> None:
    ext = path.suffix.lower()
    if ext == ".txt":
        path.write_text(str(data), encoding="utf-8")
        return
    if ext == ".json":
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if ext == ".md":
        path.write_text(str(data), encoding="utf-8")
        return
    if ext == ".csv":
        data.to_csv(path, index=False)
        return
    # TODO: implémenter .pdf → générer un PDF à partir de data
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
