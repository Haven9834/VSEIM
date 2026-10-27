"""Configuration, input validation and non-overwriting output handling."""
import csv
import hashlib
import json
import math
import re
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def identifier(value):
    value = str(value or "").strip()
    if value.lower() in {"nan", "none", "null"}:
        return ""
    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".")[0]
    return value


def safe_key(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("Identifiers must contain only ASCII letters, digits, dots, underscores or hyphens")
    stem = value.split(".")[0].upper()
    if value.endswith(".") or stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(?:COM|LPT)[1-9]", stem):
        raise ValueError("Identifier is not a portable filename")
    return value


def configured_path(value, base, *, kind="file"):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Required path is empty; complete your local configuration")
    p = Path(value).expanduser()
    p = (Path(base) / p).resolve() if not p.is_absolute() else p.resolve()
    if kind == "file" and not p.is_file():
        raise FileNotFoundError(p)
    if kind == "dir" and not p.is_dir():
        raise FileNotFoundError(p)
    return p


def confined_path(root, relative):
    """Never resolve a CSV filename outside its configured input directory."""
    relative = str(relative).replace("\\", "/")
    if not relative or Path(relative).is_absolute() or ":" in relative:
        raise ValueError("Expected a relative input filename")
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("Input filename escapes its configured directory")
    return candidate


def create_output(path):
    p = Path(path).resolve()
    if p.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {p}")
    p.mkdir(parents=True)
    return p


def csv_rows(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            yield line, row


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
