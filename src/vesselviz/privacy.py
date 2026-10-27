"""Display-only masking. Masked identifiers are not join keys."""
import re


def mask_mmsi(value):
    if not re.fullmatch(r"\d{9}", str(value)):
        raise ValueError("Expected a nine-digit MMSI")
    return value[:4] + "*****"


def short_name(value, prefix=2):
    if not isinstance(prefix, int) or not 1 <= prefix <= 4:
        raise ValueError("Name prefix length must be between 1 and 4")
    value = str(value or "").strip()
    if not value.isascii() or value.lower() in {"", "0", "nan", "null", "none", "n/a", "unknown"}:
        return "N/A"
    letters = re.sub(r"[^A-Za-z0-9]", "", value)
    if not re.search(r"[A-Za-z]", letters):
        return "N/A"
    return letters[:prefix].upper() + "**"
