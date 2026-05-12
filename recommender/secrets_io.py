from __future__ import annotations

from pathlib import Path


def read_env_style_key_file(path: str | Path, key_name: str) -> str:
    """
    Reads either:
    - `.env` style: KEY=value
    - plain text file containing only the secret
    """
    p = Path(path)
    if not p.exists():
        return ""
    # utf-8-sig strips a leading BOM (Windows editors often add it), which would otherwise
    # break `KEY=value` parsing on the first line.
    raw = p.read_text(encoding="utf-8-sig", errors="ignore")
    key_name_u = key_name.upper()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.upper().startswith(f"{key_name_u}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")

    one = raw.strip().strip('"').strip("'")
    return one
