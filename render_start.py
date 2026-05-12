from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    # Render sets PORT for web services. If it's missing, default to 8010 for local dev.
    port_raw = (os.environ.get("PORT") or "").strip() or "8010"
    try:
        port = int(port_raw)
    except ValueError:
        print(f"Invalid PORT={port_raw!r}", file=sys.stderr)
        raise SystemExit(2)

    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()

