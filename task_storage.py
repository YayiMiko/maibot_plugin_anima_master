"""Private atomic task storage shared by the command and worker processes."""

import json
import os
import uuid
from pathlib import Path


def write_record(path: Path, data: dict) -> None:
    """Persist a private snapshot atomically.

    Args:
        path: Plugin-owned destination.
        data: JSON-serializable request-local values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            temporary.chmod(0o600)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
