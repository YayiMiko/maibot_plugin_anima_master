"""Bounded cleanup of expired request-owned records and job directories."""

import json
import math
import os
import re
import time
from pathlib import Path

TERMINAL_STATES = {
    "sent",
    "completed",
    "send_disabled",
    "planning_failed",
    "input_missing",
    "delivery_unknown",
    "generation_failed",
    "failed",
    "interrupted",
}


def clean_history(
    records: Path,
    runtime: Path,
    days: int,
    active: set[str],
    logger,
    cursor: dict | None = None,
) -> int:
    """Delete only expired, completed tasks identified by their own records.

    Args:
        records: Plugin-owned task record directory.
        runtime: SDK-owned runtime root containing task ID directories.
        days: Retention duration; zero or negative disables cleanup.
        active: Task IDs still executing, including image delivery.
        logger: Logger for counts and nonfatal failures.
        cursor: Optional in-memory scan position for fair bounded hourly passes.

    Returns:
        Number of fully removed task records. No unrelated paths are scanned.
    """
    if days <= 0 or not records.exists():
        return 0
    if any(path.is_symlink() or path.is_junction() for path in (records, runtime)):
        logger.warning("Retention refused a linked storage root")
        return 0
    record_root, job_root = records.resolve(), runtime.resolve()
    if record_root == job_root or len(job_root.parts) <= 1:
        logger.warning("Retention refused overlapping or broad storage roots")
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    paths = list(records.glob("*.json"))
    after = (cursor or {}).get("after", "")
    paths.sort(key=lambda path: (path.name <= after, path.name))
    for scanned, record_path in enumerate(paths):
        if scanned >= 1000 or removed >= 64:
            break
        task_id = record_path.stem
        if cursor is not None:
            cursor["after"] = record_path.name
        if task_id in active or not re.fullmatch(r"[a-f0-9]{24}", task_id):
            continue
        try:
            if record_path.is_symlink() or record_path.is_junction():
                continue
            if record_path.stat().st_mtime >= cutoff:
                continue
            if record_path.stat().st_size > 1024 * 1024:
                continue
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (
                not isinstance(record, dict)
                or record.get("task_id") != task_id
                or record.get("status") not in TERMINAL_STATES
            ):
                continue
            created = float(record.get("created_at", time.time()))
            if not math.isfinite(created) or created >= cutoff:
                continue
            directory = runtime / task_id
            # Never use image_inputs/outputs paths embedded in a record for deletion.
            if directory.is_symlink() or directory.is_junction():
                continue
            if directory.resolve() != job_root / task_id:
                continue
            files, directories = [], []
            valid = True
            if directory.exists():
                if not directory.is_dir() or directory.stat().st_mtime >= cutoff:
                    continue
                directories.append(directory)
                for parent, dirnames, filenames in os.walk(
                    directory, followlinks=False
                ):
                    for name in dirnames + filenames:
                        path = Path(parent) / name
                        if (
                            path.is_symlink()
                            or path.is_junction()
                            or not path.resolve().is_relative_to(directory.resolve())
                            or path.stat().st_mtime >= cutoff
                            or len(files) + len(directories) >= 256
                        ):
                            valid = False
                            break
                        if name in dirnames:
                            directories.append(path)
                        elif path.is_file():
                            files.append(path)
                        else:
                            valid = False
                            break
                    if not valid:
                        dirnames[:] = []
                        break
                if not valid:
                    continue
                # Remove explicit validated paths, never recursive deletion through links.
                for path in files + sorted(
                    directories, key=lambda p: len(p.parts), reverse=True
                ):
                    if not path.resolve().is_relative_to(job_root / task_id) or any(
                        p.is_symlink() or p.is_junction() for p in (path, *path.parents)
                    ):
                        raise ValueError("Storage path changed during retention")
                    if path in files:
                        path.unlink()
                    else:
                        path.rmdir()
            if record_path.resolve().parent != record_root or record_path.is_symlink():
                raise ValueError("Record path changed during retention")
            record_path.unlink()
            removed += 1
        except (OSError, ValueError, TypeError):
            logger.warning("Retention skipped task %s", task_id, exc_info=True)
    if removed:
        logger.info("Retention removed %s expired Anima tasks", removed)
    return removed
