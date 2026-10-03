"""Inspect or explicitly apply coordinated shared QQ send timeout settings."""

import argparse
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import tomlkit


def configure(
    adapter_dir: Path,
    napcat_config: Path,
    apply: bool = False,
    backup_dir: Path | None = None,
) -> dict:
    """Validate shared files and optionally back them up and replace them.

    Args:
        adapter_dir: Installed SnowLuma adapter directory.
        napcat_config: Active account's NapCat OneBot JSON configuration.
        apply: Whether to write changes; inspection is the default.
        backup_dir: Backup root outside both installed configuration directories.

    Returns:
        Non-secret timeout settings, match status, and changed filenames.

    Raises:
        ValueError: Files use an unfamiliar layout or larger existing budgets.
    """
    paths = [adapter_dir / "config.toml", adapter_dir / "plugin.py", napcat_config]
    originals = [path.read_text(encoding="utf-8") for path in paths]
    config = tomlkit.parse(originals[0])
    action = float(config["client"]["action_timeout_sec"])
    match = re.search(
        r"    @MessageGateway\(\n(?P<body>.*?)\n    \)", originals[1], re.S
    )
    if not match or "name=SNOWLUMA_GATEWAY_NAME," not in match["body"]:
        raise ValueError("Unrecognized SnowLuma gateway declaration")
    body = match["body"]
    timeout = re.search(r"^        timeout_ms=([\d_]+),$", body, re.M)
    if "timeout_ms" in body and timeout is None:
        raise ValueError("Unrecognized gateway timeout declaration")
    gateway = int(timeout[1].replace("_", "")) if timeout else 60000
    napcat = json.loads(originals[2])
    base = int(napcat["timeout"]["baseTimeout"])
    if action > 120 or gateway > 150000 or base > 90000:
        raise ValueError(
            "Existing timeout exceeds this deployment preset; review manually"
        )
    result = {
        "action_seconds": action,
        "gateway_ms": gateway,
        "napcat_base_ms": base,
        "matches": (action, gateway, base) == (120, 150000, 90000),
        "changed": [],
    }
    if not apply:
        return result
    if backup_dir is None:
        raise ValueError("Applying changes requires an external backup directory")
    backup_dir = backup_dir.resolve()
    if backup_dir.is_relative_to(adapter_dir.resolve()) or backup_dir.is_relative_to(
        napcat_config.resolve().parent
    ):
        raise ValueError(
            "Backups must be outside the installed configuration directories"
        )
    config["client"]["action_timeout_sec"] = 120.0
    line = "        timeout_ms=150_000,"
    new_body = (
        body[: timeout.start()] + line + body[timeout.end() :]
        if timeout
        else body + "\n" + line
    )
    source = (
        originals[1][: match.start("body")]
        + new_body
        + originals[1][match.end("body") :]
    )
    napcat["timeout"]["baseTimeout"] = 90000
    updated = [
        tomlkit.dumps(config),
        source,
        json.dumps(napcat, ensure_ascii=False, indent=2) + "\n",
    ]
    needs_update = [action != 120, gateway != 150000, base != 90000]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_root = backup_dir / ("qq-send-timeouts-" + stamp)
    if any(needs_update):
        backup_root.mkdir(parents=True, mode=0o700)
    for path, content, changed in zip(paths, updated, needs_update, strict=True):
        if not changed:
            continue
        backup = backup_root / path.name
        shutil.copy2(path, backup)
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=path.name + "."
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        result["changed"].append(path.name)
    result.update(
        action_seconds=120.0, gateway_ms=150000, napcat_base_ms=90000, matches=True
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--napcat-config", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir", type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            configure(
                arguments.adapter_dir,
                arguments.napcat_config,
                arguments.apply,
                arguments.backup_dir,
            )
        )
    )
