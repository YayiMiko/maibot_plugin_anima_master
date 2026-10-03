"""Migrate Anima settings without copying host credentials or old task state."""

import json
import sys
from pathlib import Path

import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import AnimaConfig  # noqa: E402


def main() -> None:
    """Validate a detached migration and refuse to overwrite existing config."""
    source, destination = (Path(arg) for arg in sys.argv[1:])
    if destination.exists():
        raise FileExistsError("Existing MaiBot configuration must be preserved")
    old = json.loads(source.read_text(encoding="utf-8-sig"))
    flat = {
        key: value
        for name, section in old.items()
        for key, value in (
            section.items()
            if name.startswith("anima_master_") and isinstance(section, dict)
            else [(name, section)]
        )
    }
    data = AnimaConfig().model_dump()
    for name, section in data.items():
        if name in {"plugin", "maibot"}:
            continue
        for key in section:
            if key in flat:
                section[key] = flat[key]
    # AstrBot provider IDs and auto-start settings have no MaiBot runtime meaning.
    data["prompting"]["prompt_builder_provider_id"] = ""
    data["comfyui_connection"]["auto_start"] = False
    data["basic"]["reset_to_defaults"] = False
    destination.write_text(
        tomlkit.dumps(AnimaConfig.model_validate(data).model_dump()), encoding="utf-8"
    )
    destination.chmod(0o600)
    print("Anima configuration migrated and validated")


if __name__ == "__main__":
    main()
