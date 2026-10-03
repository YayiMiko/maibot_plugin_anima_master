"""Run one ComfyUI operation from an explicit private job snapshot."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "agent_tools"))
from comfyui_command_runner import run_cli_action  # noqa: E402
from comfyui_operations import generate_payload  # noqa: E402
from comfyui_status import build_status_payload  # noqa: E402


def main() -> None:
    """Emit a structured result without discovering AstrBot config or roots."""
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    config = job["config"]
    config["_task_record"] = job["record"]
    if job["action"] == "status":
        result = build_status_payload(config, config["allowed_sizes"])
    else:
        options = SimpleNamespace(
            width=job.get("width"),
            height=job.get("height"),
            override_size=bool(job.get("width")),
            steps=None,
            cfg=None,
            seed=None,
            negative_prompt=None,
        )
        result = run_cli_action(
            lambda: generate_payload(
                config, config, Path(job["outputs"]), options, job["prompt"]
            )
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
