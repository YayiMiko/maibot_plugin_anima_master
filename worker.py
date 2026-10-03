"""Run one ComfyUI operation from an explicit private job snapshot."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "agent_tools"))
from comfyui_command_runner import run_cli_action  # noqa: E402
from comfyui_history import (  # noqa: E402
    ComfyUIHistoryRunner,
    history_failed,
    output_images,
)
from comfyui_operations import generate_payload  # noqa: E402
from comfyui_status import build_status_payload  # noqa: E402


def main() -> None:
    """Emit a structured result without discovering AstrBot config or roots."""
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    config = job["config"]
    config["_task_record"] = job["record"]
    if job["action"] in {"status", "diagnose"}:
        result = build_status_payload(config, config["allowed_sizes"])
    elif job["action"] in {"check_task", "recover_task"}:
        runner = ComfyUIHistoryRunner(config, Path(job["outputs"]))
        history = runner.history(job["prompt_id"])
        if history:
            failed = history_failed(history)
            result = {
                "ok": not bool(failed),
                "remote_state": "failed" if failed else "completed",
            }
            if job["action"] == "recover_task" and not failed:
                images = output_images(history)[: int(config["max_send_images"])]
                result["outputs"] = [
                    str(runner.download_image(image, i))
                    for i, image in enumerate(images, 1)
                ]
        else:
            queue = runner.client.get_json("/queue", timeout=20)
            state = "unknown"
            for key, label in (
                ("queue_running", "running"),
                ("queue_pending", "pending"),
            ):
                if any(
                    isinstance(item, list)
                    and len(item) > 1
                    and str(item[1]) == job["prompt_id"]
                    for item in queue.get(key, [])
                ):
                    state = label
            result = {"ok": False, "remote_state": state}
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
