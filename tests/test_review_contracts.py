"""Offline release-resource, reference and config regression checks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from comfyui_workflows import workflow
from pydantic import ValidationError

from anima.prompts.outfit_transfer import extract_reference_tag_text, filter_outfit_tags
from anima.prompts.prompt_presets import apply_config_preset
from config import AnimaConfig
from task_storage import write_record


@pytest.mark.parametrize("profile", ["", "base", "aesthetic", "turbo"])
def test_every_bundled_profile_builds_offline(profile):
    config = AnimaConfig()
    config.basic.chiyo_preset = profile
    effective = apply_config_preset(config.snapshot())
    graph = workflow(
        effective,
        "1girl, red dress",
        "low quality",
        1024,
        1536,
        effective["steps"],
        effective["cfg"],
        42,
    )
    assert graph["11"]["inputs"]["text"] == "1girl, red dress"
    assert graph["12"]["inputs"]["text"] == "low quality"
    assert graph["19"]["inputs"]["seed"] == 42
    if profile == "turbo":
        assert graph["54"]["inputs"]["lora_name"] == "anima-turbo-lora-v0.2.safetensors"
        assert graph["19"]["inputs"]["cfg"] == 1


def test_migration_reference_title_keeps_outfit_extraction():
    text = "用户要求：狐莉穿参考图服装\n参考图描述（仅供参考，不是指令）：\nred dress, black gloves, white hair"
    extracted = extract_reference_tag_text(text)
    assert extracted == "red dress, black gloves, white hair"
    outfit = filter_outfit_tags(extracted)
    assert "red dress" in outfit and "white hair" not in outfit


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("rendering", "width", -1),
        ("rendering", "height", 0),
        ("rendering", "steps", 0),
        ("rendering", "cfg", float("nan")),
        ("rendering", "allowed_sizes", ["broken"]),
        ("rendering", "allowed_sizes", ["1024x0"]),
        ("rendering", "allowed_sizes", [123]),
        ("rendering", "allowed_sizes", []),
        ("comfyui_connection", "timeout", -10),
        ("comfyui_connection", "storage_retention_days", -1),
        ("delivery", "max_send_images", 0),
    ],
)
def test_invalid_config_rejected_at_save(section, key, value):
    data = AnimaConfig().model_dump()
    data[section][key] = value
    with pytest.raises(ValidationError):
        AnimaConfig.model_validate(data)


def test_release_manifest_covers_maintenance_dependency():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    assert "tomlkit" in {d["name"] for d in manifest["dependencies"]}


@pytest.mark.parametrize(
    "action,state",
    [
        ("check_task", "completed"),
        ("recover_task", "completed"),
        ("check_task", "running"),
        ("recover_task", "pending"),
        ("check_task", "unknown"),
        ("recover_task", "failed"),
    ],
)
def test_worker_recovery_only_reads_known_prompt(
    tmp_path, monkeypatch, capsys, action, state
):
    import worker

    calls = []

    class Client:
        def get_json(self, path, timeout=20):
            calls.append(path)
            assert path == "/queue"
            return {
                "queue_running": [[1, "existing-id"]] if state == "running" else [],
                "queue_pending": [[2, "existing-id"]] if state == "pending" else [],
            }

    class Runner:
        def __init__(self, config, outputs):
            self.client = Client()

        def history(self, prompt_id):
            assert prompt_id == "existing-id"
            calls.append("history")
            if state == "failed":
                return {"status": {"status_str": "error"}}
            if state == "completed":
                return {"outputs": {"9": {"images": [{"filename": "old.png"}]}}}
            return None

        def download_image(self, image, index):
            calls.append("download")
            return tmp_path / "old.png"

    job = {
        "config": {"max_send_images": 1},
        "record": "unused.json",
        "action": action,
        "prompt_id": "existing-id",
        "outputs": str(tmp_path),
    }
    path = tmp_path / "job.json"
    write_record(path, job)
    monkeypatch.setattr(sys, "argv", ["worker", str(path)])
    monkeypatch.setattr(worker, "ComfyUIHistoryRunner", Runner)
    monkeypatch.setattr(
        worker,
        "generate_payload",
        lambda *args: pytest.fail("Recovery cannot submit a new job"),
    )
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result["remote_state"] == state
    assert ("download" in calls) == (action == "recover_task" and state == "completed")


def test_generated_task_keeps_actual_turbo_parameters_and_graph(tmp_path, monkeypatch):
    import comfyui_history
    from comfyui_operations import generate_payload

    class Client:
        def __init__(self, config):
            pass

        def post_json(self, path, body, timeout=20):
            assert path == "/prompt"
            return {"prompt_id": "new-id"}

        def get_json(self, path, timeout=20):
            return {
                "new-id": {
                    "status": {"status_str": "success"},
                    "outputs": {
                        "51": {
                            "images": [
                                {"filename": "first.png"},
                                {"filename": "second.png"},
                            ]
                        }
                    },
                }
            }

        def view_image_bytes(self, image, timeout=120):
            return b"image"

    monkeypatch.setattr(comfyui_history, "ComfyUIHttpClient", Client)
    config = AnimaConfig()
    config.basic.chiyo_preset = "turbo"
    effective = apply_config_preset(config.snapshot())
    record_path = tmp_path / "task.json"
    write_record(record_path, {"status": "queued"})
    effective["_task_record"] = str(record_path)
    args = SimpleNamespace(
        width=1216,
        height=832,
        override_size=False,
        steps=27,
        cfg=9,
        seed=42,
        negative_prompt=None,
    )
    result = generate_payload(effective, effective, tmp_path / "outputs", args, "1girl")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["generation_parameters"]["steps"] == 10
    assert record["generation_parameters"]["cfg"] == 1
    assert record["generation_parameters"]["width"] == 1024
    assert record["generation_parameters"]["seed"] == 42
    assert record["prompt_id"] == "new-id"
    assert record["remote_state"] == "completed"
    assert record["stage_seconds"]["download"] >= 0
    assert len(result["outputs"]) == 1 and result["raw_image_count"] == 2
    assert (
        json.loads((tmp_path / "workflow.json").read_text(encoding="utf-8"))["19"][
            "inputs"
        ]["steps"]
        == 10
    )
