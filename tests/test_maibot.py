import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from config import AnimaConfig
from model_client import ModelClient
from plugin import AnimaPlugin, write_record
from tools.migrate_config import main as migrate


@pytest.fixture
def plugin(tmp_path):
    p = AnimaPlugin()
    p._plugin_config_instance = AnimaConfig()
    p._ctx = SimpleNamespace(
        paths=SimpleNamespace(
            data_dir=tmp_path / "data", runtime_dir=tmp_path / "runtime"
        ),
        logger=logging.getLogger("test-anima"),
        llm=SimpleNamespace(
            generate=AsyncMock(
                return_value={
                    "success": True,
                    "response": "1girl, red dress, standing, background_mode_default_portrait",
                }
            )
        ),
        send=SimpleNamespace(
            text=AsyncMock(return_value=True),
            image=AsyncMock(return_value={"sent": True, "message_id": "42"}),
        ),
    )
    p._plugin_config_instance.tag_lookup.danbooru_core_tag_lookup_enabled = False
    return p


@pytest.fixture
def command():
    return {
        "platform": "qq",
        "stream_id": "test-group",
        "user_id": "user",
        "group_id": "group",
        "text": "/anm 无优化 1girl, white background",
        "message": {"message_id": "message-1"},
    }


def fake_worker(monkeypatch):
    async def spawn(*args, **kwargs):
        job = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
        output = Path(job["outputs"]) / "image.png"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"image")
        payload = {"ok": True, "outputs": [str(output)], "prompt_id": "comfy-id"}
        return SimpleNamespace(
            returncode=0,
            communicate=AsyncMock(return_value=(json.dumps(payload).encode(), b"")),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)


def test_sdk_config_and_explicit_command(plugin):
    normalized, _ = plugin.normalize_plugin_config(plugin.config.model_dump())
    assert normalized["plugin"]["config_version"] == "0.1.0"
    component = plugin.get_components()[0]
    assert component["type"] == "COMMAND"
    pattern = component["metadata"]["command_pattern"]
    assert re.match(pattern, "/anm 白色小猫")
    assert re.match(pattern, "[image] /anm 反推")
    assert not re.match(pattern, "请用 /anm 白色小猫")
    assert not re.match(pattern, "/anmx test")


def test_config_snapshot_is_detached(plugin):
    snapshot = plugin.config.snapshot()
    snapshot["fixed_characters"].append("test=1girl")
    assert not plugin.config.style.fixed_characters


@pytest.mark.parametrize("multi_person", [False, True])
def test_raw_bypasses_llm_and_sends_once(plugin, command, monkeypatch, multi_person):
    fake_worker(monkeypatch)
    if multi_person:
        command["text"] = "/anm 多人 无优化 1girl, white background"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        plugin.ctx.llm.generate.assert_not_awaited()
        assert plugin.ctx.send.image.await_count == 1
        assert plugin.ctx.send.image.call_args.kwargs["timeout_ms"] == 180_000
        assert plugin.ctx.send.text.await_count == 1
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert record["prompt"] == "1girl, white background"
        assert record["status"] == "sent"
        assert record["delivery"][0]["message_id"] == "42"

    asyncio.run(run())


def test_normal_prompt_has_white_background_without_chat_persona(
    plugin, command, monkeypatch
):
    fake_worker(monkeypatch)
    command["text"] = "/anm 红裙少女"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert record["status"] == "sent"
        assert "white background" in record["prompt"]
        assert "simple background" in record["prompt"]
        messages = plugin.ctx.llm.generate.call_args.args[0]
        assert [m["role"] for m in messages] == ["system", "user"]
        assert plugin.ctx.llm.generate.call_args.kwargs["task_name"] == "utils"
        assert "狐莉" not in json.dumps(messages, ensure_ascii=False)

    asyncio.run(run())


@pytest.mark.parametrize(
    "text",
    [
        "/anm 改图 换衣服",
    ],
)
def test_unported_routes_never_generate(plugin, command, text):
    async def run():
        await plugin.on_load()
        command["text"] = text
        await plugin.handle_anm(**command)
        assert not plugin.jobs
        assert plugin.ctx.send.text.await_count == 1
        plugin.ctx.llm.generate.assert_not_awaited()

    asyncio.run(run())


def test_duplicate_message_and_config_snapshot(plugin, command):
    async def run():
        await plugin.on_load()
        plugin._execute = AsyncMock()
        await plugin.handle_anm(**command)
        plugin.config.rendering.cfg = 9
        await plugin.handle_anm(**command)
        await asyncio.sleep(0)
        assert plugin._execute.await_count == 1
        assert plugin._execute.call_args.args[1]["config"]["cfg"] != 9

    asyncio.run(run())


def test_restart_does_not_resubmit(plugin):
    async def run():
        path = plugin.ctx.paths.data_dir / "tasks/a.json"
        write_record(path, {"status": "generating", "prompt_id": "existing"})
        await plugin.on_load()
        assert json.loads(path.read_text())["status"] == "remote_unknown"
        assert not plugin.jobs

    asyncio.run(run())


def test_uncertain_delivery_not_resent_even_if_notice_fails(
    plugin, command, monkeypatch
):
    fake_worker(monkeypatch)
    plugin.ctx.send.image.side_effect = TimeoutError("private-address")
    plugin.ctx.send.text.side_effect = [True, TimeoutError(), TimeoutError()]

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        assert plugin.ctx.send.image.await_count == 1
        assert plugin.ctx.send.image.call_args.kwargs["timeout_ms"] == 180_000
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert record["status"] == "delivery_unknown"

    asyncio.run(run())


def test_model_failure_is_explicit(plugin):
    plugin.ctx.llm.generate.return_value = {"success": False}
    with pytest.raises(RuntimeError):
        asyncio.run(
            ModelClient(plugin.ctx, plugin.config.snapshot()).llm_generate(
                prompt="test"
            )
        )


def test_migrate_grouped_settings_and_never_copy_credentials(tmp_path, monkeypatch):
    source, target = tmp_path / "old.json", tmp_path / "config.toml"
    source.write_text(
        json.dumps(
            {
                "anima_master_style": {
                    "fixed_characters": ["test=1girl"],
                    "artist_presets": ["test=@artist,"],
                },
                "anima_master_models": {"unet_name": "existing.safetensors"},
                "anima_master_prompting": {
                    "prompt_builder_provider_id": "astrbot-only-id"
                },
                "anima_master_comfyui_connection": {"auto_start": True},
                "api_key": "secret",
            }
        ),
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(sys, "argv", ["migrate", str(source), str(target)])
    migrate()
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["models"]["unet_name"] == "existing.safetensors"
    assert data["style"]["fixed_characters"] == ["test=1girl"]
    assert not data["prompting"]["prompt_builder_provider_id"]
    assert not data["comfyui_connection"]["auto_start"]
    assert "secret" not in target.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        migrate()


def test_worker_uses_only_explicit_snapshot(tmp_path, monkeypatch, capsys):
    import worker

    job = {
        "config": {"allowed_sizes": ["1024x1024"]},
        "record": "task.json",
        "action": "generate",
        "prompt": "1girl",
        "width": 1024,
        "height": 1024,
        "outputs": str(tmp_path / "outputs"),
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker", str(path)])

    def generate(config, defaults, outputs, options, prompt):
        assert config is defaults
        assert options.override_size
        assert prompt == "1girl"
        return {"ok": True}

    monkeypatch.setattr(worker, "generate_payload", generate)
    worker.main()
    assert json.loads(capsys.readouterr().out)["ok"]


@pytest.mark.parametrize("alias", ["/anima", "/comfyui"])
def test_alias_and_size_are_request_local(plugin, command, alias):
    async def run():
        await plugin.on_load()
        plugin._execute = AsyncMock()
        command["text"] = f"{alias} 横图：无优化 1girl"
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        request = plugin._execute.call_args.args[1]
        assert request["width"] > request["height"]
        assert request["prompt"] == "无优化 1girl"

    asyncio.run(run())


@pytest.mark.parametrize("text", ["/anm 无优化 1girl", "/anm 添加角色 测试=1girl"])
def test_permissions_prevent_planning(plugin, command, text):
    plugin.config.delivery.allowed_sender_ids = ["another-user"]
    command["text"] = text

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        assert not plugin.jobs
        assert not list(plugin.records.glob("*.json"))
        plugin.ctx.llm.generate.assert_not_awaited()

    asyncio.run(run())


def test_debug_does_not_expose_other_user_or_private_values(plugin, command):
    async def run():
        await plugin.on_load()
        write_record(
            plugin.records / "foreign.json",
            {"scope": "another-user", "created_at": 999, "prompt": "private"},
        )
        command["text"] = "/anm 调试状态"
        await plugin.handle_anm(**command)
        assert plugin.ctx.send.text.call_args.args[0] == "暂无你的任务记录。"

    asyncio.run(run())


def test_explicit_model_route(plugin):
    config = plugin.config.snapshot()
    config["prompt_model_name"] = "test-model"
    asyncio.run(ModelClient(plugin.ctx, config).llm_generate(prompt="test"))
    assert plugin.ctx.llm.generate.call_args.kwargs["model_name"] == "test-model"


@pytest.mark.parametrize("requested_count", [2, 3])
def test_multi_person_uses_structured_planner_and_blocks_count_mismatch(
    plugin, command, monkeypatch, requested_count
):
    fake_worker(monkeypatch)
    plan = {
        "count_tags": ["2girls"],
        "common_tags": ["standing"],
        "background_mode": "default_portrait",
        "spatial_mode": "shared_contact",
        "relationship_tag": "holding hands",
        "interactions": ["Character A is holding Character B's hand."],
        "characters": [
            {
                "slot": slot,
                "appearance": appearance,
                "clothing": "white dress",
                "role": "girl",
                "visual_label": label,
            }
            for slot, appearance, label in [
                ("left", "black hair", "black-haired girl"),
                ("right", "blonde hair", "blonde girl"),
            ]
        ],
    }
    plugin.ctx.llm.generate.return_value = {
        "success": True,
        "response": json.dumps(plan),
    }
    command["text"] = f"/anm 多人 {requested_count}名少女牵手站立"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert record["multi_person"]
        summary = record["prompt_summary"]
        if requested_count == 2:
            assert record["status"] == "sent"
            assert summary["planned_character_count"] == 2
            assert summary["grouped_contact"]
            assert "black-haired girl" in record["prompt"]
            assert "blonde girl" in record["prompt"]
            assert "white background" in record["prompt"]
            plugin.ctx.send.image.assert_awaited_once()
        else:
            assert record["status"] == "planning_failed"
            assert summary["multi_person_plan_failed"]
            plugin.ctx.send.image.assert_not_awaited()
            assert not list(plugin.runtime.glob("*/job.json"))
            assert plugin.ctx.llm.generate.await_count == 2

    asyncio.run(run())


def test_multi_person_invalid_json_never_submits(plugin, command, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    command["text"] = "/anm 多人 两名少女"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        spawn.assert_not_awaited()
        plugin.ctx.send.image.assert_not_awaited()
        assert "本次未提交生图" in plugin.ctx.send.text.call_args.args[0]

    asyncio.run(run())


@pytest.mark.parametrize(
    "action", ["解析法术", "反推", "生图 少女站立", "无优化 1girl"]
)
def test_explicit_reference_paths(plugin, command, monkeypatch, action):
    fake_worker(monkeypatch)
    output = io.BytesIO()
    info = PngInfo()
    info.add_text(
        "parameters", "1girl, blue dress\nNegative prompt: low quality\nSteps: 20"
    )
    Image.new("RGB", (16, 24)).save(output, format="PNG", pnginfo=info)
    encoded = "base64://" + base64.b64encode(output.getvalue()).decode()
    plugin.ctx.api = SimpleNamespace(
        call=AsyncMock(
            return_value={
                "group_id": "group",
                "message": [{"type": "image", "data": {"url": encoded}}],
            }
        )
    )
    command["text"] = f"[image] /anm {action}"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        with Image.open(record["image_inputs"][0]) as saved:
            assert saved.info["parameters"].startswith("1girl, blue dress")
        if action == "解析法术":
            assert record["status"] == "completed"
            assert "1girl, blue dress" in plugin.ctx.send.text.call_args.args[0]
            plugin.ctx.llm.generate.assert_not_awaited()
            plugin.ctx.send.image.assert_not_awaited()
        elif action == "反推":
            assert record["status"] == "completed"
            messages = plugin.ctx.llm.generate.call_args.args[0]
            assert messages[1]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
            assert plugin.ctx.llm.generate.call_args.kwargs["task_name"] == "vlm"
            plugin.ctx.send.image.assert_not_awaited()
        elif action.startswith("无优化"):
            assert record["prompt"] == "1girl"
            plugin.ctx.llm.generate.assert_not_awaited()
        else:
            assert record["status"] == "sent"
            assert record["reference_context_method"] == "spell"
            messages = plugin.ctx.llm.generate.call_args.args[0]
            assert "参考图描述" in messages[1]["content"]
            assert "blue dress" in messages[1]["content"]
            assert "white background" in record["prompt"]

    asyncio.run(run())


def test_missing_image_never_uses_latest_or_submits(plugin, command, monkeypatch):
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    plugin.ctx.api = SimpleNamespace(
        call=AsyncMock(return_value={"group_id": "group", "message": []})
    )
    command["text"] = "/anm 反推"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        assert "未找到图片" in plugin.ctx.send.text.call_args.args[0]
        plugin.ctx.llm.generate.assert_not_awaited()
        spawn.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("vision_ok", [True, False])
def test_reference_without_metadata_uses_vision_or_stops(
    plugin, command, monkeypatch, vision_ok
):
    fake_worker(monkeypatch)
    output = io.BytesIO()
    Image.new("RGB", (16, 24)).save(output, format="PNG")
    encoded = "base64://" + base64.b64encode(output.getvalue()).decode()
    plugin.ctx.api = SimpleNamespace(
        call=AsyncMock(
            return_value={
                "group_id": "group",
                "message": [{"type": "image", "data": {"url": encoded}}],
            }
        )
    )
    plugin.ctx.llm.generate.side_effect = [
        {
            "success": vision_ok,
            "response": "1girl, blue dress, forest" if vision_ok else "",
        },
        {
            "success": True,
            "response": "1girl, blue dress, background_mode_default_portrait",
        },
    ]
    command["text"] = "[image] /anm 少女站立"

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert plugin.ctx.llm.generate.call_args_list[0].kwargs["task_name"] == "vlm"
        if vision_ok:
            assert record["reference_context_method"] == "reverse"
            assert record["status"] == "sent"
            assert "white background" in record["prompt"]
            plugin.ctx.send.image.assert_awaited_once()
        else:
            assert record["status"] == "failed"
            plugin.ctx.send.image.assert_not_awaited()
            assert not list(plugin.runtime.glob("*/job.json"))

    asyncio.run(run())


def test_chat_presets_persist_reload_and_snapshot_isolation(plugin, command):
    async def run():
        await plugin.on_load()
        plugin._execute = AsyncMock()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        running_snapshot = plugin._execute.call_args.args[1]["config"]
        command["text"] = "/anm 创建画师预设 测试=@example,"
        command["message"]["message_id"] = "preset-1"
        await plugin.handle_anm(**command)
        saved = json.loads(plugin.preset_path.read_text(encoding="utf-8"))
        assert saved["overrides"]["active_artist_preset"] == "测试"
        assert running_snapshot["active_artist_preset"] != "测试"
        assert plugin.config.style.active_artist_preset != "测试"
        before = plugin.ctx.send.text.await_count
        await plugin.handle_anm(**command)
        assert plugin.ctx.send.text.await_count == before
        await plugin.on_load()
        assert plugin.chat_presets == saved
        command["text"] = "/anm 无优化 1girl"
        command["message"]["message_id"] = "generation-2"
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        assert (
            plugin._execute.call_args.args[1]["config"]["active_artist_preset"]
            == "测试"
        )
        plugin.ctx.llm.generate.assert_not_awaited()

    asyncio.run(run())


def test_preset_save_failure_does_not_change_memory(plugin, command, monkeypatch):
    async def run():
        await plugin.on_load()
        command["text"] = "/anm 添加角色 测试=1girl"
        monkeypatch.setattr(
            "plugin.write_record", lambda *args: (_ for _ in ()).throw(OSError())
        )
        await plugin.handle_anm(**command)
        assert plugin.chat_presets["overrides"] == {}
        assert "保存失败" in plugin.ctx.send.text.call_args.args[0]
        assert not plugin.jobs

    asyncio.run(run())


def test_preset_reset_requires_confirmation_and_restores_host_config(plugin, command):
    async def run():
        await plugin.on_load()
        command["text"] = "/anm 添加角色 测试=1girl"
        await plugin.handle_anm(**command)
        command["message"]["message_id"] = "reset-1"
        command["text"] = "/anm 重置聊天预设"
        await plugin.handle_anm(**command)
        assert plugin.chat_presets["overrides"]
        command["text"] = "/anm 重置聊天预设 确认"
        await plugin.handle_anm(**command)
        assert plugin.chat_presets["overrides"] == {}
        assert (
            json.loads(plugin.preset_path.read_text(encoding="utf-8"))["overrides"]
            == {}
        )

    asyncio.run(run())


def test_invalid_preset_storage_is_preserved(plugin):
    async def run():
        plugin.ctx.paths.data_dir.mkdir(parents=True)
        path = plugin.ctx.paths.data_dir / "chat_presets.json"
        value = '{"overrides":{"comfyui_url":"untrusted"},"processed":[]}'
        path.write_text(value, encoding="utf-8")
        with pytest.raises(ValueError):
            await plugin.on_load()
        assert path.read_text(encoding="utf-8") == value

    asyncio.run(run())


def test_two_preset_commands_do_not_lose_updates(plugin, command):
    async def run():
        await plugin.on_load()
        first = dict(
            command,
            text="/anm 添加角色 甲=1girl, black hair",
            message={"message_id": "preset-a"},
        )
        second = dict(
            command,
            text="/anm 添加角色 乙=1girl, red hair",
            message={"message_id": "preset-b"},
        )
        await asyncio.gather(plugin.handle_anm(**first), plugin.handle_anm(**second))
        characters = plugin.chat_presets["overrides"]["fixed_characters"]
        assert any(line.startswith("甲=") for line in characters)
        assert any(line.startswith("乙=") for line in characters)
        assert not plugin.jobs
        plugin.ctx.llm.generate.assert_not_awaited()

    asyncio.run(run())


def test_unload_cancels_storage_maintenance_without_network(plugin):
    async def run():
        await plugin.on_load()
        maintenance = plugin.maintenance_task
        assert not maintenance.done()
        await plugin.on_unload()
        assert maintenance.cancelled()
        plugin.ctx.send.text.assert_not_awaited()
        plugin.ctx.llm.generate.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize("text", ["/anm 状态", "/anm 诊断"])
def test_status_has_reserved_capacity_and_handles_custom_workflow(
    plugin, command, monkeypatch, text
):
    async def spawn(*args, **kwargs):
        job = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
        assert job["action"] in {"status", "diagnose"}
        payload = {
            "ok": True,
            "comfyui_api_reachable": True,
            "custom_workflow_validation_deferred": True,
            "dns_checks": {"private-host": True},
            "base_url": "http://private-address",
        }
        return SimpleNamespace(
            returncode=0,
            communicate=AsyncMock(return_value=(json.dumps(payload).encode(), b"")),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def run():
        await plugin.on_load()
        plugin.job_actions.update(
            {str(i): "generate" for i in range(plugin.config.maibot.max_pending)}
        )
        command.update(text=text)
        command["message"]["reply_to"] = "quoted-text-not-an-image"
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        reply = plugin.ctx.send.text.call_args.args[0]
        assert "自定义工作流（提交时校验）" in reply
        assert "未就绪" not in reply and "private" not in reply
        if text.endswith("诊断"):
            assert "DNS：1/1" in reply and "生图队列：8" in reply
        plugin.ctx.llm.generate.assert_not_awaited()
        await plugin.on_unload()

    asyncio.run(run())


def test_failed_acceptance_notice_does_not_abort_generation(
    plugin, command, monkeypatch
):
    fake_worker(monkeypatch)
    plugin.ctx.send.text.side_effect = TimeoutError("notice-only")

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        plugin.ctx.send.image.assert_awaited_once()
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert record["status"] == "sent"
        assert record["delivery_started"]
        assert record["delivery"][0]["seconds"] >= 0
        assert record["stage_seconds"]["total"] >= 0

    asyncio.run(run())


def test_generation_timeout_keeps_remote_unknown_and_prompt_id(
    plugin, command, monkeypatch
):
    async def spawn(*args, **kwargs):
        job = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
        path = Path(job["record"])
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(status="generating", prompt_id="existing-job")
        write_record(path, record)
        return SimpleNamespace(
            returncode=0,
            communicate=AsyncMock(
                return_value=(
                    json.dumps({"ok": False, "error": "timeout_after_300s"}).encode(),
                    b"",
                )
            ),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert (
            record["status"] == "remote_unknown"
            and record["prompt_id"] == "existing-job"
        )
        assert "核对任务" in plugin.ctx.send.text.call_args.args[0]
        plugin.ctx.send.image.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.parametrize(
    "state,expected",
    [
        ("planning", "interrupted"),
        ("submitting", "remote_unknown"),
        ("sending", "delivery_unknown"),
    ],
)
def test_reload_distinguishes_remote_and_delivery_uncertainty(plugin, state, expected):
    async def run():
        path = plugin.ctx.paths.data_dir / "tasks/test.json"
        write_record(path, {"status": state})
        await plugin.on_load()
        assert json.loads(path.read_text())["status"] == expected
        assert not plugin.jobs

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode",
    [
        "recover",
        "check",
        "foreign",
        "delivery_unknown",
        "sent",
        "simultaneous",
        "send_timeout",
    ],
)
def test_manual_recovery_is_scoped_and_never_resamples_or_resends(
    plugin, command, monkeypatch, mode
):
    target = "a" * 24
    scope = hashlib.sha256(
        json.dumps(
            [command["platform"], command["stream_id"], command["user_id"]]
        ).encode()
    ).hexdigest()
    record_path = plugin.ctx.paths.data_dir / "tasks" / f"{target}.json"
    original = {
        "task_id": target,
        "scope": scope,
        "created_at": 1,
        "action": "generate",
        "prompt_id": "old-prompt",
        "status": "remote_unknown",
    }
    if mode == "foreign":
        original["scope"] = "another-user"
    elif mode in {"delivery_unknown", "sent"}:
        original.update(status=mode, delivery_started=True)
    write_record(record_path, original)
    write_record(
        plugin.ctx.paths.runtime_dir / target / "job.json",
        {"config": plugin.config.snapshot()},
    )
    spawned = []

    async def spawn(*args, **kwargs):
        job = json.loads(Path(args[-1]).read_text(encoding="utf-8"))
        spawned.append(job["action"])
        assert job["action"] in {"check_task", "recover_task"}
        assert job["prompt_id"] == "old-prompt"
        output = Path(job["outputs"]) / "recovered.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"image")
        result = {"ok": True, "remote_state": "completed", "outputs": [str(output)]}
        return SimpleNamespace(
            returncode=0,
            communicate=AsyncMock(return_value=(json.dumps(result).encode(), b"")),
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if mode == "send_timeout":
        plugin.ctx.send.image.side_effect = TimeoutError()

    async def run():
        await plugin.on_load()
        command["text"] = (
            f"/anm {'核对任务' if mode == 'check' else '恢复任务'} {target}"
        )
        await plugin.handle_anm(**command)
        if mode == "simultaneous":
            command["message"]["message_id"] = "second-recovery"
            await plugin.handle_anm(**command)
        if mode not in {"foreign", "delivery_unknown", "sent"}:
            assert target in plugin.history_targets.values()
        if plugin.jobs:
            await asyncio.gather(*list(plugin.jobs.values()))
        if mode in {"foreign", "delivery_unknown", "sent"}:
            assert not spawned
            plugin.ctx.send.image.assert_not_awaited()
        elif mode == "check":
            assert spawned == ["check_task"]
            plugin.ctx.send.image.assert_not_awaited()
        else:
            assert spawned == ["recover_task"]
            plugin.ctx.send.image.assert_awaited_once()
            saved = json.loads(record_path.read_text(encoding="utf-8"))
            assert saved["status"] == (
                "delivery_unknown" if mode == "send_timeout" else "sent"
            )
            command["message"]["message_id"] = "repeat-recovery"
            await plugin.handle_anm(**command)
            assert len(spawned) == 1
        plugin.ctx.llm.generate.assert_not_awaited()
        await plugin.on_unload()

    asyncio.run(run())


def test_scoped_latest_generation_index_survives_reload(plugin, command, monkeypatch):
    fake_worker(monkeypatch)

    async def run():
        await plugin.on_load()
        await plugin.handle_anm(**command)
        await asyncio.gather(*list(plugin.jobs.values()))
        generation_id = next(iter(plugin.latest_tasks.values()))
        await plugin.on_unload()
        await plugin.on_load()
        command["text"] = "/anm 调试状态"
        await plugin.handle_anm(**command)
        assert generation_id in plugin.ctx.send.text.call_args.args[0]
        assert "状态：sent" in plugin.ctx.send.text.call_args.args[0]
        await plugin.on_unload()

    asyncio.run(run())


def test_check_task_does_not_release_another_recoverys_reservation(plugin, command):
    target = "a" * 24
    scope = hashlib.sha256(
        json.dumps(
            [command["platform"], command["stream_id"], command["user_id"]]
        ).encode()
    ).hexdigest()
    write_record(
        plugin.ctx.paths.data_dir / "tasks" / f"{target}.json",
        {
            "task_id": target,
            "scope": scope,
            "created_at": 1,
            "action": "generate",
            "prompt_id": "old-id",
            "status": "remote_unknown",
        },
    )
    write_record(
        plugin.ctx.paths.runtime_dir / target / "job.json",
        {"config": plugin.config.snapshot()},
    )

    async def run():
        await plugin.on_load()
        release = asyncio.Event()

        async def execute(task_id, request):
            if request["action"] == "recover_task":
                await release.wait()

        plugin._execute = AsyncMock(side_effect=execute)
        command["text"] = f"/anm 恢复任务 {target}"
        await plugin.handle_anm(**command)
        command["text"] = f"/anm 核对任务 {target}"
        command["message"]["message_id"] = "check-second"
        await plugin.handle_anm(**command)
        checks = [
            task
            for key, task in plugin.jobs.items()
            if plugin.job_actions[key] == "check_task"
        ]
        await asyncio.gather(*checks)
        await asyncio.sleep(0)
        assert target in plugin.recovering
        command["text"] = f"/anm 恢复任务 {target}"
        command["message"]["message_id"] = "recover-third"
        await plugin.handle_anm(**command)
        assert plugin._execute.await_count == 2
        release.set()
        await asyncio.gather(*list(plugin.jobs.values()))
        await plugin.on_unload()

    asyncio.run(run())


def test_unload_while_image_send_pending_never_makes_recoverable(
    plugin, command, monkeypatch
):
    fake_worker(monkeypatch)

    async def run():
        await plugin.on_load()
        sending = asyncio.Event()

        async def delayed(*args, **kwargs):
            sending.set()
            await asyncio.Future()

        plugin.ctx.send.image.side_effect = delayed
        await plugin.handle_anm(**command)
        await asyncio.wait_for(sending.wait(), timeout=3)
        await plugin.on_unload()
        record = json.loads(
            next(plugin.records.glob("*.json")).read_text(encoding="utf-8")
        )
        assert record["status"] == "delivery_unknown"
        assert record["delivery_started"]
        plugin.ctx.send.image.assert_awaited_once()

    asyncio.run(run())
