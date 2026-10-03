"""MaiBot command entry point for the first Anima migration stage."""

import asyncio
import base64
import copy
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

from maibot_sdk import Command, MaiBotPlugin

if __package__:
    from .anima.commands.command_router import parse_generation_size, parse_hard_route
    from .anima.prompts.danbooru_resolver import DanbooruResolver
    from .anima.prompts.prompt_pipeline import PromptPipeline
    from .anima.prompts.prompt_presets import apply_config_preset, strip_raw_prefix
    from .anima.prompts.prompt_research import PromptResearcher
    from .config import AnimaConfig
    from .image_reference import inspect_image, reverse_image
    from .media import ImageResolver
    from .model_client import ModelClient
    from .preset_actions import PRESET_ACTIONS, handle_preset
    from .storage_retention import clean_history
else:
    from anima.commands.command_router import parse_generation_size, parse_hard_route
    from anima.prompts.danbooru_resolver import DanbooruResolver
    from anima.prompts.prompt_pipeline import PromptPipeline
    from anima.prompts.prompt_presets import apply_config_preset, strip_raw_prefix
    from anima.prompts.prompt_research import PromptResearcher
    from config import AnimaConfig
    from image_reference import inspect_image, reverse_image
    from media import ImageResolver
    from model_client import ModelClient
    from preset_actions import PRESET_ACTIONS, handle_preset
    from storage_retention import clean_history

COMMAND_PATTERN = r"(?i)^\s*(?:\[image\]\s*)*/(?:anm|anima|comfyui)(?:\s|$)"
HELP = (
    "Anima 指令：\n/anm <描述>\n/anm 多人 <描述>\n/anm 无优化 <tags>\n/anm 横图：<描述>\n"
    "/anm 解析法术（附图或引用图）\n/anm 反推（附图或引用图）\n"
    "/anm 查看画师预设\n/anm 查看角色\n/anm 状态\n改图尚未开放。"
)


def write_record(path: Path, data: dict) -> None:
    """Atomically persist one private request record or worker snapshot.

    Args:
        path: Plugin-owned destination.
        data: JSON-serializable request-local values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


class AnimaPlugin(MaiBotPlugin):
    """Explicit commands with isolated planning and serial ComfyUI submission."""

    config_model = AnimaConfig

    async def on_load(self) -> None:
        """Initialize owned storage and record interrupted jobs without retries."""
        if getattr(self, "maintenance_task", None):
            self.maintenance_task.cancel()
            await asyncio.gather(self.maintenance_task, return_exceptions=True)
        self.jobs = {}
        self.worker_lock = asyncio.Lock()
        self.cache = {}
        self.data = self.ctx.paths.data_dir
        self.runtime = self.ctx.paths.runtime_dir
        self.records = self.data / "tasks"
        self.records.mkdir(parents=True, exist_ok=True)
        self.preset_path = self.data / "chat_presets.json"
        self.chat_presets = {"overrides": {}, "processed": []}
        if self.preset_path.exists():
            if self.preset_path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("Chat preset storage exceeds size limit")
            stored = json.loads(self.preset_path.read_text(encoding="utf-8"))
            if not isinstance(stored, dict) or set(stored) != {
                "overrides",
                "processed",
            }:
                raise ValueError("Invalid chat preset storage; refusing to overwrite")
            overrides, processed = stored.get("overrides"), stored.get("processed")
            if (
                not isinstance(overrides, dict)
                or not isinstance(processed, list)
                or len(processed) > 128
                or any(
                    not isinstance(item, str) or not re.fullmatch(r"[a-f0-9]{24}", item)
                    for item in processed
                )
                or set(overrides)
                - {
                    "artist_presets",
                    "active_artist_preset",
                    "default_artist_tags",
                    "fixed_characters",
                }
                or any(
                    not isinstance(value, list)
                    or any(not isinstance(item, str) for item in value)
                    if key in {"artist_presets", "fixed_characters"}
                    else not isinstance(value, str)
                    for key, value in overrides.items()
                )
            ):
                raise ValueError("Invalid chat preset storage; refusing to overwrite")
            self.chat_presets = stored
        for path in self.records.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("status") in {
                    "accepted",
                    "planning",
                    "reading_image",
                    "queued",
                    "generating",
                    "sending",
                }:
                    record["status"] = "interrupted"
                    write_record(path, record)
            except (OSError, ValueError):
                self.ctx.logger.warning("Unreadable Anima task: %s", path.name)
        self.retention_cursor = {}
        clean_history(
            self.records,
            self.runtime,
            int(self.config.snapshot()["storage_retention_days"]),
            set(self.jobs),
            self.ctx.logger,
            self.retention_cursor,
        )
        self.maintenance_task = asyncio.create_task(self._maintain_storage())

    async def _maintain_storage(self) -> None:
        """Check expired task storage hourly, without network access or retries."""
        while True:
            await asyncio.sleep(3600)
            clean_history(
                self.records,
                self.runtime,
                int(self.config.snapshot()["storage_retention_days"]),
                set(self.jobs),
                self.ctx.logger,
                self.retention_cursor,
            )

    async def on_unload(self) -> None:
        """Cancel local workers without globally interrupting ComfyUI."""
        tasks = list(self.jobs.values()) + [self.maintenance_task]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def on_config_update(
        self, scope: str, config_data: dict, version: str
    ) -> None:
        """Keep running requests isolated from subsequent configuration updates."""
        self.ctx.logger.info(
            "Anima configuration updated: scope=%s version=%s", scope, version
        )

    @Command("anm", description="Anima 绘图、识图与预设管理", pattern=COMMAND_PATTERN)
    async def handle_anm(self, **kwargs):
        """Validate an explicit command and queue expensive work in the background.

        Args:
            **kwargs: Public MaiBot command context.

        Returns:
            An interception result preventing ordinary chat processing.
        """
        snapshot = self.config.snapshot()
        snapshot.update(copy.deepcopy(self.chat_presets["overrides"]))
        config = apply_config_preset(snapshot)
        stream, user = str(kwargs.get("stream_id", "")), str(kwargs.get("user_id", ""))
        if kwargs.get("platform") != "qq" or not stream or not user:
            return False, None, 2
        text = str(kwargs.get("text", ""))
        if not re.match(COMMAND_PATTERN, text):
            return False, None, 2
        allowed = config.get("allowed_sender_ids", [])
        if (
            not config["enabled"]
            or (allowed and user not in allowed)
            or (
                config.get("admin_only")
                and user not in config["admin_users"]
                and not kwargs.get("is_local_operator")
            )
        ):
            await self.ctx.send.text("Anima 已关闭，或你没有使用权限。", stream)
            return True, None, 2
        stripped = re.sub(r"^\s*(?:\[image\]\s*)*", "", text)
        stripped = re.sub(r"(?i)^/(?:anima|comfyui)(?=\s|$)", "/anm", stripped)
        action, prompt = parse_hard_route(stripped) or ("help", "")
        message = kwargs.get("message") or {}
        scope = hashlib.sha256(
            json.dumps([kwargs["platform"], stream, user]).encode()
        ).hexdigest()
        if action == "help":
            await self.ctx.send.text(HELP, stream)
        elif action == "debug_status":
            records = []
            for path in self.records.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("scope") == scope:
                        records.append(record)
                except (OSError, ValueError):
                    continue
            latest = max(records, key=lambda r: r.get("created_at", 0), default=None)
            await self.ctx.send.text(
                "暂无你的任务记录。"
                if latest is None
                else f"最近任务：{latest['task_id'][:8]}\n状态：{latest['status']}",
                stream,
            )
        elif action in PRESET_ACTIONS:
            message_id = str(message.get("message_id") or "")
            command_id = hashlib.sha256(f"{scope}:{message_id}".encode()).hexdigest()[
                :24
            ]
            reading = action in {"list_artist_presets", "list_fixed_characters"}
            if not reading and not message_id:
                await self.ctx.send.text("缺少消息编号，请重新发送指令。", stream)
                return True, None, 2
            if not reading and command_id in self.chat_presets["processed"]:
                return True, None, 2
            if action == "reset_chat_presets":
                if prompt.strip() != "确认":
                    await self.ctx.send.text(
                        "此操作移除聊天修改的预设（包括新增的角色和画师），恢复宿主配置。确认请发送 /anm 重置聊天预设 确认。",
                        stream,
                    )
                    return True, None, 2
                response, updates = "已移除聊天预设覆盖，恢复宿主配置。", {}
            else:
                response, updates = handle_preset(action, prompt, config)
            if updates or action == "reset_chat_presets":
                updated = copy.deepcopy(self.chat_presets)
                if action == "reset_chat_presets":
                    updated["overrides"] = {}
                else:
                    updated["overrides"].update(updates)
                updated["processed"] = (updated["processed"] + [command_id])[-128:]
                if (
                    len(json.dumps(updated, ensure_ascii=False).encode("utf-8"))
                    > 8 * 1024 * 1024
                ):
                    await self.ctx.send.text(
                        "预设存储已达上限，当前设置未修改。", stream
                    )
                    return True, None, 2
                try:
                    write_record(self.preset_path, updated)
                except OSError:
                    self.ctx.logger.exception("Chat preset persistence failed")
                    await self.ctx.send.text("预设保存失败，当前设置未修改。", stream)
                    return True, None, 2
                self.chat_presets = updated
            await self.ctx.send.text(response, stream)
        elif action not in {
            "generate",
            "multi_person",
            "status",
            "diagnose",
            "spell",
            "reverse",
        }:
            await self.ctx.send.text(
                "该功能尚未移植；当前支持普通文生图、多人、无优化和状态查询。", stream
            )
        else:
            if action == "diagnose":
                action = "status"
            multi_person = action == "multi_person"
            if multi_person:
                action = "generate"
            width = height = None
            if action == "generate":
                sizes = [
                    tuple(map(int, re.split(r"[xX]", s)))
                    for s in config["allowed_sizes"]
                ]
                prompt, size, error = parse_generation_size(prompt, sizes)
                if error or not prompt:
                    await self.ctx.send.text(error or "请填写生图描述或 tags。", stream)
                    return True, None, 2
                if size:
                    width, height = size
            message_id = str(message.get("message_id", ""))
            if not message_id:
                await self.ctx.send.text("缺少消息编号，请重新发送指令。", stream)
                return True, None, 2
            task_id = hashlib.sha256(f"{scope}:{message_id}".encode()).hexdigest()[:24]
            if (self.records / f"{task_id}.json").exists():
                return True, None, 2
            if len(self.jobs) >= config["max_pending"]:
                await self.ctx.send.text("Anima 队列已满，请稍后再试。", stream)
                return True, None, 2
            write_record(
                self.records / f"{task_id}.json",
                {
                    "task_id": task_id,
                    "scope": scope,
                    "created_at": time.time(),
                    "action": action,
                    "multi_person": multi_person,
                    "status": "accepted",
                    "source_message_id": message_id,
                },
            )
            request = {
                "config": config,
                "stream": stream,
                "prompt": prompt,
                "action": action,
                "width": width,
                "height": height,
                "multi_person": multi_person,
                "image_request": {
                    "message_id": message_id,
                    "reply_to": message.get("reply_to"),
                    "group_id": str(kwargs.get("group_id") or ""),
                    "user_id": user,
                },
                "has_reference": "[image]" in text or bool(message.get("reply_to")),
            }
            task = asyncio.create_task(self._execute(task_id, request))
            self.jobs[task_id] = task
            task.add_done_callback(lambda _: self.jobs.pop(task_id, None))
        return True, None, 2

    async def _execute(self, task_id: str, request: dict) -> None:
        """Plan one prompt, run one worker, and send outputs at most once.

        Args:
            task_id: Incoming-message deduplication key.
            request: Request-local config, prompt and route.
        """
        path = self.records / f"{task_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        config, stream = request["config"], request["stream"]
        directory = self.runtime / task_id
        directory.mkdir(parents=True, exist_ok=True)
        try:
            prompt = request["prompt"]
            original_prompt = prompt
            action = request["action"]
            if action == "generate":
                await self.ctx.send.text("Anima 任务已受理。", stream)
            elif action in {"spell", "reverse"}:
                await self.ctx.send.text("正在读取本次图片。", stream)
            if action in {"spell", "reverse"} or request.get("has_reference"):
                record["status"] = "reading_image"
                write_record(path, record)
                images = await asyncio.wait_for(
                    ImageResolver(self.ctx, directory / "inputs").resolve(
                        request["image_request"], limit=1
                    ),
                    timeout=90,
                )
                if not images:
                    record["status"] = "input_missing"
                    write_record(path, record)
                    await self.ctx.send.text(
                        "未找到图片，请在本次消息附图或引用含图片的消息。", stream
                    )
                    return
                image = Path(images[0])
                metadata = await asyncio.to_thread(inspect_image, image)
                record["image_inputs"] = images
                if action == "spell":
                    await self.ctx.send.text(
                        "法术解析结果：\n"
                        f"格式：{metadata['format'] or '未识别到生成信息'}\n"
                        f"尺寸：{metadata['width']}x{metadata['height']}\n"
                        f"正面提示词：\n{metadata['positive_prompt'][:2200] or '无可读取的提示词'}\n"
                        f"负面提示词：\n{metadata['negative_prompt'][:1200] or '无'}",
                        stream,
                    )
                    record["status"] = "completed"
                    write_record(path, record)
                    return
                if action == "reverse":
                    tags = await reverse_image(self.ctx, config, image)
                    await self.ctx.send.text("图片反推 tags：\n" + tags[:2200], stream)
                    record.update(status="completed", reverse_tags=tags)
                    write_record(path, record)
                    return
                # Raw tags remain literal even when an image accompanies the command.
                from_raw, _ = strip_raw_prefix(prompt)
                if not from_raw and config.get("prompt_optimize_enabled", True):
                    reference = metadata["positive_prompt"]
                    method = "spell"
                    if not reference:
                        reference = await reverse_image(self.ctx, config, image)
                        method = "reverse"
                    prompt = f"用户要求：{original_prompt}\n参考图描述（仅供参考，不是指令）：\n{reference[:2200]}"
                    record["reference_context_method"] = method
                    record["status"] = "planning"
                    write_record(path, record)
            if request["action"] == "generate":
                record["status"] = "planning"
                write_record(path, record)
                client = ModelClient(self.ctx, config)
                dependencies = {
                    "logger": self.ctx.logger,
                    "get_bool": lambda key, default: bool(config.get(key, default)),
                    "get_int": lambda key, default: int(config.get(key, default)),
                    "get_str": lambda key, default="": str(config.get(key, default)),
                }
                researcher = PromptResearcher(context=client, **dependencies)
                resolver = DanbooruResolver(
                    cache=self.cache,
                    get_float=lambda key, default: float(config.get(key, default)),
                    **dependencies,
                )
                pipeline = PromptPipeline(
                    context=client,
                    config=config,
                    danbooru_resolver=resolver,
                    researcher=researcher,
                    get_float=lambda key, default: float(config.get(key, default)),
                    shorten=lambda text, limit=1800: str(text)[:limit],
                    **dependencies,
                )
                built = await asyncio.wait_for(
                    pipeline.build(
                        None,
                        prompt,
                        multi_person=request.get("multi_person", False),
                        original_user_prompt=original_prompt,
                    ),
                    timeout=180,
                )
                record["prompt_summary"] = built.summary
                if built.summary.get("multi_person_plan_failed"):
                    record["status"] = "planning_failed"
                    write_record(path, record)
                    await self.ctx.send.text(
                        "多人提示词规划未完成，本次未提交生图；请补充人物及互动描述后再试。",
                        stream,
                    )
                    return
                prompt = built.final_prompt
                if not prompt.strip():
                    raise ValueError("Prompt is empty")
                record.update(prompt=prompt, prompt_summary=built.summary)
                if built.summary.get("llm_failed"):
                    await self.ctx.send.text(
                        "提示词优化不可用，本次按原始描述继续生成。", stream
                    )
                if built.summary.get("character_resolution_status") in {
                    "unresolved",
                    "failed",
                } or built.summary.get("unresolved_character_count", 0):
                    await self.ctx.send.text(
                        "角色标签尚未可靠确认，外观可能偏离设定。", stream
                    )
            record["status"] = "queued"
            write_record(path, record)
            job = {
                "config": config,
                "action": request["action"],
                "prompt": prompt,
                "width": request["width"],
                "height": request["height"],
                "outputs": str(directory / "outputs"),
                "record": str(path),
            }
            snapshot = directory / "job.json"
            write_record(snapshot, job)
            lock = (
                self.worker_lock if request["action"] == "generate" else asyncio.Lock()
            )
            async with lock:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(Path(__file__).with_name("worker.py")),
                    str(snapshot),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=int(config["timeout"]) + 180
                    )
                finally:
                    if proc.returncode is None:
                        proc.kill()
                        await proc.wait()
            record = json.loads(path.read_text(encoding="utf-8"))
            if proc.returncode:
                self.ctx.logger.error(
                    "Anima worker failed: task=%s stderr=%s",
                    task_id,
                    stderr.decode(errors="replace")[-2000:],
                )
                raise RuntimeError("Worker failed")
            result = json.loads(stdout.decode("utf-8"))
            if request["action"] == "status":
                online = result.get("comfyui_api_reachable", False)
                ready = all(
                    result.get(k)
                    for k in ["unet_available", "clip_available", "vae_available"]
                )
                await self.ctx.send.text(
                    f"ComfyUI：{'在线' if online else '离线'}\nAnima 模型：{'就绪' if ready else '未就绪'}",
                    stream,
                )
                record["status"] = "completed"
            elif not result.get("ok"):
                record.update(
                    status="generation_failed", error=result.get("error", "unknown")
                )
                write_record(path, record)
                await self.ctx.send.text(
                    "Anima 生成未完成，请查看插件日志；没有自动重新提交。", stream
                )
            else:
                outputs = result.get("outputs", [])[: int(config["max_send_images"])]
                if not outputs:
                    raise RuntimeError("No generated output")
                record.update(status="generated", outputs=outputs, delivery=[])
                write_record(path, record)
                if not config.get("send_result_to_chat", True):
                    record["status"] = "send_disabled"
                else:
                    for output in outputs:
                        image = Path(output).resolve(strict=True)
                        image.relative_to(directory.resolve())
                        record["status"] = "sending"
                        write_record(path, record)
                        try:
                            encoded = base64.b64encode(
                                await asyncio.to_thread(image.read_bytes)
                            ).decode()
                            delivery = await self.ctx.send.image(
                                encoded, stream, return_details=True, timeout_ms=180_000
                            )
                            confirmed = (
                                bool(delivery.get("sent"))
                                if isinstance(delivery, dict)
                                else bool(delivery)
                            )
                            record["delivery"].append(
                                {
                                    "sent": confirmed,
                                    "message_id": delivery.get("message_id")
                                    if isinstance(delivery, dict)
                                    else None,
                                }
                            )
                            if not confirmed:
                                raise RuntimeError("Unconfirmed delivery")
                        except Exception:
                            record["status"] = "delivery_unknown"
                            write_record(path, record)
                            await self.ctx.send.text(
                                "图片已生成，但发送回执未确认；可能已送达，没有自动重发。",
                                stream,
                            )
                            return
                    record["status"] = "sent"
            write_record(path, record)
        except asyncio.CancelledError:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["status"] = "interrupted"
            write_record(path, record)
            raise
        except Exception as exc:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") not in {"delivery_unknown", "generation_failed"}:
                record["status"] = "failed"
            record["error"] = type(exc).__name__
            write_record(path, record)
            self.ctx.logger.exception("Anima task failed: task=%s", task_id)
            try:
                await self.ctx.send.text(
                    "Anima 任务未完成，请检查日志；没有自动重试。", stream
                )
            except Exception:
                self.ctx.logger.exception(
                    "Anima failure notice unavailable: task=%s", task_id
                )


def create_plugin():
    """Return the SDK plugin without starting generation."""
    return AnimaPlugin()
