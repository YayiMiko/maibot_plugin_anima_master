"""Resolve only the current message, its explicit quote, and saved QQ slots."""

import asyncio
import base64
import hashlib
import io
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image

MAX_BYTES = 32 * 1024 * 1024
QQ_DOMAINS = ("qpic.cn", "qq.com", "qq.com.cn")


class ImageResolver:
    """Resolve original QQ image bytes through the public adapter APIs."""

    def __init__(self, ctx, inputs: Path):
        self.ctx = ctx
        self.inputs = inputs

    async def resolve(self, request: dict, limit: int = 3) -> list[str]:
        """Save quoted images first, followed by attachments in message order.

        Args:
            request: Authorized command context with message and group IDs.
            limit: Maximum number of retained unique images.

        Returns:
            Saved original bytes, never a global latest-image fallback.

        Raises:
            ValueError: A quote belongs to a different conversation.
        """
        current = await self.ctx.api.call(
            "adapter.napcat.message.get_msg",
            version="1",
            message_id=request["message_id"],
        )
        if not isinstance(current, dict):
            return []
        if not request["group_id"] and (
            current.get("message_type") != "private"
            or str(current.get("sender", {}).get("user_id", "")) != request["user_id"]
        ):
            raise ValueError("Message does not belong to this private conversation")
        if (
            request["group_id"]
            and str(current.get("group_id", "")) != request["group_id"]
        ):
            raise ValueError("Message does not belong to this group")
        segments = current.get("message", [])
        if not isinstance(segments, list):
            return []
        quoted = []
        reply_ids = [
            str(x.get("data", {}).get("id", ""))
            for x in segments
            if x.get("type") == "reply"
        ]
        reply_to = request.get("reply_to")
        if reply_to and not reply_ids:
            reply_ids = [str(reply_to)]
        for message_id in reply_ids[:1]:
            reply = await self.ctx.api.call(
                "adapter.napcat.message.get_msg", version="1", message_id=message_id
            )
            if not isinstance(reply, dict):
                continue
            if request["group_id"]:
                if str(reply.get("group_id", "")) != request["group_id"]:
                    raise ValueError("Quoted image is outside the current group")
            else:
                login = await self.ctx.api.call(
                    "adapter.napcat.action.call_data",
                    version="1",
                    action_name="get_login_info",
                    params={},
                )
                bot_id = str((login or {}).get("user_id", ""))
                sender = str(reply.get("sender", {}).get("user_id"))
                if reply.get("message_type") != "private" or sender not in {
                    request["user_id"],
                    bot_id,
                }:
                    raise ValueError(
                        "Quoted image is outside the current private conversation"
                    )
                if (
                    sender == bot_id
                    and str(reply.get("user_id", "")) != request["user_id"]
                ):
                    raise ValueError(
                        "Quoted outgoing image belongs to another private conversation"
                    )
            quoted.extend(reply.get("message", []))
        saved = []
        seen = set()
        for segment in quoted + segments:
            if segment.get("type") != "image":
                continue
            data = segment.get("data", {})
            detail = {}
            if data.get("file"):
                try:
                    response = await self.ctx.api.call(
                        "adapter.napcat.file.get_image",
                        version="1",
                        params={"file": data["file"]},
                    )
                    if isinstance(response, dict):
                        detail = response.get("data", response) or {}
                except Exception:
                    self.ctx.logger.warning(
                        "Original image lookup failed; trying message image URL"
                    )
            candidates = [
                detail.get("file"),
                detail.get("url"),
                data.get("url"),
                data.get("file"),
            ]
            readable = False
            for candidate in candidates:
                if not isinstance(candidate, str) or not candidate:
                    continue
                try:
                    content = await self._read(candidate)
                    digest = hashlib.sha256(content).hexdigest()
                    if digest not in seen:
                        self.inputs.mkdir(parents=True, exist_ok=True)
                        with Image.open(io.BytesIO(content)) as image:
                            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[
                                image.format
                            ]
                        path = self.inputs / f"{uuid.uuid4().hex}{suffix}"
                        path.write_bytes(content)
                        path.chmod(0o600)
                        saved.append(str(path))
                        seen.add(digest)
                    readable = True
                    break
                except Exception:
                    self.ctx.logger.debug("Image candidate unavailable")
            if not readable:
                raise ValueError(
                    "An explicit image could not be downloaded; refusing to substitute another image"
                )
            if len(saved) >= limit:
                break
        return saved

    async def _read(self, reference: str) -> bytes:
        """Validate a bounded QQ resource without discarding generation metadata.

        Args:
            reference: A QQ HTTPS URL, base64 image, or shared QQ-cache path.

        Returns:
            Original validated image bytes, including embedded metadata.

        Raises:
            ValueError: The reference is outside allowed roots or too large.
        """
        if reference.startswith("base64://"):
            if len(reference) > MAX_BYTES * 4 // 3 + 32:
                raise ValueError("Image exceeds byte limit")
            content = base64.b64decode(reference[9:], validate=True)
        elif reference.startswith("https://"):
            host = urlparse(reference).hostname or ""
            if not any(
                host == suffix or host.endswith("." + suffix) for suffix in QQ_DOMAINS
            ):
                raise ValueError("Image URL is not a QQ resource")
            chunks = []
            size = 0
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                async with client.stream("GET", reference) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError("Image exceeds byte limit")
                        chunks.append(chunk)
            content = b"".join(chunks)
        else:
            path = Path(reference.removeprefix("file://")).resolve(strict=True)
            path.relative_to(Path("/app/.config/QQ").resolve())
            if path.stat().st_size > MAX_BYTES:
                raise ValueError("Image exceeds byte limit")
            content = await asyncio.to_thread(path.read_bytes)
        if len(content) > MAX_BYTES:
            raise ValueError("Image exceeds byte limit")
        with Image.open(io.BytesIO(content)) as image:
            if image.width * image.height > 64 * 1024 * 1024:
                raise ValueError("Image exceeds pixel limit")
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Unsupported image format")
            image.verify()
        return content
