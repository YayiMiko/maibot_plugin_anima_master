"""Offline original-byte preservation and conversation boundary checks."""

import asyncio
import base64
import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image_reference import inspect_image
from media import ImageResolver


def test_comfyui_metadata_uses_sampler_links(tmp_path):
    path = tmp_path / "original.png"
    info = PngInfo()
    info.add_text(
        "prompt",
        json.dumps(
            {
                "negative": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "low quality"},
                },
                "positive": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "1girl"},
                },
                "sampler": {
                    "class_type": "KSampler",
                    "inputs": {
                        "positive": ["positive", 0],
                        "negative": ["negative", 0],
                    },
                },
            }
        ),
    )
    Image.new("RGB", (16, 24)).save(path, pnginfo=info)
    payload = inspect_image(path)
    assert payload["positive_prompt"] == "1girl"
    assert payload["negative_prompt"] == "low quality"
    assert "input" not in payload


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP"])
def test_original_bytes_are_never_reencoded(tmp_path, format):
    output = io.BytesIO()
    Image.new("RGB", (16, 24)).save(output, format=format)
    encoded = "base64://" + base64.b64encode(output.getvalue()).decode()
    resolver = ImageResolver(SimpleNamespace(), tmp_path)
    assert asyncio.run(resolver._read(encoded)) == output.getvalue()


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.com/a.png",
        "https://qpic.cn.evil.com/a.png",
        "http://127.0.0.1/a.png",
    ],
)
def test_untrusted_resources_are_rejected(tmp_path, reference):
    with pytest.raises((ValueError, OSError)):
        asyncio.run(ImageResolver(SimpleNamespace(), tmp_path)._read(reference))


@pytest.mark.parametrize("private", [False, True])
def test_quote_cannot_cross_conversations(tmp_path, private):
    ctx = SimpleNamespace(
        logger=logging.getLogger("test-media"),
        api=SimpleNamespace(
            call=AsyncMock(
                side_effect=[
                    {
                        "group_id": "" if private else "group",
                        "message_type": "private" if private else "group",
                        "sender": {"user_id": "user"},
                        "message": [{"type": "reply", "data": {"id": "other"}}],
                    },
                    {
                        "group_id": "other-group",
                        "message_type": "private",
                        "sender": {"user_id": "other-user"},
                        "message": [],
                    },
                    {"user_id": "bot"},
                ]
            )
        ),
    )
    with pytest.raises(ValueError):
        asyncio.run(
            ImageResolver(ctx, tmp_path).resolve(
                {
                    "message_id": "current",
                    "group_id": "" if private else "group",
                    "user_id": "user",
                }
            )
        )
    assert not list(tmp_path.iterdir())


def test_quoted_original_is_preferred_over_current_image(tmp_path):
    originals = []
    for color in ["red", "blue"]:
        output = io.BytesIO()
        Image.new("RGB", (16, 24), color).save(output, format="PNG")
        originals.append("base64://" + base64.b64encode(output.getvalue()).decode())
    ctx = SimpleNamespace(
        api=SimpleNamespace(
            call=AsyncMock(
                side_effect=[
                    {
                        "group_id": "group",
                        "message": [
                            {"type": "reply", "data": {"id": "quoted"}},
                            {"type": "image", "data": {"url": originals[0]}},
                        ],
                    },
                    {
                        "group_id": "group",
                        "message": [{"type": "image", "data": {"url": originals[1]}}],
                    },
                ]
            )
        )
    )
    paths = asyncio.run(
        ImageResolver(ctx, tmp_path).resolve(
            {
                "message_id": "current",
                "group_id": "group",
                "user_id": "user",
            },
            limit=1,
        )
    )
    assert len(paths) == 1
    with Image.open(paths[0]) as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)


def test_private_bot_quote_for_other_recipient_is_rejected(tmp_path):
    ctx = SimpleNamespace(
        api=SimpleNamespace(
            call=AsyncMock(
                side_effect=[
                    {
                        "message_type": "private",
                        "sender": {"user_id": "user"},
                        "message": [],
                    },
                    {
                        "message_type": "private",
                        "sender": {"user_id": "bot"},
                        "user_id": "other",
                        "message": [],
                    },
                    {"user_id": "bot"},
                ]
            )
        )
    )
    with pytest.raises(ValueError):
        asyncio.run(
            ImageResolver(ctx, tmp_path).resolve(
                {
                    "message_id": "current",
                    "group_id": "",
                    "user_id": "user",
                    "reply_to": "quote",
                }
            )
        )


def test_novelai_comment_metadata(tmp_path):
    path = tmp_path / "original.png"
    info = PngInfo()
    info.add_text(
        "Comment", json.dumps({"prompt": "1girl, blue dress", "uc": "low quality"})
    )
    Image.new("RGB", (16, 24)).save(path, pnginfo=info)
    assert inspect_image(path)["positive_prompt"] == "1girl, blue dress"
