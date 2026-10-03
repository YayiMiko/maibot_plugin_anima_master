"""MaiBot configuration models retaining upstream Anima section defaults."""

import json
import re
from pathlib import Path
from typing import Annotated

from maibot_sdk import Field, PluginConfigBase
from pydantic import AfterValidator, create_model

SCHEMA = json.loads(
    Path(__file__).with_name("config_schema.json").read_text(encoding="utf-8")
)
TYPES = {
    "string": str,
    "text": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list[str],
    "object": dict,
}


def validate_sizes(values: list[str]) -> list[str]:
    """Validate the canvas list before any command is accepted.

    Args:
        values: Width-by-height strings from the configuration UI.

    Returns:
        Validated sizes, preserving order.

    Raises:
        ValueError: A size is malformed, empty, or outside useful bounds.
    """
    if not values or len(values) > 64:
        raise ValueError("Provide between 1 and 64 allowed sizes")
    for value in values:
        if not re.fullmatch(r"\d{2,5}[xX]\d{2,5}", value):
            raise ValueError("Allowed sizes must use widthxheight")
        width, height = map(int, re.split("[xX]", value))
        if not (64 <= width <= 8192 and 64 <= height <= 8192):
            raise ValueError("Canvas dimensions must be between 64 and 8192")
    return values


BOUNDS = {
    "width": {"ge": 64, "le": 8192},
    "height": {"ge": 64, "le": 8192},
    "steps": {"ge": 1, "le": 200},
    "cfg": {"ge": 0, "le": 100, "allow_inf_nan": False},
    "timeout": {"ge": 1, "le": 86400},
    "poll_interval": {"ge": 1, "le": 120},
    "storage_retention_days": {"ge": 0, "le": 36500},
    "max_send_images": {"ge": 1, "le": 32},
    "prompt_builder_max_tokens": {"ge": 128, "le": 32768},
    "prompt_builder_max_tags": {"ge": 1, "le": 1024},
    "danbooru_tag_lookup_timeout": {"ge": 1, "le": 20, "allow_inf_nan": False},
    "danbooru_tag_max_candidates": {"ge": 1, "le": 16},
}


class PluginMetadata(PluginConfigBase):
    """Required SDK configuration version."""

    config_version: str = Field(default="0.1.0", json_schema_extra={"hidden": True})


class MaiBotSettings(PluginConfigBase):
    """Host model routing and queue configuration."""

    enabled: bool = True
    prompt_model_task: str = "utils"
    prompt_model_name: str = ""
    vision_model_task: str = "vlm"
    vision_model_name: str = ""
    admin_users: list[str] = Field(default_factory=list)
    max_pending: int = Field(default=8, ge=1, le=32)
    research_api_key: str = Field(default="", json_schema_extra={"hidden": True})


class ConfigBase(PluginConfigBase):
    """Return detached flat snapshots for existing prompt/workflow builders."""

    def snapshot(self) -> dict:
        """Flatten known sections without retaining mutable model values."""
        return {
            key: value
            for section in self.model_dump().values()
            for key, value in section.items()
        }


sections = {
    "plugin": (PluginMetadata, Field(default_factory=PluginMetadata)),
    "maibot": (MaiBotSettings, Field(default_factory=MaiBotSettings)),
}
for name, section in SCHEMA.items():
    fields = {
        key: (
            Annotated[list[str], AfterValidator(validate_sizes)]
            if key == "allowed_sizes"
            else TYPES.get(spec.get("type"), str),
            Field(
                default=spec.get("default"),
                **BOUNDS.get(key, {}),
                description=spec.get("description", key).replace("AstrBot", "MaiBot"),
                json_schema_extra={
                    "hidden": key
                    in {
                        "show_advanced_settings",
                        "manifest_max_records",
                        "reset_to_defaults",
                        "prompt_builder_provider_id",
                        "prompt_builder_deep_thinking_enabled",
                        "prompt_builder_reasoning_effort",
                        "verify_provider_id",
                        "multi_verify_provider_id",
                        "image_caption_provider_id",
                        "auto_start",
                        "startup_command",
                        "startup_workdir",
                        "startup_visible_window",
                        "startup_wait_seconds",
                        "startup_poll_interval",
                        "auto_start_admin_only",
                        "auto_start_allowed_sender_ids",
                    }
                    or name
                    in {
                        "anima_master_verify",
                        "anima_master_multi_person",
                        "anima_master_experimental",
                    }
                },
            ),
        )
        for key, spec in section.get("items", {}).items()
    }
    model = create_model(name, __base__=PluginConfigBase, **fields)
    model.__ui_label__ = section.get("description", name)
    sections[name.removeprefix("anima_master_")] = (model, Field(default_factory=model))
AnimaConfig = create_model("AnimaConfig", __base__=ConfigBase, **sections)
