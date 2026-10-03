"""MaiBot configuration models retaining upstream Anima section defaults."""

import json
from pathlib import Path

from maibot_sdk import Field, PluginConfigBase
from pydantic import create_model

SCHEMA = json.loads(
    Path(__file__).with_name("config_schema.json").read_text(encoding="utf-8")
)
TYPES = {
    "string": str,
    "text": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "object": dict,
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
            TYPES.get(spec.get("type"), str),
            Field(
                default=spec.get("default"),
                description=spec.get("description", key).replace("AstrBot", "MaiBot"),
                json_schema_extra={
                    "hidden": key
                    in {
                        "show_advanced_settings",
                        "manifest_max_records",
                        "reset_to_defaults",
                        "prompt_builder_provider_id",
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
