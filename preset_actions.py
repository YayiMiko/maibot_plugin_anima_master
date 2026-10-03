"""Pure chat preset operations, independent of Host configuration storage."""

import re

if __package__:
    from .anima.prompts.prompt_presets import (
        active_artist_preset_name,
        artist_presets,
        fixed_character_tags,
        merge_tag_text,
    )
    from .anima.prompts.tag_cleaner import (
        canonical_tag_text,
        join_prompt_parts,
        split_tags,
    )
else:
    from anima.prompts.prompt_presets import (
        active_artist_preset_name,
        artist_presets,
        fixed_character_tags,
        merge_tag_text,
    )
    from anima.prompts.tag_cleaner import (
        canonical_tag_text,
        join_prompt_parts,
        split_tags,
    )

PRESET_ACTIONS = {
    "create_artist_preset",
    "set_artist_tags",
    "append_artist_tags",
    "use_artist_preset",
    "list_artist_presets",
    "delete_artist_preset",
    "add_fixed_character",
    "list_fixed_characters",
    "delete_fixed_character",
    "reset_chat_presets",
}


def parse_name_tags(text: str) -> tuple[str, str] | None:
    """Parse named artist or character tags without splitting artist: syntax.

    Args:
        text: User-supplied name and tags separated by an equals sign or colon.

    Returns:
        A normalized name/tag pair, or None for invalid named input.
    """
    for match in re.finditer(r"[=＝：:]", text):
        name, tags = text[: match.start()].strip(), text[match.end() :].strip()
        if (
            not name
            or not tags
            or len(name) > 80
            or re.search(r"[,，\n\r@()[\]{}]", name)
            or name.lower()
            in {"artist", "tag", "tags", "prompt", "positive", "negative"}
        ):
            continue
        normalized = join_prompt_parts(
            [", ".join(canonical_tag_text(t) for t in split_tags(tags))]
        )
        if normalized:
            return name, normalized + ","
    return None


def handle_preset(action: str, prompt: str, config: dict) -> tuple[str, dict]:
    """Apply one upstream-compatible preset operation to a detached snapshot.

    Args:
        action: A preset command route.
        prompt: User arguments, never passed to a language model.
        config: Current flat config, including chat overrides.

    Returns:
        Chat response and changed keys only; no changes means no write.
    """
    text = str(prompt or "").strip()
    if len(text) > 8000:
        return "预设内容过长，请控制在 8000 字符以内。", {}
    artists = artist_presets(config)
    characters = fixed_character_tags(config)
    active = active_artist_preset_name(config)
    if action == "list_artist_presets":
        lines = [
            "画师预设：",
            f"默认画师 tags{'（当前）' if not active else ''}：{str(config.get('default_artist_tags') or '')[:160]}",
        ]
        lines.extend(
            f"{name}{'（当前）' if name == active else ''}：{tags[:120]}"
            for name, tags in sorted(artists.items())
        )
        return "\n".join(lines)[:2200], {}
    if action == "list_fixed_characters":
        if text:
            return (
                f"角色“{text}”：\n{characters[text][:1800]}"
                if text in characters
                else f"没有找到角色“{text}”。"
            ), {}
        return "固定角色（生图时点名使用）：\n" + (
            "、".join(sorted(characters))[:2000] or "无"
        ), {}
    if action == "delete_fixed_character":
        if text not in characters:
            return "没有找到该角色，请使用 /anm 查看角色。", {}
        del characters[text]
        return f"已删除角色“{text}”。", {
            "fixed_characters": [f"{n}={t}" for n, t in characters.items()]
        }
    if action == "use_artist_preset":
        if text in {"默认", "默认画师", "默认画师预设", "默认画师组", "default"}:
            return "已切回默认画师 tags。", {"active_artist_preset": ""}
        if text not in artists:
            return "没有找到该画师预设，请使用 /anm 查看画师预设。", {}
        return f"已启用画师预设“{text}”：\n{artists[text][:800]}", {
            "active_artist_preset": text
        }
    if action == "delete_artist_preset":
        if text not in artists:
            return "没有找到该画师预设，请使用 /anm 查看画师预设。", {}
        del artists[text]
        return f"已删除画师预设“{text}”。" + (
            " 已切回默认画师 tags。" if active == text else ""
        ), {
            "artist_presets": [f"{n}={t}" for n, t in artists.items()],
            "active_artist_preset": "" if active == text else active,
        }
    parsed = parse_name_tags(text)
    if action in {"create_artist_preset", "add_fixed_character"} and not parsed:
        return "请使用“名称=tags”的格式。", {}
    if action == "add_fixed_character":
        name, tags = parsed
        if name not in characters and len(characters) >= 256:
            return "角色数量已达上限，请先删除不用的角色。", {}
        characters[name] = tags
        return f"已保存角色“{name}”：\n{tags[:1000]}", {
            "fixed_characters": [f"{n}={t}" for n, t in characters.items()]
        }
    if action in {"create_artist_preset", "set_artist_tags", "append_artist_tags"}:
        if parsed:
            name, tags = parsed
        else:
            name = active if action == "append_artist_tags" else ""
            tags = join_prompt_parts(
                [", ".join(canonical_tag_text(t) for t in split_tags(text))]
            )
            if tags:
                tags += ","
        if not tags:
            return "请填写画师 tags，或使用“名称=tags”。", {}
        if action == "append_artist_tags":
            tags = merge_tag_text(
                artists.get(name) if name else config.get("default_artist_tags"), tags
            )
        if name:
            if name not in artists and len(artists) >= 256:
                return "画师预设数量已达上限，请先删除不用的预设。", {}
            artists[name] = tags
            return f"已保存并启用画师预设“{name}”：\n{tags[:800]}", {
                "artist_presets": [f"{n}={t}" for n, t in artists.items()],
                "active_artist_preset": name,
            }
        updates = {"default_artist_tags": tags}
        if action == "set_artist_tags":
            updates["active_artist_preset"] = ""
        return "已更新默认画师 tags：\n" + tags[:800], updates
    return "未知预设操作。", {}
