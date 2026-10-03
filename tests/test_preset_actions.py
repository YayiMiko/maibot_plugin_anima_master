"""Offline upstream-compatible preset parsing and mutation contracts."""

import copy

import pytest

from anima.commands.command_router import parse_hard_route
from preset_actions import handle_preset, parse_name_tags


@pytest.mark.parametrize("separator", ["=", "＝", ":", "："])
def test_named_tags_preserve_artist_colons(separator):
    result = parse_name_tags(f"测试{separator}artist:example, @another,")
    assert result[0] == "测试"
    assert "@example" in result[1]
    assert result[1].endswith(",")


def test_bare_artist_colon_is_not_a_preset_name():
    assert parse_name_tags("artist:example, @another") is None


@pytest.mark.parametrize("name", ["bad,name", "bad@name", "bad(name)", "a" * 81])
def test_invalid_names_rejected(name):
    assert parse_name_tags(f"{name}=1girl") is None


def test_artist_lifecycle_and_default_fallback():
    config = {"default_artist_tags": "@default,", "artist_presets": []}
    response, updates = handle_preset("create_artist_preset", "测试=@first,", config)
    assert "已保存并启用" in response
    config.update(updates)
    _, updates = handle_preset("append_artist_tags", "@second, @first,", config)
    config.update(updates)
    assert config["artist_presets"][0].count("@first") == 1
    assert "@second" in config["artist_presets"][0]
    _, updates = handle_preset("delete_artist_preset", "测试", config)
    assert updates["artist_presets"] == []
    assert updates["active_artist_preset"] == ""


def test_character_changes_do_not_select_an_implicit_character():
    config = {"fixed_characters": ["已有=1girl, black hair,"]}
    original = copy.deepcopy(config)
    _, updates = handle_preset("add_fixed_character", "测试=1girl, red hair", config)
    assert config == original
    assert set(updates) == {"fixed_characters"}
    config.update(updates)
    text, updates = handle_preset("list_fixed_characters", "测试", config)
    assert "red hair" in text
    assert not updates
    _, updates = handle_preset("delete_fixed_character", "测试", config)
    assert updates["fixed_characters"] == ["已有=1girl, black hair,"]


@pytest.mark.parametrize(
    "action,text",
    [
        ("create_artist_preset", "missing separator"),
        ("add_fixed_character", "invalid"),
        ("use_artist_preset", "not found"),
        ("delete_artist_preset", "not found"),
        ("delete_fixed_character", "not found"),
        ("set_artist_tags", ""),
        ("append_artist_tags", "a" * 8001),
    ],
)
def test_invalid_commands_never_mutate(action, text):
    assert handle_preset(action, text, {})[1] == {}


@pytest.mark.parametrize(
    "text,action",
    [
        ("/anm 查看角色", "list_fixed_characters"),
        ("/anm 删除角色 test", "delete_fixed_character"),
        ("/anm 重置聊天预设 确认", "reset_chat_presets"),
    ],
)
def test_new_commands_are_not_generation(text, action):
    assert parse_hard_route(text)[0] == action
