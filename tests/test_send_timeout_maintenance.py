import json

import pytest

from tools.configure_send_timeouts import configure


@pytest.fixture
def shared_files(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "config.toml").write_text(
        '[client]\naction_timeout_sec = 15.0\ntoken = "private-test-token"\n',
        encoding="utf-8",
    )
    (adapter / "plugin.py").write_text(
        '    @MessageGateway(\n        name=SNOWLUMA_GATEWAY_NAME,\n        platform="qq",\n    )\n',
        encoding="utf-8",
    )
    napcat = tmp_path / "napcat" / "onebot.json"
    napcat.parent.mkdir()
    napcat.write_text(
        json.dumps(
            {
                "timeout": {"baseTimeout": 30000, "maxTimeout": 1800000},
                "token": "secret",
            }
        ),
        encoding="utf-8",
    )
    return adapter, napcat


def test_inspect_is_read_only(shared_files):
    adapter, napcat = shared_files
    before = (adapter / "config.toml").read_bytes()
    result = configure(adapter, napcat)
    assert not result["matches"]
    assert result["changed"] == []
    assert (adapter / "config.toml").read_bytes() == before
    assert "private-test-token" not in json.dumps(result)


def test_apply_preserves_other_fields_and_is_idempotent(shared_files, tmp_path):
    adapter, napcat = shared_files
    backups = tmp_path / "backups"
    result = configure(adapter, napcat, True, backups)
    assert result["matches"]
    assert len(result["changed"]) == 3
    assert 'token = "private-test-token"' in (adapter / "config.toml").read_text()
    assert json.loads(napcat.read_text())["timeout"]["maxTimeout"] == 1800000
    assert json.loads(napcat.read_text())["token"] == "secret"
    assert "timeout_ms=150_000," in (adapter / "plugin.py").read_text()
    assert len(list(backups.glob("*/*"))) == 3
    assert configure(adapter, napcat, True, backups)["changed"] == []


def test_rejects_unfamiliar_source_before_writing(shared_files, tmp_path):
    adapter, napcat = shared_files
    (adapter / "plugin.py").write_text("unfamiliar gateway", encoding="utf-8")
    with pytest.raises(ValueError, match="Unrecognized"):
        configure(adapter, napcat, True, tmp_path / "backups")
    assert "15.0" in (adapter / "config.toml").read_text()


def test_rejects_backup_inside_installed_directory(shared_files):
    adapter, napcat = shared_files
    with pytest.raises(ValueError, match="outside"):
        configure(adapter, napcat, True, adapter / "backups")
    assert "15.0" in (adapter / "config.toml").read_text()
