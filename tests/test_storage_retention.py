"""Only temporary plugin-owned fixtures are deleted by these offline tests."""

import json
import logging
import os
import time

import pytest

from storage_retention import clean_history

LOGGER = logging.getLogger("retention-test")
OLD = time.time() - 60 * 86400
TASK = "a" * 24


@pytest.fixture
def storage(tmp_path):
    records, runtime = tmp_path / "data/tasks", tmp_path / "runtime"
    records.mkdir(parents=True)
    directory = runtime / TASK
    (directory / "outputs").mkdir(parents=True)
    image = directory / "outputs/image.png"
    image.write_bytes(b"image")
    record = records / f"{TASK}.json"
    record.write_text(
        json.dumps({"task_id": TASK, "status": "sent", "created_at": OLD}),
        encoding="utf-8",
    )
    for path in [image, directory / "outputs", directory, record]:
        os.utime(path, (OLD, OLD))
    return records, runtime, record, directory, image


def test_expired_task_cleanup_leaves_other_projects_and_presets(storage):
    records, runtime, record, directory, image = storage
    unrelated = runtime / "unrelated.txt"
    unrelated.write_text("keep")
    presets = records.parent / "chat_presets.json"
    presets.write_text("keep")
    assert clean_history(records, runtime, 30, set(), LOGGER) == 1
    assert not record.exists() and not directory.exists()
    assert unrelated.read_text() == presets.read_text() == "keep"


@pytest.mark.parametrize(
    "mode",
    [
        "disabled",
        "active",
        "recent_file",
        "recent_record",
        "nonterminal",
        "remote_unknown",
        "corrupt",
        "wrong_id",
        "nan_time",
    ],
)
def test_ineligible_tasks_are_preserved(storage, mode):
    records, runtime, record, directory, image = storage
    days, active = 30, set()
    if mode == "disabled":
        days = 0
    elif mode == "active":
        active.add(TASK)
    elif mode == "recent_file":
        os.utime(image, None)
    elif mode == "recent_record":
        os.utime(record, None)
    else:
        data = json.loads(record.read_text())
        if mode == "nonterminal":
            data["status"] = "generating"
        elif mode == "remote_unknown":
            data["status"] = "remote_unknown"
        elif mode == "wrong_id":
            data["task_id"] = "b" * 24
        elif mode == "nan_time":
            data["created_at"] = float("nan")
        record.write_text("invalid" if mode == "corrupt" else json.dumps(data))
        os.utime(record, (OLD, OLD))
    assert clean_history(records, runtime, days, active, LOGGER) == 0
    assert record.exists() and image.exists()


def test_record_output_paths_never_control_deletion(storage, tmp_path):
    records, runtime, record, directory, image = storage
    outside = tmp_path / "other-plugin-image.png"
    outside.write_bytes(b"keep")
    data = json.loads(record.read_text())
    data["outputs"] = [str(outside)]
    record.write_text(json.dumps(data))
    os.utime(record, (OLD, OLD))
    assert clean_history(records, runtime, 30, set(), LOGGER) == 1
    assert outside.read_bytes() == b"keep"


def test_linked_task_directory_is_never_followed(storage, tmp_path):
    records, runtime, record, directory, image = storage
    external = tmp_path / "outside"
    external.mkdir()
    keep = external / "keep.txt"
    keep.write_text("keep")
    linked = directory / "external"
    if os.name == "nt":
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(external)],
            check=True,
            capture_output=True,
        )
    else:
        linked.symlink_to(external, target_is_directory=True)
    os.utime(directory, (OLD, OLD))
    assert clean_history(records, runtime, 30, set(), LOGGER) == 0
    assert keep.read_text() == "keep"
    assert record.exists()
    # Remove only the link, never its target, before fixture teardown.
    if linked.is_junction():
        linked.rmdir()
    else:
        linked.unlink()


def test_rotating_scan_does_not_starve_later_tasks(storage):
    records, runtime, record, directory, image = storage
    for index in range(1000):
        (records / f"{index:024x}.json").write_text("invalid")
    cursor = {}
    # Filesystem enumeration order differs between Windows and Linux.
    first = clean_history(records, runtime, 30, set(), LOGGER, cursor)
    second = clean_history(records, runtime, 30, set(), LOGGER, cursor)
    assert first + second == 1
    assert not record.exists()
