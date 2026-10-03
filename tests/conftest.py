"""All tests are offline; accidental network access is a test failure."""

import httpx
import pytest
import requests


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    """Block live HTTP connections throughout the offline suite."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Live network access is forbidden in offline tests")

    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
