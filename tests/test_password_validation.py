import importlib
import pytest
from fastapi.testclient import TestClient


def load_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GIGASCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GIGASCRIBE_USERS_FILE", str(tmp_path / "data" / "users.json"))
    monkeypatch.setenv("GIGASCRIBE_ADMIN_PASSWORD", "adminpass")
    import server
    importlib.reload(server)
    return server


def test_admin_password_limits(tmp_path, monkeypatch):
    srv = load_server(tmp_path, monkeypatch)
    c = TestClient(srv.app)
    assert c.post("/admin/users", auth=("admin", "adminpass"), data={"username":"u", "password":"short"}).status_code == 400
    assert c.post("/admin/users", auth=("admin", "adminpass"), data={"username":"u", "password":"ю"*40}).status_code == 400
    assert c.post("/admin/users", auth=("admin", "adminpass"), data={"username":"u", "password":"valid-ю!"}).status_code == 200
