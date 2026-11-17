import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from production_rag_agent.app import create_app
from production_rag_agent.config import Settings
from production_rag_agent.runtime import BusyError, Runtime

from shared.errors import AgentError

KEY = "test-key-" + "x" * 32
AUTH = {"Authorization": "Bearer " + KEY}


class Engine:
    def __init__(self):
        self.calls = []
        self.error = None

    def start(self):
        pass

    def query(self, question, retrieve_only):
        self.calls.append((question, retrieve_only))
        if self.error:
            raise self.error
        return SimpleNamespace(
            hierarchy_sha256="a" * 64,
            answer={"status": "answered"},
            parents=[],
            logical_model_calls=1,
        )


@pytest.fixture
def settings():
    return Settings(
        _env_file=None,
        api_key=None,
        service_key=KEY,
        snapshot_name="fixture-v1",
        expected_hierarchy_sha256="a" * 64,
    )


@pytest.fixture
def engine():
    return Engine()


@pytest.fixture
def client(settings, engine):
    with TestClient(create_app(settings, engine)) as client:
        yield client


def test_health_auth_and_success(client, engine):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 401
    assert client.get("/health/ready", headers=AUTH).status_code == 200
    result = client.post("/v1/query", headers=AUTH, json={"question": "timeouts"})
    assert result.status_code == 200
    assert result.headers["cache-control"] == "no-store"
    assert len(result.headers["x-request-id"]) == 32
    assert engine.calls == [("timeouts", False)]
    assert client.get("/metrics", headers=AUTH).json()["active_workers"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {"question": "x", "model": "other"},
        {"question": "x", "store_path": "/etc"},
        {"question": "x", "retrieve_only": "yes"},
        {"question": " "},
        {"question": "x" * 1001},
    ],
)
def test_invalid_input_never_reaches_engine(client, engine, body):
    assert client.post("/v1/query", headers=AUTH, json=body).status_code == 422
    assert engine.calls == []


def test_body_limit_media_type_and_auth(client, engine):
    assert client.post("/v1/query", headers=AUTH, content="hello").status_code == 415
    assert client.post("/v1/query", json={"question": "x"}).status_code == 401
    response = client.post(
        "/v1/query", headers={**AUTH, "Content-Type": "application/json"}, content=b"x" * 9000
    )
    assert response.status_code == 413
    assert engine.calls == []


def test_errors_and_logs_do_not_disclose_input(client, engine, caplog):
    engine.error = AgentError("secret provider detail")
    with caplog.at_level(logging.INFO, logger="production_rag"):
        response = client.post("/v1/query", headers=AUTH, json={"question": "private question"})
    assert response.status_code == 502
    assert "secret provider detail" not in response.text + caplog.text
    assert "private question" not in caplog.text
    assert KEY not in caplog.text


def test_global_rate_limit(settings, engine):
    settings.requests_per_minute = 1
    with TestClient(create_app(settings, engine)) as client:
        assert client.post("/v1/query", headers=AUTH, json={"question": "x"}).status_code == 200
        assert client.post("/v1/query", headers=AUTH, json={"question": "x"}).status_code == 429


def test_timeout_keeps_worker_slot():
    async def scenario():
        runtime = Runtime(1, 10)
        release = threading.Event()
        started = threading.Event()

        def slow():
            started.set()
            release.wait(5)
            return 1

        try:
            with pytest.raises(TimeoutError):
                await runtime.execute(slow, 0.05)
            assert started.is_set()
            assert runtime.metrics()["active_workers"] == 1
            with pytest.raises(BusyError):
                await runtime.execute(lambda: 2, 1)
        finally:
            release.set()
            await runtime.close()
        assert runtime.metrics()["active_workers"] == 0

    asyncio.run(scenario())


def test_cancelled_request_keeps_worker_slot():
    async def scenario():
        runtime = Runtime(1, 10)
        release = threading.Event()
        task = asyncio.create_task(runtime.execute(lambda: release.wait(5), 10))
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(BusyError):
                await runtime.execute(lambda: None, 1)
        finally:
            release.set()
            await runtime.close()

    asyncio.run(scenario())


def test_failed_startup_does_not_serve(settings):
    class Broken(Engine):
        def start(self):
            raise AgentError("bad snapshot")

    with pytest.raises(AgentError), TestClient(create_app(settings, Broken())):
        pass


def test_http_timeout_and_overload(settings):
    release = threading.Event()

    class Slow(Engine):
        def query(self, question, retrieve_only):
            release.wait(5)
            return super().query(question, retrieve_only)

    settings.max_concurrent = 1
    settings.response_deadline_seconds = 0.03
    with TestClient(create_app(settings, Slow())) as client:
        try:
            assert client.post("/v1/query", headers=AUTH, json={"question": "x"}).status_code == 504
            assert client.post("/v1/query", headers=AUTH, json={"question": "x"}).status_code == 503
            assert client.get("/metrics", headers=AUTH).json()["active_workers"] == 1
        finally:
            release.set()


def test_slow_stream_is_rejected_before_engine(settings, engine):
    import httpx

    settings.body_timeout_seconds = 0.02
    app = create_app(settings, engine)

    async def scenario():
        async def body():
            yield b"{"
            await asyncio.sleep(0.1)
            yield b"}"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/query", headers={**AUTH, "Content-Type": "application/json"}, content=body()
            )
            assert response.status_code == 408

    with TestClient(app):
        asyncio.run(scenario())
    assert engine.calls == []


def test_unexpected_error_is_sanitized(client, engine):
    engine.error = RuntimeError("private internal state")
    response = client.post("/v1/query", headers=AUTH, json={"question": "x"})
    assert response.status_code == 500
    assert "private internal state" not in response.text


def test_client_cannot_override_request_id_or_policy(client):
    response = client.post(
        "/v1/query", headers={**AUTH, "X-Request-ID": "untrusted"}, json={"question": "x"}
    )
    assert response.headers["x-request-id"] != "untrusted"
    assert (
        client.post("/v1/query?model=other", headers=AUTH, json={"question": "x"}).status_code
        == 422
    )
