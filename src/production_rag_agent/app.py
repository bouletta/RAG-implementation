import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from anyio import to_thread
from basic_rag_agent.models import Query
from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, ValidationError
from starlette.responses import JSONResponse, Response

from shared.errors import AgentError

from .config import Settings
from .engine import Engine, RetrievalEngine
from .runtime import BusyError, Runtime

logger = logging.getLogger("production_rag")


class QueryInput(Query):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    retrieve_only: bool = False


def create_app(settings: Settings, engine: Engine | None = None) -> FastAPI:
    selected = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal selected
        selected = selected or RetrievalEngine(settings)
        await to_thread.run_sync(selected.start)
        runtime = Runtime(settings.max_concurrent, settings.requests_per_minute)
        app.state.runtime = runtime
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(
        title="Grounded RAG Service",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    def authenticate(request: Request) -> None:
        actual = request.headers.get("authorization", "").encode()
        expected = ("Bearer " + settings.service_key.get_secret_value()).encode()
        if not secrets.compare_digest(actual, expected):
            raise HTTPException(
                401, "Authentication required", headers={"WWW-Authenticate": "Bearer"}
            )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = uuid4().hex
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse({"detail": "Internal service error"}, status_code=500)
        runtime = request.app.state.runtime
        runtime.count(f"http_{response.status_code}")
        logger.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request.state.request_id,
                    "status": response.status_code,
                    "elapsed_ms": round((perf_counter() - started) * 1000, 2),
                }
            )
        )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(request: Request) -> dict[str, str]:
        authenticate(request)
        return {
            "status": "ready",
            "snapshot": settings.snapshot_name,
            "hierarchy_sha256": settings.expected_hierarchy_sha256,
            "scope": "startup validation; each query revalidates",
        }

    @app.get("/metrics")
    async def metrics(request: Request) -> dict[str, object]:
        authenticate(request)
        return request.app.state.runtime.metrics()

    @app.post("/v1/query")
    async def query(request: Request) -> dict[str, object]:
        authenticate(request)
        runtime = request.app.state.runtime
        if not runtime.allow():
            raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": "60"})
        if (
            request.headers.get("content-type", "").split(";")[0].strip().lower()
            != "application/json"
        ):
            raise HTTPException(415, "Expected application/json")
        if request.query_params:
            raise HTTPException(422, "Query parameters are not supported")
        data = bytearray()
        try:
            async with asyncio.timeout(settings.body_timeout_seconds):
                async for part in request.stream():
                    if len(data) + len(part) > settings.max_body_bytes:
                        raise HTTPException(413, "Request body too large")
                    data.extend(part)
        except TimeoutError:
            raise HTTPException(408, "Request body deadline exceeded") from None
        try:
            body = QueryInput.model_validate_json(data)
        except ValidationError:
            raise HTTPException(422, "Invalid query body") from None
        assert selected is not None
        try:
            result = await runtime.execute(
                lambda: selected.query(body.question, body.retrieve_only),
                settings.response_deadline_seconds,
            )
        except BusyError:
            raise HTTPException(503, "All workers are busy", headers={"Retry-After": "1"}) from None
        except TimeoutError:
            runtime.count("response_timeouts")
            raise HTTPException(
                504, "Response deadline exceeded; work may still be running"
            ) from None
        except AgentError:
            runtime.count("pipeline_errors")
            raise HTTPException(502, "Retrieval pipeline failed") from None
        return {
            "request_id": request.state.request_id,
            "hierarchy_sha256": result.hierarchy_sha256,
            "answer": result.answer,
            "parents": result.parents,
            "logical_model_calls": result.logical_model_calls,
        }

    return app


def factory() -> FastAPI:
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.propagate = False
    try:
        settings = Settings()  # type: ignore[call-arg]  # Required fields load from environment.
    except ValidationError:
        raise AgentError(
            "Invalid service configuration; check the documented required settings."
        ) from None
    return create_app(settings)
