# Verification record - 2026-11-17

This records local engineering checks for the public extraction, not a production approval.

| Check | Result |
| --- | --- |
| Artifact-hashed dependency installation | Passed in a clean Python 3.13 virtual environment |
| Ruff | Passed |
| Focused test suite | 37 passed in 5.29 seconds with temporary stores and injected providers |
| Container build | Passed with the pinned Python base and hashed runtime lock |
| Container smoke | Passed: offline ingestion, HTTP-to-citation flow, authentication, request validation, metrics, and UID 10001 |
| Paid provider calls | None |

Reproduction commands:

```bash
ruff check .
python -m pytest -q
docker build -t grounded-rag-service:latest .
python container_smoke.py --image grounded-rag-service:latest
```

The smoke test uses a disposable Docker volume, a read-only service root filesystem, dropped Linux capabilities, and `no-new-privileges`. It removes its generated container and volume afterward.

These checks validate engineering behavior, not live answer quality.

Still deployment-specific: corpus answer quality, provider permissions and billing, p50/p95 latency, throughput, vulnerability review, TLS, multi-tenant isolation, backup restoration, and rollback under load.
