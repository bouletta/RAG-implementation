# Grounded RAG Service

A production-oriented, single-tenant reference service for evidence-grounded question answering over reviewed documents. It combines hierarchical document retrieval, bounded model calls, checked citations, an authenticated HTTP API, and reproducible container packaging.

This repository is a focused public extraction of a larger private learning project. It contains only the components required by the final RAG pipeline and has a fresh Git history.

## What it demonstrates

- Parent-child chunking: search compact child chunks, then restore bounded parent context.
- Hybrid lexical and semantic retrieval over a persistent Chroma snapshot.
- Optional bounded multi-query expansion and exact-excerpt context compression.
- Structured answers with source IDs, exact-quote validation, and explicit abstention.
- Snapshot integrity checks, logical model-call budgets, strict request schemas, and safe errors.
- Bearer authentication, rate limiting, bounded worker concurrency, deadlines, health endpoints, and counters.
- A non-root Docker image with a pinned base digest and artifact-hashed Python dependencies.
- Unit, integration, and offline container smoke tests with injected providers.

“Production-oriented” describes the engineering concerns covered here. It does not claim multi-tenant isolation, proven answer quality on an arbitrary corpus, or a completed production deployment.

## Architecture

```text
Reviewed documents -> parent/child chunking -> embedded immutable snapshot
                                              |
Client -> authentication -> admission limits -> hybrid multi-query retrieval
                                              |
                         parent expansion -> optional compression
                                              |
                          structured answer -> checked citations / abstention
                                              |
                               safe logs, metrics, request ID
```

## Local setup

Python 3.11 or newer is required. The container release targets Python 3.13 on Linux x86_64.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
python -m pip install -r requirements-dev.txt -e .
cp .env.example .env
```

Set `OPENAI_API_KEY`, create a random `SERVICE_KEY` of at least 32 characters, and keep `.env` out of Git.

Build and inspect a snapshot:

```bash
python rag_cli.py ingest demo-v1 examples/guide.md
python rag_cli.py inspect demo-v1
```

Copy the reported hierarchy hash into `EXPECTED_HIERARCHY_SHA256`, set `SNAPSHOT_NAME=demo-v1`, then start the service:

```bash
python serve.py
```

Query it from another terminal:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready \
  -H "Authorization: Bearer $SERVICE_KEY"
curl --fail http://127.0.0.1:8000/v1/query \
  -H "Authorization: Bearer $SERVICE_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"How are retries bounded?","retrieve_only":false}'
```

Normal ingestion and queries can call paid model APIs. Tests and the container smoke path use injected providers and make no paid provider calls.

## Verification

```bash
ruff check .
python -m pytest -q
docker build -t grounded-rag-service:latest .
python container_smoke.py --image grounded-rag-service:latest
```

The tests cover the retrieval-to-citation flow, corrupted or incorrectly pinned snapshots, strict schemas, authentication, safe errors and logs, rate rejection, and worker occupancy after request timeout or cancellation.

See [VERIFICATION.md](VERIFICATION.md) for the recorded local verification and [RUNBOOK.md](RUNBOOK.md) for deployment boundaries, release gates, and rollback guidance.

## Deliberate limitations

- One process, one service key, one trust domain.
- Local Chroma storage and in-process counters and quotas.
- HTTP timeouts do not cancel already-running synchronous provider work.
- Citation shape and exact quotes do not prove semantic entailment.
- Corpus-specific quality, latency, cost, adversarial behavior, and restore procedures must be measured before deployment.

## License

MIT
