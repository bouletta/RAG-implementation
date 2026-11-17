# Grounded RAG Service operating and release runbook

## Deployment boundary

One process, one service key, one approved immutable application snapshot. Keep HTTP behind TLS ingress on a private network. Configure ingress connection/body limits and downstream request deadlines; do not assume the Python worker limit bounds all inbound connections.

## Promote a snapshot

1. In a trusted offline maintenance process, use `rag_cli.py` to ingest reviewed UTF-8 files under a new name. Never ingest while the service is using that directory.
2. Inspect the snapshot and record its hierarchy hash, embedding model/dimensions, source document hashes, chunk policy, image digest, and retrieval settings.
3. Evaluate labeled answerable and unanswerable questions, malicious source instructions, sensitive content, and citation correctness. Decide explicit acceptance thresholds before seeing results.
4. Stop writers. Back up the complete Chroma directory consistently, including SQLite and index files. A copy taken during writes is not a validated backup.
5. Give the service UID 10001 access to the deployment data directory, preserving a separate backup. Configure the new snapshot/hash and secrets outside Git.
6. Deploy one process. Confirm liveness, authenticated readiness, a bounded live canary, logs, and budget dashboards.
7. Roll back by restoring the last tested image, complete data backup, and matching configuration. Changing only the pinned hash is not a rollback.

## Example container

Prepare `/srv/grounded-rag.env` outside Git with the configured variables and restrict its permissions. Prepare `/srv/grounded-rag-data` with the reviewed Chroma directory under `chroma/`, owned by the service UID.

```bash
docker run --name grounded-rag --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --memory 1g --cpus 2 --stop-timeout 180 \
  -p 127.0.0.1:8000:8000 \
  --mount type=bind,src=/srv/grounded-rag-data,dst=/data \
  --env-file /srv/grounded-rag.env \
  -e STORE_PATH=/data/chroma \
  grounded-rag-service:latest
```

The root filesystem is read-only; the dedicated data mount is writable for Chroma bookkeeping. Restrict egress to the required provider through your infrastructure. Docker environment variables are visible to privileged host operators; use your platform's secret mechanism for deployed credentials.

## Failure response

| Signal | First action |
| --- | --- |
| Startup failure | Check approved hash, dimensions, data permissions, and key presence; do not disable validation |
| 401 | Check service-key distribution; never put tokens in tickets or logs |
| 429 | Reduce client rate; all clients share one token bucket |
| 503 | Inspect active workers; do not bypass the concurrency cap |
| 504 | Work may still be running; avoid immediate automated retries |
| 502 | Inspect safe counters and storage/provider health through trusted diagnostics |
| Rising abstentions | Compare corpus/evidence coverage before changing prompts |
| Suspected data compromise | Stop serving, preserve evidence, rotate keys, restore reviewed data |

Readiness is a startup check, not a live provider or storage health guarantee. A 200 liveness response is not proof that queries can succeed. Logs intentionally omit raw exceptions; reproduce failures in a protected diagnostic environment rather than adding unrestricted exception logging.

## Release gates

- [ ] Full repository checks, dependency consistency, and container smoke pass.
- [ ] Runtime dependencies and base-image vulnerabilities reviewed for this release.
- [ ] Fixed evaluation set meets declared answer, citation, abstention, and evidence-retention thresholds.
- [ ] Actual provider token use, cost, retry behavior, and p50/p95 latency measured.
- [ ] Load/slow-client tests confirm memory limits, ingress protections, and overload behavior.
- [ ] Timeout tests confirm callers understand work can continue after 504.
- [ ] TLS, egress, secret storage, access scope, and log retention reviewed.
- [ ] Backup restored into a fresh environment and hash verified.
- [ ] Previous release rollback and key rotation exercised.

Local regression results do not check these boxes automatically. No paid live canary or real production traffic is included in the offline test suite.

## Refresh runtime dependencies

Use CPython 3.13 on Linux x86_64. The generator resolves only binary artifacts, then records exact versions and SHA-256 hashes from pip's installation report:

```bash
python lock_runtime.py
```

Review the diff, rebuild the image, run smoke and regression checks, then rerun evaluation. This lock contains one selected wheel hash per dependency, not every platform's artifacts. Other platforms need a separately generated and tested lock. The Docker install verifies the downloaded artifacts against these hashes. Keep the OpenTelemetry exporter aligned with the repository's ADK environment when updating shared development dependencies.
