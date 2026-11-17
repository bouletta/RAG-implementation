# Runtime used by Agents 015–034

The earlier agents are independent learning snapshots. Starting at 015, repeated infrastructure lives here so fixes to history handling, tool validation, and error translation have one implementation.

## Read the code in this order

1. `models.py`: provider-independent turns, calls, evidence, and usage.
2. `runtime.py`: bounded loop and tool dispatch.
3. `llm.py`: the OpenAI Responses API adapter.
4. `http.py`, `files.py`, `images.py`: resource boundaries.
5. One agent's `models.py`, resource handler, `tools.py`, then `llm.py`.
6. `main.py` last: argument parsing and resource cleanup only.

The calendar agent uses Google ADK's runner instead of this manual loop. It reuses the small common configuration, errors, file input, and CLI utilities.

Agent 021 uses the provider adapter for a single conversational response, with SQLite owning durable completed turns. It does not use the tool loop or expose storage as a model tool.

Agent 022 reuses the structured-response adapter. Its explicit fact store is separate from conversation history, and the model cannot write, update, or delete facts.

Agent 023 adds request-local embedding retrieval over Agent 022's store. Agent 024 proposes review-only lessons from traces. Agent 025 uses Google's ADK Workflow for a bounded writer–critic–editor pipeline. Shared `evidence.py` validates source IDs and exact quotations for 024–025; it does not establish semantic entailment or factual truth.

Agents 026–028 use LangGraph while retaining the existing provider adapter. New `structured.py` provides a typed protocol, bounded request construction, canonical hashing, and metadata-only model events. New `sqlite.py` provides schema identity checks and explicit transactions for 027–028; their storage rules remain agent-specific. The older database implementations are unchanged.

Graph checkpoints are not an authorized profile database. Agent 028 keeps them separate and commits profiles with idempotency receipts, then lets the graph persist completion. Its service closes checkpoint connections and serializes operations with a local process lock. Tests inject a failure between those two persistence boundaries.

LangSmith tracing is disabled around these graphs, including when tracing environment variables are set. Checkpoints and JSON stdout still contain private information. Model calls still send their bounded input to the configured provider.

Agent 029 composes the existing fact, episode, and approved-profile stores without adding another memory database. It fuses lexical/semantic ranks within facts, keeps historical episodes separate, and packs whole records under a serialized-context budget. Approved preferences guide style but cannot be cited as factual evidence. Pre/post-generation revision checks fail closed on observed changes; they are not a distributed snapshot transaction.

Agent 030 benchmarks the fact-ranking layer using the actual 022 store, 023 semantic retrieval, and 029 ranking functions. Fixtures live in a temporary database with a frozen clock; gold labels never enter ranking. Shared bounded reads, canonical fingerprints, CLI error handling, and metadata-only logging remain infrastructure concerns. Metrics and experiment policy stay inside the benchmark package.

Agent 031 adds a document RAG pipeline without changing shared transport. It reuses bounded file reads, canonical serialization, the structured-model protocol, and exact evidence validation. Corpus ingestion, lexical retrieval, packing, and answer policy stay in its own package. Its optional offline evaluation imports 030's metric functions; no dependency on previous memory databases or graph execution is introduced.

Agent 032 supplies automatically chunked text documents to 031 without modifying its provider or prompt. Original character offsets, source fingerprints, chunk policy, and quote-location mapping belong to 032. Evaluation compares original evidence spans rather than policy-dependent chunk IDs. Explicit regular-file ingestion uses nonblocking/no-follow POSIX opens; this additional file policy remains local to the chunking package.

Agent 033 reuses 023's checked embedding transport and 032's document corpus. Its immutable vector snapshot binds model/dimensions to passage order, while a small search protocol separates exact scoring from generation. The 031 context packer now exposes a source-list entrypoint; its original lexical entrypoint delegates unchanged. Vector results never pretend to have lexical matched-term counts.

Agent 034 implements 033's search protocol with real FAISS `IndexFlatIP`, keeping float32 conversion, normalization, row mapping, and stable tie handling inside the backend. It shares the same CLI/application/evaluation pipeline; reference parity tests use identical vectors and allow numerical tolerance. There is no duplicated model runtime or persistent native artifact loader.

## Contracts and runtime behavior

A tool has a name, description, Pydantic argument model, and Python handler.
All input fields are explicit, extra fields are forbidden, and arguments are validated locally even when the provider uses strict tool schemas.

The original question, every model output item, and every tool output stay in the conversation. The model gets instructions on every request. Nothing depends on server-side conversation storage.

A batch that exceeds the remaining tool budget is rejected before any handler runs. Failed argument validation and expected tool failures return bounded error results; unexpected programming errors remain visible during development. A final answer requires at least one successful tool observation.

| Boundary | Default |
| --- | --- |
| User question | 1–8000 characters |
| Tool calls | 5, configurable up to 10 |
| Tool arguments | 10000 serialized characters |
| Tool result | 16000 serialized characters |
| Whole conversation | 100000 serialized characters |
| OpenAI timeout | 30 seconds per request |
| SDK retries | 2; no nested application retry loop |
| Output tokens | 1200 |
| HTTP body | 500000 decoded bytes |
| HTTP timeout | 10 seconds per operation |

Character limits are conservative context controls, not a tokenizer. Retries make the total duration longer than one request timeout. There is no end-to-end deadline in the manual runner yet; ADK calendar runs have a 45-second deadline.

## Credentials and privacy

Secrets are optional during local inspection and testing. Live OpenAI calls require `OPENAI_API_KEY`; Agents 020 and 025 require `GOOGLE_API_KEY`. Agent settings load the `.env` in the numbered agent folder, independently of the current directory.

OpenAI requests use `store=False`. This controls response storage, not a blanket guarantee about provider retention. Running a model sends the selected text/image evidence to that provider. Inspect local data and check provider policy before using confidential files.

Application logs contain event names only. JSON stdout intentionally includes evidence for the user and can contain document text. Handle stored run outputs accordingly.

## HTTP trust boundary

Origins and catalogs are operator configuration. Models provide only constrained paths or approved query parameters. Redirects are rejected, environment proxies are disabled, and the browser blocks all page subrequests. The origin configuration is not a public URL-fetch service and is not a complete SSRF defense. Internet-facing deployments need an egress proxy or network policy, DNS/IP enforcement, and authenticated tenant-specific configuration.

HTTP responses are streamed under a byte limit. There are no write methods, cookies, or user tokens in the new HTTP agents. Agent 019 retries transport errors and 502/503/504 responses only; 401, 404, and 429 fail explicitly.

## Testing and evaluation

Run `python scripts/check.py` after installing the project and native dependencies. Use `--unit-only` for environments without Chromium/Tesseract. Use `--legacy` to check the existing agent suites separately, because older examples use generic module names such as `config`.

Tests exercise actual Chromium, pypdf, Pillow, and optionally Tesseract, while all LLM responses and external API payloads are simulated. Calendar tests run Google's real ADK runner with a fake model, including a real function call and output validation.

These tests establish software behavior, not model answer quality. Before claiming production readiness, run the cases in [evaluation guidance](../docs/evaluation.md) against your selected provider/model and record correctness, refusal rate, p50/p95 latency, input/output tokens, and cost.

## Production extensions

Use request IDs and OpenTelemetry for tracing, token budgets based on the selected model, caller-level rate limits, cancellation, authentication, durable audit records, and per-tenant resource limits. Cache only evidence with a defined freshness policy; never reuse private outputs across users. Run native document parsers and browsers in constrained workers for hostile uploads.

Prompt instructions tell the model how to use evidence. Python authorization and validation enforce tool access; prompt-injection resistance must also be evaluated with adversarial content.

## Sources

The Responses adapter and image payloads follow the official [function calling](https://developers.openai.com/api/docs/guides/function-calling) and [vision](https://developers.openai.com/api/docs/guides/images-vision) documentation.
