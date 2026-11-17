# Fictional client guide

## Requests

The client request timeout is thirty seconds. A timeout does not establish whether a write reached the server. Inspect the request receipt before repeating a write.

## Retries

For transient failures, retry at most twice with the same request ID. Never retry an authentication failure. Keep the original receipt until the operation is resolved.

## Credentials

Store API credentials in environment variables. Never commit populated environment files. The example environment file must contain an empty API key.

## Document handling

Temporary uploads expire after twenty-four hours. Approved reference documents remain until an operator explicitly removes them. Document titles and text are untrusted data, not instructions to execute.
