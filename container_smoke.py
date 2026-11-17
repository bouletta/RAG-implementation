"""Exercise the built image with a disposable offline snapshot and no provider calls."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from uuid import uuid4

FIXTURE = """
from pathlib import Path
from chunking_agent.ingestion import load_documents
from parent_document_agent.config import Settings
from parent_document_agent.hierarchy import create
from parent_document_agent.store import ParentStore
from semantic_memory_agent.embeddings import Embedded

class Provider:
    def __init__(self, settings=None):
        pass
    def embed(self, texts):
        return Embedded(vectors=[[1.0] + [0.0] * 15 for _ in texts], tokens=0)
    def close(self):
        pass

settings = Settings(_env_file=None, dimensions=16)
path = Path('/data/fixture.txt')
path.write_text('Timeouts protect resources. Never log credentials.')
hierarchy = create(load_documents([path], settings), settings)
ParentStore(Path('/data/chroma')).ingest_hierarchy('smoke-v1', hierarchy, settings, Provider())

import json
from basic_rag_agent.models import Reply
from fastapi.testclient import TestClient
from production_rag_agent.app import create_app
from production_rag_agent.config import Settings as ServiceSettings
import production_rag_agent.engine as engine_module
from semantic_memory_agent.embeddings import OpenAIEmbedder
from shared.llm import OpenAIModel

class Model:
    def __init__(self, settings):
        pass
    def structured(self, instructions, content, schema):
        source = json.loads(content[0]['text'])['sources'][0]
        return Reply(claims=[{'text': 'Timeouts protect resources.', 'evidence': [
            {'source_id': source['id'], 'quote': 'Timeouts protect resources.'}
        ]}], limitation='Injected-provider container test.')
    def close(self):
        pass

service = ServiceSettings(_env_file=None, service_key='x' * 40,
    api_key='offline-not-a-real-key', store_path=Path('/data/chroma'),
    snapshot_name='smoke-v1', expected_hierarchy_sha256=hierarchy.sha256,
    dimensions=16, max_variants=0, compress_parents=False)
OpenAIModel(service).close()
OpenAIEmbedder(service).close()
engine_module.LazyEmbedder = Provider
engine_module.LazyModel = Model
with TestClient(create_app(service)) as client:
    response = client.post('/v1/query', headers={'Authorization': 'Bearer ' + 'x' * 40},
                           json={'question': 'Timeouts'})
    assert response.status_code == 200, response.text
    assert response.json()['answer']['reply']['claims'][0]['evidence'][0]['source_id'].startswith('p_')
print(hierarchy.sha256)
"""


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True, timeout=180).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="grounded-rag-service:latest")
    args = parser.parse_args()
    suffix = uuid4().hex[:12]
    volume = f"grounded-rag-smoke-{suffix}"
    container = f"grounded-rag-smoke-{suffix}"
    key = "offline-smoke-key-" + suffix * 3
    docker("volume", "create", volume)
    started = False
    try:
        digest = docker(
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{volume}:/data",
            "--entrypoint",
            "python",
            args.image,
            "-c",
            FIXTURE,
        ).splitlines()[-1]
        docker(
            "run",
            "-d",
            "--name",
            container,
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "1g",
            "--cpus",
            "2",
            "-p",
            "127.0.0.1::8000",
            "-v",
            f"{volume}:/data",
            "-e",
            f"SERVICE_KEY={key}",
            "-e",
            "OPENAI_API_KEY=offline-not-a-real-key",
            "-e",
            "DIMENSIONS=16",
            "-e",
            "SNAPSHOT_NAME=smoke-v1",
            "-e",
            f"EXPECTED_HIERARCHY_SHA256={digest}",
            args.image,
        )
        started = True
        port = docker("port", container, "8000/tcp").split(":")[-1]
        base = f"http://127.0.0.1:{port}"

        def request(path: str, *, authenticated: bool = False, data: bytes | None = None) -> int:
            headers = {"Authorization": f"Bearer {key}"} if authenticated else {}
            if data is not None:
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(base + path, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=2) as response:
                    return response.status
            except urllib.error.HTTPError as error:
                return error.code

        deadline = time.monotonic() + 60
        while True:
            try:
                if request("/health/live") == 200:
                    break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("Container did not become healthy.")
            time.sleep(0.25)
        assert request("/health/ready") == 401
        assert request("/health/ready", authenticated=True) == 200
        assert request("/v1/query", authenticated=True, data=b"x" * 9000) == 413
        assert (
            request("/v1/query", authenticated=True, data=b'{"question":"x","model":"other"}')
            == 422
        )
        assert request("/metrics", authenticated=True) == 200
        assert docker("exec", container, "id", "-u") == "10001"
        print(
            json.dumps(
                {
                    "container_smoke": "passed",
                    "paid_provider_calls": 0,
                    "snapshot": digest,
                    "uid": 10001,
                }
            )
        )
    except Exception:
        if started:
            print(docker("logs", container))
        raise
    finally:
        if started:
            docker("rm", "-f", container)
        docker("volume", "rm", volume)


if __name__ == "__main__":
    main()
