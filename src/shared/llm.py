"""OpenAI transport; SDK retries are the single retry owner."""

from typing import Any, TypeVar, cast

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ValidationError

from shared.config import RuntimeSettings
from shared.errors import AgentError
from shared.models import Call, Turn

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class OpenAIModel:
    def __init__(self, settings: RuntimeSettings) -> None:
        if settings.api_key is None or not settings.api_key.get_secret_value().strip():
            raise AgentError("Set OPENAI_API_KEY in the agent's .env for live model calls.")
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def close(self) -> None:
        self.client.close()

    def generate(
        self,
        instructions: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Turn:
        try:
            response = self.client.responses.create(
                model=self.settings.model,
                instructions=instructions,
                input=cast(Any, messages),
                tools=cast(Any, tools),
                parallel_tool_calls=False,
                store=False,
                max_output_tokens=self.settings.max_output_tokens,
            )
        except OpenAIError as exc:
            raise AgentError(
                "Model request failed; check credentials, quota, or connectivity."
            ) from exc
        if response.status != "completed":
            raise AgentError("Model response was incomplete; no further tools were run.")
        calls = [
            Call(item.call_id, item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        ]
        return Turn(
            text=response.output_text,
            calls=calls,
            replay=[item.model_dump(exclude_none=True) for item in response.output],
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
        )

    def structured(
        self,
        instructions: str,
        content: list[dict[str, Any]],
        schema: type[ResponseT],
    ) -> ResponseT:
        try:
            response = self.client.responses.parse(
                model=self.settings.model,
                instructions=instructions,
                input=cast(Any, [{"role": "user", "content": content}]),
                text_format=schema,
                store=False,
                max_output_tokens=self.settings.max_output_tokens,
            )
        except (OpenAIError, ValidationError) as exc:
            raise AgentError("Structured model request failed or violated its schema.") from exc
        if response.status != "completed" or not isinstance(response.output_parsed, schema):
            raise AgentError("Model refused, returned no result, or exhausted its output budget.")
        return response.output_parsed
