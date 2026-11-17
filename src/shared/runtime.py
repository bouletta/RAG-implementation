"""A bounded tool loop preserving the complete request and tool history."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.config import RuntimeSettings
from shared.errors import AgentError, ToolError
from shared.models import Model, RunResult, Step

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    arguments: type[BaseModel]
    execute: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.arguments.model_json_schema(),
            "strict": True,
        }


def run_tools(
    question: str,
    instructions: str,
    tools: list[Tool],
    model: Model,
    settings: RuntimeSettings,
) -> RunResult:
    question = question.strip()
    if not question or len(question) > 8000:
        raise AgentError("Provide a question between 1 and 8000 characters.")
    registry = {tool.name: tool for tool in tools}
    if len(registry) != len(tools):
        raise AgentError("Tool names must be unique.")
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    steps: list[Step] = []
    seen_ids: set[str] = set()
    input_tokens = output_tokens = 0
    for _ in range(settings.max_tool_calls + 1):
        if len(instructions) + len(json.dumps(messages)) > settings.max_context_chars:
            raise AgentError("Conversation context budget exceeded.")
        turn = model.generate(instructions, messages, [tool.schema() for tool in tools])
        input_tokens += turn.input_tokens
        output_tokens += turn.output_tokens
        if not turn.calls:
            if not turn.text.strip():
                raise AgentError("Model returned an empty answer.")
            if not any(step.succeeded for step in steps):
                raise AgentError("The model produced no successful tool evidence.")
            return RunResult(
                answer=turn.text.strip(),
                steps=steps,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        if len(steps) + len(turn.calls) > settings.max_tool_calls:
            raise AgentError("Tool budget exceeded before executing the requested batch.")
        messages.extend(turn.replay)
        for call in turn.calls:
            if call.id in seen_ids:
                raise AgentError("The model repeated a tool-call identifier.")
            seen_ids.add(call.id)
            try:
                tool = registry.get(call.name)
                if tool is None:
                    raise ToolError("Unknown tool.")
                if len(call.arguments) > 10000:
                    raise ToolError("Tool arguments exceed the size limit.")
                args = tool.arguments.model_validate_json(call.arguments, strict=True)
                result = tool.execute(**args.model_dump())
                if len(json.dumps(result, ensure_ascii=True)) > settings.max_tool_chars:
                    raise ToolError("Tool result exceeds the context budget; narrow the request.")
                succeeded = True
            except ValidationError:
                result, succeeded = {"error": "Invalid tool arguments."}, False
            except ToolError as exc:
                result, succeeded = {"error": str(exc)}, False
            steps.append(Step(tool=call.name, succeeded=succeeded, result=result))
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call.id,
                    "output": json.dumps(result, ensure_ascii=True),
                }
            )
            logger.info("tool_succeeded" if succeeded else "tool_rejected")
    raise AgentError("Tool round limit exceeded.")
