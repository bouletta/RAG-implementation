"""Consistent JSON stdout and error exit codes for the new agents."""

import json
import sys
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from shared.errors import AgentError
from shared.logging import configure_logging


def execute(action: Callable[[], BaseModel | dict[str, Any]]) -> int:
    configure_logging()
    try:
        result = action()
        print(
            result.model_dump_json(indent=2)
            if isinstance(result, BaseModel)
            else json.dumps(result, indent=2, ensure_ascii=False)
        )
        return 0
    except (AgentError, ValidationError) as exc:
        message = (
            "Invalid configuration or input." if isinstance(exc, ValidationError) else str(exc)
        )
        print(json.dumps({"error": message}), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
