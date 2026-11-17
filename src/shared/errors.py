"""Errors safe to surface at application boundaries."""


class AgentError(Exception):
    """A request cannot be completed within the application contract."""


class ToolError(AgentError):
    """A resource or tool request is invalid or unavailable."""
