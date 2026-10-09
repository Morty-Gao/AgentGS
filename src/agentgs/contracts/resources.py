"""Injected resource access for algorithm code; no scheduler or runtime imports."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

ResourceResolver = Callable[[Any], Path | None]

RESOURCE_MARKER = (
    "Evaluation input resource (authoritative benchmark locator; use it exactly when "
    "relevant, but retrieve and select the registered tools yourself):"
)


def retrieval_question(question, input_resource=None):
    """Construct the public retrieval input identically offline and at runtime."""
    if input_resource is not None and RESOURCE_MARKER not in question:
        return (
            question
            + "\n\n"
            + RESOURCE_MARKER
            + " "
            + json.dumps(input_resource, ensure_ascii=False)
        )
    return question


def split_question_resource(question: str) -> tuple[str, Any]:
    """Separate caller-supplied public resources from the question text."""
    if RESOURCE_MARKER not in question:
        return question.strip(), None
    body, raw = question.rsplit(RESOURCE_MARKER, 1)
    try:
        resource = json.loads(raw.strip())
    except json.JSONDecodeError:
        resource = raw.strip()
    return body.strip(), resource


_resolver: ContextVar[ResourceResolver | None] = ContextVar("agent_resource_resolver", default=None)


def resolve_resource_path(value: Any) -> Path | None:
    """Resolve through the injected adapter, or an existing local path."""
    resolver = _resolver.get()
    try:
        if resolver is not None:
            return resolver(value)
        path = Path(str(value)).expanduser()
        return path.resolve() if path.exists() else None
    except (FileNotFoundError, ValueError, OSError):
        return None


@contextmanager
def resource_access(resolver: ResourceResolver | None) -> Iterator[None]:
    """Install a resolver for this context, restoring it on normal or error exit."""
    token = _resolver.set(resolver)
    try:
        yield
    finally:
        _resolver.reset(token)
