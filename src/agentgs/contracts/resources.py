"""Injected resource access for algorithm code; no scheduler or runtime imports."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator

ResourceResolver = Callable[[Any], Path | None]

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
