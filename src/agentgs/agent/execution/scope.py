"""Default-off caller-declared scope. Model arguments never set this state."""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4

FLAG = "GEOTOOL_RS_LOCAL_MIGRATION"
HEADER = b"x-geotool-task-domains"
VERSION = "rs-local-migration-v1"
_request = ContextVar("rs_migration_request", default=None)
_call = ContextVar("rs_migration_call", default=None)
_registry = ContextVar("rs_migration_registry", default=None)
_permitted = ContextVar("explicit_reference_permission", default=False)


def _legacy_env_get(name, default):
    return "1" if _permitted.get() else default


def current_request_id():
    return _request.get()


def active():
    return current_request_id() is not None


def current_call_id():
    return _call.get() if active() else None


def set_registry(registry):
    if active():
        _registry.set(registry)


def current_registry():
    return _registry.get() if active() else None


@contextmanager
def request_scope(domains=None):
    permitted = _permitted.get() and domains == "remote_sensing"
    request = _request.set(uuid4().hex if permitted else None)
    call = _call.set(None)
    registry = _registry.set(None)
    try:
        yield current_request_id()
    finally:
        _registry.reset(registry)
        _call.reset(call)
        _request.reset(request)


@contextmanager
def call_scope(call_id=None):
    token = _call.set((call_id or uuid4().hex) if active() else None)
    try:
        yield current_call_id()
    finally:
        _call.reset(token)


def transport_context():
    if not active():
        return None
    return {
        "version": VERSION,
        "domains": ["remote_sensing"],
        "request_id": current_request_id(),
        "call_id": current_call_id() or uuid4().hex,
    }


@contextmanager
def service_scope(metadata=None):
    valid = (
        _legacy_env_get(FLAG, "0").strip() == "1"
        and isinstance(metadata, dict)
        and (set(metadata) == {"version", "domains", "request_id", "call_id"})
        and (metadata.get("version") == VERSION)
        and (metadata.get("domains") == ["remote_sensing"])
        and all(
            (
                isinstance(metadata.get(k), str) and re.fullmatch("[0-9a-f]{32}" if k == "request_id" else "[A-Za-z0-9_-]{1,128}", metadata[k])
                for k in ("request_id", "call_id")
            )
        )
    )
    req = _request.set(metadata["request_id"] if valid else None)
    call = _call.set(metadata["call_id"] if valid else None)
    registry = _registry.set(None)
    try:
        yield
    finally:
        _registry.reset(registry)
        _call.reset(call)
        _request.reset(req)
