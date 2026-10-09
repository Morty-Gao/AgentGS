"""Shared storage must not mix process liveness between worker environments."""

import asyncio

import pytest

from agentgs.runtime.concurrency import RuntimeCoordinator


async def test_different_workers_can_execute_against_same_shared_project(project, monkeypatch):
    monkeypatch.setattr("agentgs.runtime.concurrency.worker_identity", lambda: "worker-a")
    first = RuntimeCoordinator(project)
    monkeypatch.setattr("agentgs.runtime.concurrency.worker_identity", lambda: "worker-b")
    second = RuntimeCoordinator(project)
    async with first.slot("tools", 1):
        async with asyncio.timeout(1):
            async with second.slot("tools", 1):
                pass


async def test_same_worker_shares_capacity_through_project_symlink(project):
    alias = project.parent / (project.name + "-alias")
    alias.symlink_to(project, target_is_directory=True)
    first, second = RuntimeCoordinator(project), RuntimeCoordinator(alias)
    async with first.slot("tools", 1):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                async with second.slot("tools", 1):
                    pass
    async with asyncio.timeout(1):
        async with second.slot("tools", 1):
            pass
