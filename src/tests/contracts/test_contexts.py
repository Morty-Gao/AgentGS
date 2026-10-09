import asyncio

import pytest

from agentgs.contracts.remote_sensing import contract_scope, enabled, prepare_step
from agentgs.contracts.resources import resolve_resource_path, resource_access


def test_nested_scopes_restore_policy_and_resolver_after_an_error(tmp_path):
    outer = tmp_path / "outer"
    inner = tmp_path / "inner"
    initial_policy = enabled()
    with resource_access(lambda _: outer), contract_scope(False):
        with pytest.raises(ValueError, match="abort"):
            with resource_access(lambda _: inner), contract_scope(True):
                assert resolve_resource_path("asset") == inner
                assert enabled()
                raise ValueError("abort")
        assert resolve_resource_path("asset") == outer
        assert not enabled()
    assert enabled() == initial_policy
    assert resolve_resource_path("missing-contract-test-asset") is None


async def test_concurrent_requests_have_independent_contract_contexts(tmp_path):
    barrier = asyncio.Barrier(2)
    initial_policy = enabled()

    async def request(active, path):
        with resource_access(lambda _: path), contract_scope(active):
            await barrier.wait()
            assert enabled() is active
            assert resolve_resource_path("same-asset") == path

    await asyncio.gather(request(True, tmp_path / "a"), request(False, tmp_path / "b"))
    assert enabled() == initial_policy


def test_disabled_contracts_leave_step_arguments_untouched():
    params = {"nir_band_file": "scene_nir.tif", "red_band_file": "scene_red.tif"}
    with contract_scope(False):
        assert prepare_step("RemoteSensingNDVI", params, {}) == {}
    assert params == {"nir_band_file": "scene_nir.tif", "red_band_file": "scene_red.tif"}
