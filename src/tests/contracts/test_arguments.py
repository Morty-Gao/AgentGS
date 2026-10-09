from copy import deepcopy

import pytest

from agentgs.contracts.arguments import normalize_arguments, parse_arguments


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", {}),
        (" plain input ", {"input_text": "plain input"}),
        ("[1, 2]", {"x": [1, 2]}),
        ("42", {"value": 42}),
        ("null", {"value": None}),
        (
            '{"longitude": 120, "latitude": 30}',
            {"longitude": 120, "latitude": 30, "lon": 120, "lat": 30},
        ),
    ],
)
def test_supported_invocation_forms(text, expected):
    assert parse_arguments(text) == expected


def test_authoritative_fields_and_wrapper_precedence_are_preserved():
    value = {
        "scenario": "ssp245",
        "lon": "*",
        "source_locator": {"scenario": "ssp585", "lon": 120},
        "parameters": {"lon": 100, "year": 2050},
        "input": {"year": 2100, "latitude": 30},
    }
    original = deepcopy(value)
    result = normalize_arguments(value)
    assert value == original
    assert result["scenario"] == "ssp245"
    assert result["lon"] == 120
    assert result["horizon_year"] == 2050
    assert result["lat"] == 30
    assert result["source_locator"] == value["source_locator"]


def test_encoded_input_and_empty_alias_targets():
    value = {"scenario": None, "input": '{"ssp": "ssp585", "year": 2100}'}
    result = normalize_arguments(value)
    assert result["scenario"] == "ssp585"
    assert result["horizon_year"] == 2100
