"""Representation invariance and rejection of ambiguous or contradictory answers."""

import json
from pathlib import Path

import pytest

from agentgs.evaluation.datasets.base import Example
from agentgs.evaluation.datasets.geotooleval import load_tasks
from agentgs.evaluation.metrics.answers import score_answer
from agentgs.evaluation.metrics.report import score_prediction, summarize

ROOT = Path(__file__).resolve().parents[3]


def example(expected, kind="single_number", question="", **kwargs):
    return Example(1, 1, "fixture", question, expected, kind, **kwargs)


@pytest.fixture(scope="module")
def tasks():
    return {t.question_id: t for t in load_tasks(ROOT / "data/geotooleval.jsonl")}


@pytest.mark.parametrize(
    "answer",
    [
        "46.54 km",
        "**46.54 km**",
        "```text\n46.54 km\n```",
        "46.54km.",
        "46.54",
        "46540 m",
        "4.654e1 km",
        "The answer is 46.54 km.",
        '{"answer": "46.54 km"}',
        "Earlier estimate: 12 km\nFinalAnswer: 46.54 km",
    ],
)
def test_scalar_equivalent_representations(answer):
    assert score_answer(example("46.54 km"), answer)["score"] == 1


@pytest.mark.parametrize(
    "answer",
    [
        "46.54 m",
        "46.54 MW",
        "46.54 bananas",
        "47 km",
        "46.54 km or 60 km",
        "46.54 km is incorrect",
        "not 46.54 km",
        "unknown",
        "NaN",
        "inf",
        "1e309 km",
        "Results: 46.54, 60",
        "46.54 and 46.54 m",
        "[46.54, 60]",
    ],
)
def test_scalar_wrong_units_ambiguity_and_nonfinite_values(answer):
    assert score_answer(example("46.54 km"), answer)["score"] == 0


@pytest.mark.parametrize(
    "expected,answer",
    [
        ("-1.22%", "−1.22 %"),
        ("-1.22%", "-0.0122 fraction"),
        ("0.50 km", ".5 km"),
        ("6.803e18 J", "6.8030E18 J"),
        ("1234.50 MW", "1,234.5 MW"),
        ("258550097.697 km^2", "258550097.697 km²"),
        ("384.536 μatm", "384.536 µatm"),
        ("2333.7 μmol/kg", "2.3337 mmol/kg"),
    ],
)
def test_numeric_notation_and_unit_conversion(expected, answer):
    assert score_answer(example(expected), answer)["score"] == 1


def test_reference_precision_uses_exponent_and_explicit_rounding():
    assert score_answer(example("5.835e-5"), "6e-5")["score"] == 0
    assert score_answer(example("6.803e18 J"), "6.8034e18 J")["score"] == 1
    assert score_answer(example("6.803e18 J"), "6.804e18 J")["score"] == 0
    task = example("1.2", question="Round each value to three decimals.")
    assert score_answer(task, "1.201")["score"] == 0


@pytest.mark.parametrize(
    "answer,score",
    [
        ("B", 1),
        ("**B**", 1),
        ("(B)", 1),
        ("B. 2.46 mm", 1),
        ("Option A is incorrect. The answer is B.", 1),
        ("2.46 mm", 1),
        ("答案：B", 1),
        ("0.00246 m", 1),
        ("The answer is B or C", 0),
        ("Answer: A. Answer: B.", 0),
        ("Option A or option B", 0),
        ("2.45 mm", 0),
        ("2.46 km", 0),
        ("The answer is not B", 0),
    ],
)
def test_choice_parses_final_assertion_or_unique_option_value(answer, score):
    task = example("B", "choice", "Options: A. 1.87 mm; B. 2.46 mm; C. 3.05 mm; D. 3.64 mm")
    assert score_answer(task, answer)["score"] == score


def test_equal_option_values_are_ambiguous():
    task = example("B", "choice", "Options: A. 2.46 mm; B. 2.46 mm")
    assert score_answer(task, "2.46 mm")["score"] == 0


@pytest.mark.parametrize(
    "answer", ["2026/4/8", "2026年4月8日", "April 8, 2026", "8 Apr 2026", "2026-04-08"]
)
def test_calendar_date_representations(answer):
    assert score_answer(example("2026-04-08", "date"), answer)["score"] == 1


@pytest.mark.parametrize(
    "answer", ["2026-02-30", "04/08/2026", "2026-04-08 or 2026-04-09", "not 2026-04-08"]
)
def test_dates_reject_invalid_or_ambiguous_answers(answer):
    assert score_answer(example("2026-04-08", "date"), answer)["score"] == 0


@pytest.mark.parametrize("answer", ["67°45′0.00″ N", "67°45'0.0\" N", "**67°45′0″ N**"])
def test_dms_numeric_equivalence(answer):
    assert score_answer(example("67°45'0.00\" N", "single_text"), answer)["score"] == 1


@pytest.mark.parametrize("answer", ["67°45′0″ S", "67°45′1″ N", "67°60′0″ N", "not 67°45′0″ N"])
def test_dms_rejects_wrong_direction_values_or_invalid_coordinates(answer):
    assert score_answer(example("67°45'0.00\" N", "single_text"), answer)["score"] == 0


@pytest.mark.parametrize(
    "answer",
    [
        "https://example.org/Case_123",
        "[name](https://example.org/Case_123)",
        "<https://example.org/Case_123>",
        "`https://example.org/Case_123`",
    ],
)
def test_url_wrappers(answer):
    assert score_answer(example("https://example.org/Case_123", "identifier"), answer)["score"] == 1


@pytest.mark.parametrize(
    "answer",
    [
        "https://example.org/case_123",
        "https://example.org/Case_123_extra",
        "https://example.org/Case_123 or https://example.org/Other",
        "not https://example.org/Case_123",
    ],
)
def test_url_identity_and_conflicting_identifiers(answer):
    assert score_answer(example("https://example.org/Case_123", "identifier"), answer)["score"] == 0


def test_non_url_identifier_with_label():
    assert (
        score_answer(example("2011-0326-THA", "identifier"), "DisNo: `2011-0326-THA`")["score"] == 1
    )


def test_lists_ignore_bullets_but_preserve_duplicates_and_extra_items():
    task = example("Sunny Day; Unnamed; Unnamed", "multi_text")
    assert score_answer(task, "- Sunny Day\n- Unnamed\n- Unnamed")["score"] == 1
    assert score_answer(task, json.dumps(["Sunny Day", "Unnamed", "Unnamed"]))["score"] == 1
    assert score_answer(task, "Sunny Day, Unnamed, Unnamed")["score"] == 1
    assert (
        score_answer(task, "site_name: Sunny Day\nsite_name: Unnamed\nsite_name: Unnamed")["score"]
        == 1
    )
    assert score_answer(task, "Sunny Day; Unnamed")["score"] == 0.8
    assert score_answer(task, "Sunny Day; Unnamed; Unnamed; Other")["score"] < 1
    assert score_answer(example("China", "single_text"), "Chinatown")["score"] == 0
    assert score_answer(example("China", "single_text"), "Not China")["score"] == 0
    assert (
        score_answer(example("Gold:2167; Iron:499", "multi_text"), "Gold: 2167\nIron: 499")["score"]
        == 1
    )


def test_numeric_fields_remain_bound_to_their_labels():
    task = example(
        "Great-circle distance: 10388.286 km.; Initial bearing from point1 to point2: 9.06°",
        "multi_number",
    )
    assert (
        score_answer(task, "Initial bearing: 9.06°; Great-circle distance: 10388286 m")["score"]
        == 1
    )
    assert (
        score_answer(task, "Initial bearing: 10388.286°; Great-circle distance: 9.06 km")["score"]
        == 0
    )


def test_actual_earth_record_ignores_context_depth_but_checks_fields(tasks):
    good = "Moho: 31.00 km; LID-BOTTOM: 218.01 km; dlnVs: 2.56 %"
    assert score_answer(tasks[86], good)["score"] == 1
    swapped = "Moho: 218.01 km; LID-BOTTOM: 31.00 km; dlnVs: 2.56 %"
    assert score_answer(tasks[86], swapped)["score"] == pytest.approx(1 / 3, abs=1e-6)
    wrong_context = "Moho: 31.00 km; LID-BOTTOM: 218.01 km; dlnVs at 200 km: 2.56 %"
    assert score_answer(tasks[86], wrong_context)["score"] < 1
    assert score_answer(tasks[86], good + "; Moho: 99 km")["score"] < 1


def test_public_requested_fields_exclude_unrequested_reference_metadata(tasks):
    assert score_answer(tasks[199], "temperature_anomaly: 3.41 °C")["score"] == 1
    assert score_answer(tasks[214], "mean_temperature_anomaly: -0.220 °C")["score"] == 1
    assert (
        score_answer(tasks[397], "valid_count: 1918; mean_fCO2: 315.696 µatm; mean_SST: 8.227 °C")[
            "score"
        ]
        == 1
    )


def test_all_reference_answers_are_recognized(tasks):
    failed = {
        qid: score_answer(task, task.expected_answer)
        for qid, task in tasks.items()
        if score_answer(task, task.expected_answer)["score"] != 1
    }
    assert failed == {}


def test_real_dms_templates_accept_numeric_equivalents(tasks):
    for qid in [447, 448, 449, 450, 471, 472, 473, 474]:
        text = tasks[qid].expected_answer.replace("'", "′").replace('"', "″")
        row = score_prediction(tasks[qid], text)
        assert row["answer_score"]["score"] == 1
        assert row["answer_format"]["valid"]
        assert row["legacy_answer_score"]["score"] == 0


def test_format_diagnostic_and_legacy_score_are_separate(tasks):
    row = score_prediction(tasks[1], "```text\n46.54 km\n```")
    assert row["answer_score"]["score"] == row["legacy_answer_score"]["score"] == 1
    assert not row["answer_format"]["valid"]
    report = summarize([row], 1, domain_count=None)
    assert report["format_valid_rate"] == 0
    assert report["micro_fas"] == 1


def test_units_from_public_template_apply_when_gold_omits_them(tasks):
    assert score_answer(tasks[539], "-15.950 °C")["score"] == 1
    assert score_answer(tasks[539], "-15.950 km")["score"] == 0


def test_structured_scalar_uses_requested_field_and_preserves_units(tasks):
    answer = {"grid_cell": {"fco2_uatm": 384.536, "latitude": 34.5, "year": 2022}}
    assert score_answer(tasks[347], json.dumps(answer))["score"] == 1
    assert score_answer(tasks[347], str(answer))["score"] == 1
    answer["grid_cell"]["fco2_uatm"] = 100
    answer["grid_cell"]["irrelevant"] = 384.536
    assert score_answer(tasks[347], str(answer))["score"] == 0
    assert (
        score_answer(
            tasks[347], json.dumps({"rows": [{"fco2_uatm": 384.536}, {"fco2_uatm": 400}]})
        )["score"]
        == 0
    )
    assert score_answer(example("46.54 km"), '{"answer":46.54,"unit":"m"}')["score"] == 0
    assert score_answer(example("46.54 km"), '{"value":46540,"unit":"m"}')["score"] == 1
    assert score_answer(example("46.54 km"), '{"answer":"46.54 km","unit":"m"}')["score"] == 0


@pytest.mark.parametrize("answer", ["Option B or C", "Choice B/C"])
def test_choice_label_with_alternatives_is_ambiguous(answer):
    assert score_answer(example("B", "choice"), answer)["score"] == 0


def test_malformed_grouping_cannot_drop_leading_digits():
    assert score_answer(example("34 km"), "12,34 km")["score"] == 0


@pytest.mark.parametrize("kind,expected", [("single_number", "1"), ("multi_number", "1; 2")])
def test_nonfinite_predictions_can_be_saved_as_standard_json(kind, expected):
    row = score_prediction(example(expected, kind), "1e309")
    assert row["answer_score"]["score"] == 0
    json.dumps(row, allow_nan=False)
